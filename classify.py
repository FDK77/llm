import json
import os
import pathlib
import time
import threading
import queue
import requests

import httpx
import pika
import ollama
from flask import Flask, request, jsonify


OLLAMA_CONNECTION_STR = os.environ.get("OLLAMA_CONNECTION_STR", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b")
PROMPT_TEMPLATE_PATH = os.environ.get("PROMPT_TEMPLATE_PATH", "prompt2.txt")
CORE_REST_ENDPOINT = os.environ.get("CORE_REST_ENDPOINT", "http://localhost:8080/llm/results")

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost/")
QUEUE_NAME = "core.to_llm"
BATCH_INTERVAL = 5
MAX_RETRIES = 3


app = Flask(__name__)
ollama_client = ollama.Client(host=OLLAMA_CONNECTION_STR)
setup_done = False
message_buffer = queue.Queue()
message_map = {}
model_lock = threading.Lock()


def wait_for_ollama(client: ollama.Client):
    tries = 10
    while tries:
        try:
            client.ps()
            return
        except httpx.HTTPError:
            tries -= 1
            time.sleep(1)
    raise RuntimeError("Не удалось подключиться к Ollama")


def download_model(client: ollama.Client, model_name: str):
    existing_models = [m.get("name") or m.get("model") for m in client.list()["models"]]
    if model_name not in existing_models:
        print(f"Модель {model_name} не найдена. Скачиваю...")
        client.pull(model_name)


@app.before_request
def ensure_setup():
    global setup_done
    if not setup_done:
        wait_for_ollama(ollama_client)
        download_model(ollama_client, OLLAMA_MODEL)
        setup_done = True


def is_valid_response(response):
    return isinstance(response, dict) and \
           'messageid' in response and \
           'match' in response and \
           'summary' in response

def classify_with_retry(message, prompt_template):
    for attempt in range(MAX_RETRIES):
        result = classify(message, prompt_template)
        if is_valid_response(result):
            return result
        print(f"Попытка {attempt + 1}: Невалидный ответ от модели, повторная отправка...")
        time.sleep(1)
    print("Модель не вернула валидный ответ после 3 попыток")
    return None
def classify(messages: list, prompt_template: str) -> list:
    input_json = json.dumps(messages, ensure_ascii=False, indent=2)
    prompt = prompt_template.replace("$MESSAGE", input_json)

    print("\n======= PROMPT TO MODEL =======")
    print(input_json)
    print("================================")

    api_response = ollama_client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        format="json",
        stream=False,
    )

    response = api_response["response"].strip()

    print("\n======= RAW RESPONSE FROM MODEL =======")
    print(response)
    print("================================")

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Ошибка разбора JSON ответа от модели")
        return []



def rabbitmq_consumer():
    def callback(ch, method, properties, body):
        try:
            msg = json.loads(body)
            print(f"\n Получено сообщение из очереди: {json.dumps(msg, indent=2, ensure_ascii=False)}")
            if isinstance(msg, dict):
                mid = msg["messageid"]
                message_map[mid] = msg
                # подготавливаем только нужные поля для модели
                input_dto = {
                    "messageid": mid,
                    "text": msg["text"],
                    "filter": msg.get("filter", ""),
                    "summary": msg.get("summary", False)
                }
                message_buffer.put(input_dto)
            else:
                print("Получено сообщение не в формате dict, игнорируем")
        except Exception as e:
            print(f"Ошибка обработки сообщения из очереди: {e}")

    connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=True)
    print(" [*] Ожидание сообщений в очереди...")
    channel.start_consuming()


def batch_processor():
    while True:
        time.sleep(BATCH_INTERVAL)
        items = []
        while not message_buffer.empty():
            items.append(message_buffer.get())

        if not items:
            continue

        prompt_template = pathlib.Path(PROMPT_TEMPLATE_PATH).read_text(encoding="utf-8")

        to_send = []
        responses = []

        for message in items:
            with model_lock:
                result = classify_with_retry(message, prompt_template)

            if not isinstance(result, dict):
                print("Ожидался JSON-объект от модели, но получено:", type(result))
                continue

            responses.append(result)

        expected_ids = {msg["messageid"] for msg in items}
        received_ids = {r["messageid"] for r in responses if is_valid_response(r)}

        if expected_ids != received_ids:
            print(f"Не все ответы получены от модели. Ожидалось: {expected_ids}, получено: {received_ids}")
            for msg in items:
                message_buffer.put(msg)
            continue

        for result in responses:
            mid = result.get("messageid")
            match = result.get("match")
            summary = result.get("summary")

            if match is False:
                if mid in message_map:
                    del message_map[mid]
                continue

            full_dto = message_map.get(mid)
            if not full_dto:
                print(f"Не найдено сообщение по messageid={mid}")
                continue

            to_send.append({
                "messageid": mid,
                "chatid": full_dto["chatid"],
                "userid": full_dto["userid"],
                "text": full_dto["text"],
                "timestamp": full_dto.get("timestamp"),
                "summary": summary
            })

            del message_map[mid]

        if to_send:
            print(f"\nОтправка в Core {len(to_send)} сообщений:")
            print(json.dumps(to_send, indent=2, ensure_ascii=False))
            try:
                requests.post(CORE_REST_ENDPOINT, json=to_send, timeout=10)
                print(f"Отправлено {len(to_send)} сообщений в Core")

            except Exception as e:
                print(f"Ошибка при отправке в Core: {e}")



@app.route("/classify", methods=["POST"])
def classify_messages():
    prompt_template = pathlib.Path(PROMPT_TEMPLATE_PATH).read_text(encoding="utf-8")
    messages = request.json.get("messages", [])

    if not isinstance(messages, list):
        return jsonify({"error": "messages должен быть списком"}), 400

    result = classify(messages, prompt_template)
    return jsonify(result), 200


if __name__ == "__main__":
    threading.Thread(target=rabbitmq_consumer, daemon=True).start()
    threading.Thread(target=batch_processor, daemon=True).start()
    app.run(host="0.0.0.0", port=8000, debug=True)
