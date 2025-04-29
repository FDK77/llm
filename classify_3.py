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
FILTER_PROMPT_PATH = os.environ.get("FILTER_PROMPT_PATH", "prompt_filter.txt")
SUMMARY_PROMPT_PATH = os.environ.get("SUMMARY_PROMPT_PATH", "prompt_summary.txt")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b")
CORE_REST_ENDPOINT = os.environ.get("CORE_REST_ENDPOINT", "http://localhost:8080/llm/results")

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost/")
QUEUE_NAME = "core.to_llm"
BATCH_INTERVAL = 1
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

def call_ollama(prompt: str) -> str:
    api_response = ollama_client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        format="json",
        stream=False,
    )
    return api_response["response"].strip()

def call_with_retry(prompt: str) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            raw = call_ollama(prompt)
            return json.loads(raw)
        except Exception as e:
            print(f"Попытка {attempt+1} не удалась: {e}")
            time.sleep(1)
    return None

def process_message_logic(message: dict, filter_prompt: str, summary_prompt: str) -> dict:
    result = {
        "messageid": message["messageid"],
        "match": None,
        "summary": None
    }

    if message.get("filter"):
        filter_input = json.dumps({
            "messageid": message["messageid"],
            "text": message["text"],
            "filters": message["filter"]
        }, ensure_ascii=False)
        prompt = filter_prompt.replace("$MESSAGE", filter_input)
        filter_result = call_with_retry(prompt)
        if filter_result is None or filter_result.get("match") is None:
            return result  # не прошёл фильтрацию — summary не делаем
        result["match"] = filter_result["match"]
    else:
        result["match"] = None

    if message.get("summary") and (result["match"] is not None or message.get("filter") is None):
        summary_input = json.dumps({
            "messageid": message["messageid"],
            "text": message["text"]
        }, ensure_ascii=False)
        prompt = summary_prompt.replace("$MESSAGE", summary_input)
        summary_result = call_with_retry(prompt)
        if summary_result:
            result["summary"] = summary_result.get("summary")

    return result

def rabbitmq_consumer():
    def callback(ch, method, properties, body):
        try:
            msg = json.loads(body)
            message_buffer.put(msg)
            message_map[msg["messageid"]] = msg
        except Exception as e:
            print(f"Ошибка обработки сообщения из очереди: {e}")

    connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=True)
    print("[*] Ожидание сообщений в очереди...")
    channel.start_consuming()

def batch_processor():
    while True:
        time.sleep(BATCH_INTERVAL)
        items = []
        while not message_buffer.empty():
            items.append(message_buffer.get())

        if not items:
            continue

        filter_prompt = pathlib.Path(FILTER_PROMPT_PATH).read_text(encoding="utf-8")
        summary_prompt = pathlib.Path(SUMMARY_PROMPT_PATH).read_text(encoding="utf-8")

        to_send = []
        for item in items:
            with model_lock:
                result = process_message_logic(item, filter_prompt, summary_prompt)

            full_dto = message_map.get(result["messageid"])
            if not full_dto:
                continue

            to_send.append({
                "messageid": result["messageid"],
                "chatid": full_dto["chatid"],
                "userid": full_dto["userid"],
                "text": full_dto["text"],
                "timestamp": full_dto.get("timestamp"),
                "summary": result["summary"],
                "match": result["match"]
            })

        if to_send:
            print(f"\nОтправка в Core {len(to_send)} сообщений:")
            print(json.dumps(to_send, indent=2, ensure_ascii=False))
            try:
                requests.post(CORE_REST_ENDPOINT, json=to_send, timeout=10)
                print(f"✅ Отправлено {len(to_send)} сообщений в Core")
            except Exception as e:
                print(f"❌ Ошибка при отправке в Core: {e}")

@app.route("/process", methods=["POST"])
def process_api():
    filter_prompt = pathlib.Path(FILTER_PROMPT_PATH).read_text(encoding="utf-8")
    summary_prompt = pathlib.Path(SUMMARY_PROMPT_PATH).read_text(encoding="utf-8")

    messages = request.json
    if not isinstance(messages, list):
        return jsonify({"error": "ожидался список сообщений"}), 400

    results = []
    for message in messages:
        result = process_message_logic(message, filter_prompt, summary_prompt)
        full_dto = message.copy()
        full_dto["summary"] = result["summary"]
        full_dto["match"] = result["match"]
        results.append(full_dto)

    return jsonify(results), 200

if __name__ == "__main__":
    threading.Thread(target=rabbitmq_consumer, daemon=True).start()
    threading.Thread(target=batch_processor, daemon=True).start()
    app.run(host="0.0.0.0", port=8000, debug=True)
