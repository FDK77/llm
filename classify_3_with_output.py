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
    # Добавляем отладочную информацию: выводим промпт
    print("\n🔹 Запрос к модели:")
    print(prompt)

    api_response = ollama_client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        format="json",
        stream=False,
    )

    # Добавляем отладочную информацию: выводим ответ
    print("\n🔸 Ответ от модели:")
    print(api_response["response"].strip())

    return api_response["response"].strip()


def call_with_retry(prompt: str) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            raw = call_ollama(prompt)
            return json.loads(raw)
        except Exception as e:
            print(f"Попытка {attempt + 1} не удалась: {e}")
            time.sleep(1)
    return None


def process_message_logic(message: dict, filter_prompt: str, summary_prompt: str) -> dict:
    result = {
        "messageid": message["messageid"],
        "match": None,
        "summary": None
    }

    print(f"\n📩 Обработка сообщения {message['messageid']}:")
    print(f"Текст: {message['text']}")

    if message.get("filters"):
        filters_without_summary = [
            {"id": f["id"], "value": f["value"]}
            for f in message.get("filters", [])
        ]
        filter_input = json.dumps({
            "messageid": message["messageid"],
            "text": message["text"],
            "filters": filters_without_summary
        }, ensure_ascii=False, indent=2)
        prompt = filter_prompt.replace("$MESSAGE", filter_input)

        print("\n🔍 Применение фильтра...")
        filter_result = call_with_retry(prompt)

        if filter_result is None or filter_result.get("match") is None:
            print("❌ Не прошел фильтрацию")
            return result

        valid_filter_ids = {f["id"] for f in message.get("filters", [])}
        if filter_result["match"] is not None and filter_result["match"] not in valid_filter_ids:
            print(f"❌ Ошибка фильтрации: LLM вернула несуществующий ID {filter_result['match']}, перезапрашиваем...")
            filter_result = call_with_retry(prompt)
            if filter_result is None or filter_result.get("match") is None or \
                    (filter_result["match"] is not None and filter_result["match"] not in valid_filter_ids):
                print("❌ Перезапрос фильтрации не дал правильного результата")
                return result

        result["match"] = filter_result["match"]
        print(f"✅ Прошел фильтрацию, совпадение с фильтром ID: {result['match']}")
    else:
        result["match"] = None
        print("⚠️ Фильтры не указаны")

    # Новая логика: определяем нужно ли делать суммаризацию
    need_summary = False
    if result["match"] is not None and message.get("filters"):
        for f in message["filters"]:
            if f["id"] == result["match"] and f.get("summary"):
                need_summary = True
                break

    if need_summary:
        summary_input = json.dumps({
            "messageid": message["messageid"],
            "text": message["text"]
        }, ensure_ascii=False, indent=2)

        prompt = summary_prompt.replace("$MESSAGE", summary_input)

        print("\n📝 Создание сводки...")
        summary_result = call_with_retry(prompt)

        if summary_result:
            result["summary"] = summary_result.get("summary")
            if result["summary"]:
                print(f"✅ Сводка создана: {result['summary']}")
            else:
                print("ℹ️ Сводка не требуется")
        else:
            print("❌ Ошибка при создании сводки")

    print("\n📊 Итоговый результат:")
    print(f"messageid: {result['messageid']}")
    print(f"match: {result['match']}")
    print(f"summary: {result['summary']}")

    return result



def on_message_callback(ch, method, properties, body):
    try:
        msg = json.loads(body)
        message_buffer.put(msg)
        message_map[msg["messageid"]] = msg
    except Exception as e:
        print(f"Ошибка обработки сообщения из очереди: {e}")

def rabbitmq_consumer():
    while True:
        try:
            connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message_callback, auto_ack=True)
            print("[*] Ожидание сообщений в очереди...")
            channel.start_consuming()
        except Exception as e:
            print(f"❗ Ошибка в rabbitmq_consumer: {e}. Переподключение через 5 секунд...")
            time.sleep(5)


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

            if result["match"] is None:
                continue

            full_dto = message_map.get(result["messageid"])
            if not full_dto:
                continue

            # Сохраняем исходный формат для Core
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
            print(f"\n📤 Отправка в Core {len(to_send)} сообщений:")
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

    print(f"\n📥 Получено {len(messages)} сообщений для обработки")

    results = []
    for message in messages:
        result = process_message_logic(message, filter_prompt, summary_prompt)

        # Упрощаем структуру ответа только для API
        simplified_result = {
            "messageid": message["messageid"],
            "text": message["text"],
            "match": result["match"],
            "summary": result["summary"]
        }
        results.append(simplified_result)

    return jsonify(results), 200


if __name__ == "__main__":
    threading.Thread(target=rabbitmq_consumer, daemon=True).start()
    threading.Thread(target=batch_processor, daemon=True).start()
    app.run(host="0.0.0.0", port=8000, debug=True)