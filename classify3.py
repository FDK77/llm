import json
import os
import pathlib
import time
from typing import List

import httpx
import ollama
from flask import Flask, request, jsonify

OLLAMA_CONNECTION_STR = os.environ.get("OLLAMA_CONNECTION_STR", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
PROMPT_TEMPLATE_PATH = os.environ.get("PROMPT_TEMPLATE_PATH", "prompt.txt")

app = Flask(__name__)
ollama_client = ollama.Client(host=OLLAMA_CONNECTION_STR)
setup_done = False


def wait_for_ollama(client: ollama.Client):
    tries = 10
    while True:
        try:
            client.ps()
            return
        except httpx.HTTPError:
            if tries:
                tries -= 1
                time.sleep(1)
            else:
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


def classify(message: dict, prompt_template: str, client: ollama.Client) -> dict:
    message_str = json.dumps(message, ensure_ascii=False).replace('"', "'")
    prompt = prompt_template.replace("$MESSAGE", message_str)

    print("\n======= PROMPT TO MODEL =======")
    print(prompt)
    print("================================")

    api_response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        format="json",
        stream=False,
    )

    response = api_response["response"].strip()

    print("\n======= RAW RESPONSE =======")
    print(response)
    print("================================")

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "messageid": message.get("messageid", -1),
            "match": None,
            "summary": None,
            "error": "Ошибка разбора JSON"
        }


@app.route('/classify', methods=['POST'])
def classify_messages():
    prompt_template = pathlib.Path(PROMPT_TEMPLATE_PATH).read_text(encoding="utf-8")
    messages = request.json.get("messages", [])

    if not isinstance(messages, list):
        return jsonify({"error": "messages должен быть списком"}), 400

    results = []
    for message in messages:
        result = classify(message, prompt_template, ollama_client)
        results.append(result)

    return jsonify(results), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
