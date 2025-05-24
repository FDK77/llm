import time
import json
import httpx
import ollama

from config import OLLAMA_CONNECTION_STR, OLLAMA_MODEL, MAX_RETRIES

client = ollama.Client(host=OLLAMA_CONNECTION_STR)

def wait_for_ollama():
    tries = 10
    while tries:
        try:
            client.ps()
            return
        except httpx.HTTPError:
            tries -= 1
            time.sleep(1)
    raise RuntimeError("Не удалось подключиться к Ollama")

def download_model(model_name: str):
    existing_models = [m.get("name") or m.get("model") for m in client.list()["models"]]
    if model_name not in existing_models:
        print(f"Модель {model_name} не найдена. Скачиваю...")
        client.pull(model_name)

def call_ollama(prompt: str) -> str:
    print(prompt)

    api_response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        format="json",
        stream=False,
    )

    print("\nОтвет от модели:")
    print(api_response["response"].strip())

    return api_response["response"].strip()

def call_with_retry(prompt: str) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            raw = call_ollama(prompt)
            return json.loads(raw)
        except Exception as e:
            print(f"Попытка {attempt + 1} не удалась: {e}")
            time.sleep(1)
    return None
