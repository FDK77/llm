import json
import os
import pathlib
import time

import httpx
import ollama

OLLAMA_CONNECTION_STR = os.environ.get(
    "OLLAMA_CONNECTION_STR", "http://localhost:11434"
)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
PROMPT_TEMPLATE_PATH = os.environ.get("PROMPT_TEMPLATE_PATH", "promt3.txt")


SAMPLE_MASSAGES = [
""" {
    "messageid": 123,
    "text": "Сегодня собираемся в парке с ребятами, будем играть в шахматы. Приходите все кто хочет!",
    "filter": "Хочу видеть сообщения с фильмами",
    "summary": false
} """
]


def wait_for_ollama(ollama_client: ollama.Client):
    tries = 10
    while True:
        try:
            ollama_client.ps()
            break
        except httpx.HTTPError:
            if tries:
                tries -= 1
                time.sleep(1)
            else:
                raise


def main():
    ollama_client = ollama.Client(host=OLLAMA_CONNECTION_STR)
    wait_for_ollama(ollama_client)


def download_model(client: ollama.Client, model_name: str):
    existing_models = [m.get("name") or m.get("model") for m in client.list()["models"]]
    if model_name not in existing_models:
        print(f"Модель {model_name} не найдена. Скачиваю...")
        client.pull(model_name)


def classify(
    message:str, prompt_template: str, ollama_client: ollama.Client
) -> str:
    message = message.replace('"', "'")
    prompt = prompt_template
    prompt = prompt.replace("$MESSAGE", message)


    api_response = ollama_client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        format="json",
        stream=False,
    )
    response = api_response["response"]
    data = json.loads(response)
    messageid = data["messageid"]
    match = data["match"]
    summary = data["summary"]
    return messageid, match, summary


def main():
    ollama_client = ollama.Client(host=OLLAMA_CONNECTION_STR)
    wait_for_ollama(ollama_client)
    download_model(ollama_client, OLLAMA_MODEL)
    prompt_template = pathlib.Path(PROMPT_TEMPLATE_PATH).read_text(encoding="UTF-8")
    massages = SAMPLE_MASSAGES

    for massage in massages:
        mid,match,summary = classify(massage, prompt_template, ollama_client)
        print (mid,match,summary)


if __name__ == "__main__":
    main()
