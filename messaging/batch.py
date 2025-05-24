import time
import pathlib
import threading
import requests
import json

from config import BATCH_INTERVAL, FILTER_PROMPT_PATH, SUMMARY_PROMPT_PATH, CORE_REST_ENDPOINT
from messaging.consumer import message_buffer, message_map
from model.processor import process_message_logic

model_lock = threading.Lock()

def start_batch_processor():
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
            except Exception as e:
                print(f"\nОшибка при отправке в Core: {e}")
