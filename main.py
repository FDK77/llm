from flask import Flask, request, jsonify
import threading
import pathlib

from config import FILTER_PROMPT_PATH, SUMMARY_PROMPT_PATH
from model.client import wait_for_ollama, download_model
from model.processor import process_message_logic
from messaging.consumer import start_consumer
from messaging.batch import start_batch_processor

from config import OLLAMA_MODEL

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_api():
    filter_prompt = pathlib.Path(FILTER_PROMPT_PATH).read_text(encoding="utf-8")
    summary_prompt = pathlib.Path(SUMMARY_PROMPT_PATH).read_text(encoding="utf-8")

    messages = request.json
    if not isinstance(messages, list):
        return jsonify({"error": "ожидался список сообщений"}), 400

    print(f"\n Получено {len(messages)} сообщений для обработки")

    results = []
    for message in messages:
        result = process_message_logic(message, filter_prompt, summary_prompt)
        results.append({
            "messageid": message["messageid"],
            "text": message["text"],
            "match": result["match"],
            "summary": result["summary"]
        })

    return jsonify(results), 200

if __name__ == "__main__":
    wait_for_ollama()
    download_model(OLLAMA_MODEL)

    threading.Thread(target=start_consumer, daemon=True).start()
    threading.Thread(target=start_batch_processor, daemon=True).start()

    app.run(host="0.0.0.0", port=8000, debug=True)
