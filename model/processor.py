import json
from model.client import call_with_retry
from config import CHUNK_SIZE

def process_message_logic(message: dict, filter_prompt: str, summary_prompt: str) -> dict:
    result = {
        "messageid": message["messageid"],
        "match": None,
        "summary": None,
    }
    print(f"\nОбработка сообщения {message['messageid']}:")
    filters = message.get("filters", [])
    if not filters:
        return result
    for offset in range(0, len(filters), CHUNK_SIZE):
        sub_filters = filters[offset : offset + CHUNK_SIZE]
        filters_without_summary = [{"id": f["id"], "value": f["value"]} for f in sub_filters]
        filter_input = json.dumps(
            {
                "messageid": message["messageid"],
                "text": message["text"],
                "filters": filters_without_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        prompt = filter_prompt.replace("$MESSAGE", filter_input)
        print("\n Отправка запроса в LLM")
        filter_result = call_with_retry(prompt)
        if filter_result is None or filter_result.get("match") is None:
            continue
        valid_ids = {f["id"] for f in sub_filters}
        if filter_result["match"] not in valid_ids:
            filter_result = call_with_retry(prompt)
            if filter_result is None or filter_result.get("match") not in valid_ids:
                print("Повтор не дал корректного ID :( – пропуск")
                continue

        result["match"] = filter_result["match"]
        break
    if result["match"] is None:
        return result
    need_summary = False
    for f in filters:
        if f["id"] == result["match"] and f.get("summary"):
            need_summary = True
            break
    if need_summary:
        summary_input = json.dumps(
            {
                "messageid": message["messageid"],
                "text": message["text"],
            },
            ensure_ascii=False,
            indent=2,
        )
        prompt = summary_prompt.replace("$MESSAGE", summary_input)
        summary_result = call_with_retry(prompt)
        if summary_result:
            result["summary"] = summary_result.get("summary")
            if result["summary"]:
                print(f"Суммари создан: {result['summary']}")
            else:
                print("Суммари не требуется")
        else:
            print("Ошибка при создании суммари")
    print("\nИтоговый результат:")
    print(f"messageid: {result['messageid']}")
    print(f"match: {result['match']}")
    print(f"summary: {result['summary']}")
    return result
