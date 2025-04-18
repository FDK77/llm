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
PROMPT_TEMPLATE_PATH = os.environ.get("PROMPT_TEMPLATE_PATH", "prompt.txt")

SAMPLE_DESCRIPTION = """
Join us on an enchanting journey into the world of birds, where we explore the beauty, diversity, and wonder of these incredible creatures. From the vibrant feathers of tropical parrots to the majestic flight of eagles, this video showcases some of the most stunning and unique bird species from around the globe.

Learn fascinating facts about their behaviors, habitats, and the vital roles they play in our ecosystems. Whether you're a bird enthusiast or simply curious about the natural world, this video will captivate your imagination and deepen your appreciation for these winged wonders.

Don't forget to like, comment, and subscribe for more wildlife adventures! 🦅✨
"""

SAMPLE_COMMENTS = [
    "This video was so relaxing! The footage of the eagles soaring was breathtaking.",
    "I've always loved birds, but this video taught me so many new things! Thank you!",
    "The variety of bird species you featured is incredible. Nature is truly amazing!",
    "I never realized how important birds are to our ecosystems. Great educational content!",
    "The parrot at 3:45 is stunning! Those colors are unreal!",
    "Amazing video! The music paired with the bird footage was perfect.",
    "I wish I could have a backyard full of these beautiful birds. So inspiring!",
    "Birds are such fascinating creatures. Thank you for sharing this wonderful content!",
    "This makes me want to go birdwatching this weekend. Anyone else?",
    "The way you captured the details of their feathers and movements is just phenomenal!",
    "More pics in my bio.",
    "Giving away $1000, dm me on whatsapp",
    "Birds are stupid and people that like birds are stupid.",
    "Check out the Bird Watcher 3000 binoculars, they are the best product for watching birds.",
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
    comment: str, description: str, prompt_template: str, ollama_client: ollama.Client
) -> str:
    comment = comment.replace('"', "'")
    description = description.replace('"', "'")
    prompt = prompt_template
    prompt = prompt.replace("$COMMENT", comment)
    prompt = prompt.replace("$DESCRIPTION", description)

    api_response = ollama_client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        format="json",
        stream=False,
    )
    response = api_response["response"]
    data = json.loads(response)
    breaks_rules = data["breaks_rules"]
    if not isinstance(breaks_rules, bool):
        raise TypeError("expected bool")
    return breaks_rules


def main():
    ollama_client = ollama.Client(host=OLLAMA_CONNECTION_STR)
    wait_for_ollama(ollama_client)
    download_model(ollama_client, OLLAMA_MODEL)
    prompt_template = pathlib.Path(PROMPT_TEMPLATE_PATH).read_text()
    description = SAMPLE_DESCRIPTION
    comments = SAMPLE_COMMENTS

    for comment in comments:
        breaks_rules = classify(comment, description, prompt_template, ollama_client)
        if breaks_rules:
            print(f"Bad: {comment[:80]!r}")


if __name__ == "__main__":
    main()
