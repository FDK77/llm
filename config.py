import os

# Настройки Ollama
OLLAMA_CONNECTION_STR = os.environ.get("OLLAMA_CONNECTION_STR", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b")

# Пути к промптам
FILTER_PROMPT_PATH = os.environ.get("FILTER_PROMPT_PATH", "prompts/prompt_filter.txt")
SUMMARY_PROMPT_PATH = os.environ.get("SUMMARY_PROMPT_PATH", "prompts/prompt_summary.txt")

# Core-сервис
CORE_REST_ENDPOINT = os.environ.get("CORE_REST_ENDPOINT", "http://localhost:8080/llm/results")

# RabbitMQ
RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost/")
QUEUE_NAME = "core.to_llm"

# Параметры обработки
BATCH_INTERVAL = 1
MAX_RETRIES = 3
CHUNK_SIZE = 2
