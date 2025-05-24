import json
import time
import pika
import queue

from config import RABBITMQ_URL, QUEUE_NAME

message_buffer = queue.Queue()
message_map = {}

def on_message_callback(ch, method, properties, body):
    try:
        msg = json.loads(body)
        message_buffer.put(msg)
        message_map[msg["messageid"]] = msg
    except Exception as e:
        print(f"Ошибка обработки сообщения из очереди: {e}")

def start_consumer():
    while True:
        try:
            connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message_callback, auto_ack=True)
            print("Ожидание сообщений в очереди...")
            channel.start_consuming()
        except Exception as e:
            print(f"Ошибка : {e}. Переподключение...")
            time.sleep(5)
