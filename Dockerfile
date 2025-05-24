FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc && \
    pip install --upgrade pip

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "./main.py"]
