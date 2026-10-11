# Bestand plaatsen als: Dockerfile (root van de repository transcription-workflow)
FROM python:3.12-slim

WORKDIR /app

COPY worker.py ./worker.py

RUN pip install --no-cache-dir \
    "mistralai>=2.0.0" \
    "mistralai-workflows>=3.10" \
    "httpx>=0.27" \
    "pydantic>=2.0"

ENV PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]
