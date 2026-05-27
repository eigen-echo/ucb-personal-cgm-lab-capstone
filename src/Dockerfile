FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY train_models.py .

ENV PYTHONPATH=/app
ENV DB_PATH=/app/data/cgm_app.db
ENV DATA_RAW_DIR=/app/data/raw
ENV DATA_PROCESSED_DIR=/app/data/processed
ENV MODELS_DIR=/app/models
ENV SCRIPTS_DIR=/app/scripts

RUN mkdir -p /app/data/raw /app/data/processed /app/models

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
