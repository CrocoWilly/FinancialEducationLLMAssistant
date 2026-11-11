FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY broker.py /app/broker.py
ENV UPSTREAM_LLM_BASE=http://127.0.0.1:8000/v1 \
    BROKER_MAX_INFLIGHT=32 \
    RATE_LIMIT_PER_MIN=0 \
    RATE_LIMIT_BURST=5 \
    QUEUE_WAIT_TIMEOUT=45 \
    UPSTREAM_CONNECT_TIMEOUT=10 \
    UPSTREAM_READ_TIMEOUT=120
EXPOSE 8080
CMD ["uvicorn","broker:app","--host","0.0.0.0","--port","8080"]
