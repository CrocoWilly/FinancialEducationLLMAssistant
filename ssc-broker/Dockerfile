FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY broker.py /app/broker.py
ENV UPSTREAM_LLM_BASE=http://127.0.0.1:8000/v1 \
    BROKER_MAX_INFLIGHT=8 \
    RATE_LIMIT_SECONDS=10 \
    QUEUE_WAIT_TIMEOUT=45 \
    OPENCC_USER_DICT=/app/opencc_user.txt
EXPOSE 8080
CMD ["uvicorn","broker:app","--host","0.0.0.0","--port","8080"]
