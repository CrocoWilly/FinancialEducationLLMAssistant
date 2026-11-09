# ssc-broker

FastAPI service between Dify and vLLM. Exposes an OpenAI-compatible API, converts Traditional ↔ Simplified Chinese, and caps how many requests reach the GPU at once.

## Endpoints

| Method | Path | Behaviour |
|---|---|---|
| `GET` | `/health` | Broker status, upstream URL, concurrency cap |
| `GET` | `/v1/models` | Proxied to vLLM |
| `POST` | `/v1/chat/completions` (also `/chat/completions`) | Proxied to vLLM, stream or non-stream; model name forced to `DEFAULT_MODEL`; input 繁→簡, output 簡→繁 (`s2twp`) |
| `GET` / `HEAD` | `/v1/chat/completions` | Returns `200` without calling vLLM (Dify's connectivity probe; vLLM would return 404) |

## Configuration

Set via environment variables. `docker-compose.yml` values override the `Dockerfile` defaults; `UPSTREAM_LLM_BASE` comes from `.env` (copy `.env.example`).

| Variable | compose | Dockerfile default | Meaning |
|---|---|---|---|
| `UPSTREAM_LLM_BASE` | from `.env` | `http://127.0.0.1:8000/v1` | vLLM base URL, must include `/v1`, no trailing slash |
| `DEFAULT_MODEL` | – | `ssc-FinLLM` (in code) | Model name forced on every request |
| `BROKER_MAX_INFLIGHT` | `16` | `8` | Max concurrent upstream requests |
| `RATE_LIMIT_SECONDS` | `0` (off) | `10` | Min seconds between requests per client IP |
| `QUEUE_WAIT_TIMEOUT` | `60` | `45` | Seconds to wait for a concurrency slot |
| `UPSTREAM_CONNECT_TIMEOUT` | `10` | `10` | Seconds to connect to vLLM |
| `UPSTREAM_READ_TIMEOUT` | `120` | `120` | Max seconds between two reads from vLLM (per chunk when streaming) |

## Error responses

| Situation | Response |
|---|---|
| Per-IP rate limit hit | `429` |
| No concurrency slot within `QUEUE_WAIT_TIMEOUT` | `503` + `Retry-After: 5` |
| vLLM unreachable | `502` |
| vLLM timeout before streaming starts / non-stream | `504` |
| vLLM timeout or disconnect mid-stream | final SSE event `data: {"error": ...}` |
| vLLM returns non-200 | same status, error body passed through |

## Run

```bash
cp .env.example .env          # set UPSTREAM_LLM_BASE to the GPU machine's Tailscale address
docker compose up -d --build  # joins Dify's docker_default network; Dify uses http://broker:8080/v1
curl http://localhost:8080/health
```

Without Docker:

```bash
pip install -r requirements.txt
UPSTREAM_LLM_BASE=http://127.0.0.1:8000/v1 uvicorn broker:app --port 8080
```
