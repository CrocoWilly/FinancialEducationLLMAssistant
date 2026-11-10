# ssc-broker

FastAPI service between Dify and vLLM. Exposes an OpenAI-compatible API, converts Traditional ↔ Simplified Chinese, and caps how many requests reach the GPU at once.

## Endpoints

| Method | Path | Behaviour |
|---|---|---|
| `GET` | `/health` | Broker status, upstream URL, concurrency cap |
| `GET` | `/v1/models` | Proxied to vLLM (requires API key) |
| `POST` | `/v1/chat/completions` (also `/chat/completions`) | Requires API key. Proxied to vLLM, stream or non-stream; model name forced to `DEFAULT_MODEL`; input 繁→簡, output 簡→繁 (`s2twp`) |
| `GET` / `HEAD` | `/v1/chat/completions` | Returns `200` without calling vLLM (Dify's connectivity probe; vLLM would return 404) |

## Configuration

Set via environment variables. `docker-compose.yml` values override the `Dockerfile` defaults; `UPSTREAM_LLM_BASE` comes from `.env` (copy `.env.example`).

| Variable | compose | Dockerfile default | Meaning |
|---|---|---|---|
| `UPSTREAM_LLM_BASE` | from `.env` | `http://127.0.0.1:8000/v1` | vLLM base URL, must include `/v1`, no trailing slash |
| `BROKER_API_KEY` | from `.env` (required) | – (empty = no auth, logs a warning) | Clients must send `Authorization: Bearer <key>` to `/v1/*` |
| `DEFAULT_MODEL` | – | `ssc-FinLLM` (in code) | Model name forced on every request |
| `BROKER_MAX_INFLIGHT` | `32` | `32` | Max concurrent upstream requests (32 measured best on one RTX 3090; size it to your GPU's KV cache) |
| `RATE_LIMIT_SECONDS` | `0` (off) | `0` (off) | Min seconds between requests per client IP — keep off behind Dify (see below) |
| `QUEUE_WAIT_TIMEOUT` | `60` | `45` | Seconds to wait for a concurrency slot |
| `UPSTREAM_CONNECT_TIMEOUT` | `10` | `10` | Seconds to connect to vLLM |
| `UPSTREAM_READ_TIMEOUT` | `120` | `120` | Max seconds between two reads from vLLM (per chunk when streaming) |

## Note on rate limiting

The per-IP limit keys on the address of whoever connects to the broker. Behind Dify that is always the Dify API container, so every user shares one key and `RATE_LIMIT_SECONDS=10` would mean *one request per 10 s for the whole system*. That's why it is off by default. Per-user limits belong in the layer that sees real client IPs: Nginx (`limit_req_zone $binary_remote_addr ...`) or Dify itself.

## Error responses

| Situation | Response |
|---|---|
| Missing or wrong API key | `401` + `WWW-Authenticate: Bearer` (checked before rate limit and queue) |
| Per-IP rate limit hit | `429` |
| No concurrency slot within `QUEUE_WAIT_TIMEOUT` | `503` + `Retry-After: 5` |
| vLLM unreachable | `502` |
| vLLM timeout before streaming starts / non-stream | `504` |
| vLLM timeout or disconnect mid-stream | final SSE event `data: {"error": ...}` |
| vLLM returns non-200 | same status, error body passed through |

## Run

```bash
cp .env.example .env          # set UPSTREAM_LLM_BASE and BROKER_API_KEY
docker compose up -d --build  # joins Dify's docker_default network; Dify uses http://broker:8080/v1
curl http://localhost:8080/health
```

In Dify, add the broker as an **OpenAI-API-compatible** model provider with API base `http://broker:8080/v1` and the same `BROKER_API_KEY` as the API key; Dify then sends it as a Bearer token.

The port is published only on `127.0.0.1`, so the broker is reachable from Dify (Docker network) and from the EC2 host itself, but not from the internet.

Logs (one line per proxied request, plus a startup warning if no API key is set) go to stderr: `docker compose logs -f broker`.

Without Docker:

```bash
pip install -r requirements.txt
UPSTREAM_LLM_BASE=http://127.0.0.1:8000/v1 BROKER_API_KEY=dev-key uvicorn broker:app --port 8080
```
