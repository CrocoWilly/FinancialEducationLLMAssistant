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
| `RATE_LIMIT_PER_MIN` | `10` | `0` (off) | Per-user token bucket: average requests per minute (see below) |
| `RATE_LIMIT_BURST` | `5` | `5` | Per-user token bucket: requests allowed back to back |
| `QUEUE_WAIT_TIMEOUT` | `60` | `45` | Seconds to wait for a concurrency slot |
| `UPSTREAM_CONNECT_TIMEOUT` | `10` | `10` | Seconds to connect to vLLM |
| `UPSTREAM_READ_TIMEOUT` | `120` | `120` | Max seconds between two reads from vLLM (per chunk when streaming) |

## Note on rate limiting

The concurrency cap protects the GPU; the rate limit keeps one user from taking every slot. Each user gets a token bucket: up to `RATE_LIMIT_BURST` requests back to back, refilled at `RATE_LIMIT_PER_MIN`. A request over the limit gets `429` with `Retry-After` before it can queue for a slot.

Users are told apart by the request's `user` field. Dify fills it with its end-user ID when it calls an OpenAI-compatible model (`user=user_id` in Dify 1.8.1's LLM node; `data["user"] = user` in the plugin SDK). Requests without it fall back to the client IP, which behind Dify is the Dify container, so they share one bucket. Idle buckets are pruned once a minute.

The code default is off, so local load tests (one IP, no `user` field) aren't throttled; `docker-compose.yml` turns it on for production. A real student pauses about 24 s between questions (median in production), so 10 per minute with a burst of 5 only stops scripted traffic.

## Error responses

| Situation | Response |
|---|---|
| Missing or wrong API key | `401` + `WWW-Authenticate: Bearer` (checked before rate limit and queue) |
| Per-user rate limit hit | `429` + `Retry-After` (seconds until the next token) |
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
