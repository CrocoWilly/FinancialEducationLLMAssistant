"""
Mock OpenAI-compatible upstream (stands in for vLLM, no GPU needed).

Streams fake Simplified-Chinese tokens at a fixed rate, and tracks how many
requests are in flight at once so you can check whether the broker's
concurrency cap actually holds.

    uvicorn mock_upstream:app --port 8000
    MOCK_TOKENS_PER_SEC=60 MOCK_TTFT_MS=200 uvicorn mock_upstream:app --port 8000

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (stream / non-stream, honours max_tokens and
                                 stream_options.include_usage)
    GET  /stats                 current / peak in-flight requests, totals
                                (MOCK_FAIL_STATUS=500 makes every chat request fail)
    POST /stats/reset
"""
import asyncio
import json
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

TOKENS_PER_SEC = float(os.getenv("MOCK_TOKENS_PER_SEC", "60"))
TTFT_MS = float(os.getenv("MOCK_TTFT_MS", "200"))
DEFAULT_MAX_TOKENS = int(os.getenv("MOCK_DEFAULT_MAX_TOKENS", "256"))
MODEL = os.getenv("MOCK_MODEL", "ssc-FinLLM")
# If non-zero, every chat request fails with this HTTP status (for error-path tests).
FAIL_STATUS = int(os.getenv("MOCK_FAIL_STATUS", "0"))

# Simplified Chinese on purpose, so the broker's 简→繁 conversion is exercised.
VOCAB = list("股票债券基金风险收益资产负债利率通货膨胀投资组合分散价格市场经济银行")

app = FastAPI()
stats = {"inflight": 0, "peak_inflight": 0, "total": 0, "completed": 0}
last_request: dict = {}  # body of the most recent chat request, as received


def reset():
    """Clear all counters and the recorded request (used between tests)."""
    stats.update(inflight=0, peak_inflight=0, total=0, completed=0)
    last_request.clear()


def _enter():
    stats["inflight"] += 1
    stats["total"] += 1
    stats["peak_inflight"] = max(stats["peak_inflight"], stats["inflight"])


def _leave():
    stats["inflight"] -= 1
    stats["completed"] += 1


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}


@app.get("/stats")
async def get_stats():
    return stats


@app.get("/health")
async def health():
    """Liveness endpoint, like vLLM's (scripts wait on it before benchmarking)."""
    return {}


@app.get("/metrics")
async def metrics():
    """A few gauges in vLLM's Prometheus format, so metric scraping can be tested without a GPU."""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        f'vllm:num_requests_running{{model_name="{MODEL}"}} {stats["inflight"]}\n'
        f'vllm:num_requests_waiting{{model_name="{MODEL}"}} 0\n'
        f'vllm:kv_cache_usage_perc{{model_name="{MODEL}"}} {min(stats["inflight"] / 20, 1.0)}\n'
        f'vllm:num_preemptions_total{{model_name="{MODEL}"}} 0\n'
    )


@app.post("/stats/reset")
async def reset_stats():
    stats.update(inflight=stats["inflight"], peak_inflight=stats["inflight"],
                 total=0, completed=0)
    return stats


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    last_request.clear()
    last_request.update(body)
    if FAIL_STATUS:
        return JSONResponse({"error": {"message": "mock failure"}}, status_code=FAIL_STATUS)
    n_tokens = int(body.get("max_tokens") or DEFAULT_MAX_TOKENS)
    stream = bool(body.get("stream", False))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    usage = {"prompt_tokens": 10, "completion_tokens": n_tokens,
             "total_tokens": 10 + n_tokens}

    def chunk(delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    if not stream:
        _enter()
        try:
            await asyncio.sleep(TTFT_MS / 1000 + n_tokens / TOKENS_PER_SEC)
        finally:
            _leave()
        text = "".join(VOCAB[i % len(VOCAB)] for i in range(n_tokens))
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "length",
                         "message": {"role": "assistant", "content": text}}],
            "usage": usage,
        })

    async def gen():
        _enter()
        try:
            await asyncio.sleep(TTFT_MS / 1000)
            yield f"data: {json.dumps(chunk({'role': 'assistant', 'content': ''}))}\n\n"
            interval = 1.0 / TOKENS_PER_SEC
            start = time.monotonic()
            for i in range(n_tokens):
                # schedule against the start time so the rate doesn't drift
                delay = start + i * interval - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                tok = VOCAB[i % len(VOCAB)]
                yield f"data: {json.dumps(chunk({'content': tok}), ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps(chunk({}, 'length'))}\n\n"
            if include_usage:
                u = {"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": MODEL, "choices": [], "usage": usage}
                yield f"data: {json.dumps(u)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            _leave()

    return StreamingResponse(gen(), media_type="text/event-stream")
