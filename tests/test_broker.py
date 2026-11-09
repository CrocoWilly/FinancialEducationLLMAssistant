"""
Tests for ssc-broker. Each test starts a real broker in front of a mock vLLM
(see conftest.py) and checks one behaviour through plain HTTP requests.
"""
import asyncio
import json

import httpx
import pytest

import broker
import mock_upstream
from conftest import free_port

TRADITIONAL_QUESTION = "債券的風險是什麼"
SIMPLIFIED_QUESTION = "债券的风险是什么"


# ---------- helpers ----------

def chat_body(*, stream, max_tokens=20, text=TRADITIONAL_QUESTION, model="gpt-4o-mini"):
    return {"model": model, "stream": stream, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": text}]}


def read_stream(response) -> tuple[str, list[dict]]:
    """Collect the SSE events of a streaming response: (joined content, all events)."""
    events, parts = [], []
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        events.append(event)
        for choice in event.get("choices") or []:
            parts.append((choice.get("delta") or {}).get("content") or "")
    return "".join(parts), events


def chat(base_url, *, stream, **kwargs) -> tuple[int, str, list[dict]]:
    """Send one chat request; return (status code, answer text, SSE events)."""
    body = chat_body(stream=stream, **kwargs)
    with httpx.Client(timeout=30) as client:
        if not stream:
            r = client.post(f"{base_url}/chat/completions", json=body)
            text = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else ""
            return r.status_code, text, []
        with client.stream("POST", f"{base_url}/chat/completions", json=body) as r:
            if r.status_code != 200:
                r.read()
                return r.status_code, "", []
            text, events = read_stream(r)
            return r.status_code, text, events


def chat_concurrently(base_url, n, *, stream, max_tokens=20) -> list[int]:
    """Fire n requests at the same time; return their status codes."""
    async def one(client):
        body = chat_body(stream=stream, max_tokens=max_tokens)
        async with client.stream("POST", f"{base_url}/chat/completions", json=body) as r:
            await r.aread()          # wait for the whole response (stream or not)
            return r.status_code

    async def main():
        async with httpx.AsyncClient(timeout=30) as client:
            return await asyncio.gather(*(one(client) for _ in range(n)))

    return asyncio.run(main())


# ---------- basic endpoints ----------

def test_health_reports_upstream_and_cap(upstream, start_broker):
    base = start_broker(upstream.url, max_inflight=3)
    r = httpx.get(base.removesuffix("/v1") + "/health")
    assert r.status_code == 200
    assert r.json()["upstream"] == upstream.url + "/v1"
    assert r.json()["max_inflight"] == 3


def test_dify_probe_is_answered_without_calling_upstream(start_broker):
    # Upstream port has nothing listening: the probe must not depend on it.
    base = start_broker(f"http://127.0.0.1:{free_port()}")
    assert httpx.get(f"{base}/chat/completions").status_code == 200
    assert httpx.head(f"{base}/chat/completions").status_code == 200


# ---------- request / response rewriting ----------

def test_model_name_is_forced_to_default(upstream, start_broker):
    base = start_broker(upstream.url)
    chat(base, stream=False, model="gpt-4o-mini")
    assert mock_upstream.last_request["model"] == "ssc-FinLLM"


def test_input_is_converted_to_simplified(upstream, start_broker):
    base = start_broker(upstream.url)
    chat(base, stream=False, text=TRADITIONAL_QUESTION)
    assert mock_upstream.last_request["messages"][0]["content"] == SIMPLIFIED_QUESTION


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "non-stream"])
def test_output_is_converted_to_traditional(upstream, start_broker, stream):
    # The mock answers in Simplified Chinese: 股票债券基金风险...
    base = start_broker(upstream.url)
    status, text, _ = chat(base, stream=stream, max_tokens=10)
    assert status == 200
    assert text.startswith("股票債券基金風險")


# ---------- concurrency cap (regression tests for the semaphore bug) ----------

def test_concurrent_streams_never_exceed_cap(upstream, start_broker):
    base = start_broker(upstream.url, max_inflight=2)
    statuses = chat_concurrently(base, 6, stream=True)
    assert statuses == [200] * 6
    assert mock_upstream.stats["peak_inflight"] == 2


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "non-stream"])
def test_slots_are_returned_when_requests_finish_together(upstream, start_broker, stream):
    # Old bug: when requests finished with nobody queued, their slots were lost
    # and later requests ran one at a time.
    base = start_broker(upstream.url, max_inflight=2)
    chat_concurrently(base, 2, stream=stream)        # first wave fills both slots
    mock_upstream.reset()
    chat_concurrently(base, 2, stream=stream)        # second wave should still run in parallel
    assert mock_upstream.stats["peak_inflight"] == 2


def test_releasing_more_slots_than_the_cap_raises():
    # A double release must fail loudly instead of silently raising the cap
    # (that silent drift is exactly how the original stream bug hid).
    sem = type(broker.sem)(1)
    with pytest.raises(ValueError):
        sem.release()


def test_full_queue_returns_503(upstream, start_broker):
    base = start_broker(upstream.url, max_inflight=1, queue_timeout=0.2)
    statuses = chat_concurrently(base, 3, stream=False, max_tokens=100)   # ~0.55 s each
    assert sorted(statuses) == [200, 503, 503]


def test_slot_is_released_when_client_disconnects_mid_stream(upstream, start_broker):
    base = start_broker(upstream.url, max_inflight=1, queue_timeout=1.0)
    body = chat_body(stream=True, max_tokens=400)      # would take ~2 s to finish
    with httpx.Client(timeout=10) as client:
        with client.stream("POST", f"{base}/chat/completions", json=body) as r:
            next(r.iter_lines())                       # read one line, then hang up
    # If the slot were still held by the abandoned stream, this would wait 1 s and get 503.
    status, _, _ = chat(base, stream=False)
    assert status == 200


# ---------- upstream failures ----------

@pytest.mark.parametrize("stream", [True, False], ids=["stream", "non-stream"])
def test_unreachable_upstream_returns_502(start_broker, stream):
    base = start_broker(f"http://127.0.0.1:{free_port()}")
    status, _, _ = chat(base, stream=stream)
    assert status == 502


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "non-stream"])
def test_upstream_error_status_is_passed_through(upstream, start_broker, monkeypatch, stream):
    monkeypatch.setattr(mock_upstream, "FAIL_STATUS", 400)
    base = start_broker(upstream.url)
    status, _, _ = chat(base, stream=stream)
    assert status == 400


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "non-stream"])
def test_slots_are_returned_after_upstream_errors(upstream, start_broker, monkeypatch, stream):
    # Every error path must give its slot back, or failures would slowly use up the cap.
    base = start_broker(upstream.url, max_inflight=1, queue_timeout=0.3)
    monkeypatch.setattr(mock_upstream, "FAIL_STATUS", 500)
    assert chat(base, stream=stream)[0] == 500
    assert chat(base, stream=stream)[0] == 500
    monkeypatch.setattr(mock_upstream, "FAIL_STATUS", 0)
    assert chat(base, stream=stream)[0] == 200     # would be 503 if a slot had leaked


def test_upstream_timeout_returns_504(upstream, start_broker, monkeypatch):
    monkeypatch.setattr(mock_upstream, "TTFT_MS", 500.0)
    base = start_broker(upstream.url, read_timeout=0.1)
    status, _, _ = chat(base, stream=False)
    assert status == 504


def test_upstream_timeout_mid_stream_ends_with_error_event(upstream, start_broker, monkeypatch):
    # Headers are already sent when the upstream stalls, so the broker can't change
    # the status code; it must end the stream with an error event instead.
    monkeypatch.setattr(mock_upstream, "TTFT_MS", 500.0)
    base = start_broker(upstream.url, read_timeout=0.1)
    status, _, events = chat(base, stream=True)
    assert status == 200
    assert "error" in events[-1]


# ---------- rate limit ----------

def test_second_request_within_rate_limit_window_returns_429(upstream, start_broker):
    base = start_broker(upstream.url, rate_limit=10)
    assert chat(base, stream=False)[0] == 200
    assert chat(base, stream=False)[0] == 429
