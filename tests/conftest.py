"""
Shared test fixtures.

Every test gets real HTTP servers on free localhost ports, so requests go
through the same code paths as in production (uvicorn, streaming, timeouts):

    test  ──HTTP──▶  broker (ssc-broker/broker.py)  ──HTTP──▶  mock vLLM (benchmark/mock_upstream.py)

Servers are started fresh for each test, so no state leaks between tests.
"""
import socket
import threading
import time

import pytest
import uvicorn

import broker
import mock_upstream


def free_port() -> int:
    """Ask the OS for an unused TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    """Run an ASGI app with uvicorn in a background thread."""

    def __init__(self, app):
        self.port = free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.01)
        return self

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)


@pytest.fixture
def upstream(monkeypatch):
    """A fake vLLM: 200 tok/s, 50 ms to first token. Yields the server."""
    monkeypatch.setattr(mock_upstream, "TOKENS_PER_SEC", 200.0)
    monkeypatch.setattr(mock_upstream, "TTFT_MS", 50.0)
    monkeypatch.setattr(mock_upstream, "FAIL_STATUS", 0)
    mock_upstream.reset()
    server = ServerThread(mock_upstream.app).start()
    yield server
    server.stop()


@pytest.fixture
def start_broker(monkeypatch):
    """
    Factory: start_broker(upstream_url, max_inflight=..., ...) -> broker base URL.

    The broker reads its settings from module-level variables at import time,
    so we override those variables (monkeypatch undoes it after the test).
    """
    servers = []

    def _start(upstream_url, *, max_inflight=4, queue_timeout=5.0,
               read_timeout=10.0, rate_per_min=0.0, rate_burst=5, api_key=""):
        monkeypatch.setattr(broker, "API_KEY", api_key, raising=False)
        monkeypatch.setattr(broker, "UP", upstream_url + "/v1")
        monkeypatch.setattr(broker, "MAX_INF", max_inflight)
        # Same semaphore class the broker uses, just with a test-sized cap.
        monkeypatch.setattr(broker, "sem", type(broker.sem)(max_inflight))
        monkeypatch.setattr(broker, "Q_TIMEOUT", queue_timeout)
        monkeypatch.setattr(broker, "UP_READ_TIMEOUT", read_timeout, raising=False)
        monkeypatch.setattr(broker, "limiter", broker.TokenBucketLimiter(rate_per_min, rate_burst))
        server = ServerThread(broker.app).start()
        servers.append(server)
        return server.url + "/v1"

    yield _start
    for server in servers:
        server.stop()
