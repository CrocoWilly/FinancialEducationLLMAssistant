import os, time, asyncio, orjson, logging
from typing import List, Dict, Any
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from opencc import OpenCC

# ---------- 基本設定 ----------
# 必須「含 /v1」，且不帶尾巴斜線
UP = os.getenv("UPSTREAM_LLM_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
# 你的上游唯一模型名稱（Dify 會亂丟 gpt-4o-mini；這裡統一蓋掉）
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "ssc-FinLLM")

MAX_INF   = int(os.getenv("BROKER_MAX_INFLIGHT", "16"))
RL_SECS   = float(os.getenv("RATE_LIMIT_SECONDS", "0"))
Q_TIMEOUT = int(os.getenv("QUEUE_WAIT_TIMEOUT", "60"))

cc_t2s = OpenCC("t2s")
cc_s2t = OpenCC("s2twp")

last_req: Dict[str, float] = {}
sem = asyncio.Semaphore(MAX_INF)

log = logging.getLogger("broker")
log.setLevel(logging.INFO)

def _up(path: str) -> str:
    """把 '/something' 接到上游 base（UP 已含 /v1，且無尾斜線）"""
    return f"{UP}{path}"

# ---------- 文字轉換 ----------
def t2s(x: str) -> str:
    try:
        return cc_t2s.convert(x)
    except Exception:
        return x

def s2t(x: str) -> str:
    try:
        return cc_s2t.convert(x)
    except Exception:
        return x

def t2s_msgs(msgs: List[Dict[str, Any]]):
    out = []
    for m in msgs or []:
        mm = m.copy()
        if isinstance(mm.get("content"), str):
            mm["content"] = t2s(mm["content"])
        out.append(mm)
    return out

app = FastAPI()

# ---------- 健康檢查 ----------
@app.get("/health")
async def health():
    return {"ok": True, "upstream": UP, "default_model": DEFAULT_MODEL}

# ---------- 轉發：/v1/models ----------
@app.get("/v1/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(_up("/models"))
        return Response(content=r.content, status_code=r.status_code,
                        media_type="application/json")
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"upstream_error: {e}"}, status_code=502)

# ---------- Dify 驗證探測：GET/HEAD /v1/chat/completions ----------
@app.get("/v1/chat/completions")
def chat_probe_ok():
    # 不打上游，直接回 200，避免 vLLM 對 GET 回 404
    return {"ok": True, "message": "chat endpoint ready"}

@app.head("/v1/chat/completions")
def chat_probe_head():
    return Response(status_code=200)

# ---------- 內部實作：POST chat/completions ----------
async def _chat_impl(req: Request):
    ip = req.client.host if req.client else "unknown"
    now = time.monotonic()

    # 每 IP 簡易節流
    if RL_SECS > 0 and (now - last_req.get(ip, 0.0) < RL_SECS):
        return JSONResponse({"error": "rate_limited"}, status_code=429)
    last_req[ip] = now

    body = await req.json()

    # 1) 統一模型名稱（避免 Dify 用 gpt-* 測試導致上游 404）
    body["model"] = DEFAULT_MODEL

    # 2) 文字簡轉繁（輸入）
    body["messages"] = t2s_msgs(body.get("messages", []))

    stream = bool(body.get("stream", True))

    # 取得併發名額
    await asyncio.wait_for(sem.acquire(), timeout=Q_TIMEOUT)

    # 轉發時的 headers
    headers = {"Content-Type": "application/json"}
    auth = req.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth

    try:
        url = _up("/chat/completions")

        if stream:
            async def gen():
                try:
                    log.info(f"proxy → {url} (stream)")
                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream(
                            "POST", url, json=body,
                            headers={"Accept": "text/event-stream", **headers}
                        ) as upstream:
                            async for line in upstream.aiter_lines():
                                if not line:
                                    continue
                                if line.startswith("data:"):
                                    payload = line[len("data:"):].strip()
                                    if payload == "[DONE]":
                                        yield "data: [DONE]\n\n"
                                        break
                                    # 嘗試對 chunk 做繁轉
                                    try:
                                        obj = orjson.loads(payload)
                                        ch = obj.get("choices") or []
                                        if ch and "delta" in ch[0]:
                                            d = ch[0]["delta"]
                                            if isinstance(d.get("content"), str):
                                                d["content"] = s2t(d["content"])
                                        yield "data: " + orjson.dumps(obj).decode() + "\n\n"
                                    except Exception:
                                        yield "data: " + payload + "\n\n"
                                else:
                                    yield line + "\n"
                finally:
                    sem.release()

            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # ---- 非串流 ----
        log.info(f"proxy → {url}")
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, json=body, headers=headers)
        r.raise_for_status()

        data = r.json()
        # 回傳做繁轉
        ch = data.get("choices") or []
        if ch and "message" in ch[0]:
            msg = ch[0]["message"]
            if isinstance(msg.get("content"), str):
                msg["content"] = s2t(msg["content"])
        return JSONResponse(data)

    except httpx.HTTPError as e:
        return JSONResponse({"error": f"upstream_error: {e}"}, status_code=502)

    finally:
        # 非串流路徑需要這個；串流路徑在 gen() 的 finally 已釋放
        if sem.locked():
            sem.release()

# v1 路徑（Dify 會用）
@app.post("/v1/chat/completions")
async def chat_v1(req: Request):
    return await _chat_impl(req)

# 非 v1 路徑（保險：其他工具可能打這條）
@app.post("/chat/completions")
async def chat_alias(req: Request):
    return await _chat_impl(req)

