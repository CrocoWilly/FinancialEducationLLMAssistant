import os, time, asyncio, orjson, logging, hmac
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from opencc import OpenCC
from starlette.background import BackgroundTask

# ---------- 基本設定 ----------
# 必須「含 /v1」，且不帶尾巴斜線
UP = os.getenv("UPSTREAM_LLM_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
# 你的上游唯一模型名稱（Dify 會亂丟 gpt-4o-mini；這裡統一蓋掉）
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "ssc-FinLLM")

MAX_INF   = int(os.getenv("BROKER_MAX_INFLIGHT", "16"))
RL_SECS   = float(os.getenv("RATE_LIMIT_SECONDS", "0"))
Q_TIMEOUT = float(os.getenv("QUEUE_WAIT_TIMEOUT", "60"))

# 上游逾時：connect = 連線建立；read = 兩次收到資料之間最多等多久（串流時是每個 chunk 之間）
UP_CONNECT_TIMEOUT = float(os.getenv("UPSTREAM_CONNECT_TIMEOUT", "10"))
UP_READ_TIMEOUT    = float(os.getenv("UPSTREAM_READ_TIMEOUT", "120"))

cc_t2s = OpenCC("t2s")
cc_s2t = OpenCC("s2twp")

last_req: Dict[str, float] = {}
# Bounded：若名額被多還（release 次數超過 acquire），直接丟 ValueError，
# 而不是默默讓上限變大
sem = asyncio.BoundedSemaphore(MAX_INF)

# 呼叫 /v1/* 需要的 API key（Authorization: Bearer <key>）；空字串 = 不驗證
API_KEY = os.getenv("BROKER_API_KEY", "")

# 自己掛 handler：uvicorn 只設定它自己的 logger，否則 broker 的 INFO log 會被丟掉
log = logging.getLogger("broker")
log.setLevel(logging.INFO)
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s broker: %(message)s"))
    log.addHandler(_handler)
    log.propagate = False

client: httpx.AsyncClient  # 在 lifespan 中建立，所有請求共用連線池


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global client
    if not API_KEY:
        log.warning("BROKER_API_KEY is not set: /v1/* accepts requests without authentication")
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=UP_CONNECT_TIMEOUT, read=UP_READ_TIMEOUT,
            write=30.0, pool=UP_CONNECT_TIMEOUT,
        ),
        limits=httpx.Limits(max_connections=MAX_INF + 4, max_keepalive_connections=MAX_INF),
    )
    try:
        yield
    finally:
        await client.aclose()


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


class _Slot:
    """一個併發名額；release() 可重複呼叫，但只會真的釋放一次。"""

    def __init__(self):
        self._held = True

    def release(self):
        if self._held:
            self._held = False
            sem.release()


def _upstream_error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"error": f"upstream_error: {detail}"}, status_code=status)


def require_api_key(req: Request):
    """FastAPI dependency：API_KEY 有設定時，要求 Authorization: Bearer <API_KEY>。"""
    if not API_KEY:
        return
    scheme, _, token = req.headers.get("Authorization", "").partition(" ")
    # compare_digest：比對時間不隨「前面幾個字元相同」而變，避免 timing attack
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key",
                            headers={"WWW-Authenticate": "Bearer"})


app = FastAPI(lifespan=lifespan)

# ---------- 健康檢查 ----------
@app.get("/health")
async def health():
    return {"ok": True, "upstream": UP, "default_model": DEFAULT_MODEL,
            "max_inflight": MAX_INF}

# ---------- 轉發：/v1/models ----------
@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models():
    try:
        r = await client.get(_up("/models"))
        return Response(content=r.content, status_code=r.status_code,
                        media_type="application/json")
    except httpx.HTTPError as e:
        return _upstream_error(502, repr(e))

# ---------- Dify 驗證探測：GET/HEAD /v1/chat/completions ----------
@app.get("/v1/chat/completions")
def chat_probe_ok():
    # 不打上游，直接回 200，避免 vLLM 對 GET 回 404
    return {"ok": True, "message": "chat endpoint ready"}

@app.head("/v1/chat/completions")
def chat_probe_head():
    return Response(status_code=200)


def _convert_chunk(payload: str) -> str:
    """把一個 SSE data payload 的 delta.content 做簡→繁；解析失敗就原樣回傳。"""
    try:
        obj = orjson.loads(payload)
    except Exception:
        return payload
    for c in obj.get("choices") or []:
        d = c.get("delta") or {}
        if isinstance(d.get("content"), str):
            d["content"] = s2t(d["content"])
    return orjson.dumps(obj).decode()


# ---------- 內部實作：POST chat/completions ----------
async def _chat_impl(req: Request):
    ip = req.client.host if req.client else "unknown"
    now = time.monotonic()

    # 每 IP 簡易節流
    if RL_SECS > 0 and (now - last_req.get(ip, 0.0) < RL_SECS):
        return JSONResponse({"error": "rate_limited"}, status_code=429)
    last_req[ip] = now

    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    # 1) 統一模型名稱（避免 Dify 用 gpt-* 測試導致上游 404）
    body["model"] = DEFAULT_MODEL

    # 2) 輸入繁轉簡（模型以簡體訓練）
    body["messages"] = t2s_msgs(body.get("messages", []))

    stream = bool(body.get("stream", True))

    # 取得併發名額；排隊逾時回 503
    try:
        await asyncio.wait_for(sem.acquire(), timeout=Q_TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse({"error": "busy", "detail": "queue wait timeout"},
                            status_code=503, headers={"Retry-After": "5"})
    slot = _Slot()

    # 轉發時的 headers
    headers = {"Content-Type": "application/json"}
    auth = req.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth

    url = _up("/chat/completions")

    # ---- 串流 ----
    if stream:
        try:
            log.info(f"proxy → {url} (stream)")
            upstream_req = client.build_request(
                "POST", url, json=body,
                headers={"Accept": "text/event-stream", **headers},
            )
            upstream = await client.send(upstream_req, stream=True)
        except httpx.TimeoutException as e:
            slot.release()
            return _upstream_error(504, repr(e))
        except httpx.HTTPError as e:
            slot.release()
            return _upstream_error(502, repr(e))

        # 上游非 200：讀完錯誤內容、關閉連線、回傳錯誤（不要當成正常串流）
        if upstream.status_code != 200:
            try:
                detail = (await upstream.aread()).decode(errors="replace")[:500]
            finally:
                await upstream.aclose()
                slot.release()
            return _upstream_error(upstream.status_code, detail)

        async def gen():
            try:
                async for line in upstream.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        yield "data: " + _convert_chunk(payload) + "\n\n"
                    else:
                        yield line + "\n"
            except httpx.HTTPError as e:
                # 串流中途斷線/逾時：送一個 error 事件給客戶端
                err = orjson.dumps({"error": f"upstream_error: {e!r}"}).decode()
                yield "data: " + err + "\n\n"
            finally:
                await upstream.aclose()
                slot.release()

        async def cleanup():
            # 保險：若 client 在 gen() 開始前就斷線，gen 的 finally 不會執行
            await upstream.aclose()
            slot.release()

        return StreamingResponse(
            gen(),
            background=BackgroundTask(cleanup),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ---- 非串流 ----
    try:
        log.info(f"proxy → {url}")
        r = await client.post(url, json=body, headers=headers)
        if r.status_code != 200:
            return _upstream_error(r.status_code, r.text[:500])

        data = r.json()
        # 回傳做簡轉繁
        for c in data.get("choices") or []:
            msg = c.get("message") or {}
            if isinstance(msg.get("content"), str):
                msg["content"] = s2t(msg["content"])
        return JSONResponse(data)

    except httpx.TimeoutException as e:
        return _upstream_error(504, repr(e))
    except httpx.HTTPError as e:
        return _upstream_error(502, repr(e))
    finally:
        slot.release()

# v1 路徑（Dify 會用）
@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_v1(req: Request):
    return await _chat_impl(req)

# 非 v1 路徑（保險：其他工具可能打這條）
@app.post("/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_alias(req: Request):
    return await _chat_impl(req)
