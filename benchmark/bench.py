"""
Load / throughput benchmark for any OpenAI-compatible chat endpoint
(the broker, vLLM directly, or the mock upstream).

Examples:
    # through the broker
    python bench.py --base-url http://localhost:8080/v1 --concurrency 1,4,8,16,32

    # straight at vLLM on a GPU box
    python bench.py --base-url http://<gpu-host>:8000/v1 --model ssc-FinLLM \
        --concurrency 1,2,4,8,16 --max-tokens 256 --json-out results.json

Metrics per concurrency level:
    ok / errors         success count and error breakdown (HTTP status / exception)
    TTFT p50/p95        time to first content token (streaming only)
    decode tok/s        per-request completion_tokens / (last token - first token)
    e2e p50/p95         full request latency
    agg tok/s           sum of completion tokens / wall time for the level
Token counts come from the server's `usage` (requested via
stream_options.include_usage); if absent, content chunks are counted instead.
"""
import argparse
import asyncio
import json
import statistics
import time

import httpx

from prompts import build_messages

DEFAULT_PROMPT = "請用淺顯的方式解釋什麼是股票、債券和基金，以及它們的風險差異。"


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def one_request(client, args):
    body = {
        "model": args.model,
        "messages": (build_messages(args.prompt_tokens) if args.prompt_tokens
                     else [{"role": "user", "content": args.prompt}]),
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "stream": not args.no_stream,
    }
    if not args.no_stream:
        body["stream_options"] = {"include_usage": True}
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    t0 = time.perf_counter()
    res = {"ok": False, "error": None, "ttft": None, "e2e": None,
           "tokens": 0, "decode_tps": None, "text": ""}
    try:
        if args.no_stream:
            r = await client.post("/chat/completions", json=body, headers=headers)
            if r.status_code != 200:
                res["error"] = f"HTTP {r.status_code}"
                return res
            data = r.json()
            res["text"] = data["choices"][0]["message"]["content"] or ""
            res["tokens"] = (data.get("usage") or {}).get("completion_tokens") or len(res["text"])
            res["e2e"] = time.perf_counter() - t0
            res["ok"] = True
            return res

        first = last = None
        chunks = 0
        usage_tokens = None
        parts = []
        async with client.stream("POST", "/chat/completions", json=body, headers=headers) as r:
            if r.status_code != 200:
                await r.aread()
                res["error"] = f"HTTP {r.status_code}"
                return res
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if "error" in obj:
                    res["error"] = "stream error"
                    return res
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens")
                for c in obj.get("choices") or []:
                    content = (c.get("delta") or {}).get("content")
                    if content:
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        chunks += 1
                        parts.append(content)
        end = time.perf_counter()
        if first is None:
            res["error"] = "no tokens"
            return res
        res["tokens"] = usage_tokens or chunks
        res["ttft"] = first - t0
        res["e2e"] = end - t0
        if last > first and res["tokens"] > 1:
            res["decode_tps"] = (res["tokens"] - 1) / (last - first)
        res["text"] = "".join(parts)
        res["ok"] = True
        return res
    except httpx.TimeoutException:
        res["error"] = "timeout"
        return res
    except httpx.HTTPError as e:
        res["error"] = type(e).__name__
        return res


async def run_level(args, conc):
    n = args.requests or conc * args.rounds
    limits = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"),
                                 limits=limits, timeout=timeout) as client:
        gate = asyncio.Semaphore(conc)  # exactly `conc` requests in flight

        async def worker():
            async with gate:
                return await one_request(client, args)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(worker() for _ in range(n)))
        wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    errors = {}
    for r in results:
        if not r["ok"]:
            errors[r["error"]] = errors.get(r["error"], 0) + 1
    ttft = [r["ttft"] for r in ok if r["ttft"] is not None]
    e2e = [r["e2e"] for r in ok]
    dtps = [r["decode_tps"] for r in ok if r["decode_tps"]]
    total_tokens = sum(r["tokens"] for r in ok)
    return {
        "concurrency": conc, "requests": n, "ok": len(ok), "errors": errors,
        "wall_s": wall,
        "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95),
        "e2e_p50": pct(e2e, 50), "e2e_p95": pct(e2e, 95),
        "decode_tps_p50": pct(dtps, 50),
        "decode_tps_mean": statistics.mean(dtps) if dtps else float("nan"),
        "agg_tps": total_tokens / wall if wall else 0.0,
        "sample": ok[0]["text"][:40] if ok else "",
    }


def fmt(x, nd=2):
    return "-" if x != x else f"{x:.{nd}f}"  # NaN -> "-"


async def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="ssc-FinLLM")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="use a production-shaped prompt of ~N tokens (system prompt + "
                         "context + question) instead of --prompt; production median is 727")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", default="1,4,8,16",
                    help="comma-separated levels, e.g. 1,4,8,16,32")
    ap.add_argument("--rounds", type=int, default=3,
                    help="requests per level = concurrency × rounds (default 3)")
    ap.add_argument("--requests", type=int, default=0,
                    help="fixed number of requests per level (overrides --rounds)")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    rows = []
    hdr = (f"{'conc':>4} {'reqs':>5} {'ok':>4} {'TTFT p50':>9} {'TTFT p95':>9} "
           f"{'decode tok/s':>12} {'e2e p50':>8} {'e2e p95':>8} {'agg tok/s':>10}  errors")
    print(f"target={args.base_url}  stream={not args.no_stream}  max_tokens={args.max_tokens}"
          f"  prompt={'~%d tokens' % args.prompt_tokens if args.prompt_tokens else 'short'}")
    print(hdr)
    print("-" * len(hdr))
    for conc in levels:
        row = await run_level(args, conc)
        rows.append(row)
        print(f"{conc:>4} {row['requests']:>5} {row['ok']:>4} "
              f"{fmt(row['ttft_p50']):>9} {fmt(row['ttft_p95']):>9} "
              f"{fmt(row['decode_tps_p50'], 1):>12} "
              f"{fmt(row['e2e_p50']):>8} {fmt(row['e2e_p95']):>8} "
              f"{fmt(row['agg_tps'], 1):>10}  {row['errors'] or ''}", flush=True)
    if rows and rows[0]["sample"]:
        print(f"\nsample output: {rows[0]['sample']}…")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"saved {args.json_out}")


if __name__ == "__main__":
    asyncio.run(main())
