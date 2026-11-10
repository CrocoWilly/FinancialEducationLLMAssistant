"""
Simulated chat sessions: N users who ask, wait for the streamed answer, pause,
and ask again — the way people used the production app.

Defaults come from the production Dify database (September 2025):
  prompt ~727 tokens (system prompt + retrieved chunks + question),
  think time between turns: median 24 s (p25 15 s, p75 32 s) -> log-normal,
  answers capped at max_tokens=512.

Examples:
    # through the broker (cap 16), 10 / 20 / 50 sessions for 3 minutes each
    python sessions.py --base-url http://localhost:8080/v1 --api-key KEY \
        --sessions 10,20,50 --duration 180 --metrics-url http://localhost:8000/metrics

Per level it reports: requests, errors (by status), TTFT p50/p95, per-stream
decode tok/s, end-to-end latency, average / max requests in flight (client
side, includes time queued in the broker) and, if --metrics-url points at
vLLM's /metrics, max running / waiting requests, max KV-cache usage and
preemptions during the run.
"""
import argparse
import asyncio
import json
import math
import random
import statistics
import time

import httpx

from prompts import build_messages


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fmt(x, nd=2):
    return "-" if x != x else f"{x:.{nd}f}"


# ---------- vLLM /metrics scraping ----------

GAUGES = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
}


def parse_metrics(text: str) -> dict:
    """Sum the values of the metrics we care about (across label sets)."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        for key, names in GAUGES.items():
            if name in names:
                try:
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


async def scrape_metrics(client, url, samples, stop):
    while not stop.is_set():
        try:
            r = await client.get(url, timeout=5)
            samples.append(parse_metrics(r.text))
        except httpx.HTTPError:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


# ---------- one chat turn ----------

async def chat_turn(client, args, rng, inflight):
    body = {
        "model": args.model,
        "messages": build_messages(args.prompt_tokens, rng),
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.min_tokens:
        # vLLM extension: keep generating until min_tokens (worst-case long answers)
        body["min_tokens"] = args.min_tokens
    res = {"status": None, "error": None, "ttft": None, "e2e": None, "tokens": 0,
           "prompt_tokens": None, "decode_tps": None}
    t0 = time.perf_counter()
    inflight[0] += 1
    try:
        async with client.stream("POST", "/chat/completions", json=body) as r:
            res["status"] = r.status_code
            if r.status_code != 200:
                await r.aread()
                return res
            first = last = None
            chunks = 0
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
                    res["tokens"] = obj["usage"].get("completion_tokens") or 0
                    res["prompt_tokens"] = obj["usage"].get("prompt_tokens")
                for c in obj.get("choices") or []:
                    if (c.get("delta") or {}).get("content"):
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        chunks += 1
            if first is None:
                res["error"] = "no tokens"
                return res
            res["tokens"] = res["tokens"] or chunks
            res["ttft"] = first - t0
            res["e2e"] = time.perf_counter() - t0
            if last > first and res["tokens"] > 1:
                res["decode_tps"] = (res["tokens"] - 1) / (last - first)
            return res
    except httpx.TimeoutException:
        res["error"] = "timeout"
        return res
    except httpx.HTTPError as e:
        res["error"] = type(e).__name__
        return res
    finally:
        inflight[0] -= 1


# ---------- sessions ----------

def think_time(rng, args):
    # log-normal with the production median; sigma from the p25/p75 spread
    return rng.lognormvariate(math.log(args.think_median), args.think_sigma)


async def session(client, args, rng, deadline, results, inflight):
    # stagger starts so sessions don't all fire at t=0
    await asyncio.sleep(rng.uniform(0, args.think_median))
    while time.monotonic() < deadline:
        results.append(await chat_turn(client, args, rng, inflight))
        pause = think_time(rng, args)
        if time.monotonic() + pause >= deadline:
            break
        await asyncio.sleep(pause)


async def run_level(args, n_sessions):
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    limits = httpx.Limits(max_connections=n_sessions + 8, max_keepalive_connections=n_sessions)
    results, inflight, inflight_samples, metric_samples = [], [0], [], []
    stop = asyncio.Event()
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), headers=headers,
                                 limits=limits, timeout=httpx.Timeout(args.timeout)) as client:
        scraper = (asyncio.create_task(scrape_metrics(client, args.metrics_url, metric_samples, stop))
                   if args.metrics_url else None)

        async def sample_inflight():
            while not stop.is_set():
                inflight_samples.append(inflight[0])
                await asyncio.sleep(0.5)

        sampler = asyncio.create_task(sample_inflight())
        deadline = time.monotonic() + args.duration
        rngs = [random.Random(args.seed * 1000 + i) for i in range(n_sessions)]
        t0 = time.perf_counter()
        await asyncio.gather(*(session(client, args, rngs[i], deadline, results, inflight)
                               for i in range(n_sessions)))
        wall = time.perf_counter() - t0
        stop.set()
        await sampler
        if scraper:
            await scraper

    ok = [r for r in results if r["status"] == 200 and not r["error"]]
    errors = {}
    for r in results:
        if r not in ok:
            key = r["error"] or f"HTTP {r['status']}"
            errors[key] = errors.get(key, 0) + 1
    row = {
        "sessions": n_sessions, "duration_s": args.duration, "wall_s": round(wall, 1),
        "requests": len(results), "ok": len(ok), "errors": errors,
        "ttft_p50": pct([r["ttft"] for r in ok], 50), "ttft_p95": pct([r["ttft"] for r in ok], 95),
        "e2e_p50": pct([r["e2e"] for r in ok], 50), "e2e_p95": pct([r["e2e"] for r in ok], 95),
        "decode_tps_p50": pct([r["decode_tps"] for r in ok if r["decode_tps"]], 50),
        "answer_tokens_p50": pct([r["tokens"] for r in ok], 50),
        "prompt_tokens_p50": pct([r["prompt_tokens"] for r in ok if r["prompt_tokens"]], 50),
        "agg_tps": sum(r["tokens"] for r in ok) / wall if wall else 0.0,
        "inflight_avg": statistics.mean(inflight_samples) if inflight_samples else 0.0,
        "inflight_max": max(inflight_samples) if inflight_samples else 0,
    }
    if metric_samples:
        def col(k):
            return [s[k] for s in metric_samples if k in s]
        row["vllm_running_max"] = max(col("running"), default=None)
        row["vllm_waiting_max"] = max(col("waiting"), default=None)
        kv = col("kv_usage")
        row["vllm_kv_usage_max"] = max(kv) if kv else None
        pre = col("preemptions")
        row["vllm_preemptions"] = (pre[-1] - pre[0]) if len(pre) >= 2 else None
    return row


async def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="ssc-FinLLM")
    ap.add_argument("--sessions", default="10,20,50", help="comma-separated levels")
    ap.add_argument("--duration", type=float, default=180, help="seconds per level")
    ap.add_argument("--prompt-tokens", type=int, default=727)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="force at least this many answer tokens (vLLM min_tokens); "
                         "set to --max-tokens for a worst-case load")
    ap.add_argument("--think-median", type=float, default=24.0)
    ap.add_argument("--think-sigma", type=float, default=0.56,
                    help="log-normal sigma; 0.56 matches production p25=15 s / p75=32 s")
    ap.add_argument("--metrics-url", default="", help="vLLM /metrics URL (optional)")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="", help="free text stored with the results")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    levels = [int(x) for x in args.sessions.split(",") if x.strip()]
    print(f"target={args.base_url} prompt~{args.prompt_tokens} tok, max_tokens={args.max_tokens}, "
          f"{f'min_tokens={args.min_tokens}, ' if args.min_tokens else ''}"
          f"think median {args.think_median}s, {args.duration:.0f}s per level {args.label}")
    hdr = (f"{'sess':>4} {'reqs':>5} {'ok':>4} {'TTFT p50':>8} {'TTFT p95':>8} {'tok/s':>6} "
           f"{'e2e p50':>7} {'e2e p95':>7} {'infl avg/max':>12} {'vLLM run/wait':>13} "
           f"{'KV max':>6} {'preempt':>7}  errors")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for n in levels:
        row = await run_level(args, n)
        row["label"] = args.label
        row["min_tokens"] = args.min_tokens
        rows.append(row)
        kv = row.get("vllm_kv_usage_max")
        run, wait = row.get("vllm_running_max"), row.get("vllm_waiting_max")
        print(f"{n:>4} {row['requests']:>5} {row['ok']:>4} {fmt(row['ttft_p50']):>8} "
              f"{fmt(row['ttft_p95']):>8} {fmt(row['decode_tps_p50'], 1):>6} "
              f"{fmt(row['e2e_p50']):>7} {fmt(row['e2e_p95']):>7} "
              f"{row['inflight_avg']:>5.1f}/{row['inflight_max']:<6} "
              f"{('-' if run is None else int(run)):>6}/{('-' if wait is None else int(wait)):<6} "
              f"{('-' if kv is None else f'{kv:.0%}'):>6} "
              f"{('-' if row.get('vllm_preemptions') is None else int(row['vllm_preemptions'])):>7}  "
              f"{row['errors'] or ''}", flush=True)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"saved {args.json_out}")


if __name__ == "__main__":
    asyncio.run(main())
