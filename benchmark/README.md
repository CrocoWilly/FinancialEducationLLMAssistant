# Benchmark

Two tools:

| File | What it is |
|---|---|
| `bench.py` | Load generator for any OpenAI-compatible endpoint (broker, vLLM, mock). Reports TTFT, per-request decode tok/s, e2e latency, aggregate tok/s and errors per concurrency level. |
| `mock_upstream.py` | Fake vLLM (no GPU). Streams Simplified-Chinese tokens at a fixed rate and exposes `/stats` with the **peak number of in-flight requests**, so you can verify the broker's concurrency cap. |

```bash
pip install -r requirements.txt            # plus ../ssc-broker/requirements.txt to run the broker
```

## 1. Broker-layer test (no GPU)

```bash
# terminal 1 – fake upstream: 60 tok/s, 200 ms TTFT
MOCK_TOKENS_PER_SEC=60 MOCK_TTFT_MS=200 uvicorn mock_upstream:app --port 8000

# terminal 2 – broker pointed at it
cd ../ssc-broker
UPSTREAM_LLM_BASE=http://127.0.0.1:8000/v1 BROKER_MAX_INFLIGHT=4 uvicorn broker:app --port 8080

# terminal 3
curl -XPOST localhost:8000/stats/reset
python bench.py --base-url http://localhost:8080/v1 --concurrency 1,4,8,16 --max-tokens 60
curl localhost:8000/stats                 # peak_inflight must be <= BROKER_MAX_INFLIGHT
```

## 2. Real model throughput (needs a GPU)

ssc-FinLLM 7B in FP16 needs ~15 GB of VRAM for weights alone, so use a 24 GB card
(RTX 3090 / 4090 / A10 / L4 …). A rented cloud GPU works fine. The `vllm serve` command below is a suggested starting point and has not been verified in this repo.

```bash
pip install vllm
vllm serve ~/models/ssc-finllm-model \
    --served-model-name ssc-FinLLM \
    --trust-remote-code --dtype float16 --max-model-len 4096 --port 8000

# single-stream speed (the "tokens/sec" a user sees)
python bench.py --base-url http://localhost:8000/v1 --concurrency 1 --rounds 5 --max-tokens 256

# concurrency sweep
python bench.py --base-url http://localhost:8000/v1 --concurrency 1,2,4,8,16,32 \
    --max-tokens 256 --json-out results_gpu.json
```

How to read it:

- **decode tok/s @ conc=1** → the single-user generation speed (what "~62 tokens/sec" refers to).
- **agg tok/s** → total server throughput; rises with concurrency until the GPU saturates.
- **decode tok/s @ conc=N** → what each user experiences when N people are generating at once.
- A "supports N concurrent users" claim should come with the conc=N row (TTFT p95, per-user tok/s, zero errors).

Token counts use the server's `usage.completion_tokens` (requested via `stream_options.include_usage`);
if the server omits it, the number of content chunks is used instead.
