# Benchmark

| File | What it is |
|---|---|
| `bench.py` | Load generator for any OpenAI-compatible endpoint (broker, vLLM, mock). Reports TTFT, per-request decode tok/s, e2e latency, aggregate tok/s and errors per concurrency level. |
| `sessions.py` | Simulated chat sessions: N users who ask, read the streamed answer, pause, and ask again. Defaults come from production usage (prompt ~727 tokens, think time median 24 s). Can scrape vLLM's `/metrics` for running/waiting requests, KV-cache usage and preemptions. |
| `prompts.py` | Production-shaped prompts: the real system prompt, original filler text in place of the retrieved course-note chunks, and a question. Length is calibrated against the model's tokenizer. |
| `mock_upstream.py` | Fake vLLM (no GPU). Streams Simplified-Chinese tokens at a fixed rate; `/stats` reports the **peak number of in-flight requests**, `/metrics` mimics a few vLLM gauges. |
| `gpu_run.sh` | One-shot GPU benchmark: download the model, start vLLM with the production flags, start the broker, run everything above, pack the results. |

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

## 2. Real model on a GPU

ssc-FinLLM 7B in FP16 needs ~14.4 GiB of VRAM for the weights alone, so use a 24 GB card
(RTX 3090 / 4090 / A10 / L4 …). A rented cloud GPU works fine.

```bash
git clone https://github.com/CrocoWilly/FinancialEducationLLMAssistant && cd FinancialEducationLLMAssistant
export HF_TOKEN=hf_...          # read access to the model repo (HF_REPO, default CrocoWilly/ssc-FinLLM-7B)
bash benchmark/gpu_run.sh       # ~30-40 min; writes /workspace/results-<time>.tar.gz
```

The script starts vLLM with **the same command the production GPU machine used**:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_DIR --tokenizer $MODEL_DIR --served-model-name ssc-FinLLM \
  --trust-remote-code --dtype float16 --max-model-len 4096 --host 0.0.0.0 --port 8000
```

and then runs, all with production-sized prompts (~727 tokens) and `max_tokens=512`:

| Step | What it measures |
|---|---|
| single stream, direct to vLLM | the generation speed one user sees |
| concurrency sweep 1–32, direct to vLLM | aggregate throughput and per-user speed as batching grows |
| 10 / 20 / 50 / 80 chat sessions through the broker (cap 16) | latency, errors, queueing, vLLM KV-cache usage — checks the 20–50 sessions estimate |
| 50 / 80 chat sessions direct to vLLM (no cap) | what happens without the broker's cap (KV-cache pressure, preemptions) |

`MOCK=1 bash benchmark/gpu_run.sh` dry-runs the whole script without a GPU, using the mock instead of vLLM.

How to read it:

- **decode tok/s @ conc=1** → the single-user generation speed.
- **agg tok/s** → total server throughput; rises with concurrency until the GPU saturates.
- **decode tok/s @ conc=N** → what each user experiences when N people are generating at once.
- **sessions**: TTFT includes time queued in the broker; `infl avg/max` is client-side requests in flight, `vLLM run/wait` is what vLLM itself was running and queueing.

Token counts use the server's `usage.completion_tokens` (requested via `stream_options.include_usage`);
if the server omits it, the number of content chunks is used instead.
