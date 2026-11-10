# Financial Education LLM Assistant

[![tests](https://github.com/CrocoWilly/FinancialEducationLLMAssistant/actions/workflows/tests.yml/badge.svg)](https://github.com/CrocoWilly/FinancialEducationLLMAssistant/actions/workflows/tests.yml)

A Traditional-Chinese financial Q&A assistant: a web/mobile chat UI on AWS, answering in real time from a 7B financial LLM running on a home GPU.

> **Status:** ran in September 2025 and was used by 8 people ([numbers below](#in-production-september-2025)). The GPU machine has since been retired, so the live deployment is offline; the demo video, the broker, and its tests remain.

## TL;DR

**Solo project.** I designed, deployed, and tested the whole pipeline. Dify (chat UI and orchestration), vLLM (inference) and the ssc-FinLLM model are off-the-shelf; **the FastAPI broker that connects them and the benchmark tooling are my code.**

- **Cloud-to-home streaming pipeline.** Dify on AWS EC2 → my broker → Tailscale → vLLM on a local RTX 3090, with tokens streamed end to end over SSE (median ~61 tok/s per answer across 59 real answers).
- **Low-overhead broker.** At 50 concurrent streams it adds ≤ ~10 ms to time-to-first-token (p95), with 0 errors and no per-stream slowdown, while converting every chunk from Simplified to Traditional Chinese.
- **Found and fixed a concurrency bug by load testing.** The original semaphore logic let streams bypass the cap entirely and leaked slots on idle, dropping throughput from ~197 to 66 tok/s. I reproduced it with a mock upstream, fixed it, and verified the fix with the same test. [Details ↓](#case-study-a-concurrency-cap-that-didnt-cap)

## Demo

[![Demo video](./doc_image/UI.png)](https://drive.google.com/file/d/13gUPVPvrnMjAFcjswvMSbEEROo9Srip6/view?usp=sharing)

Click the screenshot to watch the real-time demo ([short link](https://reurl.cc/EQx22R)).

## Architecture

```mermaid
flowchart LR
    U[Browser<br/>web / mobile] -->|HTTPS| D

    subgraph EC2[AWS EC2 · Docker]
        D[Dify v1.8.1<br/>Nginx + UI + API] -->|OpenAI-compatible<br/>HTTP + SSE| B[ssc-broker<br/>FastAPI · my code]
    end

    B -->|Tailscale<br/>private tunnel| V

    subgraph PC[Home PC · RTX 3090]
        V[vLLM<br/>ssc-FinLLM 7B FP16]
    end

    V -.->|token stream| B -.->|簡→繁 per chunk| D -.-> U
```

| Component | Role | Source |
|---|---|---|
| Dify v1.8.1 | Chat UI, conversation handling, the chatflow (retrieval → LLM → answer); configured with the broker as an "OpenAI-API-compatible" model provider | [langgenius/dify@1.8.1](https://github.com/langgenius/dify/tree/1.8.1), standard Docker deployment |
| **ssc-broker** | Protocol adapter, 繁↔簡 conversion, concurrency control, timeouts, error mapping | **[`ssc-broker/`](ssc-broker/)** |
| vLLM | Serves the model over an OpenAI-compatible API with token streaming | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| ssc-FinLLM 7B | Chinese financial LLM (InternLM2 architecture, FP16) | [haidizym/ssc-FinLLM](https://github.com/haidizym/ssc-FinLLM) |

## Design decisions

**Why a broker instead of pointing Dify straight at vLLM?**
Dify and vLLM don't quite fit together. Dify probes the endpoint with `GET /v1/chat/completions` when you add a model, and vLLM answers 404. Dify may also send other model names (e.g. `gpt-4o-mini`) that vLLM doesn't serve. The broker answers the probe itself and forces the model name. It also gives me one place for language conversion and for protecting the GPU, without modifying either project.

**Why convert 繁→簡 on input and 簡→繁 on output?**
ssc-FinLLM is a Simplified-Chinese model, but the users read Traditional Chinese. Input is converted with OpenCC `t2s`; output uses `s2twp`, which also maps vocabulary to Taiwan usage (e.g. 软件 → 軟體), not just characters. Output is converted **per SSE chunk**, so streaming stays real-time instead of waiting for the full answer.

**Why cap concurrency in the broker when vLLM already batches requests?**
vLLM accepts every request and queues internally. On a single 24 GB consumer GPU, too many concurrent sequences compete for KV-cache memory and every user slows down. A semaphore in the broker (`BROKER_MAX_INFLIGHT`) bounds how much work reaches the GPU. Requests beyond the cap wait in a queue with a timeout and then get a fast `503` with `Retry-After`, instead of hanging.

**Why Tailscale between EC2 and the home PC?**
The GPU machine needs no public port or router configuration. Traffic goes through an encrypted WireGuard tunnel, and EC2 reaches the PC at a stable private address. Running the GPU at home avoids paying for a cloud GPU around the clock.

**Who can reach the GPU?**
The broker only needs to be reachable by Dify, which talks to it over the Docker network, so its port is published on `127.0.0.1` only. On top of that, `/v1/*` requires a shared API key (`Authorization: Bearer …`, compared with `hmac.compare_digest` to avoid timing leaks), checked before rate limiting and before a concurrency slot is taken, so unauthenticated traffic never queues for the GPU. `docker compose` refuses to start without a key.

**Timeouts and failure modes.**
Connect and read timeouts are separate. The read timeout applies *between chunks*, so a long but healthy stream isn't cut off, while a stalled one is. Errors map to distinct statuses: `502` when vLLM is unreachable, `504` on timeout, `503` when the queue is full, `429` when rate-limited, `401` without a valid key. If vLLM fails mid-stream, the client gets a final SSE error event. When a client disconnects, the slot is still released (verified by test).

## Chatflow and knowledge retrieval

The Dify app 「金融知識小幫手」 is a chatflow: **start → knowledge retrieval → LLM → answer**.

- **Knowledge:** three personal-finance course notes (introduction, time value of money, annuities), split into 262 chunks. Retrieval is Dify's *economy* mode, a keyword index without embeddings, returning the top 4 chunks.
- **LLM step:** ssc-FinLLM through the broker, temperature 0.7, max 512 tokens, with the retrieved chunks injected into the system prompt.
- **System prompt rules:** answer in Traditional Chinese, step by step; show the assumptions and steps for any calculation; if the retrieved excerpts don't support an answer, say 「資料不足」 and list what information is missing; no investment solicitation, return guarantees, or stock tips.

**What went wrong with vector search.** I first built a second knowledge base in *high-quality* mode with `text-embedding-3-large`. All three documents failed to index with `404 {"detail": "Not Found"}`. That body is FastAPI's default 404, which points to the cause: Dify's `OPENAI_API_BASE` was set to the broker, so the embedding calls most likely went to the broker, which only implements chat endpoints. I switched to the keyword-indexed knowledge base, which is what the app ran with. The fix for vector search would be to point embeddings at a real embedding endpoint, or add `/v1/embeddings` to the broker.

## Case study: a concurrency cap that didn't cap

**Symptom found in review.** The original broker released its semaphore slot with:

```python
finally:
    if sem.locked():   # only true when the count is exactly 0
        sem.release()
```

**Reproduction.** I wrote a mock vLLM ([`benchmark/mock_upstream.py`](benchmark/mock_upstream.py)) that streams at a fixed 60 tok/s and records how many requests it is serving at once. Because the upstream speed is fixed, any change in the numbers comes from the broker. Setup: cap = 4, ~1.2 s per request, a fresh broker and mock process for every cell, and non-stream cells repeated 3× with identical results.

| Clients | Before — stream<br/>peak in-flight | Before — non-stream<br/>peak / throughput | After — stream<br/>peak in-flight | After — non-stream<br/>peak / throughput |
|---|---|---|---|---|
| 4  | 4 | 4 / **66 tok/s** ⚠️ | 4 | 4 / 197 tok/s |
| 8  | **8** ❌ | 4 / 194 tok/s | 4 | 4 / 197 tok/s |
| 16 | **16** ❌ | 4 / 194 tok/s | 4 | 4 / 197 tok/s |

*With the cap working, throughput should be about 4 slots × 60 tok/s ≈ 197 tok/s.*

**Root cause: two opposite failures.**
- **Stream: the cap was never enforced.** The handler returns a `StreamingResponse` immediately, so the `finally` runs *before* streaming starts and releases the slot early. The generator then releases it again. Every client got through.
- **Non-stream: slots leaked when nobody was queued.** A slot was returned only if the count happened to be 0. When 4 requests finished together with no one waiting, 3 slots were lost and the broker dropped to one request at a time. Under constant overload it looked fine. In production, every lull in traffic would leak slots until the broker was permanently serialized.

**Lesson.** Peak in-flight alone looked healthy in the non-stream case. Only throughput exposed the leak.

**Fix.** Each request owns exactly one slot, wrapped in an idempotent `release()`, and the semaphore is an `asyncio.BoundedSemaphore`, so any future double release raises `ValueError` instead of silently raising the cap. Non-stream requests release it in `finally`. Streams release it when the generator finishes, with a background-task fallback in case the client disconnects before streaming starts. The same test now shows the peak stays at 4 and throughput at ~197 tok/s, and extra requests queue as expected.

**Regression tests.** Both failure modes are now pytest cases that run in CI (see [Testing](#testing)). Run against the original `broker.py`, they fail with `assert 6 == 2` (stream cap bypassed) and `assert 1 == 2` (slots leaked).

## In production (September 2025)

Numbers from the deployment's Dify database and broker log, recovered before the server was shut down:

| Metric | Value |
|---|---|
| Test accounts / users who asked questions | 10 / 8 |
| Conversations / answers | 20 / 61, over 2 days (2025-09-26 and 27) |
| Most users active at the same time | 2 within any 5 min, 3 within any 15 min |
| Broker chat requests | 75, all `200` |
| Answer latency (LLM call) | median 4.98 s, p90 7.08 s |
| Generation speed per answer | median 61.2 tok/s (59 answers) |
| Peak concurrent generations | 2 |

Generation speed is answer tokens ÷ LLM latency. The token counts are Dify's own estimate, not the model's tokenizer, so the speed is approximate. The median matches what I saw during development, and a 7B FP16 model on an RTX 3090 is memory-bandwidth-bound at roughly that speed: every generated token reads all 15.48 GB of weights, and 936 GB/s ÷ 15.48 GB ≈ 60 tok/s. Real traffic never exceeded 2 concurrent generations, so the concurrency behaviour is covered by the broker benchmarks and tests below rather than by production load.

## Performance

**Broker overhead.** Measured with [`benchmark/bench.py`](benchmark/bench.py) against the mock upstream (60 tok/s, 200 ms TTFT), 256 tokens per request, cap 64, everything on one Apple M4 machine:

| Concurrent streams | Errors | TTFT p95: direct → via broker | Per-stream tok/s | Aggregate tok/s |
|---|---|---|---|---|
| 1  | 0 | 0.23 s → 0.23 s | 60.0 | 57 |
| 16 | 0 | 0.22 s → 0.22 s | 60.0 | 915 |
| 32 | 0 | 0.22 s → 0.23 s | 60.0 | 1,828 |
| 50 | 0 | 0.23 s → 0.24 s | 60.0 | 2,848 |

The broker is not the bottleneck. System capacity is bounded by GPU throughput.

**Model speed.** Median ~61 tok/s per answer in production on the RTX 3090 (see above). A controlled GPU sweep with `bench.py` is listed under next steps.

### Capacity estimate (not measured)

A back-of-envelope estimate of how many chat sessions one RTX 3090 can serve, from the model config and the production usage above.

**1. How many generations fit in GPU memory at once?**

All sizes in GiB (2³⁰ bytes); the RTX 3090's 24 GB is 24 GiB.

| | | Source |
|---|---|---|
| VRAM vLLM may use | 21.6 GiB | 90% of 24 GiB, vLLM's default `gpu_memory_utilization` (production didn't override it) |
| − model weights | 14.41 GiB | 15,475,417,088 bytes, `total_size` in [`pytorch_model.bin.index.json`](models/ssc-finllm-model/pytorch_model.bin.index.json) |
| − runtime overhead (activations, CUDA graphs) | ~1–1.5 GiB | rule of thumb |
| **= KV cache** | **~5.7–6.2 GiB** | |
| KV cache per token | 128 KiB | 2 (K and V) × 32 layers × 8 KV heads × 128 dims × 2 bytes |
| **= tokens that fit** | **~47–51 k** | |

The per-token figure comes from [`config.json`](models/ssc-finllm-model/config.json): `num_hidden_layers` 32, `num_key_value_heads` 8 (grouped-query attention, so 8 K/V heads are shared by the 32 query heads), head size `hidden_size` 4096 ÷ `num_attention_heads` 32 = 128, and `torch_dtype` float16 = 2 bytes.

A production request is ~1,000 tokens at the median (727 prompt incl. retrieved chunks + 280 answer) and ~2,300 at p90 (1,800 prompt + 512 max answer). So **~20–22 p90-sized generations** fit at once (~46–50 at the median).

**2. How busy is one session?** Generating an answer takes ~5 s (280 tokens at ~61 tok/s), and users waited a median 24 s between turns. A session is generating ~5 / (5 + 24) ≈ 17% of the time.

**3. Sessions supported.** 20 sessions → ~3.4 concurrent generations on average; 50 sessions → ~8.6 on average, peaks around 16. Even that peak stays under the ~20–22 limit from step 1, so **one RTX 3090 should serve roughly 20–50 active chat sessions.**

The broker's default cap, `BROKER_MAX_INFLIGHT=16`, is consistent with this estimate: it sits a little below the ~20–22 p90-sized generations that fit in the KV cache, leaving headroom for longer prompts and the overhead guess. Requests beyond it wait in the broker's queue (and get a `503` after the timeout) instead of being admitted to vLLM, where they would compete for KV cache and could be preempted.

**Caveats.**
- Per-stream speed drops as vLLM batches more sequences, which lengthens each generation and raises the busy fraction. Only a GPU benchmark can pin this down.
- Usage patterns come from two days of trial use; real users probably pause longer, which would make this estimate conservative.
- Token counts are Dify's estimates, not the model's tokenizer.
- These are *sessions*, not simultaneous generations: 50 sessions does **not** mean 50 requests generating at once.

## Quickstart (no GPU needed)

Runs the broker against the mock upstream and load-tests it:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r ssc-broker/requirements.txt -r benchmark/requirements.txt

uvicorn mock_upstream:app --app-dir benchmark --port 8000 --log-level warning &
UPSTREAM_LLM_BASE=http://127.0.0.1:8000/v1 BROKER_MAX_INFLIGHT=4 \
    uvicorn broker:app --app-dir ssc-broker --port 8080 --log-level warning &
sleep 3

python benchmark/bench.py --base-url http://localhost:8080/v1 --concurrency 1,4,8 --max-tokens 60
curl localhost:8000/stats      # peak_inflight stays at 4
kill %1 %2
```

The output is streamed in Traditional Chinese, converted by the broker from the mock's Simplified text. For production deployment and configuration, see [`ssc-broker/README.md`](ssc-broker/README.md). For GPU benchmarking, see [`benchmark/README.md`](benchmark/README.md).

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v
```

31 tests in [`tests/test_broker.py`](tests/test_broker.py) run in about 15 s and need no GPU. GitHub Actions runs them on every push and pull request. Each test starts a real broker and mock vLLM on free local ports and talks to them over HTTP, so streaming, timeouts and disconnects go through the same code paths as in production. The logging tests go one step further and launch `uvicorn` as a separate process, because a missing log handler only shows up there.

| Area | What is checked |
|---|---|
| Protocol | Dify probe answered locally, model name forced, `/health` |
| Language | input 繁→簡, output 簡→繁 (stream and non-stream) |
| Concurrency | cap never exceeded; slots returned after normal completion, upstream errors and client disconnects; over-release raises instead of silently raising the cap; `503` when the queue is full |
| Failures | `502` unreachable upstream, upstream status passed through, `504` timeout, SSE error event on mid-stream timeout |
| Rate limit | `429` inside the window |
| Auth | `401` without / with a wrong key on `/v1/*`; correct key accepted; `/health` and Dify probe stay open; rejected requests never reach the GPU or take a slot |
| Logging | request log lines and the missing-key warning actually appear in uvicorn's output |

To check that the tests actually catch bugs, I ran them against the original `broker.py` (19 of 31 fail, each matching a known bug or a feature it lacked). I also deliberately broke each slot-release path, the API-key check and the log handler in the fixed broker; every one of those changes is caught by at least one test.

## Limitations & next steps

- **GPU benchmark:** the production speed comes from Dify's approximate token counts; a controlled single-stream and concurrency sweep on a 24 GB GPU with `bench.py` would replace it. The original GPU machine is retired, so this needs a rented GPU.
- **Retries:** failed upstream calls are not retried. A retry is only safe before streaming starts.
- **Rate limiting:** the broker only sees Dify's IP, so its per-IP limit would throttle all users together; it is off by default. Per-user limits should be set in Nginx (which sees client IPs; its config lives in the Dify deployment on EC2, not in this repo) or in Dify. The limiter's per-IP table is also in-memory and never pruned.
- **Multiple workers / instances:** the concurrency cap and rate limit are per process, so N uvicorn workers allow N × `BROKER_MAX_INFLIGHT`. A shared cap would need Redis or similar.
- **Reproducibility:** Dify ran as the stock 1.8.1 Docker deployment with two changes: `OPENAI_API_BASE=http://broker:8080` in `.env`, and `extra_hosts: host.docker.internal:host-gateway` on the `api`, `worker` and `worker_beat` services. vLLM ran with `--dtype float16 --max-model-len 4096 --trust-remote-code` and default memory settings; the full command is in [`benchmark/README.md`](benchmark/README.md).

## Repository structure

```
ssc-broker/          FastAPI broker (my code): broker.py, Dockerfile, docker-compose.yml
benchmark/           bench.py load tester + mock_upstream.py fake vLLM
tests/               pytest suite for the broker (runs in CI)
.github/workflows/   GitHub Actions: run pytest on every push / PR
models/ssc-finllm-model/   model config & tokenizer files (weights not committed)
doc_image/           demo screenshot & video
```

## Tech stack

Python · FastAPI · httpx · asyncio · SSE · OpenCC · pytest · GitHub Actions · Docker · AWS EC2 (Ubuntu 22.04) · Tailscale · vLLM · Dify

## References

- [Dify v1.8.1](https://github.com/langgenius/dify/tree/1.8.1) · [docs](https://docs.dify.ai/)
- [vLLM](https://github.com/vllm-project/vllm)
- [ssc-FinLLM](https://github.com/haidizym/ssc-FinLLM)
- [FastAPI](https://fastapi.tiangolo.com/) · [OpenCC](https://github.com/BYVoid/OpenCC) · [Tailscale](https://tailscale.com/)
