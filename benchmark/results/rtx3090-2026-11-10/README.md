# RTX 3090 run, 2026-11-10

Raw output of [`gpu_run.sh`](../../gpu_run.sh) on a rented RTX 3090 (Vast.ai, 24 GB, CUDA 13.1 driver),
vLLM 0.33.0 started with the production flags. Versions are in `env.txt`; vLLM's KV-cache sizing is
in `vllm_kv_cache.txt`. Every session level ran for 180 s.

| File | Commit | Command | What |
|---|---|---|---|
| `single.json`, `sweep.json` | `32b9545` | `STAGES="single sweep sessions direct stress"` (cap was fixed at 16 then) | 1 stream; 1–32 concurrent requests, direct to vLLM |
| `sessions_broker.json` | `32b9545` | same | 10/20/50/80 sessions, broker cap 16 |
| `sessions_direct.json` | `32b9545` | same | 50/80 sessions, no cap |
| `stress_natural_*.json` | `32b9545` | same | 150/300 sessions (broker cap 16), 300 (no cap) |
| `stress_long_*.json` | `32b9545` | same | 50/100 sessions with 512-token answers, cap 16 vs no cap |
| `cap32_*.json` | `3c6265b` | `STAGES=cap BROKER_CAP=32` | the broker-side stress levels again with cap 32 |
| `fill_*.json` | `38f4e7f` | `STAGES=fill BROKER_CAP=32` | 150 sessions with no cap; 50 sessions with cap 32 |

Earlier runs on the same machine are not included: one used the defaults before the stress stage
existed, and in both of them every prompt repeated the same filler text, so vLLM's prefix cache hit
95–98% and the numbers were too optimistic (fixed in `32b9545`).

To rerun the cap-16 levels on the current script, which defaults to cap 32, pass `BROKER_CAP=16`.
