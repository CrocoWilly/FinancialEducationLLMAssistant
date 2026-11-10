#!/usr/bin/env bash
# One-shot GPU benchmark: download the model, start vLLM with the production
# flags, start the broker in front of it, run every benchmark, pack the results.
#
# On a fresh GPU box (24 GB card, e.g. RTX 3090), from a clone of this repo:
#     export HF_TOKEN=hf_...            # read access to the model repo
#     bash benchmark/gpu_run.sh
#
# Optional env: HF_REPO, WORK (default /workspace), VLLM_VERSION (default: latest),
#               SESSION_SECONDS (default 180), SKIP_INSTALL=1,
#               BROKER_PORT (default 8090; 8080 is often taken by Jupyter on rented boxes),
#               BROKER_CAP (broker's in-flight cap, default 32 like the broker),
#               MOCK=1 (dry run without a GPU: mock vLLM instead of the real one),
#               STAGES (default "single sweep sessions direct"; add "stress" for the
#                       overload tests, e.g. SKIP_INSTALL=1 STAGES=stress on a box
#                       that already ran the defaults; "cap" reruns only the
#                       broker-side stress levels, e.g. BROKER_CAP=32 STAGES=cap;
#                       "fill" runs 150 sessions with no cap and 50 through the
#                       broker, the two cells the other stages leave out).
set -euo pipefail

HF_REPO=${HF_REPO:-CrocoWilly/ssc-FinLLM-7B}
WORK=${WORK:-/workspace}
SESSION_SECONDS=${SESSION_SECONDS:-180}
REPO=$(cd "$(dirname "$0")/.." && pwd)
MODEL_DIR=$WORK/ssc-finllm-model
OUT=$WORK/results-$(date +%Y%m%d-%H%M)
KEY=bench-key
STAGES=${STAGES:-single sweep sessions direct}
has() { [[ " $STAGES " == *" $1 "* ]]; }
BROKER_PORT=${BROKER_PORT:-8090}
BROKER_CAP=${BROKER_CAP:-32}            # broker's BROKER_MAX_INFLIGHT
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
ulimit -n 65536 2>/dev/null || true   # hundreds of concurrent connections

step() { echo; echo "=== $(date +%T) $* ==="; }
wait_http() {  # wait_http URL SECONDS
  for _ in $(seq "$2"); do curl -sf "$1" >/dev/null && return 0; sleep 1; done
  echo "timed out waiting for $1"; return 1
}

# ---------------------------------------------------------------- setup
if [ "${SKIP_INSTALL:-0}" != 1 ] && [ "${MOCK:-0}" != 1 ]; then
  step "install vLLM ${VLLM_VERSION:-latest}, broker and benchmark deps"
  pip install -q -U pip
  pip install -q "vllm${VLLM_VERSION:+==$VLLM_VERSION}" huggingface_hub \
      -r "$REPO/ssc-broker/requirements.txt" -r "$REPO/benchmark/requirements.txt"
fi

if [ "${MOCK:-0}" != 1 ]; then
  step "download $HF_REPO"
  hf download "$HF_REPO" --local-dir "$MODEL_DIR"
  ls -la "$MODEL_DIR"

  step "environment"
  {
    nvidia-smi
    python --version
    python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
    git -C "$REPO" rev-parse --short HEAD
  } | tee "$OUT/env.txt"
fi

# ---------------------------------------------------------------- servers
for port in 8000 "$BROKER_PORT"; do
  if curl -s -o /dev/null "http://127.0.0.1:$port/"; then
    echo "port $port is already in use on this machine; free it or set BROKER_PORT"; exit 1
  fi
done

step "start vLLM with the production flags"
if [ "${MOCK:-0}" = 1 ]; then
  ( cd "$REPO/benchmark" && exec python -m uvicorn mock_upstream:app --port 8000 --log-level warning ) > "$OUT/vllm.log" 2>&1 &
else
# Same command the production GPU machine used (gpu_memory_utilization left at its default).
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --tokenizer "$MODEL_DIR" \
  --served-model-name ssc-FinLLM \
  --trust-remote-code \
  --dtype float16 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000 > "$OUT/vllm.log" 2>&1 &
fi
VLLM_PID=$!
if ! wait_http http://127.0.0.1:8000/health 1200; then tail -50 "$OUT/vllm.log"; exit 1; fi
grep -i -E "KV cache|GPU blocks|maximum concurrency|Memory profiling" "$OUT/vllm.log" | tee "$OUT/vllm_kv_cache.txt" || true

step "start broker (cap $BROKER_CAP, API key)"
( cd "$REPO/ssc-broker" && UPSTREAM_LLM_BASE=http://127.0.0.1:8000/v1 BROKER_API_KEY=$KEY \
    BROKER_MAX_INFLIGHT=$BROKER_CAP QUEUE_WAIT_TIMEOUT=60 \
    exec uvicorn broker:app --port $BROKER_PORT --log-level warning ) > "$OUT/broker.log" 2>&1 &
BROKER_PID=$!
trap 'kill $BROKER_PID $VLLM_PID 2>/dev/null || true' EXIT
wait_http http://127.0.0.1:$BROKER_PORT/health 30

cd "$REPO/benchmark"
METRICS=http://127.0.0.1:8000/metrics

# ---------------------------------------------------------------- benchmarks
step "warm-up"
python bench.py --base-url http://127.0.0.1:8000/v1 --concurrency 1 --rounds 2 --prompt-tokens 727 --max-tokens 64 >/dev/null

if has single; then
step "single stream, direct to vLLM (production-sized prompt)"
python bench.py --base-url http://127.0.0.1:8000/v1 --concurrency 1 --rounds 5 \
  --prompt-tokens 727 --max-tokens 512 --json-out "$OUT/single.json"
fi

if has sweep; then
step "concurrency sweep, direct to vLLM"
python bench.py --base-url http://127.0.0.1:8000/v1 --concurrency 1,2,4,8,16,32 --rounds 3 \
  --prompt-tokens 727 --max-tokens 512 --json-out "$OUT/sweep.json"
fi

if has sessions; then
step "chat sessions through the broker (cap $BROKER_CAP)"
python sessions.py --base-url http://127.0.0.1:$BROKER_PORT/v1 --api-key $KEY \
  --sessions 10,20,50,80 --duration "$SESSION_SECONDS" --metrics-url $METRICS \
  --label "via broker, cap $BROKER_CAP" --json-out "$OUT/sessions_broker.json"
fi

if has direct; then
step "chat sessions direct to vLLM (no cap)"
python sessions.py --base-url http://127.0.0.1:8000/v1 \
  --sessions 50,80 --duration "$SESSION_SECONDS" --metrics-url $METRICS \
  --label "direct to vLLM, no cap" --json-out "$OUT/sessions_direct.json"
fi

if has stress; then
# Natural-length answers, more sessions: where does the system start to queue?
step "stress: natural answers, through the broker"
python sessions.py --base-url http://127.0.0.1:$BROKER_PORT/v1 --api-key $KEY \
  --sessions 150,300 --duration "$SESSION_SECONDS" --metrics-url $METRICS \
  --label "stress, via broker, cap $BROKER_CAP" --json-out "$OUT/stress_natural_broker.json"
step "stress: natural answers, direct to vLLM"
python sessions.py --base-url http://127.0.0.1:8000/v1 \
  --sessions 300 --duration "$SESSION_SECONDS" --metrics-url $METRICS \
  --label "stress, direct to vLLM, no cap" --json-out "$OUT/stress_natural_direct.json"
# Worst case: every answer runs to 512 tokens, so KV cache fills up.
step "stress: 512-token answers, through the broker"
python sessions.py --base-url http://127.0.0.1:$BROKER_PORT/v1 --api-key $KEY \
  --sessions 50,100 --min-tokens 512 --duration "$SESSION_SECONDS" --metrics-url $METRICS \
  --label "stress, 512-token answers, via broker, cap $BROKER_CAP" --json-out "$OUT/stress_long_broker.json"
step "stress: 512-token answers, direct to vLLM"
python sessions.py --base-url http://127.0.0.1:8000/v1 \
  --sessions 50,100 --min-tokens 512 --duration "$SESSION_SECONDS" --metrics-url $METRICS \
  --label "stress, 512-token answers, direct to vLLM, no cap" --json-out "$OUT/stress_long_direct.json"
fi

if has cap; then
# Same broker-side levels as "stress", to compare one cap against another.
step "cap $BROKER_CAP: natural answers, through the broker"
python sessions.py --base-url http://127.0.0.1:$BROKER_PORT/v1 --api-key $KEY \
  --sessions 80,150,300 --duration "$SESSION_SECONDS" --metrics-url $METRICS \
  --label "via broker, cap $BROKER_CAP" --json-out "$OUT/cap${BROKER_CAP}_natural_broker.json"
step "cap $BROKER_CAP: 512-token answers, through the broker"
python sessions.py --base-url http://127.0.0.1:$BROKER_PORT/v1 --api-key $KEY \
  --sessions 50,100 --min-tokens 512 --duration "$SESSION_SECONDS" --metrics-url $METRICS \
  --label "512-token answers, via broker, cap $BROKER_CAP" --json-out "$OUT/cap${BROKER_CAP}_long_broker.json"
fi

if has fill; then
step "fill: natural answers, 150 sessions direct to vLLM"
python sessions.py --base-url http://127.0.0.1:8000/v1 \
  --sessions 150 --duration "$SESSION_SECONDS" --metrics-url $METRICS \
  --label "stress, direct to vLLM, no cap" --json-out "$OUT/fill_natural_direct_150.json"
step "fill: natural answers, 50 sessions through the broker (cap $BROKER_CAP)"
python sessions.py --base-url http://127.0.0.1:$BROKER_PORT/v1 --api-key $KEY \
  --sessions 50 --duration "$SESSION_SECONDS" --metrics-url $METRICS \
  --label "via broker, cap $BROKER_CAP" --json-out "$OUT/fill_cap${BROKER_CAP}_natural_broker_50.json"
fi

# ---------------------------------------------------------------- pack
step "pack results"
tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "results: $OUT.tar.gz"
