#!/usr/bin/env bash
# Run every ablation config end-to-end then build the final table.
#
# Usage:
#     bash scripts/run_everything.sh                    # full 500-topic run
#     TOPIC_LIMIT=10 bash scripts/run_everything.sh     # quick 10-topic test
#     CONFIGS=single_llm,cot,full bash scripts/run_everything.sh  # subset
#
# All flags can be overridden via env vars:
#     SPLIT=ddo_sample
#     MODEL=~/models/Qwen2.5-3B-Instruct
#     GPUS=0,1,2,3
#     TOPIC_LIMIT=0           # 0=all
#     CONFIGS=                # empty=all
#     VLLM_PY=~/env-vllm/bin/python
#     TOPIC_FILE=data/eval/ddo_sample_topics.jsonl

set -euo pipefail

# ---- Config with sensible defaults ----
SPLIT="${SPLIT:-ddo_sample}"
MODEL="${MODEL:-$HOME/models/Qwen2.5-3B-Instruct}"
GPUS="${GPUS:-0,1,2}"
TOPIC_LIMIT="${TOPIC_LIMIT:-0}"
CONFIGS="${CONFIGS:-}"
VLLM_PY="${VLLM_PY:-$HOME/env-vllm/bin/python}"
TOPIC_FILE="${TOPIC_FILE:-data/eval/${SPLIT}_topics.jsonl}"

# ---- Resolve the project root from this script's directory ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "MAJ-Debate — full ablation run"
echo "========================================================================"
echo "Project root : $PROJECT_ROOT"
echo "Split        : $SPLIT"
echo "Model        : $MODEL"
echo "GPUs         : $GPUS"
echo "Topic limit  : $TOPIC_LIMIT  (0 = all)"
echo "Configs      : ${CONFIGS:-all 8}"
echo "vLLM python  : $VLLM_PY"
echo "Topic file   : $TOPIC_FILE"
echo "========================================================================"

# ---- Sanity checks ----
for p in "$VLLM_PY" "$MODEL" "$TOPIC_FILE" "$SCRIPT_DIR/run_all_ablations.py"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: required path does not exist: $p" >&2
        exit 1
    fi
done

# ---- Build orchestrator args ----
ARGS=(
    --split "$SPLIT"
    --model "$MODEL"
    --gpus "$GPUS"
    --topic-limit "$TOPIC_LIMIT"
    --topic-file "$TOPIC_FILE"
    --vllm-python "$VLLM_PY"
)
if [ -n "$CONFIGS" ]; then
    ARGS+=(--configs "$CONFIGS")
fi

# ---- Step 0: clean up zombie GPU processes from prior failed runs ----
echo
echo ">>> [0/2] Cleaning up zombie GPU processes..."
echo
"$VLLM_PY" "$SCRIPT_DIR/cleanup_gpus.py" --kill -v || echo "  (cleanup had issues, continuing anyway)"

# ---- Step 1: run ablations ----
echo
echo ">>> [1/2] Running all ablation configurations..."
echo
"$VLLM_PY" "$SCRIPT_DIR/run_all_ablations.py" "${ARGS[@]}"

# ---- Step 2: aggregate metrics ----
echo
echo ">>> [2/2] Aggregating metrics into ablation table..."
echo
"$VLLM_PY" "$SCRIPT_DIR/evaluate_ablations.py" --split "$SPLIT"

# ---- Final pointer ----
OUT_DIR="outputs/ablations/$SPLIT"
echo
echo "========================================================================"
echo "ALL DONE"
echo "========================================================================"
echo
echo "Final artifacts:"
echo "  CSV (for LaTeX table):   $OUT_DIR/ablation_table.csv"
echo "  Markdown (quick view):   $OUT_DIR/ablation_table.md"
echo "  Full metrics JSON:        $OUT_DIR/ablation_table.json"
echo "  Per-run manifest:         $OUT_DIR/ablation_runs.json"
echo "  Per-config logs:          $OUT_DIR/logs/"
echo
echo "Pretty-print the markdown table:"
echo "    cat $OUT_DIR/ablation_table.md"