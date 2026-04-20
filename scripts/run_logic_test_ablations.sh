#!/usr/bin/env bash
# Run all 8 ablation configurations on the 50-topic logic test set.
#
# This is the "correctness agreement" benchmark — topics where the ground-truth
# answer is logically or empirically verifiable, unlike DDO's opinion-based
# crowd votes or the original human_eval opinion topics.
#
# Categories: syllogism, paradox, fallacy, math_fact, empirical_fact (10 each)
# Label balance: 24 PRO / 26 CON (near-balanced)
#
# Runtime estimate:
#   - Stage 1: ~3 min per fresh config (2 configs have own stage 1: two_agents, dung_no_agents)
#   - Stage 2: ~8 min per fresh config (data-parallel across 3 GPUs)
#   - Stage 3: <1 min (CPU)
#   - Stage 4: ~2 min per config (single batch)
#   - Full run: ~30-45 minutes
#
# Usage:
#     bash scripts/run_logic_test_ablations.sh
#
# After it finishes, compare to DDO:
#     python scripts/compare_benchmarks.py

set -euo pipefail

SPLIT="logic_test"
MODEL="${MODEL:-$HOME/models/Qwen2.5-3B-Instruct}"
GPUS="${GPUS:-0,1,2}"
VLLM_PY="${VLLM_PY:-$HOME/env-vllm/bin/python}"
TOPIC_FILE="${TOPIC_FILE:-data/eval/logic_test_topics.jsonl}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "MAJ-Debate — logic test benchmark (50 topics, correctness-verifiable)"
echo "========================================================================"
echo "Project root : $PROJECT_ROOT"
echo "Split        : $SPLIT"
echo "Topic file   : $TOPIC_FILE"
echo "Model        : $MODEL"
echo "GPUs         : $GPUS"
echo "========================================================================"
echo

if [ ! -f "$TOPIC_FILE" ]; then
    echo "ERROR: topic file not found: $TOPIC_FILE"
    echo
    echo "Save the logic test set JSONL from the chat to that path:"
    echo "    mkdir -p data/eval"
    echo "    <copy logic_test_topics.jsonl to data/eval/>"
    exit 1
fi

N_TOPICS=$(wc -l < "$TOPIC_FILE")
echo "Found $N_TOPICS topics in $TOPIC_FILE"

# Distribution summary
echo
echo "Category breakdown:"
"$VLLM_PY" -c "
import json
from collections import Counter
rows = []
for line in open('$TOPIC_FILE'):
    line = line.strip()
    if line:
        rows.append(json.loads(line))
cats = Counter(r['domain'] for r in rows)
labels = Counter(r['benchmark_label'] for r in rows)
print(f'  Domains: {dict(cats)}')
print(f'  Labels: {dict(labels)}')
"

# Clean GPUs before launch
echo
"$VLLM_PY" "$SCRIPT_DIR/cleanup_gpus.py" --kill -v || true

echo
echo ">>> Running all 8 ablation configurations on logic_test split..."
echo
"$VLLM_PY" "$SCRIPT_DIR/run_all_ablations.py" \
    --split "$SPLIT" \
    --topic-file "$TOPIC_FILE" \
    --model "$MODEL" \
    --gpus "$GPUS" \
    --vllm-python "$VLLM_PY"

echo
echo ">>> Aggregating metrics for logic_test..."
echo
"$VLLM_PY" "$SCRIPT_DIR/evaluate_ablations.py" --split "$SPLIT"

echo
echo ">>> Building persuasion-vs-correctness comparison..."
echo
"$VLLM_PY" "$SCRIPT_DIR/compare_benchmarks.py" \
    --persuasion-split ddo_sample \
    --correctness-split "$SPLIT" || echo "  (compare failed — need DDO table too)"

echo
echo "========================================================================"
echo "DONE"
echo "========================================================================"
echo
echo "Key outputs:"
echo "    outputs/ablations/$SPLIT/ablation_table.md    (correctness-only)"
echo "    outputs/ablations/persuasion_vs_correctness.md  (DDO vs logic side-by-side)"
echo
echo "Inspect for duplicate-rationale bugs:"
echo "    $VLLM_PY scripts/inspect_ablations.py --split $SPLIT"
echo "    $VLLM_PY scripts/diagnose_duplicates.py --split $SPLIT"
