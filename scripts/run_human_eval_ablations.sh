#!/usr/bin/env bash
# Run all 8 MAJ-Debate ablation configurations on the 50-topic human test set.
#
# This is a thin wrapper that:
#   1. Verifies the human topics .jsonl exists
#   2. Calls run_all_ablations.py with --split human_eval and --topic-file pointing at it
#   3. Uses the same ablation configs as the DDO run, just on a different topic file
#
# All outputs land under:
#   outputs/stage{1,2,3,4}/human_eval[_<config>]/
#   outputs/ablations/human_eval/
#
# Usage:
#     bash scripts/run_human_eval_ablations.sh
#
# Environment overrides (same as run_everything.sh):
#     MODEL=~/models/Qwen2.5-3B-Instruct
#     GPUS=0,1,2
#     VLLM_PY=~/env-vllm/bin/python
#     TOPIC_FILE=data/eval/human_eval_topics.jsonl

set -euo pipefail

SPLIT="human_eval"
MODEL="${MODEL:-$HOME/models/Qwen2.5-3B-Instruct}"
GPUS="${GPUS:-0,1,2}"
VLLM_PY="${VLLM_PY:-$HOME/env-vllm/bin/python}"
TOPIC_FILE="${TOPIC_FILE:-data/eval/human_eval_topics.jsonl}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "MAJ-Debate — human evaluation ablation run (50 topics)"
echo "========================================================================"
echo "Project root : $PROJECT_ROOT"
echo "Split        : $SPLIT"
echo "Topic file   : $TOPIC_FILE"
echo "Model        : $MODEL"
echo "GPUs         : $GPUS"
echo "========================================================================"

if [ ! -f "$TOPIC_FILE" ]; then
    echo "ERROR: topic file not found: $TOPIC_FILE"
    echo
    echo "Save the human test set JSONL from the chat to that path:"
    echo "    mkdir -p data/eval"
    echo "    <paste the 50-topic JSONL into data/eval/human_eval_topics.jsonl>"
    exit 1
fi

N_TOPICS=$(wc -l < "$TOPIC_FILE")
echo "Found $N_TOPICS topics in $TOPIC_FILE"
echo

# Ensure GPUs are clean before launch
"$VLLM_PY" "$SCRIPT_DIR/cleanup_gpus.py" --kill -v || true

echo
echo ">>> Running all 8 ablation configurations on the human test set..."
echo

"$VLLM_PY" "$SCRIPT_DIR/run_all_ablations.py" \
    --split "$SPLIT" \
    --topic-file "$TOPIC_FILE" \
    --model "$MODEL" \
    --gpus "$GPUS" \
    --vllm-python "$VLLM_PY"

echo
echo ">>> Aggregating metrics..."
echo
"$VLLM_PY" "$SCRIPT_DIR/evaluate_ablations.py" --split "$SPLIT"

echo
echo "========================================================================"
echo "DONE — human-eval ablation run complete"
echo "========================================================================"
echo
echo "Model verdicts are now in:"
echo "    outputs/stage4/human_eval[_<config>]/stage4_judgments.json"
echo
echo "Next steps:"
echo "    # Build annotator sheets (one CSV per annotator, 50 rows × args):"
echo "    $VLLM_PY $SCRIPT_DIR/build_annotation_sheet.py --n-annotators 3"
echo
echo "    # Have the 3 annotators fill in their verdict + 1-5 Likert."
echo "    # When they return filled CSVs, put them at:"
echo "    #     annotations/human_eval_annotator_1.csv"
echo "    #     annotations/human_eval_annotator_2.csv"
echo "    #     annotations/human_eval_annotator_3.csv"
echo
echo "    # Then score everything:"
echo "    $VLLM_PY $SCRIPT_DIR/score_human_eval.py"
