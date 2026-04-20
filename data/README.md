# Data Layout For Real Proposal-Scale Runs

This project now expects real input files instead of mock/demo data.

Create these files before running the full pipeline:

- `data/eval/ddo_sample_topics.jsonl`
- `data/eval/human_eval_topics.jsonl`

Each line must be a JSON object with at least:

```json
{
  "topic_id": "DDO_0001",
  "topic_text": "Resolved: ...",
  "domain": "policy",
  "benchmark_label": "PRO",
  "source_dataset": "DDO",
  "source_ref": "optional original debate id or topic id"
}
```

Notes:

- `benchmark_label` is required for DDO benchmark topics.
- For `human_eval_topics.jsonl`, you may leave `benchmark_label` as `null` because final correctness comes from human annotation.
- Use exactly `PRO` or `CON` when a label is present.
- Keep `topic_id` stable across all stages and all reruns.

Recommended final target:

- `ddo_sample_topics.jsonl`: `500` records
- `human_eval_topics.jsonl`: `50` records
