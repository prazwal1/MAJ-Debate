#!/usr/bin/env python3
"""
Dump everything needed for the progress report into a single markdown file.

Run this AFTER both ablation runs (ddo_sample + logic_test) finish.
It collects all the numbers, summaries, and diagnostic output into one
self-contained markdown file you can paste or upload.

Usage:
    python scripts/dump_progress_report_data.py
    -> outputs/progress_report_bundle.md
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / 'outputs').exists() or (p / 'scripts').exists():
            return p
    return cwd


ROOT = find_project_root()


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception as e:
        return {'error': str(e), 'path': str(p)}


def read_text(p):
    try:
        return Path(p).read_text()
    except Exception as e:
        return f'(not available: {e})'


def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return (r.stdout or '') + (r.stderr or '')
    except Exception as e:
        return f'(command failed: {e})'


def stage2_summary(path):
    """Extract just the summary (not the full 500-topic payload) from a
    stage 2 relations file. Returns a dict with topic_count, label_counts,
    confidence_stats, targeted_mode."""
    try:
        with open(path) as f:
            doc = json.load(f)
        summary = {
            'path': str(path),
            'topic_count': len(doc.get('topics', [])),
            'summary': doc.get('summary', {}),
        }
        # Sample one topic for auditability
        topics = doc.get('topics', [])
        if topics:
            t = topics[0]
            summary['first_topic_sample'] = {
                'topic_id': t.get('topic_id'),
                'n_relations': len(t.get('relations', [])),
                'first_relation_sample': t['relations'][0] if t.get('relations') else None,
            }
        return summary
    except Exception as e:
        return {'error': str(e), 'path': str(path)}


def stage3_summary(path):
    """Extract just the top-level summary and one sample graph."""
    try:
        with open(path) as f:
            doc = json.load(f)
        graphs = doc.get('graphs', [])
        # Compute verdict distribution
        verdict_counts = {}
        basis_counts = {}
        empty_grounded = 0
        for g in graphs:
            gv = g.get('graph_verdict', {})
            w = gv.get('winner', 'UNKNOWN')
            b = gv.get('basis', 'UNKNOWN')
            verdict_counts[w] = verdict_counts.get(w, 0) + 1
            basis_counts[b] = basis_counts.get(b, 0) + 1
            if not g.get('grounded_extension'):
                empty_grounded += 1
        n = len(graphs) or 1
        return {
            'path': str(path),
            'n_graphs': len(graphs),
            'verdict_distribution': verdict_counts,
            'basis_distribution': basis_counts,
            'pct_empty_grounded': round(100 * empty_grounded / n, 1),
            'summary': doc.get('summary', {}),
            'first_graph_sample': {
                'topic_id': graphs[0].get('topic_id') if graphs else None,
                'grounded_extension_size': len(graphs[0].get('grounded_extension', [])) if graphs else 0,
                'graph_verdict': graphs[0].get('graph_verdict') if graphs else None,
            } if graphs else None,
        }
    except Exception as e:
        return {'error': str(e), 'path': str(path)}


def main():
    out = []
    out.append('# MAJ-Debate Progress Report Data Bundle')
    out.append('')
    out.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    out.append(f'Project root: `{ROOT}`')
    out.append('')

    # ---- Section 1: ablation tables ----
    out.append('## 1. DDO ablation table (persuasion benchmark, n=500)')
    out.append('')
    out.append('### ablation_table.json')
    out.append('```json')
    out.append(json.dumps(read_json(ROOT / 'outputs/ablations/ddo_sample/ablation_table.json'),
                          indent=2))
    out.append('```')
    out.append('')
    out.append('### ablation_table.md (human-readable)')
    out.append('```')
    out.append(read_text(ROOT / 'outputs/ablations/ddo_sample/ablation_table.md'))
    out.append('```')
    out.append('')

    out.append('## 2. Logic-test ablation table (correctness benchmark, n=50)')
    out.append('')
    out.append('### ablation_table.json')
    out.append('```json')
    out.append(json.dumps(read_json(ROOT / 'outputs/ablations/logic_test/ablation_table.json'),
                          indent=2))
    out.append('```')
    out.append('')
    out.append('### ablation_table.md')
    out.append('```')
    out.append(read_text(ROOT / 'outputs/ablations/logic_test/ablation_table.md'))
    out.append('```')
    out.append('')

    # ---- Section 3: side-by-side ----
    out.append('## 3. Persuasion vs correctness comparison')
    out.append('')
    out.append('```')
    out.append(read_text(ROOT / 'outputs/ablations/persuasion_vs_correctness.md'))
    out.append('```')
    out.append('')

    # ---- Section 4: stage 2 summaries ----
    out.append('## 4. Stage 2 (relation labelling) summaries')
    out.append('')
    out.append('Three stage-2 outputs exist for DDO under v2.1. The Attack label count is')
    out.append('the key RQ3 signal.')
    out.append('')
    for config in ['ddo_sample', 'ddo_sample_direct_judge', 'ddo_sample_targeted_attacks']:
        p = ROOT / f'outputs/stage2/{config}/stage2_relations.json'
        out.append(f'### {config} ({"shared full/six_agents" if config == "ddo_sample" else config.split("_", 2)[-1]})')
        out.append('```json')
        out.append(json.dumps(stage2_summary(p), indent=2))
        out.append('```')
        out.append('')

    # ---- Section 5: stage 3 summary ----
    out.append('## 5. Stage 3 (Dung graph) summary')
    out.append('')
    out.append('The verdict distribution and empty-grounded % reveal the graph layers')
    out.append('bias. A heavy PRO skew suggests the labeller over-generates Attack edges')
    out.append('against CON arguments.')
    out.append('')
    for split in ['ddo_sample', 'logic_test']:
        p = ROOT / f'outputs/stage3/{split}/stage3_graphs.json'
        out.append(f'### {split} (full pipeline graph)')
        out.append('```json')
        out.append(json.dumps(stage3_summary(p), indent=2))
        out.append('```')
        out.append('')

    # ---- Section 6: inspector diagnostic output ----
    out.append('## 6. Inspector diagnostic output')
    out.append('')
    out.append('Cross-config duplicate check — confirms ablations are distinct.')
    out.append('')
    for split in ['ddo_sample', 'logic_test']:
        out.append(f'### {split}')
        out.append('```')
        py = str(ROOT / 'scripts/inspect_ablations.py')
        env_py = Path.home() / 'env-vllm/bin/python'
        if not env_py.exists():
            env_py = 'python3'
        out.append(run_cmd([str(env_py), py, '--split', split]))
        out.append('```')
        out.append('')

    # ---- Section 7: run manifests (which configs have v2.1 outputs) ----
    out.append('## 7. Run manifests')
    out.append('')
    for split in ['ddo_sample', 'logic_test']:
        manifest_p = ROOT / f'outputs/ablations/{split}/ablation_runs.json'
        out.append(f'### {split} ablation_runs.json')
        out.append('```json')
        out.append(json.dumps(read_json(manifest_p), indent=2))
        out.append('```')
        out.append('')

    # ---- Section 8: environment / reproducibility ----
    out.append('## 8. Environment')
    out.append('')
    out.append('```')
    out.append(f'Date: {datetime.now().isoformat(timespec="seconds")}')
    out.append(f'Host: puffer')
    try:
        out.append(run_cmd(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                            '--format=csv,noheader']))
    except Exception:
        pass
    try:
        out.append('--- pip packages (relevant) ---')
        env_py = Path.home() / 'env-vllm/bin/python'
        if env_py.exists():
            out.append(run_cmd([str(env_py), '-m', 'pip', 'show', 'vllm', 'torch', 'transformers']))
    except Exception:
        pass
    out.append('```')
    out.append('')

    # ---- Write bundle ----
    bundle_path = ROOT / 'outputs' / 'progress_report_bundle.md'
    bundle_path.write_text('\n'.join(out))
    print(f'Wrote: {bundle_path}')
    print(f'Size: {bundle_path.stat().st_size // 1024} KB')
    print()
    print('Upload this single file plus your proposal .tex to the next chat.')


if __name__ == '__main__':
    main()
