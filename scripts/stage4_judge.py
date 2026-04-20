#!/usr/bin/env python3
"""
Stage 4 — Judge Brain (vLLM).

Reads the stage-2 relations (and optionally the stage-3 graph output)
and asks a Qwen judge to produce an explainable verdict per topic.

When --stage3 is provided (default in "full" ablation), the prompt includes
the grounded extension, preferred extensions, and per-argument acceptance
status, anchoring the LLM's verdict in Dung semantics.

When --stage3 is NOT provided (direct_judge / targeted_no_graph ablations),
the judge only sees the argument list and labelled attack/support relations —
this isolates the graph's contribution (RQ1).

Output schema: outputs/stage4/<split>/stage4_judgments.json

    {
      "judgments": [
        {
          "topic_id": "...",
          "topic_text": "...",
          "benchmark_label": "PRO",        # DDO crowd vote
          "verdict": "PRO",                # judge's call
          "confidence": 0.78,
          "rationale": "Short natural-language explanation ...",
          "killing_attacks": ["AID_A007 -> AID_A003", ...],
          "used_graph": true,
          "graph_verdict": "PRO"           # if stage3 given
        },
        ...
      ],
      "summary": {
          "n_topics": ...,
          "verdict_counts": {...},
          "agreement_with_benchmark_pct": 72.4,
          "used_graph": true
      }
    }
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path


log = logging.getLogger('stage4')


SYSTEM_PROMPT = (
    'You are an impartial debate judge. You evaluate arguments on the basis '
    'of logical structure and which side has more undefeated claims. '
    'Respond ONLY with valid JSON — no preamble, no markdown fences, no '
    'chain-of-thought narration. BE CONCISE: keep rationale under 30 words, '
    'list at most 5 killing_attacks. Output must be complete valid JSON '
    '(closing brace required). Just the JSON object requested.'
)


def build_judge_prompt(topic, stage3_graph=None, conf_threshold=0.65,
                        max_args=30, max_rels=40):
    """Compose a compact prompt. Truncates aggressively so we fit into a
    reasonable context window without dropping critical structure."""
    tid = topic['topic_id']
    topic_text = topic['topic_text']
    args = topic.get('arguments', [])
    rels = topic.get('relations', [])
    strength = topic.get('argument_strength') or {}

    # Argument list (id | stance | text)
    args_lines = []
    for a in args[:max_args]:
        txt = (a.get('text') or '')[:160].replace('\n', ' ')
        st = strength.get(a['arg_id'])
        sw = st['strength'] if isinstance(st, dict) else None
        if sw is not None:
            args_lines.append(f'  {a["arg_id"]} [{a["stance"]}, s={sw:.2f}]: {txt}')
        else:
            args_lines.append(f'  {a["arg_id"]} [{a["stance"]}]: {txt}')
    args_block = '\n'.join(args_lines)
    if len(args) > max_args:
        args_block += f'\n  ... (and {len(args) - max_args} more)'

    # Relations (only kept + label != None, sorted by confidence desc)
    kept = [r for r in rels
            if r.get('kept') and r.get('label') in ('Attack', 'Support')]
    kept.sort(key=lambda r: r.get('confidence', 0), reverse=True)
    rel_lines = []
    for r in kept[:max_rels]:
        rel_lines.append(
            f'  {r["source_arg_id"]} --{r["label"]}--> {r["target_arg_id"]} '
            f'(c={r.get("confidence", 0):.2f})'
        )
    rel_block = '\n'.join(rel_lines) if rel_lines else '  (no high-confidence labelled relations)'
    if len(kept) > max_rels:
        rel_block += f'\n  ... (and {len(kept) - max_rels} more)'

    # Optional graph block
    graph_block = ''
    if stage3_graph:
        gv = stage3_graph.get('graph_verdict', {})
        grounded = stage3_graph.get('grounded_extension', [])
        pref = stage3_graph.get('preferred_extensions', [])
        accept = stage3_graph.get('acceptance', {})
        accepted_args = [a for a, v in accept.items() if v.get('skeptical')]
        graph_block = (
            '\nDUNG-SEMANTICS RESULTS:\n'
            f'  grounded extension ({len(grounded)}): {grounded[:15]}\n'
            f'  # preferred extensions : {len(pref)}\n'
            f'  skeptically accepted   : {accepted_args[:15]}\n'
            f'  formal graph verdict   : {gv.get("winner")} '
            f'(pro={gv.get("pro_score")}, con={gv.get("con_score")}, '
            f'basis={gv.get("basis")})\n'
            'Use these formal results as strong prior evidence, but you may '
            'override them if the arguments clearly support the other side.\n'
        )

    return (
        f'Topic: {topic_text}\n'
        f'Topic ID: {tid}\n\n'
        f'ARGUMENTS:\n{args_block}\n\n'
        f'HIGH-CONFIDENCE RELATIONS (confidence >= {conf_threshold}):\n'
        f'{rel_block}\n'
        f'{graph_block}\n'
        f'Decide which side wins and output complete JSON. Keep rationale '
        f'under 30 words. Do not cite more than 5 arg_ids in total.\n'
        f'{{"verdict": "PRO" or "CON" or "TIE", '
        f'"confidence": 0.0-1.0, '
        f'"rationale": "30 words max", '
        f'"killing_attacks": ["<src_id> -> <tgt_id>", ...max 5]}}'
    )


def parse_judgment(raw):
    """Return dict with keys verdict/confidence/rationale/killing_attacks.

    Tries, in order:
      1. Full strict JSON parse
      2. Brace-balanced extraction
      3. Partial-JSON recovery via regex (for truncated outputs — we can
         still use a verdict if the model wrote '"verdict": "PRO"' before
         hitting the token limit)
    """
    if not raw:
        return None
    t = re.sub(r'```(?:json)?', '', raw).strip()
    # 1. Full parse
    try:
        v = json.loads(t)
        if isinstance(v, dict) and 'verdict' in v:
            return v
    except Exception:
        pass
    # 2. Brace-balanced
    start = t.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == '{':
                depth += 1
            elif t[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(t[start:i + 1])
                        if isinstance(v, dict) and 'verdict' in v:
                            return v
                    except Exception:
                        break
    # 3. Partial recovery — extract fields by regex
    m = re.search(r'"verdict"\s*:\s*"(PRO|CON|TIE)"', t, re.IGNORECASE)
    if m:
        verdict = m.group(1).upper()
        cm = re.search(r'"confidence"\s*:\s*([0-9.]+)', t)
        confidence = float(cm.group(1)) if cm else 0.5
        # Rationale may be truncated mid-string; take what's there
        rm = re.search(r'"rationale"\s*:\s*"([^"]{0,400})', t)
        rationale = (rm.group(1) if rm else '') + ' [recovered from truncated output]'
        return {
            'verdict': verdict,
            'confidence': confidence,
            'rationale': rationale,
            'killing_attacks': [],
            '_partial_recovery': True,
        }
    return None


def coerce_verdict(j):
    v = str(j.get('verdict', 'TIE')).strip().upper()
    if v not in ('PRO', 'CON', 'TIE'):
        v = 'TIE'
    try:
        conf = float(j.get('confidence', 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return {
        'verdict': v,
        'confidence': round(conf, 3),
        'rationale': str(j.get('rationale', ''))[:400],
        'killing_attacks': [str(k) for k in (j.get('killing_attacks') or [])][:10],
    }


def _format_chat(prompt, tokenizer):
    msgs = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False)


def compute_agreement(judgments):
    """% agreement with benchmark_label (DDO crowd vote) among judged topics
    that have a benchmark label."""
    total = agree = 0
    for j in judgments:
        gold = j.get('benchmark_label')
        pred = j.get('verdict')
        if gold and pred and gold != 'TIE':
            total += 1
            if gold == pred:
                agree += 1
    return round(100.0 * agree / total, 2) if total else None, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2', required=True)
    ap.add_argument('--stage3', default='', help='optional — enables graph-grounded judging')
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--topic-limit', type=int, default=0)
    ap.add_argument('--gpu-mem-util', type=float, default=0.80)
    ap.add_argument('--max-model-len', type=int, default=4096)
    ap.add_argument('--max-tokens', type=int, default=700,
                    help='generation budget per topic (default 700). '
                         'Stage 4 verdict JSON with verbose rationale + '
                         'killing_attacks list can easily hit 400-500 tokens. '
                         'Previously 300 caused 40-60% parse failures on '
                         'configs with many arguments per topic.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s stage4 %(levelname)-7s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)])
    for lib in ['transformers', 'torch', 'vllm', 'urllib3', 'httpx']:
        logging.getLogger(lib).setLevel(logging.WARNING)

    stage2 = json.loads(Path(args.stage2).read_text())
    topics = stage2.get('topics', [])
    log.info('Loaded %d stage-2 topics', len(topics))

    graphs_by_id = {}
    use_graph = bool(args.stage3)
    if use_graph:
        stage3 = json.loads(Path(args.stage3).read_text())
        graphs_by_id = {g['topic_id']: g for g in stage3.get('graphs', [])}
        if not graphs_by_id:
            log.error('CRITICAL: --stage3 was passed but the file contains '
                      '0 graphs. Every prompt will be generated WITHOUT '
                      'graph context, making this config indistinguishable '
                      'from a non-graph baseline. Refusing to continue. '
                      'Regenerate stage 3 output with: '
                      'python scripts/stage3_graph.py --input <stage2> '
                      '--output %s', args.stage3)
            sys.exit(2)
        log.info('Loaded %d stage-3 graphs (graph-grounded judging ENABLED)',
                 len(graphs_by_id))
    else:
        log.info('No stage-3 provided (graph-grounded judging DISABLED)')

    if args.topic_limit > 0:
        topics = topics[:args.topic_limit]
        log.info('Topic limit -> %d', len(topics))

    # Resume
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing, done = [], set()
    if out_path.exists():
        try:
            doc = json.loads(out_path.read_text())
            existing = doc.get('judgments', [])
            done = {j['topic_id'] for j in existing}
        except Exception:
            pass
    log.info('Resume: %d topics already judged', len(done))

    remaining = [t for t in topics if t['topic_id'] not in done]
    if not remaining:
        log.info('Nothing to do')
        _write_final(out_path, existing, use_graph)
        return

    # vLLM init
    log.info('Initialising vLLM (%s)...', args.model)
    t0 = time.monotonic()
    from vllm import LLM, SamplingParams
    engine = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype='float16',
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )
    tok = engine.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        repetition_penalty=1.05,
    )
    log.info('Engine ready in %.1fs', time.monotonic() - t0)

    # Batched generation for ALL topics at once
    prompts = []
    graph_hits = 0
    for t in remaining:
        g = graphs_by_id.get(t['topic_id']) if use_graph else None
        if g is not None:
            graph_hits += 1
        prompts.append(_format_chat(build_judge_prompt(t, g), tok))

    if use_graph:
        log.info('Graph context attached to %d/%d prompts (%.1f%%)',
                 graph_hits, len(remaining),
                 100 * graph_hits / max(len(remaining), 1))
        if graph_hits == 0:
            log.error('CRITICAL: use_graph=True but 0 prompts got graph '
                      'context. topic_ids in stage 2 do not match topic_ids '
                      'in stage 3 graphs file. Stopping.')
            sys.exit(2)
        elif graph_hits < len(remaining) * 0.9:
            log.warning('Only %.1f%% of prompts received graph context. '
                        'Check that stage 2 and stage 3 cover the same topics.',
                        100 * graph_hits / max(len(remaining), 1))

    log.info('Judging %d topics in a single batch...', len(prompts))
    t = time.monotonic()
    outs = engine.generate(prompts, sampling)
    raws = [o.outputs[0].text.strip() for o in outs]
    log.info('Generated in %.1fs (%.2fs/topic)',
             time.monotonic() - t, (time.monotonic() - t) / max(len(prompts), 1))

    judgments = list(existing)
    parse_failures = 0
    for topic, raw in zip(remaining, raws):
        parsed = parse_judgment(raw)
        if parsed is None:
            parse_failures += 1
            parsed = {'verdict': 'TIE', 'confidence': 0.0,
                      'rationale': 'parse_failed', 'killing_attacks': []}
        j = coerce_verdict(parsed)
        g = graphs_by_id.get(topic['topic_id']) if use_graph else None
        judgments.append({
            'topic_id': topic['topic_id'],
            'topic_text': topic.get('topic_text'),
            'domain': topic.get('domain'),
            'benchmark_label': topic.get('benchmark_label'),
            'source_dataset': topic.get('source_dataset'),
            'verdict': j['verdict'],
            'confidence': j['confidence'],
            'rationale': j['rationale'],
            'killing_attacks': j['killing_attacks'],
            'used_graph': use_graph,
            'graph_verdict': (g.get('graph_verdict', {}).get('winner')
                              if g else None),
            'raw_output_preview': raw[:200],
        })

    _write_final(out_path, judgments, use_graph)
    log.info('Parse failures: %d/%d', parse_failures, len(remaining))


def _write_final(out_path, judgments, use_graph):
    from collections import Counter
    vc = Counter(j['verdict'] for j in judgments)
    agreement_pct, agreement_n = compute_agreement(judgments)
    final = {
        'judgments': judgments,
        'summary': {
            'n_topics': len(judgments),
            'verdict_counts': dict(vc),
            'used_graph': use_graph,
            'agreement_with_benchmark_pct': agreement_pct,
            'agreement_with_benchmark_n': agreement_n,
            'generated_at': datetime.now().isoformat(timespec='seconds'),
        },
    }
    tmp = out_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(final, indent=2))
    tmp.replace(out_path)
    log.info('Wrote %s (%d judgments, agreement=%s%%)',
             out_path, len(judgments), agreement_pct)


if __name__ == '__main__':
    main()