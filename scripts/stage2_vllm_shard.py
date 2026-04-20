#!/usr/bin/env python3
"""
Stage 2 vLLM single-shard worker — PATCHED for ablation support.

Identical to the student's working stage2_vllm_shard.py, with one addition:
respects MAJ_STAGE2_TARGETED env var (default 1). When set to 0, the pair
classifier uses a *zero-shot* relation prompt (no implicit-premise targeting);
this supports the RQ3 ablation comparing targeted vs zero-shot attacks.

All other behavior, schemas, confidence thresholds, prefiltering and
resume semantics are unchanged from the working stage 2 shard.

Invoked by the orchestrator (run_all_ablations.py) as a subprocess with
CUDA_VISIBLE_DEVICES set.
"""

import argparse
import json
import logging
import os
import re
import statistics
import sys
import time
from datetime import datetime
from itertools import permutations
from pathlib import Path


# ============================================================================
# Config (env vars, unchanged from original)
# ============================================================================

def _env_int(name, default):
    return int(os.environ.get(name, str(default)))

def _env_float(name, default):
    return float(os.environ.get(name, str(default)))

def _env_bool(name, default=True):
    return os.environ.get(name, '1' if default else '0') not in {'0', 'false', 'False'}


MAX_MODEL_LEN        = _env_int('MAJ_STAGE2_MAX_MODEL_LEN', 4096)
GPU_MEM_UTIL         = _env_float('MAJ_STAGE2_GPU_MEM_UTIL', 0.80)
MAX_TOKENS           = _env_int('MAJ_STAGE2_MAX_TOKENS', 280)
PAIR_BATCH_SIZE      = _env_int('MAJ_STAGE2_PAIR_BATCH', 8)
STRENGTH_BATCH_SIZE  = _env_int('MAJ_STAGE2_STRENGTH_BATCH', 15)
CONFIDENCE_THRESHOLD = _env_float('MAJ_STAGE2_CONFIDENCE', 0.65)

EMBED_MODEL_NAME        = os.environ.get('MAJ_STAGE2_EMBED_MODEL',
                                         'sentence-transformers/all-MiniLM-L6-v2')
PREFILTER_ENABLED       = _env_bool('MAJ_STAGE2_PREFILTER', True)
PF_MAX_ROUND_GAP        = _env_int('MAJ_STAGE2_PF_MAX_ROUND_GAP', 4)
PF_MIN_SIMILARITY       = _env_float('MAJ_STAGE2_PF_MIN_SIM', 0.15)
PF_SAME_STANCE_MIN_SIM  = _env_float('MAJ_STAGE2_PF_SAME_STANCE_MIN_SIM', 0.35)
TOPIC_LIMIT             = _env_int('MAJ_STAGE2_TOPIC_LIMIT', 0)

# *** NEW in this patched version: RQ3 switch ***
TARGETED_ATTACKS = _env_bool('MAJ_STAGE2_TARGETED', True)


LABELS = ['Attack', 'Support', 'Neutral', 'None']

SYSTEM_PROMPT = (
    'You are a careful debate analyst. Respond ONLY with the requested JSON. '
    'No reasoning prose, no markdown fences, just the JSON.'
)


# ============================================================================
# Prompts — both variants
# ============================================================================

def build_pair_batch_prompt_zeroshot(topic_text, pairs):
    """Original zero-shot prompt. Used when MAJ_STAGE2_TARGETED=0."""
    lines = []
    for i, (a, b) in enumerate(pairs):
        sa = (a.get('text') or a.get('argument') or '').replace('\n', ' ')
        sb = (b.get('text') or b.get('argument') or '').replace('\n', ' ')
        lines.append(
            f'PAIR {i}:\n'
            f'  SOURCE ({a.get("arg_id")}): {sa}\n'
            f'  TARGET ({b.get("arg_id")}): {sb}'
        )
    return (
        f'Topic: {topic_text}\n\n'
        f'For each pair, decide whether SOURCE attacks, supports, is neutral to, '
        f'or has no relation with TARGET.\n\n'
        f'{chr(10).join(lines)}\n\n'
        f'Output a JSON array (one object per pair, in order):\n'
        f'[{{"pair": 0, "label": "Attack|Support|Neutral|None", '
        f'"confidence": 0.0-1.0}}, ...]'
    )


def build_pair_batch_prompt_targeted(topic_text, pairs):
    """Targeted implicit-premise prompt (Ozaki 2025 style).
    Default; used when MAJ_STAGE2_TARGETED=1.
    The classifier is told to name the implicit premise being attacked
    before returning its label, which empirically catches more true Attacks."""
    lines = []
    for i, (a, b) in enumerate(pairs):
        sa = (a.get('text') or a.get('argument') or '').replace('\n', ' ')
        sb = (b.get('text') or b.get('argument') or '').replace('\n', ' ')
        lines.append(
            f'PAIR {i}:\n'
            f'  SOURCE ({a.get("arg_id")}, stance={a.get("stance")}): {sa}\n'
            f'  TARGET ({b.get("arg_id")}, stance={b.get("stance")}): {sb}'
        )
    return (
        f'Topic: {topic_text}\n\n'
        f'For each pair, identify the implicit premise of TARGET that SOURCE '
        f'is challenging or reinforcing, then decide the relation:\n'
        f'  Attack   = SOURCE undermines an implicit or explicit premise of TARGET.\n'
        f'  Support  = SOURCE reinforces a premise or conclusion of TARGET.\n'
        f'  Neutral  = SOURCE is on-topic but independent of TARGET.\n'
        f'  None     = SOURCE is off-topic or unrelated to TARGET.\n\n'
        f'{chr(10).join(lines)}\n\n'
        f'Output a JSON array (one object per pair, in order):\n'
        f'[{{"pair": 0, "label": "Attack|Support|Neutral|None", '
        f'"confidence": 0.0-1.0, '
        f'"premise": "one short phrase naming the premise involved"}}, ...]'
    )


def build_pair_batch_prompt(topic_text, pairs):
    if TARGETED_ATTACKS:
        return build_pair_batch_prompt_targeted(topic_text, pairs)
    return build_pair_batch_prompt_zeroshot(topic_text, pairs)


def build_strength_batch_prompt(topic_text, args):
    lines = []
    for i, a in enumerate(args):
        t = (a.get('text') or a.get('argument') or '').replace('\n', ' ')
        lines.append(f'ARG {i} ({a.get("arg_id")}): {t}')
    return (
        f'Topic: {topic_text}\n\n'
        f'Rate each argument\'s strength (0.0=weak, 1.0=strong):\n\n'
        f'{chr(10).join(lines)}\n\n'
        f'Output a JSON array:\n'
        f'[{{"arg": 0, "strength": 0.0-1.0}}, ...]'
    )


def format_chat(prompt, tokenizer):
    msgs = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False)


def parse_json_array(text):
    if not text:
        return []
    text = re.sub(r'```(?:json)?', '', text).strip()
    try:
        v = json.loads(text)
        return v if isinstance(v, list) else []
    except Exception:
        pass
    start = text.find('[')
    if start < 0:
        return []
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
            if depth == 0:
                try:
                    v = json.loads(text[start:i + 1])
                    return v if isinstance(v, list) else []
                except Exception:
                    return []
    return []


def coerce_label(raw):
    s = str(raw).strip().title()
    return s if s in LABELS else 'None'


def coerce_float01(raw, default=0.0):
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


# ============================================================================
# Prefilter (unchanged)
# ============================================================================

_EMBEDDER = None


def get_embedder(log):
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
        log.info('Loading embedder on CPU...')
        _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
        log.info('Embedder ready')
    except Exception as ex:
        log.warning('Embedder load failed (%s); prefilter DISABLED', ex)
        _EMBEDDER = False
    return _EMBEDDER


def compute_sim_matrix(args, log):
    emb = get_embedder(log)
    if not emb:
        return None
    texts = [a.get('text') or a.get('argument') or '' for a in args]
    vecs = emb.encode(texts, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
    return vecs @ vecs.T


def _none_relation(a, b, reason):
    return {
        'source_arg_id': a['arg_id'], 'target_arg_id': b['arg_id'],
        'source_stance': a.get('stance'), 'target_stance': b.get('stance'),
        'source_round': a.get('round'), 'target_round': b.get('round'),
        'label': 'None', 'confidence': 0.0, 'kept': False,
        'justification': f'prefiltered: {reason}', 'prefiltered': True,
    }


def prefilter_pairs(args, sim):
    n = len(args)
    if sim is None:
        return [(a, b) for a, b in permutations(args, 2)], []
    keep, auto_none = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = args[i], args[j]
            s = float(sim[i, j])
            ra, rb = a.get('round', 0), b.get('round', 0)
            if ra is not None and rb is not None and abs(ra - rb) > PF_MAX_ROUND_GAP:
                auto_none.append(_none_relation(a, b, 'round_gap'))
                continue
            if a.get('stance') == b.get('stance') and s < PF_SAME_STANCE_MIN_SIM:
                auto_none.append(_none_relation(a, b, 'same_stance_low_sim'))
                continue
            if s < PF_MIN_SIMILARITY:
                auto_none.append(_none_relation(a, b, 'low_sim'))
                continue
            keep.append((a, b))
    return keep, auto_none


# ============================================================================
# Per-topic processing (unchanged from original)
# ============================================================================

def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _relation_base(a, b):
    return {
        'source_arg_id': a['arg_id'], 'target_arg_id': b['arg_id'],
        'source_stance': a.get('stance'), 'target_stance': b.get('stance'),
        'source_round': a.get('round'), 'target_round': b.get('round'),
    }


def _relation_from_obj(a, b, obj):
    label = coerce_label(obj.get('label', 'None'))
    conf = coerce_float01(obj.get('confidence', 0.0))
    rel = {
        **_relation_base(a, b),
        'label': label,
        'confidence': round(conf, 3),
        'kept': conf >= CONFIDENCE_THRESHOLD and label != 'None',
    }
    if 'premise' in obj:
        rel['premise'] = str(obj['premise'])[:200]
    return rel


def _relation_failure(a, b):
    return {
        **_relation_base(a, b),
        'label': 'None', 'confidence': 0.0, 'kept': False,
        'justification': 'parse_failed', 'failed': True,
    }


def process_topic(engine, sampling_params, tokenizer, topic, log):
    args = topic['arguments']
    topic_text = topic['topic_text']
    topic_id = topic['topic_id']
    n = len(args)
    t0 = time.monotonic()

    sim = compute_sim_matrix(args, log) if PREFILTER_ENABLED else None
    keep_pairs, auto_none = prefilter_pairs(args, sim)
    total_pairs = n * max(n - 1, 0)
    log.info('topic %s: %d args, %d pairs, keep %d, auto-None %d (targeted=%s)',
             topic_id, n, total_pairs, len(keep_pairs), len(auto_none),
             TARGETED_ATTACKS)

    pair_batches = list(_chunks(keep_pairs, PAIR_BATCH_SIZE))
    strength_batches = list(_chunks(args, STRENGTH_BATCH_SIZE))

    pair_prompts = [build_pair_batch_prompt(topic_text, pb) for pb in pair_batches]
    strength_prompts = [build_strength_batch_prompt(topic_text, sb) for sb in strength_batches]

    all_prompts = strength_prompts + pair_prompts
    if not all_prompts:
        all_outputs = []
    else:
        formatted = [format_chat(p, tokenizer) for p in all_prompts]
        outs = engine.generate(formatted, sampling_params)
        all_outputs = [o.outputs[0].text.strip() for o in outs]

    strength_outputs = all_outputs[:len(strength_prompts)]
    pair_outputs = all_outputs[len(strength_prompts):]

    strength_scores = {}
    for batch, raw in zip(strength_batches, strength_outputs):
        arr = parse_json_array(raw)
        by_idx = {}
        for obj in arr:
            if isinstance(obj, dict) and 'arg' in obj:
                try:
                    by_idx[int(obj['arg'])] = obj
                except (TypeError, ValueError):
                    pass
        if not by_idx and len(arr) == len(batch):
            by_idx = {i: obj for i, obj in enumerate(arr) if isinstance(obj, dict)}
        for i, arg in enumerate(batch):
            obj = by_idx.get(i, {})
            strength_scores[arg['arg_id']] = {
                'strength': round(coerce_float01(obj.get('strength', 0.5)), 3),
                'rationale': obj.get('rationale', 'No rationale.'),
            }

    relations = list(auto_none)
    failed_pairs = 0
    for batch, raw in zip(pair_batches, pair_outputs):
        arr = parse_json_array(raw)
        by_idx = {}
        for obj in arr:
            if isinstance(obj, dict) and 'pair' in obj:
                try:
                    by_idx[int(obj['pair'])] = obj
                except (TypeError, ValueError):
                    pass
        if not by_idx and len(arr) == len(batch):
            by_idx = {i: obj for i, obj in enumerate(arr) if isinstance(obj, dict)}
        for i, (a, b) in enumerate(batch):
            obj = by_idx.get(i)
            if obj is None:
                relations.append(_relation_failure(a, b))
                failed_pairs += 1
            else:
                relations.append(_relation_from_obj(a, b, obj))

    elapsed = time.monotonic() - t0
    counts = {k: 0 for k in LABELS}
    kept = prefiltered = 0
    for r in relations:
        counts[r.get('label', 'None')] += 1
        kept += int(r.get('kept', False))
        prefiltered += int(r.get('prefiltered', False))
    avg_strength = round(
        statistics.mean(v['strength'] for v in strength_scores.values()), 3
    ) if strength_scores else 0.0

    log.info('topic %s DONE in %.1fs | total=%d llm=%d prefilt=%d kept=%d fail=%d',
             topic_id, elapsed, len(relations), len(keep_pairs),
             prefiltered, kept, failed_pairs)

    return {
        'topic_id': topic_id,
        'topic_text': topic_text,
        'domain': topic.get('domain'),
        'benchmark_label': topic.get('benchmark_label'),
        'source_dataset': topic.get('source_dataset'),
        'source_ref': topic.get('source_ref'),
        'evaluation_split': topic.get('evaluation_split'),
        'run_name': topic.get('run_name'),
        'arguments': args,
        'argument_strength': strength_scores,
        'relations': relations,
        'summary': {
            'n_arguments': n,
            'n_ordered_pairs': total_pairs,
            'n_llm_classified': len(keep_pairs),
            'n_prefiltered': prefiltered,
            'kept_relations': kept,
            'failed_pairs': failed_pairs,
            'label_counts': counts,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'avg_strength': avg_strength,
            'targeted_attacks': TARGETED_ATTACKS,
            'elapsed_seconds': round(elapsed, 2),
        },
    }


# ============================================================================
# Main (unchanged)
# ============================================================================

def setup_logging(shard_id, log_file):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter(
        f'%(asctime)s shard{shard_id} %(levelname)-7s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(str(log_file), mode='a')
        fh.setFormatter(fmt)
        root.addHandler(fh)
    for lib in ['transformers', 'torch', 'sentence_transformers',
                'urllib3', 'httpx', 'vllm']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    return logging.getLogger(f'stage2_shard{shard_id}')


def save_atomic(path, doc):
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(doc, f, indent=2)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-id', type=int, required=True)
    parser.add_argument('--total-shards', type=int, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--log-file', type=str, default='')
    args = parser.parse_args()

    log = setup_logging(args.shard_id, Path(args.log_file) if args.log_file else None)
    log.info('=' * 70)
    log.info('Stage 2 Shard | shard %d of %d | targeted=%s',
             args.shard_id, args.total_shards, TARGETED_ATTACKS)
    log.info('CUDA_VISIBLE_DEVICES=%s', os.environ.get('CUDA_VISIBLE_DEVICES', 'UNSET'))
    log.info('Model: %s', args.model)
    log.info('Input: %s', args.input)
    log.info('Output: %s', args.output)
    log.info('=' * 70)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path) as f:
        stage1 = json.load(f)
    all_topics = sorted(stage1.get('topics', []),
                         key=lambda t: len(t.get('arguments', [])))
    my_topics = [t for i, t in enumerate(all_topics)
                 if i % args.total_shards == args.shard_id]
    log.info('My shard: %d topics', len(my_topics))

    if TOPIC_LIMIT:
        my_topics = my_topics[:TOPIC_LIMIT]
        log.info('TOPIC_LIMIT=%d; shard processing %d', TOPIC_LIMIT, len(my_topics))

    # Resume
    if out_path.exists():
        try:
            ckpt = json.loads(out_path.read_text())
            done_ids = {t['topic_id'] for t in ckpt.get('topics', [])}
            log.info('Resuming: %d topics already in shard output', len(done_ids))
        except Exception:
            ckpt = {'topics': [], 'summary': {}}
            done_ids = set()
    else:
        ckpt = {'topics': [], 'summary': {}}
        done_ids = set()

    if PREFILTER_ENABLED:
        get_embedder(log)

    log.info('Initializing vLLM engine (TP=1)...')
    t_init = time.monotonic()
    from vllm import LLM, SamplingParams
    engine = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype='float16',
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )
    tokenizer = engine.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        repetition_penalty=1.05,
    )
    log.info('Engine ready in %.1fs', time.monotonic() - t_init)

    warm = engine.generate([format_chat('Reply only: OK', tokenizer)], sampling_params)
    log.info('Warmup: %r', warm[0].outputs[0].text.strip()[:50])

    run_t0 = time.monotonic()
    completed = failed = 0
    for idx, topic in enumerate(my_topics, 1):
        tid = topic['topic_id']
        if tid in done_ids:
            log.info('[%d/%d] %s: SKIP (done)', idx, len(my_topics), tid)
            continue
        log.info('[%d/%d] %s: START (%d args)',
                 idx, len(my_topics), tid, len(topic.get('arguments', [])))
        try:
            result = process_topic(engine, sampling_params, tokenizer, topic, log)
            ckpt['topics'].append(result)
            save_atomic(out_path, ckpt)
            completed += 1
        except Exception as ex:
            failed += 1
            log.exception('topic %s FAILED: %s', tid, ex)

    total = time.monotonic() - run_t0
    ckpt['summary'] = {
        'shard_id': args.shard_id,
        'total_shards': args.total_shards,
        'completed_at': datetime.now().isoformat(timespec='seconds'),
        'n_topics_in_shard': len(my_topics),
        'completed_in_this_run': completed,
        'failed_in_this_run': failed,
        'already_done_at_start': len(done_ids),
        'targeted_attacks': TARGETED_ATTACKS,
        'elapsed_seconds': round(total, 1),
    }
    save_atomic(out_path, ckpt)
    log.info('SHARD %d COMPLETE | completed=%d failed=%d | %.1fs',
             args.shard_id, completed, failed, total)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
