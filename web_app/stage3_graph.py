#!/usr/bin/env python3
"""
Stage 3 — Dung Abstract Argumentation Graph Engine.

Bundled directly into web_app/ so the app is fully self-contained.

Computes Dung semantics (grounded, preferred, stable extensions) from
stage-2 attack relations.  No LLM calls — pure graph theory.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path


log = logging.getLogger('stage3')


# ---------------------------------------------------------------------------
# Dung semantics
# ---------------------------------------------------------------------------

class AF:
    """Abstract argumentation framework (A, R)."""

    MAX_ARGS_FOR_PREFERRED = 20

    def __init__(self, args, attacks):
        self.args = sorted(set(args))
        self.n = len(self.args)
        self.idx = {a: i for i, a in enumerate(self.args)}
        self.attackers  = {a: set() for a in self.args}
        self.attacked_by = {a: set() for a in self.args}
        for src, tgt in attacks:
            if src in self.idx and tgt in self.idx:
                self.attackers[tgt].add(src)
                self.attacked_by[src].add(tgt)

    def attacked_by_any(self, S):
        out = set()
        for a in S:
            out |= self.attacked_by[a]
        return out

    def characteristic(self, S):
        S = set(S)
        attacked_by_S = self.attacked_by_any(S)
        return {a for a in self.args if self.attackers[a].issubset(attacked_by_S)}

    def grounded(self):
        S = set()
        while True:
            nxt = self.characteristic(S)
            if nxt == S:
                return S
            S = nxt

    def is_conflict_free(self, S):
        S = set(S)
        for a in S:
            if self.attacked_by[a] & S:
                return False
        return True

    def is_admissible(self, S):
        S = set(S)
        if not self.is_conflict_free(S):
            return False
        attacked_by_S = self.attacked_by_any(S)
        for a in S:
            if not self.attackers[a].issubset(attacked_by_S):
                return False
        return True

    def preferred_extensions(self):
        if self.n == 0:
            return [set()]
        if self.n > self.MAX_ARGS_FOR_PREFERRED:
            g = self.grounded()
            greedy = self._greedy_admissible_extension(seed=g)
            return [greedy] if greedy else [g]
        exts = []
        for r in range(self.n, -1, -1):
            for combo in combinations(self.args, r):
                S = set(combo)
                if not self.is_admissible(S):
                    continue
                if any(S < e for e in exts):
                    continue
                exts = [e for e in exts if not (e < S)]
                exts.append(S)
        uniq = []
        for e in exts:
            if e not in uniq:
                uniq.append(e)
        return uniq

    def _greedy_admissible_extension(self, seed=None):
        S = set(seed) if seed else set()
        changed = True
        while changed:
            changed = False
            for a in self.args:
                if a in S:
                    continue
                cand = S | {a}
                if self.is_admissible(cand):
                    S = cand
                    changed = True
        return S

    def stable_extensions(self, pref_exts=None):
        pref_exts = pref_exts or self.preferred_extensions()
        out = []
        all_args = set(self.args)
        for e in pref_exts:
            outside = all_args - e
            if outside == self.attacked_by_any(e):
                out.append(e)
        return out

    def acceptance(self, pref_exts=None, grounded=None):
        pref_exts = pref_exts if pref_exts is not None else self.preferred_extensions()
        grounded  = grounded  if grounded  is not None else self.grounded()
        if not pref_exts:
            pref_exts = [set()]
        in_all = set.intersection(*pref_exts) if pref_exts else set()
        in_any = set.union(*pref_exts)        if pref_exts else set()
        return {
            a: {
                'grounded':  a in grounded,
                'skeptical': a in in_all,
                'credulous': a in in_any,
            }
            for a in self.args
        }


# ---------------------------------------------------------------------------
# Build AF from stage-2 topic record
# ---------------------------------------------------------------------------

def build_af_from_topic(topic, conf_threshold=0.65, cross_stance_only=False):
    arg_ids    = [a['arg_id'] for a in topic.get('arguments', [])]
    arg_stance = {a['arg_id']: a.get('stance') for a in topic.get('arguments', [])}
    edges = []
    dropped_same_stance = 0
    for r in topic.get('relations', []):
        if r.get('label') != 'Attack':
            continue
        if not r.get('kept', False):
            continue
        if r.get('confidence', 0.0) < conf_threshold:
            continue
        src = r['source_arg_id']
        tgt = r['target_arg_id']
        src_stance = r.get('source_stance') or arg_stance.get(src)
        tgt_stance = r.get('target_stance') or arg_stance.get(tgt)
        if cross_stance_only and src_stance == tgt_stance:
            dropped_same_stance += 1
            continue
        edges.append((src, tgt))
    return arg_ids, edges, dropped_same_stance


def verdict_from_extensions(topic, grounded, pref_exts, acceptance):
    arg_stance = {a['arg_id']: a.get('stance') for a in topic.get('arguments', [])}
    strength   = topic.get('argument_strength') or {}

    def score_set(S):
        pro = con = 0.0
        for a in S:
            s = arg_stance.get(a)
            w = (strength.get(a) or {}).get('strength', 0.5) if isinstance(strength.get(a), dict) else 0.5
            if s == 'PRO':
                pro += w
            elif s == 'CON':
                con += w
        return pro, con

    if grounded:
        pro, con = score_set(grounded)
        basis = 'grounded'
    elif pref_exts:
        pro = con = 0.0
        for e in pref_exts:
            p, c = score_set(e)
            pro += p; con += c
        pro /= max(len(pref_exts), 1)
        con /= max(len(pref_exts), 1)
        basis = 'preferred_majority'
    else:
        pro, con = score_set(set())
        basis = 'empty'

    winner = 'PRO' if pro > con else ('CON' if con > pro else 'TIE')
    return {
        'winner': winner, 'basis': basis,
        'pro_score': round(pro, 3), 'con_score': round(con, 3),
        'margin': round(pro - con, 3),
    }


def process_topic(topic, conf_threshold, cross_stance_only=False):
    tid = topic['topic_id']
    arg_ids, edges, dropped = build_af_from_topic(
        topic, conf_threshold, cross_stance_only=cross_stance_only)
    af = AF(arg_ids, edges)

    t0 = time.monotonic()
    grounded = af.grounded()
    pref     = af.preferred_extensions()
    stable   = af.stable_extensions(pref)
    accept   = af.acceptance(pref, grounded)
    verdict  = verdict_from_extensions(topic, grounded, pref, accept)

    return {
        'topic_id': tid,
        'topic_text': topic.get('topic_text'),
        'domain': topic.get('domain'),
        'benchmark_label': topic.get('benchmark_label'),
        'source_dataset': topic.get('source_dataset'),
        'n_arguments': len(arg_ids),
        'n_attack_edges': len(edges),
        'cross_stance_only': cross_stance_only,
        'dropped_same_stance_attack_edges': dropped,
        'grounded_extension': sorted(grounded),
        'grounded_size': len(grounded),
        'preferred_extensions': [sorted(e) for e in pref],
        'n_preferred': len(pref),
        'stable_extensions': [sorted(e) for e in stable],
        'n_stable': len(stable),
        'acceptance': accept,
        'graph_verdict': verdict,
        'elapsed_seconds': round(time.monotonic() - t0, 3),
    }


# ---------------------------------------------------------------------------
# CLI (standalone usage — not required by web app)
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--conf-threshold', type=float, default=0.65)
    ap.add_argument('--cross-stance-only', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s stage3 %(levelname)-7s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    stage2 = json.loads(Path(args.input).read_text())
    topics = stage2.get('topics', [])
    results = []
    for t in topics:
        results.append(process_topic(t, args.conf_threshold,
                                     cross_stance_only=args.cross_stance_only))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({'graphs': results}, indent=2))
    log.info('Wrote %s', out_path)


if __name__ == '__main__':
    main()
