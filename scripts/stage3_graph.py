#!/usr/bin/env python3
"""
Stage 3 — Dung Abstract Argumentation Graph Engine.

Reads stage2_relations.json, builds a directed attack graph per topic from
Attack-labeled relations (with confidence >= threshold), then computes
Dung semantics:
  - conflict-free sets
  - admissible sets (defend themselves)
  - grounded extension (minimal complete / unique)
  - preferred extensions (maximal admissible)
  - stable extensions (attack everything outside)
  - credulous / skeptical acceptance per argument

Writes outputs/stage3/<split>/stage3_graphs.json.

This stage does not call any LLM — it's pure graph theory. Typical runtime
is a few seconds per topic even on a CPU.

Usage:
    python stage3_graph.py \\
        --input outputs/stage2/ddo_sample/stage2_relations.json \\
        --output outputs/stage3/ddo_sample/stage3_graphs.json
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
    """An abstract argumentation framework: (A, R) where R is a set of attack
    edges. All extension computations are exact; the only fudge factor is
    MAX_ARGS_FOR_PREFERRED below (for large frameworks we approximate)."""

    MAX_ARGS_FOR_PREFERRED = 20   # 2^n grows fast; ~20 args = ~1M subsets

    def __init__(self, args, attacks):
        self.args = sorted(set(args))
        self.n = len(self.args)
        self.idx = {a: i for i, a in enumerate(self.args)}
        # attackers[b] = set of args that attack b
        self.attackers = {a: set() for a in self.args}
        self.attacked_by = {a: set() for a in self.args}
        for src, tgt in attacks:
            if src in self.idx and tgt in self.idx:
                self.attackers[tgt].add(src)
                self.attacked_by[src].add(tgt)

    def attacks(self, S, T):
        """Does S attack any member of T?"""
        for a in S:
            if self.attacked_by[a] & T:
                return True
        return False

    def is_conflict_free(self, S):
        S = set(S)
        for a in S:
            if self.attacked_by[a] & S:
                return False
        return True

    def defends(self, S, a):
        """Does S defend argument a? i.e. for every attacker b of a, some
        c in S attacks b."""
        for b in self.attackers[a]:
            if not (self.attacked_by_any(S) & {b}):
                return False
        return True

    def attacked_by_any(self, S):
        out = set()
        for a in S:
            out |= self.attacked_by[a]
        return out

    def characteristic(self, S):
        """F(S) = {a | S defends a}."""
        S = set(S)
        attacked_by_S = self.attacked_by_any(S)
        return {a for a in self.args
                if self.attackers[a].issubset(attacked_by_S)}

    def grounded(self):
        """Least fixed point of F, starting from empty set.
        This is always unique and always exists."""
        S = set()
        while True:
            nxt = self.characteristic(S)
            if nxt == S:
                return S
            S = nxt

    def is_admissible(self, S):
        S = set(S)
        if not self.is_conflict_free(S):
            return False
        # must defend every member
        attacked_by_S = self.attacked_by_any(S)
        for a in S:
            if not self.attackers[a].issubset(attacked_by_S):
                return False
        return True

    def preferred_extensions(self):
        """All maximal admissible sets. Exponential; we cap at MAX_ARGS_FOR_PREFERRED."""
        if self.n == 0:
            return [set()]
        if self.n > self.MAX_ARGS_FOR_PREFERRED:
            # Approximate: return just the grounded + a greedy-maximal admissible.
            g = self.grounded()
            greedy = self._greedy_admissible_extension(seed=g)
            return [greedy] if greedy else [g]
        exts = []
        # iterate subsets by decreasing size so that supersets get checked first
        for r in range(self.n, -1, -1):
            for combo in combinations(self.args, r):
                S = set(combo)
                if not self.is_admissible(S):
                    continue
                # skip if strict subset of an already-kept ext
                if any(S < e for e in exts):
                    continue
                # drop previously-kept exts that are strict subsets of S
                exts = [e for e in exts if not (e < S)]
                exts.append(S)
        # dedupe
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
        """Stable = conflict-free + attacks every argument outside."""
        pref_exts = pref_exts or self.preferred_extensions()
        out = []
        all_args = set(self.args)
        for e in pref_exts:
            outside = all_args - e
            if outside == self.attacked_by_any(e):
                out.append(e)
        return out

    def acceptance(self, pref_exts=None, grounded=None):
        """Per-argument acceptance: credulous (in some preferred),
        skeptical (in every preferred)."""
        pref_exts = pref_exts if pref_exts is not None else self.preferred_extensions()
        grounded = grounded if grounded is not None else self.grounded()
        if not pref_exts:
            pref_exts = [set()]
        in_all = set.intersection(*pref_exts) if pref_exts else set()
        in_any = set.union(*pref_exts) if pref_exts else set()
        return {
            a: {
                'grounded': a in grounded,
                'skeptical': a in in_all,
                'credulous': a in in_any,
            }
            for a in self.args
        }


# ---------------------------------------------------------------------------
# Build + run per topic
# ---------------------------------------------------------------------------

def build_af_from_topic(topic, conf_threshold=0.65):
    """Extract (arg_ids, attack_edges) from a stage-2 topic record."""
    arg_ids = [a['arg_id'] for a in topic.get('arguments', [])]
    edges = []
    for r in topic.get('relations', []):
        if r.get('label') != 'Attack':
            continue
        if not r.get('kept', False):
            continue
        if r.get('confidence', 0.0) < conf_threshold:
            continue
        edges.append((r['source_arg_id'], r['target_arg_id']))
    return arg_ids, edges


def verdict_from_extensions(topic, grounded, pref_exts, acceptance):
    """Which side (PRO/CON) wins based on the grounded extension?
    Fallback to preferred-majority if grounded is empty."""
    arg_stance = {a['arg_id']: a.get('stance') for a in topic.get('arguments', [])}
    strength = topic.get('argument_strength') or {}

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
        # average over preferred extensions
        pro = con = 0.0
        for e in pref_exts:
            p, c = score_set(e)
            pro += p
            con += c
        pro /= max(len(pref_exts), 1)
        con /= max(len(pref_exts), 1)
        basis = 'preferred_majority'
    else:
        pro, con = score_set(set())
        basis = 'empty'

    if pro > con:
        winner = 'PRO'
    elif con > pro:
        winner = 'CON'
    else:
        winner = 'TIE'
    margin = pro - con
    return {
        'winner': winner,
        'basis': basis,
        'pro_score': round(pro, 3),
        'con_score': round(con, 3),
        'margin': round(margin, 3),
    }


def process_topic(topic, conf_threshold):
    tid = topic['topic_id']
    arg_ids, edges = build_af_from_topic(topic, conf_threshold)
    af = AF(arg_ids, edges)

    t0 = time.monotonic()
    grounded = af.grounded()
    pref = af.preferred_extensions()
    stable = af.stable_extensions(pref)
    accept = af.acceptance(pref, grounded)
    verdict = verdict_from_extensions(topic, grounded, pref, accept)
    elapsed = time.monotonic() - t0

    return {
        'topic_id': tid,
        'topic_text': topic.get('topic_text'),
        'domain': topic.get('domain'),
        'benchmark_label': topic.get('benchmark_label'),
        'source_dataset': topic.get('source_dataset'),
        'n_arguments': len(arg_ids),
        'n_attack_edges': len(edges),
        'grounded_extension': sorted(grounded),
        'grounded_size': len(grounded),
        'preferred_extensions': [sorted(e) for e in pref],
        'n_preferred': len(pref),
        'stable_extensions': [sorted(e) for e in stable],
        'n_stable': len(stable),
        'acceptance': accept,
        'graph_verdict': verdict,
        'elapsed_seconds': round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True,
                    help='stage2_relations.json')
    ap.add_argument('--output', required=True,
                    help='stage3_graphs.json')
    ap.add_argument('--conf-threshold', type=float, default=0.65)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s stage3 %(levelname)-7s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)])

    log.info('Loading %s', args.input)
    stage2 = json.loads(Path(args.input).read_text())
    topics = stage2.get('topics', [])
    log.info('%d topics', len(topics))

    results = []
    verdict_counts = {'PRO': 0, 'CON': 0, 'TIE': 0}
    for i, t in enumerate(topics, 1):
        try:
            g = process_topic(t, args.conf_threshold)
            results.append(g)
            verdict_counts[g['graph_verdict']['winner']] += 1
            if i % 25 == 0 or i == len(topics):
                log.info('[%d/%d] %s: winner=%s basis=%s '
                         'grounded=%d/%d pref=%d',
                         i, len(topics), g['topic_id'],
                         g['graph_verdict']['winner'],
                         g['graph_verdict']['basis'],
                         g['grounded_size'], g['n_arguments'],
                         g['n_preferred'])
        except Exception as ex:
            log.exception('topic %s failed: %s', t.get('topic_id'), ex)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final = {
        'graphs': results,
        'summary': {
            'n_topics': len(results),
            'conf_threshold': args.conf_threshold,
            'verdict_counts': verdict_counts,
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'source': str(args.input),
        },
    }
    tmp = out_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(final, indent=2))
    tmp.replace(out_path)
    log.info('Wrote %s', out_path)
    log.info('Verdict distribution: %s', verdict_counts)


if __name__ == '__main__':
    main()
