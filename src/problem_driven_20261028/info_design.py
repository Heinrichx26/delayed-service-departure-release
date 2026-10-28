"""Decision-timed acquisition of service confirmations on a fixed support.

Experiments read frozen inputs. Every policy is replayed independently before
its communication count or waiting is reported. Plotting lives separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
import json
from math import lcm
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/trb_reassessment_20261027'))
import fixed_support_frontier as base

DATA = ROOT / 'data/problem_driven_20261028'
OUT = ROOT / 'results/revision_20261028/problem_driven'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(case, prefix=()):
    paths, caps, n, lag = case['service'], case['caps'], case['n'], case['lag']
    references = [base.physical(base.earliest(n, caps), p, n, lag)[0] for p in paths]
    if not all(r[-1] == n for r in references):
        return None
    required_and_bounds = [base.bound(r, caps, lag) for r in references]
    bounds = [b for _,b in required_and_bounds]
    assert len(prefix) <= len(caps)
    assert all(type(v) is int for v in prefix)
    assert all(0 <= u <= k for u,k in zip(base.increments(prefix), caps))
    assert not prefix or prefix[-1] <= n
    assert all(prefix[t] >= required[t] for required,_ in required_and_bounds
               for t in range(len(prefix)))
    assert not prefix or prefix[-1] >= max(b[len(prefix)-1] for b in bounds)
    star, target, witnesses, count = base.threshold(paths, bounds, list(prefix))
    probabilities = [Fraction(p) for p in case['probabilities']]
    denominator = lcm(*(p.denominator for p in probabilities))
    weights = [int(p * denominator) for p in probabilities]
    assert sum(weights) == denominator and all(w > 0 for w in weights)
    return dict(case=case, paths=paths, caps=caps, n=n, lag=lag,
                references=references, bounds=bounds, prefix=list(prefix),
                t0=len(prefix), horizon=len(caps), target=target, L_star=star,
                weights=weights, denominator=denominator)


def critical_pairs(problem):
    paths, target, t0 = problem['paths'], problem['target'], problem['t0']
    result = []
    for s in range(len(paths)):
        for q in range(s):
            first_service = next((j for j,(a,b) in enumerate(zip(paths[s],paths[q])) if a != b), None)
            first_action = next((t for t in range(t0,problem['horizon']) if target[s][t] != target[q][t]), None)
            if first_action is not None and first_service is not None and first_action <= first_service:
                result.append((q,s,first_action))
    return result


def menu_cover(problem, lead):
    """Exact unit-cost set cover, with dominance and integer recursion."""
    pairs, paths, t0 = critical_pairs(problem), problem['paths'], problem['t0']
    covers = []
    for j in range(t0,problem['horizon']):
        release = max(t0,j-lead)
        mask = sum(1 << i for i,(s,q,t) in enumerate(pairs)
                   if release <= t and paths[s][j] != paths[q][j])
        if mask:
            covers.append((j,mask))
    universe = (1 << len(pairs)) - 1
    available = 0
    for _,mask in covers:
        available |= mask
    if available != universe:
        return dict(feasible=False, critical_pairs=len(pairs), relevant_messages=len(covers),
                    uncovered_pairs=[list(pairs[i]) for i in range(len(pairs)) if not (available >> i)&1])
    # Equal-cost supersets dominate subsets for the static menu only.
    unique = {}
    for j,mask in covers:
        unique.setdefault(mask,j)
    reduced = [(j,m) for m,j in unique.items()
               if not any(m != other and m & other == m for other in unique)]
    containing = [[i for i,(_,m) in enumerate(reduced) if (m >> p)&1] for p in range(len(pairs))]

    @lru_cache(None)
    def solve(uncovered):
        if not uncovered:
            return ()
        options = min((containing[p] for p in range(len(pairs)) if (uncovered >> p)&1), key=len)
        best = None
        for i in sorted(options,key=lambda k:-(reduced[k][1]&uncovered).bit_count()):
            remaining = uncovered & ~reduced[i][1]
            suffix = solve(remaining)
            candidate = (i,) + suffix
            if best is None or len(candidate) < len(best):
                best = candidate
                if len(best) == 1:
                    break
        return best

    selected = solve(universe)
    return dict(feasible=True, critical_pairs=len(pairs), relevant_messages=len(covers),
                reduced_messages=len(reduced), optimal_message_count=len(selected),
                selected_slots=sorted(reduced[i][0] for i in selected),
                candidate_slots=[j for j,_ in covers], exact_cover_states=solve.cache_info().currsize)


class AdaptiveDesign:
    def __init__(self, problem, lead):
        self.problem, self.lead = problem, lead
        self.paths, self.target = problem['paths'], problem['target']
        self.h, self.t0 = problem['horizon'], problem['t0']
        self.full = (1 << len(self.paths)) - 1
        self.weights, self.denominator = problem['weights'], problem['denominator']
        self.choice, self.values = {}, {}
        self.mass = lru_cache(None)(self.mass)
        self.value = lru_cache(None)(self.value)
        self.tests = []
        for j in range(self.t0,self.h):
            groups = defaultdict(int)
            for s,path in enumerate(self.paths):
                groups[path[j]] |= 1 << s
            if len(groups) > 1:
                self.tests.append((j,max(self.t0,j-lead),tuple(groups.values())))

    def mass(self, mask):
        return sum(w for s,w in enumerate(self.weights) if (mask >> s)&1)

    def partition(self,mask,groups):
        return tuple(sub for group in groups if (sub := mask & group))

    def natural(self,t,mask):
        groups = defaultdict(int)
        for s,path in enumerate(self.paths):
            if (mask >> s)&1:
                groups[path[t]] |= 1 << s
        return tuple(groups.values())

    def constant_action(self,t,mask):
        values = {self.target[s][t] for s in range(len(self.paths)) if (mask >> s)&1}
        return len(values) == 1

    def value(self,t,mask):
        if t == self.h:
            self.values[(t,mask)] = 0
            self.choice[(t,mask)] = ('end',)
            return 0
        if self.constant_action(t,mask):
            children = [self.value(t+1,sub) for sub in self.natural(t,mask)]
            result = None if any(v is None for v in children) else sum(children)
            self.choice[(t,mask)] = ('advance',) if result is not None else ('infeasible',)
        else:
            result = None
            selected = None
            seen_partitions = set()
            for j,available,groups in self.tests:
                if available > t:
                    continue
                parts = self.partition(mask,groups)
                if len(parts) <= 1 or parts in seen_partitions:
                    continue
                seen_partitions.add(parts)
                children = [self.value(t,sub) for sub in parts]
                if any(v is None for v in children):
                    continue
                candidate = self.mass(mask) + sum(children)
                if result is None or candidate < result:
                    result, selected = candidate,j
            self.choice[(t,mask)] = ('query',selected) if selected is not None else ('infeasible',)
        self.values[(t,mask)] = result
        return result

    def initial_groups(self):
        groups = defaultdict(int)
        for s,path in enumerate(self.paths):
            groups[tuple(path[:self.t0])] |= 1 << s
        return tuple(groups.values())

    def solve(self):
        values = [self.value(self.t0,mask) for mask in self.initial_groups()]
        feasible = all(v is not None for v in values)
        total = sum(values) if feasible else None
        return dict(feasible=feasible, weighted_cost=total,
                    expected_queries_exact=str(Fraction(total,self.denominator)) if feasible else None,
                    expected_queries=float(Fraction(total,self.denominator)) if feasible else None,
                    states=len(self.values), queried_slots=len(self.tests))

    def certificate(self):
        return [dict(t=t,mask=mask,value=value,choice=list(self.choice[(t,mask)]))
                for (t,mask),value in sorted(self.values.items())]

    def replay(self):
        """Reconstruct every path; use target-independent inventory recursion."""
        rows = []
        for s,path in enumerate(self.paths):
            mask = next(m for m in self.initial_groups() if (m >> s)&1)
            actual = list(self.problem['prefix'])
            queries = []
            for t in range(self.t0,self.h):
                while self.choice[(t,mask)][0] == 'query':
                    j = self.choice[(t,mask)][1]
                    assert max(self.t0,j-self.lead) <= t
                    queries.append(dict(t=t,j=j,value=path[j],compatible_before=mask))
                    mask = sum(1 << q for q in range(len(self.paths))
                               if (mask >> q)&1 and self.paths[q][j] == path[j])
                assert self.choice[(t,mask)][0] == 'advance'
                assert self.constant_action(t,mask)
                actual.append(self.target[s][t])
                mask = sum(1 << q for q in range(len(self.paths))
                           if (mask >> q)&1 and self.paths[q][t] == path[t])
            completed,inventory,_ = base.physical(actual,path,self.problem['n'],self.problem['lag'])
            assert all(0 <= u <= k for u,k in zip(base.increments(actual),self.problem['caps']))
            assert completed == self.problem['references'][s]
            assert actual == self.target[s]
            waiting = self.problem['case']['delta'] * sum(inventory[self.t0:])
            rows.append(dict(s=s,queries=queries,query_count=len(queries),
                             completed=completed,cumulative_release=actual,waiting=waiting))
        weighted = sum(self.weights[r['s']]*r['query_count'] for r in rows)
        optimum = sum(self.value(self.t0,m) for m in self.initial_groups())
        assert weighted == optimum
        return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke'],required=True)
    args = parser.parse_args()
    print('Implementation scaffold ready; smoke runner follows theory review.')


if __name__ == '__main__':
    main()
