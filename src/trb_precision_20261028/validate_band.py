"""Independent enumeration of quantities and requests with interval answers."""
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import json
import random
from time import perf_counter
from band_design import BandDesign, ROOT, old
from validate_joint import physical


def brute(problem, lead, delay, budget, width):
    paths = problem['paths']
    h, n, weights = problem['horizon'], problem['n'], problem['weights']

    @lru_cache(None)
    def recurse(t, members, pending, prefix):
        if t == h:
            valid = all(physical(prefix, paths[s], problem['lag']) ==
                        problem['references'][s] for s in members)
            return (0, 0) if valid else None
        previous = prefix[-1] if prefix else 0
        slots = [j for j in range(t + delay, min(h, t + lead + 1))
                 if j not in {j for _, j in pending}]
        best = None
        for u in range(previous, min(n, previous + problem['caps'][t]) + 1):
            extended = prefix + (u,)
            if any(physical(extended, paths[s], problem['lag']) !=
                   problem['references'][s][:t+1] for s in members):
                continue
            for size in range(min(budget, len(slots)) + 1):
                for request in combinations(slots, size):
                    combined = pending + tuple((t + delay, j) for j in request)
                    due = [j for arrival, j in combined if arrival == t + 1]
                    future = tuple(sorted((a, j) for a, j in combined if a > t + 1))
                    groups = defaultdict(list)
                    for s in members:
                        observed = (paths[s][t],) + tuple(paths[s][j] // width for j in due)
                        groups[observed].append(s)
                    cost = [sum(weights[s] * (u-problem['references'][s][t])
                                for s in members),
                            size * sum(weights[s] for s in members)]
                    for group in groups.values():
                        child = recurse(t + 1, tuple(group), future, extended)
                        if child is None:
                            break
                        cost[0] += child[0]
                        cost[1] += child[1]
                    else:
                        if best is None or tuple(cost) < best:
                            best = tuple(cost)
        return best
    groups = defaultdict(list)
    for s, path in enumerate(paths):
        groups[tuple(path[:problem['t0']])].append(s)
    values = [recurse(problem['t0'], tuple(m), (), tuple(problem['prefix']))
              for m in groups.values()]
    return tuple(sum(v[k] for v in values) for k in (0, 1))


def replay(problem, rows, lead, delay, budget, width):
    histories = {}
    for row in rows:
        path = problem['paths'][row['s']]
        queries, release = row['queries'], row['release']
        assert physical(release, path, problem['lag']) == problem['references'][row['s']]
        previous = 0
        for t, u in enumerate(release):
            assert 0 <= u-previous <= problem['caps'][t]
            previous = u
            if t < problem['t0']:
                continue
            batch = tuple(q['slot'] for q in queries if q['dispatch'] == t)
            assert len(batch) <= budget
            for q in queries:
                assert q['arrival'] == q['dispatch'] + delay
                assert q['slot']-lead <= q['dispatch'] <= q['slot']-delay
            arrived = tuple((q['dispatch'], q['slot'], path[q['slot']] // width)
                            for q in queries if q['arrival'] <= t)
            sent = tuple((q['dispatch'], q['slot'])
                         for q in queries if q['dispatch'] < t)
            key = (t, tuple(path[:t]), arrived, sent)
            action = (u, batch)
            assert key not in histories or histories[key] == action
            histories[key] = action
    return len(histories)


def main():
    start = perf_counter()
    rng = random.Random(2026102829)
    rows = []
    for i in range(30):
        n, m, h = rng.choice([2, 3]), rng.choice([2, 3, 4]), 6
        paths = []
        for _ in range(m):
            path = [0]*h
            for _ in range(n):
                path[rng.randrange(2, h)] += 1
            paths.append(path)
        # Unequal positive probabilities also test the weighted objective.
        w = [rng.randrange(1, 5) for _ in range(m)]
        case = dict(n=n, lag=1, caps=[rng.choice([1, 2]) for _ in range(h)],
                    service=paths, probabilities=[f'{x}/{sum(w)}' for x in w], delta=2)
        for prefix in [(), tuple(old.base.earliest(n, case['caps'])[:2])] if i < 10 else [()]:
            problem = old.prepare(case, prefix=prefix)
            assert problem is not None
            for delay in [1, 2]:
                for budget in [1, 2]:
                    prior = None
                    for width in [1, 2, 4]:
                        model = BandDesign(problem, 3, delay, budget, width)
                        result = model.solve()
                        oracle = brute(problem, 3, delay, budget, width)
                        assert (result['weighted_waiting'], result['weighted_queries']) == oracle
                        replay(problem, model.replay(), 3, delay, budget, width)
                        if prior is not None:
                            assert prior <= oracle[0]
                        prior = oracle[0]
                        rows.append(dict(case=i, prefix=len(prefix), delay=delay,
                                         budget=budget, width=width, value=oracle))
    output = dict(passed=True, conditions=len(rows), seconds=perf_counter()-start,
                  seed=2026102829, rows=rows)
    (ROOT/'results/trb_precision_20261028/theory_smoke.json').write_text(
        json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
