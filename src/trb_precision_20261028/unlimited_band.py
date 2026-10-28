"""Independent release lower bound with every admissible interval response."""
from collections import defaultdict
from band_design import old


def unlimited(problem, lead, delay, width):
    paths, h, t0 = problem['paths'], problem['horizon'], problem['t0']
    releases = [list(problem['prefix']) for _ in paths]
    groups_by_time = []
    for t in range(t0, h):
        reported = range(max(t, t0+delay), min(h, t+lead-delay+1)) if t >= t0+delay else ()
        groups = defaultdict(list)
        for s, path in enumerate(paths):
            key = (tuple(path[:t]), tuple(path[j]//width for j in reported))
            groups[key].append(s)
        for group in groups.values():
            lower = max(problem['bounds'][s][t] for s in group)
            for s in group:
                previous = releases[s][-1] if releases[s] else 0
                releases[s].append(max(previous, lower))
        groups_by_time.append(list(groups.values()))
    rows = []
    for s, path in enumerate(paths):
        done, inventory, _ = old.base.physical(releases[s], path, problem['n'], problem['lag'])
        assert done == problem['references'][s]
        assert all(0 <= v <= k for v, k in zip(
            old.base.increments(releases[s]), problem['caps']))
        rows.append(dict(s=s, release=releases[s], completed=done,
                         post_release=problem['case']['delta']*sum(inventory[t0:])/problem['n']))
    weighted = sum(w*sum(u-d for u, d in zip(releases[s][t0:],
                    problem['references'][s][t0:])) for s, w in enumerate(problem['weights']))
    return dict(weighted_waiting=weighted,
                waiting=weighted*problem['case']['delta']/problem['denominator']/problem['n'],
                paths=rows, groups=groups_by_time)
