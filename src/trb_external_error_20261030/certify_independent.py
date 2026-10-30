"""Check finite-lookahead lower bounds on deduplicated independent conditions."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/trb_error_20261030'))
from robust_reports import prepare, SearchLimit
from budgeted_reports import BudgetedReports
from independent_panel import read, save, sha, DATA, OUT


def one(row):
    path = ROOT/row['input_path']
    assert sha(path) == row['input_sha256']
    c = read(path)
    p = prepare(c['service'], c['n'], (row['cap'],)*len(c['service'][0]), c['lag'])
    candidate = row['methods']['6']
    assert candidate['resolved']
    upper = candidate['lookahead']['value']
    optimum = candidate['exact']['value']
    q = candidate['lookahead']['max_requests']
    model = BudgetedReports(p, c['lag']+2, row['delay'], 1, 1,
                            total_requests=q, seconds_limit=20, state_limit=500000)
    started = perf_counter()

    @lru_cache(None)
    def lower(depth, t, e, pending, queried, u):
        if t == model.h:
            return 0
        if depth == 0:
            return model.lower(t, e, u)
        if perf_counter()-started > 20 or lower.cache_info().misses > 500000:
            raise SearchLimit('independent bound resource allowance reached')
        z = model.release(t, e, u)
        if z == model.n:
            return model.lower(t, e, u)
        best = None
        for batch in model.batches(t, e, pending, queried):
            children = model.transitions(t, e, pending, queried, batch)
            cheap = max(z-ref+model.lower(t+1, en, z) for _, _, en, _, _, ref in children)
            if best is not None and cheap >= best:
                continue
            worst = 0
            for _, _, en, pn, an, ref in children:
                worst = max(worst, z-ref+lower(depth-1, t+1, en, pn, an, z))
                if best is not None and worst >= best:
                    break
            if best is None or worst < best:
                best = worst
        return best

    result = {k:row[k] for k in ('case_id','airport','hour','n','cap','delay','numerical_signature')}
    initial_lower = model.lower(0, model.initial()[1], 0)
    result.update(upper=upper, optimum=optimum, total_request_limit=q,
                  depths=[], initial_lower=initial_lower, certified=False)
    previous = initial_lower
    try:
        for depth in range(1, 9):
            value = lower(depth, *model.initial())
            assert previous <= value <= optimum <= upper
            result['depths'].append(dict(depth=depth, lower=value))
            previous = value
            if value == upper:
                result.update(certified=True, first_certifying_depth=depth)
                break
        result['resolved'] = True
    except SearchLimit as exc:
        result.update(resolved=False, reason=str(exc))
    result.update(remaining_bound_gap=upper-previous, true_gap=upper-optimum,
                  seconds=perf_counter()-started, states=lower.cache_info().misses)
    assert result['remaining_bound_gap'] >= result['true_gap']
    return result


def run(stage):
    full = read(OUT/'full.json')
    assert full['passed'] and len(full['rows']) == 70
    rows = full['rows']
    if stage == 'smoke':
        selected = [next(r for r in rows if r['airport']==a and r['delay']==d)
                    for a in ('ATL','CLT','DFW') for d in (1,2)]
        selected += [r for r in rows if r['methods']['6']['lookahead']['value'] >
                     r['methods']['6']['exact']['value']]
        rows = list({r['numerical_signature']:r for r in selected}.values())
        cached = []
    else:
        smoke = read(OUT/'certificate_smoke.json')
        assert smoke['passed']
        cached = smoke['rows']
    done = {r['numerical_signature']:r for r in cached}
    tasks = [r for r in rows if r['numerical_signature'] not in done]
    save(DATA/('certificate_'+stage+'_design.json'), dict(
        maximum_depth=8, bounds='Perfect-information continuation after d request decisions',
        selection='All unique independent shared-error conditions' if stage=='full' else
                  'First condition at each airport and response time, plus every suboptimal six-step condition',
        seconds_limit=20, state_limit=500000,
        numerical_signatures=[r['numerical_signature'] for r in rows]))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i, result in enumerate(pool.map(one, tasks), 1):
            done[result['numerical_signature']] = result
            if stage == 'smoke' or i % 10 == 0:
                print(json.dumps(dict(completed=len(done), total=len(rows))), flush=True)
    ordered = [done[r['numerical_signature']] for r in rows]
    result = dict(passed=all(r['resolved'] for r in ordered), conditions=len(ordered),
                  certified=sum(r['certified'] for r in ordered), rows=ordered)
    save(OUT/('certificate_'+stage+'.json'), result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['smoke','full'])
    run(ap.parse_args().stage)
