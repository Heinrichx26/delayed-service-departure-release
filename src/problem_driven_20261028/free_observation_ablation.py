"""A frozen-input ablation of post-reset free service observations.

The existing acquisition implementation is imported without alteration.
Each stored Bellman state is checked from raw service arrays, and every
feasible policy is replayed on every original path before aggregation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction
import gzip
import json
import os
from pathlib import Path
from time import perf_counter

import info_design as design
import experiment

ROOT = design.ROOT
DATA = design.DATA / 'free_observation_ablation'
OUT = design.OUT / 'free_observation_ablation'
INPUT_MANIFEST = design.DATA / 'design.json'
SOURCE = Path(__file__).resolve()


class NoFreeObservations(design.AdaptiveDesign):
    """Retain initial observed histories; suppress subsequent free reports."""

    def natural(self, t, mask):
        return (mask,)

    def replay(self):
        rows = []
        for s, path in enumerate(self.paths):
            mask = next(m for m in self.initial_groups() if m & (1 << s))
            actual, queries = list(self.problem['prefix']), []
            for t in range(self.t0, self.h):
                while self.choice[(t, mask)][0] == 'query':
                    j = self.choice[(t, mask)][1]
                    assert self.t0 <= j < self.h
                    assert max(self.t0, j-self.lead) <= t
                    before = mask
                    queries.append(dict(t=t, j=j, value=path[j], compatible_before=mask))
                    mask = sum(1 << q for q in range(len(self.paths))
                               if before & (1 << q) and self.paths[q][j] == path[j])
                    assert 0 < mask.bit_count() < before.bit_count()
                assert self.choice[(t, mask)] == ('advance',)
                members = [q for q in range(len(self.paths)) if mask & (1 << q)]
                assert len({self.target[q][t] for q in members}) == 1
                actual.append(self.target[s][t])
                # No support contraction by the realised service path here.
            assert self.choice[(self.h, mask)] == ('end',)
            completed, inventory, _ = design.base.physical(
                actual, path, self.problem['n'], self.problem['lag'])
            assert all(0 <= u <= k for u, k in
                       zip(design.base.increments(actual), self.problem['caps']))
            assert completed == self.problem['references'][s]
            assert actual == self.target[s]
            waiting = self.problem['case']['delta'] * sum(inventory[self.t0:])
            rows.append(dict(s=s, queries=queries, query_count=len(queries),
                             completed=completed, cumulative_release=actual,
                             waiting=waiting))
        weighted = sum(self.weights[r['s']]*r['query_count'] for r in rows)
        initial_cost = sum(self.values[(self.t0, m)] for m in self.initial_groups())
        assert weighted == initial_cost
        return rows


def certificate_check(problem, lead, certificate, result, free_after_reset):
    """Rebuild both observation and query partitions independently of the engine."""
    states = {(r['t'], r['mask']): r for r in certificate}
    assert len(states) == len(certificate)
    paths, targets, weights = problem['paths'], problem['target'], problem['weights']
    t0, h = problem['t0'], problem['horizon']
    alternatives = 0

    def groups(mask, key):
        bins = defaultdict(int)
        for s, path in enumerate(paths):
            if mask & (1 << s):
                bins[key(s, path)] |= 1 << s
        return tuple(bins.values())

    for (t, mask), row in states.items():
        members = [s for s in range(len(paths)) if mask & (1 << s)]
        assert members and t0 <= t <= h
        assert len({tuple(targets[s][t0:t]) for s in members}) == 1
        observed_end = t if free_after_reset else t0
        assert len({tuple(paths[s][:observed_end]) for s in members}) == 1
        if t == h:
            expected, expected_choice = 0, ['end']
        elif len({targets[s][t] for s in members}) == 1:
            children = (groups(mask, lambda s, p: p[t])
                        if free_after_reset else (mask,))
            values = [states[(t+1, child)]['value'] for child in children]
            expected = None if None in values else sum(values)
            expected_choice = ['advance'] if expected is not None else ['infeasible']
        else:
            candidates = []
            for j in range(t0, h):
                if max(t0, j-lead) > t:
                    continue
                children = groups(mask, lambda s, p: p[j])
                if len(children) <= 1:
                    continue
                values = [states[(t, child)]['value'] for child in children]
                alternatives += 1
                cost = None if None in values else sum(weights[s] for s in members)+sum(values)
                if cost is not None:
                    candidates.append((cost, j))
            expected = min((cost for cost, _ in candidates), default=None)
            expected_choice = row['choice']
            if expected is None:
                assert expected_choice == ['infeasible']
            else:
                assert expected_choice[0] == 'query'
                assert (expected, expected_choice[1]) in candidates
        assert row['value'] == expected
        assert row['choice'] == expected_choice
    initial = groups((1 << len(paths))-1, lambda s, p: tuple(p[:t0]))
    costs = [states[(t0, mask)]['value'] for mask in initial]
    expected_total = None if None in costs else sum(costs)
    assert result['weighted_cost'] == expected_total
    assert result['feasible'] == (expected_total is not None)
    return dict(passed=True, states=len(states), query_alternatives=alternatives,
                free_after_reset=free_after_reset)


def solve_pair(problem, lead):
    begin = perf_counter()
    details = {}
    for name, cls, free in (('exact', design.AdaptiveDesign, True),
                            ('no_free', NoFreeObservations, False)):
        engine = cls(problem, lead)
        optimum = engine.solve()
        certificate = engine.certificate()
        checked = certificate_check(problem, lead, certificate, optimum, free)
        replay = engine.replay() if optimum['feasible'] else []
        details[name] = dict(optimum=optimum, certificate_check=checked,
                             certificate=certificate, replay=replay)
    exact, no_free = (details[name]['optimum'] for name in ('exact', 'no_free'))
    assert exact['feasible'] == (lead >= problem['L_star'])
    if no_free['feasible']:
        assert exact['feasible']
        assert no_free['weighted_cost'] >= exact['weighted_cost']
        assert details['exact']['replay']
        for e, n in zip(details['exact']['replay'], details['no_free']['replay']):
            assert e['completed'] == n['completed']
            assert e['cumulative_release'] == n['cumulative_release']
            assert e['waiting'] == n['waiting']
    if lead >= 0:
        assert no_free['feasible'] == exact['feasible']
    details['elapsed_seconds'] = perf_counter()-begin
    return details


def record(row, case, problem, lead, result):
    exact, no_free = (result[name]['optimum'] for name in ('exact', 'no_free'))
    gap = (Fraction(no_free['weighted_cost']-exact['weighted_cost'], problem['denominator'])
           if no_free['feasible'] else None)
    return dict(case_id=row['case_id'], key=row['key'], airport=case['airport'],
                day=case['day'], capacity_label=row['capacity_label'], t0=problem['t0'],
                N=case['n'], lag=case['lag'], horizon=problem['horizon'], lead=lead,
                lead_regime='lag_plus_2' if lead == case['lag']+2 else 'full_horizon',
                L_star=problem['L_star'], exact_feasible=exact['feasible'],
                no_free_feasible=no_free['feasible'],
                exact_queries=exact['expected_queries'],
                exact_queries_exact=exact['expected_queries_exact'],
                no_free_queries=no_free['expected_queries'],
                no_free_queries_exact=no_free['expected_queries_exact'],
                free_observation_saving=float(gap) if gap is not None else None,
                free_observation_saving_exact=str(gap) if gap is not None else None,
                strict_saving=gap is not None and gap > 0,
                checked_states=sum(result[k]['certificate_check']['states'] for k in ('exact','no_free')),
                checked_alternatives=sum(result[k]['certificate_check']['query_alternatives'] for k in ('exact','no_free')),
                replayed_paths=sum(len(result[k]['replay']) for k in ('exact','no_free')),
                elapsed_seconds=result['elapsed_seconds'])


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    return dict(ablation_sha256=design.digest(SOURCE),
                acquisition_sha256=design.digest(Path(design.__file__)),
                original_runner_sha256=design.digest(Path(experiment.__file__)),
                input_manifest_sha256=design.digest(INPUT_MANIFEST))


def prefixes(case):
    empty = design.prepare(case)
    if empty is None:
        return None
    prefix = design.base.frontier(empty['paths'], empty['bounds'], [], -1)[0][:1]
    return ([], prefix)


def smoke():
    begin = perf_counter()
    control_case = dict(n=1, caps=[1]*5, lag=1, delta=1,
                        service=[[1,0,0,1,0],[0,0,0,0,1]],
                        probabilities=['1/2','1/2'])
    control_problem = design.prepare(control_case)
    control = solve_pair(control_problem, 3)
    assert control['exact']['optimum']['weighted_cost'] == 0
    assert control['no_free']['optimum']['weighted_cost'] == control_problem['denominator']
    manifest = design.read(INPUT_MANIFEST)
    physical = design.read(ROOT/manifest['physical_input_manifest'])
    keys = set(physical['smoke_keys'])
    selected = [r for r in manifest['rows'] if r['key'] in keys and r['capacity_label'] == 'original']
    assert len(selected) == 2
    rows, details = [], []
    for row in selected:
        path = ROOT/row['input_path']
        assert design.digest(path) == row['input_sha256']
        case = design.read(path)
        choices = prefixes(case)
        assert choices is not None
        for executed in choices:
            problem = design.prepare(case, executed)
            leads = [case['lag']+2, problem['horizon']-1]
            assert len(set(leads)) == 2
            for lead in leads:
                result = solve_pair(problem, lead)
                flat = record(row, case, problem, lead, result)
                rows.append(flat)
                details.append(dict(input=row, t0=problem['t0'], lead=lead, **result))
                print(row['case_id'], problem['t0'], lead, flat['exact_queries_exact'],
                      flat['no_free_queries_exact'], round(result['elapsed_seconds'],3), flush=True)
    useful = any(r['strict_saving'] for r in rows)
    report = dict(passed=True, date='2026-10-28', **source_hashes(),
                  public_cases=len(selected), comparison_points=len(rows), useful=useful,
                  positive_control=control,
                  strict_saving_points=sum(r['strict_saving'] for r in rows),
                  max_seconds=max(r['elapsed_seconds'] for r in rows),
                  elapsed_seconds=perf_counter()-begin, rows=rows, details=details)
    design.save(OUT/'smoke.json', report)
    csv_write(OUT/'smoke.csv', rows)
    print({k:v for k,v in report.items() if k not in ('rows','details','positive_control')}, flush=True)


def freeze():
    report = design.read(OUT/'smoke.json')
    assert report['passed'] and report['useful']
    for key, digest in source_hashes().items():
        assert report[key] == digest
    manifest = design.read(INPUT_MANIFEST)
    eligibility = []
    for row in manifest['rows']:
        path = ROOT/row['input_path']
        assert design.digest(path) == row['input_sha256']
        complete = prefixes(design.read(path)) is not None
        eligibility.append(dict(**row, complete=complete))
    complete = [r for r in eligibility if r['complete']]
    assert len(complete) == 200
    frozen = dict(date='2026-10-28', frozen_before_panel=True, **source_hashes(),
                  smoke_sha256=design.digest(OUT/'smoke.json'),
                  candidate_supports=len(eligibility), complete_supports=len(complete),
                  excluded_supports=len(eligibility)-len(complete), comparisons=800,
                  rows=eligibility, prefixes=[0,1], leads=['lag+2','H-1'], workers=2,
                  intervention='Remove only free service observations at and after reset; retain complete pre-reset history and memory of purchased responses.',
                  controls='Identical original paths, probabilities, target trajectories, executed prefixes, query availability, and unit per-coordinate fees.',
                  verification='Independent raw-array Bellman check for every stored state and every available informative query; physical replay on every original path for both policies.',
                  eligibility='All original positive-probability paths complete their fixed cohort under the capacity-specific earliest reference; excluded supports retained in coverage.')
    design.save(DATA/'design.json', frozen)
    print(dict(frozen=True, complete_supports=len(complete), comparisons=800), flush=True)


def worker(row):
    begin = perf_counter()
    path = ROOT/row['input_path']
    assert design.digest(path) == row['input_sha256']
    case = design.read(path)
    choices = prefixes(case)
    assert choices is not None
    rows, details = [], []
    for executed in choices:
        problem = design.prepare(case, executed)
        leads = [case['lag']+2, problem['horizon']-1]
        assert len(set(leads)) == 2
        for lead in leads:
            result = solve_pair(problem, lead)
            rows.append(record(row, case, problem, lead, result))
            details.append(dict(t0=problem['t0'], lead=lead, **result))
    output = OUT/'runs'/f"{row['case_id']}.json.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(json.dumps(dict(input=row,results=details),
                                               separators=(',',':')).encode(), mtime=0))
    return dict(case_id=row['case_id'], run_path=str(output.relative_to(ROOT)),
                run_sha256=design.digest(output), elapsed_seconds=perf_counter()-begin), rows


def panel():
    begin = perf_counter()
    manifest = design.read(DATA/'design.json')
    for key, digest in source_hashes().items():
        assert manifest[key] == digest
    assert manifest['smoke_sha256'] == design.digest(OUT/'smoke.json')
    complete = [r for r in manifest['rows'] if r['complete']]
    coverage = [dict(case_id=r['case_id'],key=r['key'],capacity_label=r['capacity_label'],
                     complete=r['complete']) for r in manifest['rows']]
    rows, runs = [], []
    with ProcessPoolExecutor(max_workers=min(manifest['workers'], os.cpu_count() or 1)) as pool:
        jobs = {pool.submit(worker, row): row for row in complete}
        for future in as_completed(jobs):
            run, records = future.result()
            runs.append(run)
            rows.extend(records)
            if len(runs) % 20 == 0 or len(runs) == len(jobs):
                print('completed', len(runs), 'of', len(jobs),
                      'seconds', round(perf_counter()-begin,1), flush=True)
    rows.sort(key=lambda r:(r['case_id'],r['t0'],r['lead']))
    runs.sort(key=lambda r:r['case_id'])
    assert len(rows) == manifest['comparisons']
    csv_write(OUT/'coverage.csv', coverage)
    csv_write(OUT/'comparisons.csv', rows)
    by_regime = []
    for capacity in sorted({r['capacity_label'] for r in rows}):
        for t0 in (0,1):
            for regime in ('lag_plus_2','full_horizon'):
                group = [r for r in rows if r['capacity_label']==capacity and
                         r['t0']==t0 and r['lead_regime']==regime]
                valid = [r for r in group if r['no_free_feasible']]
                by_regime.append(dict(capacity_label=capacity,t0=t0,lead_regime=regime,
                                      windows=len(group),paired_feasible=len(valid),
                                      strict_saving_points=sum(r['strict_saving'] for r in valid),
                                      mean_exact_queries=sum(r['exact_queries'] for r in valid)/len(valid),
                                      mean_no_free_queries=sum(r['no_free_queries'] for r in valid)/len(valid),
                                      mean_saving=sum(r['free_observation_saving'] for r in valid)/len(valid)))
    csv_write(OUT/'summary.csv', by_regime)
    report = dict(passed=True,date='2026-10-28',design_sha256=design.digest(DATA/'design.json'),
                  candidate_supports=len(coverage),complete_supports=len(complete),
                  excluded_supports=len(coverage)-len(complete),comparison_points=len(rows),
                  paired_feasible_points=sum(r['no_free_feasible'] for r in rows),
                  strict_saving_points=sum(r['strict_saving'] for r in rows),
                  checked_bellman_states=sum(r['checked_states'] for r in rows),
                  checked_query_alternatives=sum(r['checked_alternatives'] for r in rows),
                  replayed_paths=sum(r['replayed_paths'] for r in rows),
                  elapsed_seconds=perf_counter()-begin,summary=by_regime,runs=runs)
    design.save(OUT/'panel.json', report)
    print({k:v for k,v in report.items() if k not in ('summary','runs')}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke','freeze','panel'], required=True)
    args = parser.parse_args()
    globals()[args.phase]()


if __name__ == '__main__':
    main()
