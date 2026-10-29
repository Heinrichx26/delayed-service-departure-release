"""Frozen-support announcement-frontier matcher, separate from production control.

The full stage uses only integer recurrences and independent physical checks.
Small tree linear programs are used only as smoke-stage optimality oracles.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import gzip
import hashlib
import json
from math import ceil
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/trb_reassessment_20261027/fixed_support_frontier'
OUT = ROOT/'results/revision_20261027/trb_reassessment/fixed_support_frontier'
FROZEN = ROOT/'data/trb_reassessment_20261027/limited_announcement_frozen_design.json'
CAPACITY = ROOT/'data/trb_reassessment_20261027/capacity_mechanism_four_window_design.json'
MANIFEST = DATA/'design.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def packed(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(value, separators=(',', ':')).encode(), mtime=0))


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def increments(cumulative):
    previous = 0
    result = []
    for value in cumulative:
        result.append(value-previous)
        previous = value
    return result


def earliest(n, caps, prefix=()):
    result = list(prefix)
    previous = result[-1] if result else 0
    for cap in caps[len(prefix):]:
        previous = min(n, previous+cap)
        result.append(previous)
    return result


def physical(cumulative, service, n, lag):
    """Inventory simulation without deadline or frontier helper functions."""
    held, queue, done = n, 0, 0
    pipeline = [0]*lag
    completions, inventories, states = [], [], []
    for u, c in zip(increments(cumulative), service):
        assert type(u) is int and 0 <= u <= held
        held -= u
        mature = pipeline.pop(0) if lag else u
        if lag:
            pipeline.append(u)
        queue += mature
        served = min(queue, c)
        queue -= served
        done += served
        inventory = queue+sum(pipeline)
        assert held+inventory+done == n
        completions.append(done)
        inventories.append(inventory)
        states.append(dict(held=held, queue=queue, completed=done, pipeline=list(pipeline)))
    return completions, inventories, states


def completion_only(cumulative, service, lag):
    # A separate cumulative-work recursion is used for the action-boundary test.
    done, result = 0, []
    for t, capacity in enumerate(service):
        eligible = cumulative[t-lag] if t >= lag else 0
        done = min(eligible, done+capacity)
        result.append(done)
    return result


def bound(reference, caps, lag):
    h = len(caps)
    required = [reference[min(t+lag, h-1)] for t in range(h)]
    backward = required[:]
    for t in range(h-2, -1, -1):
        backward[t] = max(required[t], backward[t+1]-caps[t+1])
    for t in range(h):
        direct = max(required[v]-sum(caps[t+1:v+1]) for v in range(t, h))
        assert direct == backward[t]
    return required, backward


def groups(paths, t, lead):
    stop = min(len(paths[0]), max(0, t+lead+1))
    result = defaultdict(list)
    for s, path in enumerate(paths):
        result[tuple(path[:stop])].append(s)
    return list(result.values())


def frontier(paths, bounds, prefix, lead):
    result = [list(prefix) for _ in paths]
    for t in range(len(prefix), len(paths[0])):
        for group in groups(paths, t, lead):
            histories = {tuple(result[s]) for s in group}
            assert len(histories) == 1
            previous = result[group[0]][-1] if result[group[0]] else 0
            value = max(previous, max(bounds[s][t] for s in group))
            for s in group:
                result[s].append(value)
    return result


def threshold(paths, bounds, prefix):
    oracle = []
    for row in bounds:
        before = prefix[-1] if prefix else 0
        plan = list(prefix)
        for value in row[len(prefix):]:
            before = max(before, value)
            plan.append(before)
        oracle.append(plan)
    maximum, witnesses, count = -1, [], 0
    for s in range(len(paths)):
        for q in range(s):
            first = next((t for t, (a, b) in enumerate(zip(paths[s], paths[q])) if a != b), None)
            for t in range(len(prefix), len(paths[0])):
                if oracle[s][t] == oracle[q][t]:
                    continue
                assert first is not None
                needed = first-t
                if needed > maximum:
                    maximum, witnesses, count = needed, [], 0
                if needed == maximum:
                    count += 1
                    if len(witnesses) < 20:
                        witnesses.append(dict(paths=[q, s], action_slot=t,
                                              first_service_difference=first,
                                              oracle_cumulative=[oracle[q][t], oracle[s][t]]))
    return maximum, oracle, witnesses, count


def lp_oracle(case, references, prefix, lead, expected, expected_wait):
    """Independent information-tree LP; no backward closure enters the rows."""
    import warnings
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    paths, caps, lag, n = case['service'], case['caps'], case['lag'], case['n']
    h, count, t0 = len(caps), len(paths), len(prefix)
    probabilities = [float(Fraction(p)) for p in case['probabilities']]
    objective = np.array([case['delta']*probabilities[s] if t >= t0 else 0
                          for s in range(count) for t in range(h)])
    lower = []
    for s in range(count):
        for t in range(h):
            requirement = references[s][min(t+lag, h-1)]
            if t < t0:
                assert prefix[t] >= requirement
                lower.append((prefix[t], prefix[t]))
            else:
                lower.append((requirement, n))
    rr, cc, vv, rhs = [], [], [], []
    er, ec, ev, equal_rhs = [], [], [], []
    def inequality(pairs, value):
        row = len(rhs)
        for index, coefficient in pairs:
            rr.append(row); cc.append(index); vv.append(coefficient)
        rhs.append(value)
    for s in range(count):
        for t in range(h):
            index = s*h+t
            previous = index-1 if t else None
            inequality([(index, 1)]+([(previous, -1)] if t else []), caps[t])
            if t:
                inequality([(index, -1), (previous, 1)], 0)
    for t in range(t0, h):
        for group in groups(paths, t, lead):
            for s in group[1:]:
                row = len(equal_rhs)
                er.extend([row, row]); ec.extend([s*h+t, group[0]*h+t]); ev.extend([1, -1])
                equal_rhs.append(0)
    a = coo_matrix((vv, (rr, cc)), shape=(len(rhs), count*h)).tocsr()
    e = coo_matrix((ev, (er, ec)), shape=(len(equal_rhs), count*h)).tocsr() if equal_rhs else None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        solution = linprog(objective, A_ub=a, b_ub=rhs, A_eq=e,
                           b_eq=equal_rhs if equal_rhs else None, bounds=lower, method='highs',
                           options={'threads': 1, 'primal_feasibility_tolerance': 1e-9,
                                    'dual_feasibility_tolerance': 1e-9})
    assert solution.success, solution.message
    actual = solution.x.reshape(count, h)
    assert np.max(np.abs(actual-np.asarray(expected))) < 1e-6
    constant = case['delta']*sum(probabilities[s]*sum(references[s][t0:]) for s in range(count))
    value = float(solution.fun-constant)
    assert abs(value-float(expected_wait)) < 1e-6
    return dict(lead=lead, passed=True, variables=count*h, equality_rows=len(equal_rhs),
                inequality_rows=len(rhs), waiting=value, max_vector_error=float(np.max(np.abs(actual-np.asarray(expected)))))


def freeze():
    original, capacity = read(FROZEN), read(CAPACITY)
    assert len(original['paired_inputs']) == 94 and len(capacity['selected']) == 4
    cases, records = {}, []
    for pair in original['paired_inputs']:
        item = pair['forecast']
        cp, ip = ROOT/item['case_path'], ROOT/item['input_path']
        assert sha(cp) == item['case_sha256'] and sha(ip) == item['input_sha256']
        case = read(cp)
        payload = json.loads(gzip.decompress(ip.read_bytes()))
        n, lag, h = case['N'], case['lag_slots'], len(case['service'])
        paths = [([0]*lag+s['service'][:h-lag] if lag else s['service'][:]) for s in payload['scenarios']]
        caps = payload['scenarios'][0]['release_caps']
        assert all(s['release_caps'] == caps for s in payload['scenarios'])
        assert all(k == n for k in caps)
        assert len(paths) == len(payload['probabilities']) == 16
        probabilities = [str(Fraction(str(p))) for p in payload['probabilities']]
        assert all(Fraction(p) > 0 for p in probabilities) and sum(Fraction(p) for p in probabilities) == 1
        value = dict(key=case['key'], airport=case['airport'], day=case['day'], hour=case['hour'],
                     n=n, G=case['G'], lag=lag, delta=payload['scenarios'][0]['slot_minutes'],
                     costs=case['costs'], service=paths, probabilities=probabilities,
                     donor_days=payload['donor_days'], caps=caps,
                     origin=dict(case_path=item['case_path'], case_sha256=item['case_sha256'],
                                 input_path=item['input_path'], input_sha256=item['input_sha256']),
                     support_rule='Original sixteen shifted forecast-training service paths, with original weights; fixed for every lead.',
                     actual_calendar_included=False)
        cases[case['key']] = value
        records.append((value, 'original'))
    for selected in capacity['selected']:
        value = cases[selected['row']['key']]
        for label, denominator in (('ceil_N_over_4', 4), ('ceil_N_over_8', 8)):
            altered = dict(value)
            altered['caps'] = [ceil(value['n']/denominator)]*len(value['caps'])
            records.append((altered, label))
    rows = []
    for case, label in records:
        case = dict(case, capacity_label=label)
        filename = case['key']+'__'+label+'.json'
        path = DATA/'cases'/filename
        if path.exists():
            assert read(path) == case
        else:
            write(path, case)
        rows.append(dict(key=case['key'], capacity_label=label,
                         input_path=str(path.relative_to(ROOT)).replace('\\', '/'), input_sha256=sha(path)))
    manifest = dict(frozen_before_frontier_outcomes=True, date='2026-10-27',
                    original_design_sha256=sha(FROZEN), capacity_design_sha256=sha(CAPACITY),
                    original_physical_windows=94, additional_tight_capacity_configurations=8,
                    smoke_keys=original['smoke_keys'], prefixes=[0, 1],
                    common_prefix_rule='t0=1 fixes the first action of the L=-1 frontier over the full fixed support; identical for every compared lead.',
                    completion_eligibility='Every original support scenario must complete all N by H-1 under its own capacity-specific earliest release plan.',
                    common_deadlines='Same path-specific earliest completion references and transported requirements for every lead within each window-capacity configuration.',
                    lead_grid='Every integer lead from -1 through H-1 inclusive',
                    production_controller_runs=0, rows=rows)
    if MANIFEST.exists():
        assert read(MANIFEST) == manifest
    else:
        write(MANIFEST, manifest)
    return manifest


def input_case(row):
    path = ROOT/row['input_path']
    assert sha(path) == row['input_sha256']
    return read(path)


def audit_case(case, with_lp=False):
    n, lag, caps, paths = case['n'], case['lag'], case['caps'], case['service']
    h, count, delta = len(caps), len(paths), case['delta']
    probabilities = [Fraction(p) for p in case['probabilities']]
    reference_plan = earliest(n, caps)
    reference = [physical(reference_plan, path, n, lag)[0] for path in paths]
    terminal_deficits = [n-row[-1] for row in reference]
    coverage = dict(key=case['key'], capacity_label=case['capacity_label'],
                    N=n, lag_slots=lag, horizon=h, release_cap_min=min(caps), release_cap_max=max(caps),
                    support_scenarios=count, unique_service_paths=len({tuple(p) for p in paths}),
                    support_paths_clearing=sum(value == 0 for value in terminal_deficits),
                    all_support_paths_clear=not any(terminal_deficits),
                    terminal_unserved_by_support=terminal_deficits,
                    maximum_terminal_unserved=max(terminal_deficits))
    if any(terminal_deficits):
        coverage['status'] = 'incomplete_reference_support'
        coverage['prefixes_tested'] = 0
        return dict(coverage=coverage, references=reference, earliest_cumulative=reference_plan,
                    prefix_results=[], theorem_frontier_evaluated=False)
    needs_bounds = [bound(d, caps, lag) for d in reference]
    needs, bounds = [r[0] for r in needs_bounds], [r[1] for r in needs_bounds]
    history_frontier = frontier(paths, bounds, [], -1)
    common_first = history_frontier[0][0]
    assert all(row[0] == common_first for row in history_frontier)
    results = []
    for t0, prefix in ((0, []), (1, [common_first])):
        past_states = [physical(prefix, p[:t0], n, lag)[2] for p in paths]
        if t0:
            assert all(prefix[t] >= needs[s][t] for s in range(count) for t in range(t0))
            assert prefix[-1] >= max(bounds[s][t0-1] for s in range(count))
            assert all(states[-1] == past_states[0][-1] for states in past_states)
        minimum_lead, oracle, witnesses, witness_count = threshold(paths, bounds, prefix)
        previous, first_equal, levels, lp_checks = None, None, [], []
        selected_lp = set((-1, 0, lag, lag+2, h-1, minimum_lead, minimum_lead-1))
        selected_lp = {lead for lead in selected_lp if -1 <= lead < h}
        all_boundary_checks, all_feasible_checks, action_checks = 0, 0, 0
        for lead in range(-1, h):
            policy = frontier(paths, bounds, prefix, lead)
            path_wait = []
            for s, cumulative in enumerate(policy):
                assert cumulative[:t0] == prefix
                assert all(0 <= u <= cap for u, cap in zip(increments(cumulative), caps))
                completed, inventory, states = physical(cumulative, paths[s], n, lag)
                assert completed == reference[s]
                assert completed == completion_only(cumulative, paths[s], lag)
                assert inventory == [u-d for u, d in zip(cumulative, reference[s])]
                assert all(cumulative[t] >= oracle[s][t] for t in range(h))
                if previous is not None:
                    assert all(cumulative[t] <= previous[s][t] for t in range(h))
                path_wait.append(delta*sum(inventory[t0:]))
                action_checks += h-t0
            boundary_checks, feasible_checks, examples = 0, 0, []
            for t in range(t0, h):
                for group in groups(paths, t, lead):
                    s = group[0]
                    history = policy[s][:t]
                    assert all(policy[q][:t+1] == policy[s][:t+1] for q in group)
                    current = policy[s][t]
                    feasible = earliest(n, caps, history+[current])
                    for q in group:
                        assert completion_only(feasible, paths[q], lag) == reference[q]
                        feasible_checks += 1
                    before = history[-1] if history else 0
                    if current == before:
                        continue
                    altered = earliest(n, caps, history+[current-1])
                    failures = []
                    for q in group:
                        completed = completion_only(altered, paths[q], lag)
                        deficit = [a-b for a, b in zip(reference[q], completed)]
                        assert min(deficit) >= 0
                        bad = next((v for v, amount in enumerate(deficit) if amount > 0), None)
                        if bad is not None:
                            failures.append(dict(support_path=q, first_missed_completion=bad,
                                                 reference=reference[q][bad], completed=completed[bad]))
                    assert failures, (case['key'], t0, lead, t)
                    boundary_checks += 1
                    if len(examples) < 3:
                        examples.append(dict(action_slot=t, info_class=group, cumulative_required=current,
                                             tested_cumulative=current-1, witness=failures[0]))
            all_boundary_checks += boundary_checks
            all_feasible_checks += feasible_checks
            wait = sum(p*w for p, w in zip(probabilities, path_wait))
            oracle_wait = delta*sum(probabilities[s]*sum(oracle[s][t]-reference[s][t] for t in range(t0, h)) for s in range(count))
            formula = delta*sum(probabilities[s]*sum(policy[s][t]-reference[s][t] for t in range(t0, h)) for s in range(count))
            assert wait == formula and wait >= oracle_wait
            equal = policy == oracle
            assert equal == (lead >= minimum_lead)
            if equal and first_equal is None:
                first_equal = lead
            if previous is not None:
                assert all(path_wait[s] <= levels[-1]['path_waiting_minutes'][s] for s in range(count))
            level = dict(lead=lead, expected_waiting_minutes_exact=str(wait), expected_waiting_minutes=float(wait),
                         normalized_expected_waiting_minutes=float(wait/n),
                         excess_over_full_information_exact=str(wait-oracle_wait),
                         oracle_equal=equal, path_waiting_minutes=path_wait, cumulative_release=policy,
                         feasible_current_earliest_continuations=feasible_checks,
                         positive_action_lower_boundary_checks=boundary_checks,
                         lower_boundary_examples=examples)
            levels.append(level)
            if with_lp and lead in selected_lp:
                lp_checks.append(lp_oracle(case, reference, prefix, lead, policy, wait))
            previous = policy
        assert first_equal == minimum_lead
        if minimum_lead >= 0:
            assert witnesses
            for witness in witnesses:
                s, q = witness['paths']
                t = witness['action_slot']
                stop = min(h, max(0, t+minimum_lead))
                assert paths[s][:stop] == paths[q][:stop]
                assert oracle[s][t] != oracle[q][t]
        result = dict(t0=t0, common_executed_cumulative=prefix,
                      inherited_state=(past_states[0][-1] if t0 else dict(held=n, queue=0, completed=0, pipeline=[0]*lag)),
                      common_prefix_feasible=True, L_star=minimum_lead, first_equal_lead=first_equal,
                      threshold_witnesses=witnesses, threshold_witness_count=witness_count,
                      full_information_cumulative=oracle, frontier=levels,
                      strict_information_benefit=levels[0]['expected_waiting_minutes'] > levels[-1]['expected_waiting_minutes'],
                      waiting_reduction_minutes=levels[0]['expected_waiting_minutes']-levels[-1]['expected_waiting_minutes'],
                      waiting_reduction_per_aircraft=(levels[0]['expected_waiting_minutes']-levels[-1]['expected_waiting_minutes'])/n,
                      path_action_checks=action_checks, feasible_current_checks=all_feasible_checks,
                      positive_action_lower_boundary_checks=all_boundary_checks,
                      tree_lp_checks=lp_checks, passed=True)
        results.append(result)
    coverage.update(status='frontier_audited', prefixes_tested=len(results))
    return dict(coverage=coverage, references=reference, earliest_cumulative=reference_plan,
                transported_requirements=needs, singleton_backward_bounds=bounds,
                prefix_results=results, theorem_frontier_evaluated=True)


def self_test():
    case = dict(key='unit_burst', capacity_label='unit', n=2, G=2, lag=1, delta=1,
                caps=[1]*6, service=[[0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 2]], probabilities=['1/2', '1/2'])
    result = audit_case(case, with_lp=True)
    assert [r['L_star'] for r in result['prefix_results']] == [2, 1]
    assert result['prefix_results'][0]['frontier'][0]['expected_waiting_minutes'] == 6
    assert result['prefix_results'][0]['frontier'][-1]['expected_waiting_minutes'] == 3
    return dict(passed=True, capacity_burst_thresholds=[2, 1], tree_lp_checks=sum(len(r['tree_lp_checks']) for r in result['prefix_results']))


def run(stage):
    design = read(MANIFEST)
    assert sha(FROZEN) == design['original_design_sha256']
    assert sha(CAPACITY) == design['capacity_design_sha256']
    if stage == 'panel':
        smoke = read(OUT/'smoke.json')
        assert smoke['passed'] and smoke['source_sha256'] == sha(Path(__file__))
        assert smoke['manifest_sha256'] == sha(MANIFEST)
    start = perf_counter()
    unit = self_test() if stage == 'smoke' else None
    rows = design['rows'] if stage == 'panel' else [r for r in design['rows'] if r['key'] in design['smoke_keys']]
    results, summaries = [], []
    for i, row in enumerate(rows, 1):
        case = input_case(row)
        result = audit_case(case, with_lp=(stage == 'smoke'))
        path = OUT/(stage+'_runs')/(row['key']+'__'+row['capacity_label']+'.json.gz')
        packed(path, result)
        results.append(result)
        summaries.append(result['coverage'])
        print(json.dumps(dict(stage=stage, completed=i, total=len(rows), key=row['key'],
                              capacity=row['capacity_label'], status=result['coverage']['status'],
                              elapsed_seconds=perf_counter()-start)), flush=True)
    groups_summary = []
    for label in ('original', 'ceil_N_over_4', 'ceil_N_over_8'):
        selected = [r for r in results if r['coverage']['capacity_label'] == label]
        for t0 in (0, 1):
            eligible = [r for r in selected if r['theorem_frontier_evaluated']]
            prefixes = [next(p for p in r['prefix_results'] if p['t0'] == t0) for r in eligible]
            groups_summary.append(dict(capacity_label=label, t0=t0, candidate_window_configurations=len(selected),
                                       complete_support_configurations=len(eligible),
                                       incomplete_support_configurations=len(selected)-len(eligible),
                                       feasible_common_prefix_configurations=len(prefixes),
                                       strict_information_benefit_configurations=sum(p['strict_information_benefit'] for p in prefixes),
                                       L_star_counts=dict(sorted(Counter(p['L_star'] for p in prefixes).items())),
                                       positive_action_boundary_checks=sum(p['positive_action_lower_boundary_checks'] for p in prefixes),
                                       tree_lp_checks=sum(len(p['tree_lp_checks']) for p in prefixes)))
    report = dict(passed=True, stage=stage, source_sha256=sha(Path(__file__)), manifest_sha256=sha(MANIFEST),
                  design=design, unit_smoke=unit, configurations=len(results),
                  eligible_configurations=sum(r['theorem_frontier_evaluated'] for r in results),
                  audited_prefix_configurations=sum(len(r['prefix_results']) for r in results),
                  positive_action_lower_boundary_checks=sum(p['positive_action_lower_boundary_checks'] for r in results for p in r['prefix_results']),
                  feasible_current_checks=sum(p['feasible_current_checks'] for r in results for p in r['prefix_results']),
                  action_checks=sum(p['path_action_checks'] for r in results for p in r['prefix_results']),
                  tree_lp_checks=sum(len(p['tree_lp_checks']) for r in results for p in r['prefix_results']),
                  production_controller_runs=0, elapsed_seconds=perf_counter()-start,
                  summary=groups_summary, coverage=summaries)
    write(OUT/(stage+'.json'), report)
    csv_write(OUT/(stage+'_coverage.csv'), summaries)
    csv_write(OUT/(stage+'_summary.csv'), groups_summary)
    prefix_rows = []
    curve_rows = []
    for result in results:
        for prefix in result['prefix_results']:
            base = {k: result['coverage'][k] for k in ('key', 'capacity_label', 'N', 'lag_slots', 'horizon')}
            base.update(t0=prefix['t0'], common_prefix=prefix['common_executed_cumulative'],
                        L_star=prefix['L_star'], strict_information_benefit=prefix['strict_information_benefit'],
                        waiting_reduction_per_aircraft=prefix['waiting_reduction_per_aircraft'])
            prefix_rows.append(base)
            for level in prefix['frontier']:
                curve_rows.append(dict(base, lead=level['lead'],
                                       waiting_minutes_exact=level['expected_waiting_minutes_exact'],
                                       normalized_waiting_minutes=level['normalized_expected_waiting_minutes'],
                                       excess_minutes_exact=level['excess_over_full_information_exact'],
                                       oracle_equal=level['oracle_equal']))
    csv_write(OUT/(stage+'_prefix_results.csv'), prefix_rows)
    csv_write(OUT/(stage+'_frontier_curves.csv'), curve_rows)
    print(json.dumps({k: v for k, v in report.items() if k not in ('design', 'coverage')}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('freeze', 'smoke', 'panel'), required=True)
    args = parser.parse_args()
    if args.phase == 'freeze':
        result = freeze()
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
    else:
        run(args.phase)
