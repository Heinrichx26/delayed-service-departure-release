"""Held-out trace diagnostics for finite calendar supports and interval boxes.

This independent smoke keeps the legacy counterfactual reset and lag convention.
The actual cohort calendar is never used to construct the uncertainty boxes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import ceil
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/problem_driven_20261028'))
import info_design as acquisition

SOURCE = Path(__file__).resolve()
DATA = ROOT / 'data/substantive_20261028/holdout_boxes'
OUT = ROOT / 'results/revision_20261028/substantive_strengthening/holdout_boxes'
LEGACY = ROOT / 'data/trb_reassessment_20261028/fixed_support_capacity_panel/design.json'
RADII = [0, 1, 2, 4]
CAPACITIES = [('original', 1), ('ceil_N_over_4', 4), ('ceil_N_over_8', 8)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes():
    return dict(source_sha256=digest(SOURCE), acquisition_sha256=digest(Path(acquisition.__file__)),
                physical_sha256=digest(Path(acquisition.base.__file__)), legacy_manifest_sha256=digest(LEGACY),
                selective_source_sha256=digest(SOURCE.with_name('selective_interval.py')))


def prepare():
    old = read(LEGACY)
    selected = [r for r in old['rows'] if r['capacity_label'] == 'original' and r['key'] in old['smoke_keys']]
    assert len(selected) == 2
    rows = []
    for row in selected:
        source_path = ROOT / row['input_path']
        assert digest(source_path) == row['input_sha256']
        case = read(source_path)
        observed_path = ROOT / case['origin']['case_path']
        assert digest(observed_path) == case['origin']['case_sha256']
        observed = read(observed_path)
        lag, h = case['lag'], len(case['caps'])
        assert 0 <= lag <= h and len(observed['service']) >= h-lag
        actual = ([0] * lag + observed['service'])[:h]
        assert len(actual) == h
        dropped_tail = observed['service'][h-lag:]
        shift_record = dict(lag_slots=lag, horizon_slots=h,
                            unshifted_slots=len(observed['service']),
                            unshifted_service_total=sum(observed['service']),
                            shifted_service_total=sum(actual),
                            dropped_tail_slots=len(dropped_tail),
                            dropped_tail_service_total=sum(dropped_tail))
        assert shift_record['unshifted_service_total'] == sum(actual)+sum(dropped_tail)
        assert observed['day'] == case['day'] and observed['N'] == case['n']
        assert case['day'] not in case['donor_days']
        assert all(3 <= int(day[-2:]) <= 18 for day in case['donor_days'])
        value = dict(case, actual_service=actual, actual_unshifted_service=observed['service'],
                     actual_shift_record=shift_record,
                     actual_source_path=str(observed_path.relative_to(ROOT)),
                     actual_source_sha256=digest(observed_path),
                     provenance='Held-out recorded cohort calendar, shifted by the same frozen training-only lag as every donor.')
        path = DATA / 'smoke_cases' / (row['case_id'] + '.json')
        save(path, value)
        rows.append(dict(case_id=row['case_id'], key=row['key'], input_path=str(path.relative_to(ROOT)),
                         input_sha256=digest(path)))
    manifest = dict(date='2026-10-28', frozen_before_smoke_outcomes=True, **hashes(),
                    selection='Two unchanged legacy diagnostic smoke windows; descriptive, not a representative estimate.',
                    radii=RADII, capacities=[name for name, _ in CAPACITIES],
                    availability='Same L=lag+2 availability for every method.',
                    methods=['past_only', 'training_static_menu', 'all_available', 'selective_available'],
                    prefix='Empty reset, t0=0, shared by all policies.',
                    boxes='Union of sixteen coordinate boxes centred on the training calendars; +/-r aircraft per two-minute slot, clamped to [0,N]. The structural first-lag zeros remain zero. No total-calendar-mass constraint is added.',
                    completion_domain='A box is retained if its upper endpoint clears N under earliest release. This is equivalent to containing at least one completing calendar. All physically evaluated traces and incomplete references remain reported.',
                    controller='Receive free past first. Then intersect boxes with exact responses from the method-selected available coordinates. Compute each upper calendar bound and use their maximum. On empty compatible support, release remaining aircraft as early as allowed without further lookahead.',
                    confirmation_cost='All-available and training-menu policies skip coordinates logically constant over the current compatible box union. A skipped coordinate is still checked when its free actual observation arrives. No further confirmations are requested after all aircraft have been released. The static ex-ante fee and actual sent confirmations are separate metrics.',
                    selective='Stop when the current upper release and the minimum corner lower release after replacing all available unobserved coordinates by their lower endpoints agree. Otherwise query a coordinate using deadline sensitivity. Covered completing calendars require the same release as all-available confirmation. Actual confirmation counts are compared empirically. Outside-support action equality is not required.',
                    point_policy='Replay the unchanged optimal adaptive point-support policy at lag+2 and H-1. An unsupported response or free observation triggers the same earliest-remaining fallback.',
                    metrics='Full-calendar coverage, first incompatibility, causal fallback, completion deficits, absolute terminal unserved, raw waiting, and excess waiting only when completion is preserved and the actual reference clears N. Incomplete references retain their partial-reference waiting separately.',
                    rows=rows)
    save(DATA / 'smoke_design.json', manifest)
    print(dict(prepared=True, windows=len(rows), radii=RADII), flush=True)


def reference(case, caps, service):
    earliest = acquisition.base.earliest(case['n'], caps)
    completed, inventory, _ = acquisition.base.physical(earliest, service, case['n'], case['lag'])
    assert completed == cumulative_physical(earliest, service, case['lag'])
    return completed


def cumulative_physical(cumulative, service, lag):
    done, result = 0, []
    for t, capacity in enumerate(service):
        eligible = cumulative[t-lag] if t >= lag else 0
        done = min(eligible, done+capacity)
        result.append(done)
    return result


def deadline_bound(completed, caps, lag):
    h = len(caps)
    required = [completed[min(t+lag, h-1)] for t in range(h)]
    closure = required[:]
    for t in range(h-2, -1, -1):
        closure[t] = max(required[t], closure[t+1]-caps[t+1])
    for t in {0, h//2, h-1}:
        assert closure[t] == max(required[v]-sum(caps[t+1:v+1]) for v in range(t, h))
    return closure


def oracle_plan(case, caps):
    completed = reference(case, caps, case['actual_service'])
    bound = deadline_bound(completed, caps, case['lag'])
    cumulative, previous = [], 0
    for value in bound:
        previous = max(previous, value)
        cumulative.append(previous)
    assert all(0 <= u <= k for u, k in zip(acquisition.base.increments(cumulative), caps))
    assert cumulative_physical(cumulative, case['actual_service'], case['lag']) == completed
    return cumulative, completed


def evaluate(case, caps, cumulative, fallback, queries=0):
    actual = case['actual_service']
    completed, inventories, _ = acquisition.base.physical(cumulative, actual, case['n'], case['lag'])
    assert completed == cumulative_physical(cumulative, actual, case['lag'])
    assert all(0 <= u <= k for u, k in zip(acquisition.base.increments(cumulative), caps))
    oracle, earliest = oracle_plan(case, caps)
    assert all(got <= ref for got, ref in zip(completed, earliest))
    _, oracle_inventory, _ = acquisition.base.physical(oracle, actual, case['n'], case['lag'])
    deficits = [ref-got for ref, got in zip(earliest, completed)]
    preserving = max(deficits, default=0) == 0
    waiting = case['delta'] * sum(inventories)
    oracle_waiting = case['delta'] * sum(oracle_inventory)
    if preserving:
        assert all(u >= v for u, v in zip(cumulative, oracle))
        assert waiting >= oracle_waiting
    actual_reference_clears = earliest[-1] == case['n']
    return dict(actual_reference_clears=actual_reference_clears, completion_preserving=preserving,
                completion_preserving_and_reference_clears=preserving and actual_reference_clears,
                actual_reference_terminal_completed=earliest[-1],
                actual_reference_terminal_unserved=case['n']-earliest[-1],
                maximum_completion_deficit=max(deficits, default=0),
                completion_deficit_aircraft_minutes=case['delta']*sum(deficits),
                first_completion_deficit=next((t for t, d in enumerate(deficits) if d), None),
                absolute_terminal_unserved=case['n']-completed[-1],
                raw_waiting_aircraft_minutes=waiting,
                oracle_waiting_aircraft_minutes=oracle_waiting if actual_reference_clears else None,
                partial_reference_waiting_aircraft_minutes=oracle_waiting if not actual_reference_clears else None,
                excess_waiting_if_preserving=waiting-oracle_waiting if preserving and actual_reference_clears else None,
                waiting_per_aircraft=waiting/case['n'], query_count=queries,
                fallback_used=fallback is not None, fallback=fallback,
                cumulative_release=cumulative, completed=completed, oracle_cumulative_release=oracle)


def boxes(case, radius):
    n, lag = case['n'], case['lag']
    result = []
    for path in case['service']:
        lo = [0 if j < lag else max(0, value-radius) for j, value in enumerate(path)]
        hi = [0 if j < lag else min(n, value+radius) for j, value in enumerate(path)]
        result.append((lo, hi))
    return result


def box_controller(case, caps, radius, lead, method, menu=()):
    if method == 'selective_available':
        from selective_interval import selective_controller
        return selective_controller(case, caps, radius, lead)
    assert method in ('past_only', 'training_static_menu', 'all_available', 'unconditional_reports'), method
    actual, h = case['actual_service'], len(caps)
    initial = boxes(case, radius)
    covered = any(all(a <= v <= b for a, v, b in zip(lo, actual, hi)) for lo, hi in initial)
    active = [(lo[:], hi[:]) for lo, hi in initial if reference(case, caps, hi)[-1] == case['n']]
    eligible_initial = len(active)
    cumulative, previous, fallback, snapshots, queries, known = [], 0, None, [], [], {}

    def intersect(active, j, value):
        updated = []
        for lo, hi in active:
            if not lo[j] <= value <= hi[j]:
                continue
            lo, hi = lo[:], hi[:]
            lo[j] = hi[j] = value
            if reference(case, caps, hi)[-1] == case['n']:
                updated.append((lo, hi))
        return updated

    for t in range(h):
        if fallback is None:
            if t and t-1 not in known:
                known[t-1] = actual[t-1]
                active = intersect(active, t-1, actual[t-1])
            if not active:
                fallback = dict(decision_slot=t, coordinate=t-1 if t else None,
                                reason='No completing calendar remains after free observation')
        if fallback is None and (previous < case['n'] or method == 'unconditional_reports'):
            candidates = range(h) if method in ('all_available', 'unconditional_reports') else menu if method == 'training_static_menu' else ()
            for j in candidates:
                if max(0, j-lead) > t or j in known:
                    continue
                assert j >= t
                if method != 'unconditional_reports' and max(hi[j] for lo, hi in active) == min(lo[j] for lo, hi in active):
                    continue
                known[j] = actual[j]
                queries.append(dict(t=t, j=j, value=actual[j]))
                active = intersect(active, j, actual[j])
                if not active:
                    fallback = dict(decision_slot=t, coordinate=j,
                                    reason='No completing calendar remains after confirmation')
                    break
        if fallback is not None:
            previous = min(case['n'], previous+caps[t])
        else:
            bound = max(deadline_bound(reference(case, caps, hi), caps, case['lag'])[t]
                        for _, hi in active)
            action = max(previous, bound)
            assert previous <= action <= min(case['n'], previous+caps[t])
            previous = action
        cumulative.append(previous)
        snapshots.append(dict(t=t, compatible_boxes=len(active), known_coordinates=sorted(known),
                              cumulative_release=previous))
    result = evaluate(case, caps, cumulative, fallback, len(queries))
    if covered and result['actual_reference_clears']:
        assert not result['fallback_used'] and result['completion_preserving']
    result.update(radius=radius, lead=lead, method=method, full_calendar_covered=covered,
                  initial_boxes=len(initial), initial_completing_boxes=eligible_initial,
                  fixed_menu_fee=len(menu) if method == 'training_static_menu' else None,
                  selected_menu=list(menu) if method == 'training_static_menu' else None,
                  queries=queries, snapshots=snapshots)
    return result


def point_policy(case, lead):
    problem = acquisition.prepare(case)
    assert problem is not None
    engine = acquisition.AdaptiveDesign(problem, lead)
    optimum = engine.solve()
    assert optimum['feasible']
    mask, h = engine.full, problem['horizon']
    previous, cumulative, queries, fallback = 0, [], [], None
    actual = case['actual_service']
    for t in range(h):
        if fallback is None:
            while engine.choice[(t, mask)][0] == 'query':
                j = engine.choice[(t, mask)][1]
                assert max(0, j-lead) <= t
                queries.append(dict(t=t, j=j, value=actual[j]))
                mask = sum(1 << q for q in range(len(case['service']))
                           if mask & (1 << q) and case['service'][q][j] == actual[j])
                if not mask:
                    fallback = dict(decision_slot=t, coordinate=j, response=actual[j], reason='Unsupported paid response')
                    break
        if fallback is not None:
            previous = min(case['n'], previous+case['caps'][t])
        else:
            assert engine.choice[(t, mask)][0] == 'advance'
            members = [q for q in range(len(case['service'])) if mask & (1 << q)]
            targets = {problem['target'][q][t] for q in members}
            assert len(targets) == 1
            previous = next(iter(targets))
        cumulative.append(previous)
        if fallback is None:
            mask = sum(1 << q for q in range(len(case['service']))
                       if mask & (1 << q) and case['service'][q][t] == actual[t])
            if not mask and t+1 < h:
                fallback = dict(decision_slot=t+1, coordinate=t, response=actual[t], reason='Unsupported free observation')
    result = evaluate(case, case['caps'], cumulative, fallback, len(queries))
    result.update(lead=lead, queries=queries,
                  full_calendar_covered=any(actual == path for path in case['service']),
                  in_support_expected_queries=optimum['expected_queries_exact'])
    return result


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_smoke():
    case = dict(n=1, lag=1, delta=1, caps=[1]*5,
                service=[[0,0,0,0,1]], probabilities=['1'],
                actual_service=[0,1,0,0,0])
    late = box_controller(case, case['caps'], 0, 3, 'past_only')
    early = box_controller(case, case['caps'], 0, 3, 'unconditional_reports')
    protected = box_controller(case, case['caps'], 1, 3, 'past_only')
    assert late['fallback_used'] and late['maximum_completion_deficit'] == 1
    assert late['first_completion_deficit'] < late['fallback']['decision_slot']
    assert early['fallback_used'] and early['completion_preserving']
    assert protected['full_calendar_covered'] and protected['completion_preserving']
    assert not protected['fallback_used']
    outside = dict(n=1, lag=1, delta=1, caps=[1]*6,
                   service=[[0,0,0,0,1,0], [0,0,0,1,0,0]],
                   probabilities=['1/2', '1/2'], actual_service=[0,1,0,2,0,0])
    full = box_controller(outside, outside['caps'], 0, 3, 'all_available')
    selective = box_controller(outside, outside['caps'], 0, 3, 'selective_available')
    assert full['fallback_used'] and full['completion_preserving']
    assert selective['fallback_used'] and selective['maximum_completion_deficit'] == 1
    assert full['fallback']['decision_slot'] < selective['fallback']['decision_slot']
    assert full['cumulative_release'] != selective['cumulative_release']
    negative_controls = dict(
        unconditional_reports=dict(case=case, past_only=late, unconditional_reports=early),
        selective_outside_support=dict(case=outside, all_available=full, selective_available=selective))
    # Zero-width boxes and a covering training menu reproduce both exact targets.
    training = dict(n=2, lag=1, delta=1, caps=[1]*6,
                    service=[[0,0,2,0,0,0], [0,0,0,0,0,2]],
                    probabilities=['1/2','1/2'])
    problem = acquisition.prepare(training)
    menu = acquisition.menu_cover(problem, 3)
    assert menu['feasible']
    for s, path in enumerate(training['service']):
        result = box_controller(dict(training, actual_service=path), training['caps'],
                                0, 3, 'training_static_menu', menu['selected_slots'])
        assert result['cumulative_release'] == problem['target'][s]
        assert result['completion_preserving']
        actual_case = dict(training, actual_service=path)
        full = box_controller(actual_case, training['caps'], 0, 3, 'all_available')
        selective = box_controller(actual_case, training['caps'], 0, 3, 'selective_available')
        assert full['cumulative_release'] == selective['cumulative_release']
    incomplete = dict(n=2, lag=1, delta=1, caps=[2]*5,
                      service=[[0,0,1,0,0]], actual_service=[0,0,1,0,0])
    partial = evaluate(incomplete, incomplete['caps'], [2]*5, None)
    assert partial['completion_preserving'] and not partial['actual_reference_clears']
    assert partial['oracle_waiting_aircraft_minutes'] is None
    assert partial['excess_waiting_if_preserving'] is None
    assert partial['partial_reference_waiting_aircraft_minutes'] is not None
    return dict(passed=True, physical_runs=12, irreversible_late_fallback_checked=True,
                advance_detection_checked=True, interval_coverage_checked=True,
                zero_width_training_target_checks=2,
                incomplete_reference_waiting_suppressed=True,
                selective_covered_equal_release_checks=2,
                selective_example_query_counts_recorded=True,
                selective_outside_support_completion_loss_retained=True,
                unconditional_reports_original_negative_control_retained=True,
                negative_controls=negative_controls)


def smoke():
    begin = perf_counter()
    manifest = read(DATA / 'smoke_design.json')
    for key, value in hashes().items():
        assert manifest[key] == value
    synthetic = synthetic_smoke()
    details, flat, point, menu_exclusions = [], [], [], []
    for row in manifest['rows']:
        path = ROOT / row['input_path']
        assert digest(path) == row['input_sha256']
        case = read(path)
        for label, divisor in CAPACITIES:
            caps = [ceil(case['n']/divisor)] * len(case['caps'])
            lead = case['lag']+2
            training = acquisition.prepare(dict(case, caps=caps))
            menu = acquisition.menu_cover(training, lead) if training is not None else None
            static_available = menu is not None and menu['feasible']
            if not static_available:
                menu_exclusions.append(dict(case_id=row['case_id'], capacity_label=label,
                    reason='Incomplete training reference' if training is None else 'Insufficient availability lead'))
            for radius in manifest['radii']:
                runs = {}
                for method in manifest['methods']:
                    if method == 'training_static_menu' and not static_available:
                        continue
                    slots = menu['selected_slots'] if method == 'training_static_menu' else ()
                    result = box_controller(case, caps, radius, lead, method, slots)
                    runs[method] = result
                    if method == 'selective_available':
                        full = runs['all_available']
                        required = result['full_calendar_covered'] and result['actual_reference_clears']
                        equal = result['cumulative_release'] == full['cumulative_release']
                        no_more_queries = result['query_count'] <= full['query_count']
                        if required:
                            assert equal
                        result.update(matches_all_available_release=equal,
                                      no_more_queries_than_all_available=no_more_queries,
                                      covered_complete_equality_required=required)
                    details.append(dict(case_id=row['case_id'], capacity_label=label, **result))
                    flat.append(dict(case_id=row['case_id'], key=case['key'], capacity_label=label,
                                     N=case['n'], cap=caps[0], radius=radius, lead=lead,
                                     **{k: v for k, v in result.items() if k not in ('snapshots', 'queries', 'selected_menu', 'cumulative_release', 'completed', 'oracle_cumulative_release', 'fallback', 'radius', 'lead')}))
        for lead in (case['lag']+2, len(case['caps'])-1):
            result = point_policy(case, lead)
            point.append(dict(case_id=row['case_id'], key=case['key'], **result))
        print(row['case_id'], 'finished', flush=True)
    covered = [r for r in flat if r['full_calendar_covered'] and r['actual_reference_clears']]
    assert all(r['completion_preserving'] and not r['fallback_used'] for r in covered)
    summary = []
    for radius in manifest['radii']:
        for method in manifest['methods']:
            group = [r for r in flat if r['radius'] == radius and r['capacity_label'] == 'original' and r['method'] == method]
            summary.append(dict(radius=radius, method=method, windows=len(group), covered=sum(r['full_calendar_covered'] for r in group),
                            fallback=sum(r['fallback_used'] for r in group),
                            preserving=sum(r['completion_preserving'] for r in group),
                            maximum_completion_deficit=max(r['maximum_completion_deficit'] for r in group),
                            mean_queries=sum(r['query_count'] for r in group)/len(group),
                            mean_ex_ante_static_fee=sum(r['fixed_menu_fee'] for r in group)/len(group) if method == 'training_static_menu' else None,
                            mean_raw_waiting_per_aircraft=sum(r['waiting_per_aircraft'] for r in group)/len(group)))
    report = dict(passed=True, date='2026-10-28', **hashes(),
                  design_sha256=digest(DATA/'smoke_design.json'), windows=len(manifest['rows']),
                  box_policy_runs=len(flat), point_policy_runs=len(point),
                  synthetic=synthetic, training_static_menu_exclusions=menu_exclusions,
                  covered_eligible_runs=len(covered), covered_eligible_all_preserving=True,
                  point_support_violations=sum(not r['full_calendar_covered'] for r in point),
                  point_fallback_runs=sum(r['fallback_used'] for r in point),
                  point_completion_loss_runs=sum(not r['completion_preserving'] for r in point),
                  elapsed_seconds=perf_counter()-begin, summary=summary, box_rows=flat,
                  point_rows=point, box_details=details)
    save(OUT/'smoke.json', report)
    csv_write(OUT/'box_smoke.csv', flat)
    csv_write(OUT/'summary_smoke.csv', summary)
    csv_write(OUT/'point_smoke.csv', [{k: v for k, v in r.items() if k not in ('queries', 'fallback', 'cumulative_release', 'completed', 'oracle_cumulative_release')} for r in point])
    public = {k: v for k, v in report.items() if k not in ('box_rows', 'point_rows', 'box_details')}
    public['synthetic'] = {k: v for k, v in synthetic.items() if k != 'negative_controls'}
    print(public, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'smoke'], required=True)
    args = parser.parse_args()
    globals()[args.phase]()


if __name__ == '__main__':
    main()
