"""Frozen held-out calendar panel, extending the independent smoke protocol."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path
from statistics import mean
from time import perf_counter

import holdout_box_smoke as diagnostic

ROOT = diagnostic.ROOT
DATA = diagnostic.DATA
OUT = diagnostic.OUT
SOURCE = Path(__file__).resolve()
METHODS = ['past_only', 'training_static_menu', 'all_available', 'selective_available']
WORKERS = 2


def source_hashes():
    return dict(diagnostic.hashes(), panel_source_sha256=diagnostic.digest(SOURCE))


def build_inputs():
    legacy = diagnostic.read(diagnostic.LEGACY)
    rows = [row for row in legacy['rows'] if row['capacity_label'] == 'original']
    assert len(rows) == 94
    prepared = []
    for row in rows:
        source = ROOT / row['input_path']
        assert diagnostic.digest(source) == row['input_sha256']
        case = diagnostic.read(source)
        observed_path = ROOT / case['origin']['case_path']
        assert diagnostic.digest(observed_path) == case['origin']['case_sha256']
        observed = diagnostic.read(observed_path)
        lag, horizon = case['lag'], len(case['caps'])
        assert 0 <= lag <= horizon and len(observed['service']) >= horizon-lag
        actual = ([0] * lag + observed['service'])[:horizon]
        dropped = observed['service'][horizon-lag:]
        assert len(actual) == horizon
        assert observed['N'] == case['n'] and observed['day'] == case['day']
        assert case['day'] not in case['donor_days']
        assert all(3 <= int(day[-2:]) <= 18 for day in case['donor_days'])
        shift = dict(lag_slots=lag, horizon_slots=horizon,
                     unshifted_slots=len(observed['service']),
                     unshifted_service_total=sum(observed['service']),
                     shifted_service_total=sum(actual),
                     dropped_tail_slots=len(dropped),
                     dropped_tail_service_total=sum(dropped))
        assert sum(observed['service']) == sum(actual)+sum(dropped)
        value = dict(case, actual_service=actual,
                     actual_unshifted_service=observed['service'],
                     actual_shift_record=shift,
                     actual_source_path=str(observed_path.relative_to(ROOT)),
                     actual_source_sha256=diagnostic.digest(observed_path),
                     provenance='Held-out recorded cohort calendar, shifted by the same frozen training-only lag as every donor.')
        path = DATA / 'panel_cases' / (row['case_id']+'.json')
        diagnostic.save(path, value)
        prepared.append(dict(case_id=row['case_id'], key=row['key'],
                             airport=case['airport'], day=case['day'], hour=case['hour'],
                             cluster=case['airport']+'_'+case['day'],
                             N=case['n'], lag=lag, horizon=horizon,
                             actual_shift_record=shift,
                             source_input_path=str(source.relative_to(ROOT)),
                             source_input_sha256=diagnostic.digest(source),
                             input_path=str(path.relative_to(ROOT)),
                             input_sha256=diagnostic.digest(path)))
    assert len({row['cluster'] for row in prepared}) == 32
    return prepared


def prepare():
    """Freeze only after the four-policy smoke matches the current sources."""
    smoke = diagnostic.read(OUT / 'smoke.json')
    assert smoke['passed']
    for key, value in diagnostic.hashes().items():
        assert smoke[key] == value, key
    smoke_design = diagnostic.read(DATA / 'smoke_design.json')
    assert smoke['design_sha256'] == diagnostic.digest(DATA / 'smoke_design.json')
    assert smoke_design['methods'] == METHODS
    assert smoke['synthetic']['passed']
    rows = build_inputs()
    manifest = dict(date='2026-10-28', frozen_before_full_outcomes=True,
                    **source_hashes(), smoke_sha256=diagnostic.digest(OUT/'smoke.json'),
                    smoke_design_sha256=diagnostic.digest(DATA/'smoke_design.json'),
                    windows=len(rows), airport_day_clusters=len({r['cluster'] for r in rows}),
                    workers=WORKERS, radii=diagnostic.RADII,
                    capacities=[label for label, _ in diagnostic.CAPACITIES],
                    methods=METHODS,
                    selection='All ninety-four original-capacity windows in the frozen legacy manifest, with no outcome-based exclusions.',
                    donor_rule='Sixteen same-year January 3--18 donor calendars from the unchanged source inputs.',
                    shift='The actual held-out calendar receives the same frozen training-only lag and horizon truncation as the donor calendars. Every dropped tail count is recorded.',
                    availability='The same L=lag+2 availability for every box method; original point-policy diagnostics also use H-1.',
                    boxes=smoke_design['boxes'],
                    completion_domain=smoke_design['completion_domain'],
                    prefix=smoke_design['prefix'],
                    controller=smoke_design['controller'],
                    selective_scope='Action equality with all-available confirmation is checked on covered completing actual calendars. Outside support, earlier detection can change fallback actions and outcomes.',
                    static_menu_cost='The ex-ante fixed menu fee and actual confirmed coordinates before fallback are separate fields.',
                    failure_reporting='All incomplete actual references, static-menu unavailability, point-policy unavailability, fallback traces, and completion losses remain in the outputs.',
                    aggregation='Window-equal descriptive means. Airport-day cluster summaries are also saved. No confidence or population-coverage claim is assigned to the radius grid.',
                    rows=rows)
    diagnostic.save(DATA/'panel_design.json', manifest)
    print(dict(prepared=True, windows=len(rows), clusters=32,
               source_hashes=source_hashes(),
               manifest_sha256=diagnostic.digest(DATA/'panel_design.json')), flush=True)


def draft():
    """Stage the declared input population before selective-policy integration."""
    rows = build_inputs()
    value = dict(date='2026-10-28', status='Input freeze preparation; not the final runnable design',
                 windows=len(rows), airport_day_clusters=len({r['cluster'] for r in rows}),
                 methods_pending_four_policy_smoke=METHODS, radii=diagnostic.RADII,
                 capacities=[label for label, _ in diagnostic.CAPACITIES], workers=WORKERS,
                 source_hashes_at_preparation=source_hashes(),
                 nonzero_truncation_windows=sum(r['actual_shift_record']['dropped_tail_service_total'] > 0 for r in rows),
                 rows=rows)
    diagnostic.save(DATA/'panel_preparation.json', value)
    print({key: value for key, value in value.items() if key != 'rows'}, flush=True)


def flat_result(row, label, caps, radius, lead, result):
    skip = {'snapshots', 'queries', 'selected_menu', 'cumulative_release', 'completed',
            'oracle_cumulative_release', 'fallback', 'radius', 'lead'}
    return dict(case_id=row['case_id'], key=row['key'], airport=row['airport'],
                day=row['day'], hour=row['hour'], cluster=row['cluster'],
                capacity_label=label, N=row['N'], cap=caps[0], radius=radius, lead=lead,
                **{key: value for key, value in result.items() if key not in skip})


def run_window(row):
    path = ROOT/row['input_path']
    assert diagnostic.digest(path) == row['input_sha256']
    case = diagnostic.read(path)
    flat, details, exclusions, point = [], [], [], []
    for label, divisor in diagnostic.CAPACITIES:
        caps = [ceil(case['n']/divisor)] * len(case['caps'])
        lead = case['lag']+2
        training = diagnostic.acquisition.prepare(dict(case, caps=caps))
        menu = diagnostic.acquisition.menu_cover(training, lead) if training is not None else None
        static_available = menu is not None and menu['feasible']
        if not static_available:
            exclusions.append(dict(case_id=row['case_id'], cluster=row['cluster'],
                                   capacity_label=label, method='training_static_menu',
                                   reason='Incomplete training reference' if training is None else 'Insufficient availability lead'))
        for radius in diagnostic.RADII:
            runs = {}
            for method in METHODS:
                if method == 'training_static_menu' and not static_available:
                    continue
                slots = menu['selected_slots'] if method == 'training_static_menu' else ()
                result = diagnostic.box_controller(case, caps, radius, lead, method, slots)
                runs[method] = result
            complete_and_covered = runs['all_available']['actual_reference_clears'] and runs['all_available']['full_calendar_covered']
            equal = runs['all_available']['cumulative_release'] == runs['selective_available']['cumulative_release']
            no_more_queries = runs['selective_available']['query_count'] <= runs['all_available']['query_count']
            if complete_and_covered:
                assert equal, (row['case_id'], label, radius)
            runs['selective_available']['matches_all_available_release'] = equal
            runs['selective_available']['no_more_queries_than_all_available'] = no_more_queries
            runs['selective_available']['covered_complete_equality_required'] = complete_and_covered
            for method, result in runs.items():
                flat.append(flat_result(row, label, caps, radius, lead, result))
                details.append(dict(capacity_label=label, **result))
    problem = diagnostic.acquisition.prepare(case)
    for lead in (case['lag']+2, len(case['caps'])-1):
        available = problem is not None
        if available:
            engine = diagnostic.acquisition.AdaptiveDesign(problem, lead)
            available = engine.solve()['feasible']
        if available:
            result = diagnostic.point_policy(case, lead)
            point.append(dict(case_id=row['case_id'], key=row['key'], cluster=row['cluster'],
                              method='adaptive_point_support', available=True, **result))
        else:
            point.append(dict(case_id=row['case_id'], key=row['key'], cluster=row['cluster'],
                              method='adaptive_point_support', available=False, lead=lead,
                              reason='Incomplete training reference' if problem is None else 'Insufficient availability lead'))
    detail = dict(case_id=row['case_id'], key=row['key'], cluster=row['cluster'],
                  actual_shift_record=case['actual_shift_record'], box_details=details,
                  point_rows=point, exclusions=exclusions)
    output_path = OUT/'panel_windows'/(row['case_id']+'.json')
    diagnostic.save(output_path, detail)
    return dict(case_id=row['case_id'], flat=flat, point=point, exclusions=exclusions,
                detail_path=str(output_path.relative_to(ROOT)), detail_sha256=diagnostic.digest(output_path))


def summarize(group, identity):
    complete = [row for row in group if row['actual_reference_clears']]
    covered = [row for row in complete if row['full_calendar_covered']]
    preserving = [row for row in complete if row['completion_preserving']]
    excess = [row['excess_waiting_if_preserving']/row['N'] for row in preserving]
    fixed_fees = [row['fixed_menu_fee'] for row in group if row.get('fixed_menu_fee') is not None]
    return dict(identity, windows=len(group), actual_reference_complete=len(complete),
                actual_reference_incomplete=len(group)-len(complete),
                covered=len([row for row in group if row['full_calendar_covered']]),
                covered_complete=len(covered),
                covered_complete_preserving=sum(row['completion_preserving'] for row in covered),
                completion_preserving_complete=len(preserving),
                completion_loss_complete=len(complete)-len(preserving),
                completion_loss_all=sum(not row['completion_preserving'] for row in group),
                fallback_runs=sum(row['fallback_used'] for row in group),
                maximum_completion_deficit=max(row['maximum_completion_deficit'] for row in group),
                mean_completion_deficit_aircraft_minutes=mean(row['completion_deficit_aircraft_minutes'] for row in group),
                mean_absolute_terminal_unserved=mean(row['absolute_terminal_unserved'] for row in group),
                mean_waiting_minutes_per_aircraft=mean(row['waiting_per_aircraft'] for row in group),
                mean_excess_waiting_minutes_per_aircraft_preserving_complete=mean(excess) if excess else None,
                mean_actual_confirmed_coordinates=mean(row['query_count'] for row in group),
                mean_ex_ante_static_fee=mean(fixed_fees) if fixed_fees else None)


def run():
    begin = perf_counter()
    manifest = diagnostic.read(DATA/'panel_design.json')
    for key, value in source_hashes().items():
        assert manifest[key] == value, key
    assert manifest['methods'] == METHODS
    assert manifest['smoke_sha256'] == diagnostic.digest(OUT/'smoke.json')
    assert manifest['smoke_design_sha256'] == diagnostic.digest(DATA/'smoke_design.json')
    flat, point, exclusions, outputs = [], [], [], []
    with ProcessPoolExecutor(max_workers=manifest['workers']) as pool:
        for index, result in enumerate(pool.map(run_window, manifest['rows']), 1):
            flat.extend(result['flat'])
            point.extend(result['point'])
            exclusions.extend(result['exclusions'])
            outputs.append({key: value for key, value in result.items() if key not in ('flat', 'point', 'exclusions')})
            print(str(index)+'/'+str(len(manifest['rows'])), result['case_id'], flush=True)
    summary, cluster_summary = [], []
    for label, _ in diagnostic.CAPACITIES:
        for radius in diagnostic.RADII:
            for method in METHODS:
                group = [row for row in flat if row['capacity_label'] == label and row['radius'] == radius and row['method'] == method]
                if not group:
                    continue
                identity = dict(capacity_label=label, radius=radius, method=method)
                summary.append(summarize(group, identity))
                for cluster in sorted({row['cluster'] for row in group}):
                    cluster_summary.append(summarize([row for row in group if row['cluster'] == cluster], dict(identity, cluster=cluster)))
    covered = [row for row in flat if row['full_calendar_covered'] and row['actual_reference_clears']]
    assert all(row['completion_preserving'] and not row['fallback_used'] for row in covered)
    selected = [row for row in flat if row['method'] == 'selective_available']
    assert all(row['matches_all_available_release'] for row in selected if row['covered_complete_equality_required'])
    report = dict(passed=True, date='2026-10-28', **source_hashes(),
                  design_sha256=diagnostic.digest(DATA/'panel_design.json'),
                  windows=manifest['windows'], airport_day_clusters=manifest['airport_day_clusters'],
                  box_policy_runs=len(flat), point_policy_rows=len(point),
                  point_policy_available=sum(row['available'] for row in point),
                  covered_complete_runs=len(covered), covered_complete_all_preserving=True,
                  selective_covered_complete_equal=True,
                  selective_out_of_support_release_divergences=sum(not row['matches_all_available_release'] for row in selected if not row['covered_complete_equality_required']),
                  elapsed_seconds=perf_counter()-begin, training_static_menu_exclusions=exclusions,
                  nonzero_truncation_windows=sum(row['actual_shift_record']['dropped_tail_service_total'] > 0 for row in manifest['rows']),
                  summary=summary, window_outputs=outputs)
    diagnostic.csv_write(OUT/'box_panel.csv', flat)
    diagnostic.csv_write(OUT/'summary_panel.csv', summary)
    diagnostic.csv_write(OUT/'cluster_summary_panel.csv', cluster_summary)
    diagnostic.csv_write(OUT/'point_panel.csv', [{key: value for key, value in row.items() if key not in ('queries', 'fallback', 'cumulative_release', 'completed', 'oracle_cumulative_release')} for row in point])
    diagnostic.csv_write(OUT/'exclusions_panel.csv', exclusions)
    diagnostic.save(OUT/'panel.json', report)
    print({key: value for key, value in report.items() if key not in ('summary', 'window_outputs', 'training_static_menu_exclusions')}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['draft', 'prepare', 'run'], required=True)
    args = parser.parse_args()
    globals()[args.phase]()


if __name__ == '__main__':
    main()
