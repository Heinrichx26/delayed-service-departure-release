"""Held-out replay with deterministic capacity-range responses and free inference.

Plotting is separate. Width one reproduces exact responses. Natural past
observations always refine a previous range report to the realized count.
"""
from pathlib import Path
from statistics import mean
from concurrent.futures import ProcessPoolExecutor
import argparse
import gzip
import json

import holdout_box_smoke as base
from selective_interval import choose_confirmation

ROOT = base.ROOT
DATA = ROOT / 'data/substantive_20261028'
OUT = ROOT / 'results/revision_20261028/substantive_strengthening/interval_precision_v2'
PARENT = DATA / 'holdout_boxes/panel_design.json'
RADII, WIDTHS = [1, 2, 4], [1, 2, 4]
METHODS = ['past_only', 'training_static_menu', 'all_available', 'selective_available']


def response(value, width, n):
    lower = width * (value // width)
    return lower, min(n, lower + width - 1)


def inferable_reports(active, width, n, available, known):
    inferred = {}
    for j in available:
        if j in known:
            continue
        minimum = min(lo[j] for lo, hi in active)
        maximum = max(hi[j] for lo, hi in active)
        lower_response = response(minimum, width, n)
        if lower_response == response(maximum, width, n):
            inferred[j] = lower_response
    return inferred


def controller(case, radius, width, method, menu=()):
    caps, n, lag = case['caps'], case['n'], case['lag']
    h, lead, actual = len(caps), lag + 2, case['actual_service']
    initial = base.boxes(case, radius)
    covered = any(all(a <= v <= b for a, v, b in zip(lo, actual, hi)) for lo, hi in initial)
    active = [(lo[:], hi[:]) for lo, hi in initial if base.reference(case, caps, hi)[-1] == n]
    initial_count = len(active)
    reports, queries, snapshots, cumulative, inferred = {}, [], [], [], []
    previous, fallback = 0, None

    def intersect(boxes, j, lower, upper):
        result = []
        for lo, hi in boxes:
            left, right = max(lo[j], lower), min(hi[j], upper)
            if left <= right:
                lo, hi = lo[:], hi[:]
                lo[j], hi[j] = left, right
                if base.reference(case, caps, hi)[-1] == n:
                    result.append((lo, hi))
        return result

    def obtain(j, t):
        nonlocal active, fallback
        lo, hi = response(actual[j], width, n)
        assert lo <= actual[j] <= hi and j not in reports
        reports[j] = (lo, hi)
        queries.append({'t': t, 'j': j, 'lower': lo, 'upper': hi})
        active = intersect(active, j, lo, hi)
        if not active:
            fallback = {'decision_slot': t, 'coordinate': j, 'reason': 'Empty completing support after range report'}

    def infer_free(t):
        available = range(t, min(h, t + lead + 1))
        facts = inferable_reports(active, width, n, available, reports)
        for j, bounds in facts.items():
            reports[j] = bounds
            inferred.append({'t': t, 'j': j, 'lower': bounds[0], 'upper': bounds[1]})

    for t in range(h):
        certificate = None
        if fallback is None:
            if t:
                # A past actual count refines even an already received coarse report.
                active = intersect(active, t - 1, actual[t - 1], actual[t - 1])
            if not active:
                fallback = {'decision_slot': t, 'coordinate': t - 1 if t else None,
                            'reason': 'Empty completing support after exact past observation'}
        if fallback is None and previous < n:
            if method == 'selective_available':
                while True:
                    infer_free(t)
                    j, certificate = choose_confirmation(active, n, caps, lag, t, previous, lead, reports)
                    if j is None:
                        break
                    obtain(j, t)
                    if fallback is not None:
                        break
            else:
                candidates = range(h) if method == 'all_available' else menu if method == 'training_static_menu' else ()
                for j in candidates:
                    infer_free(t)
                    if j < t or j > t + lead or j in reports:
                        continue
                    obtain(j, t)
                    if fallback is not None:
                        break
        if fallback is not None:
            previous = min(n, previous + caps[t])
        else:
            target = (certificate['upper_release'] if certificate is not None else
                      max(previous, max(base.deadline_bound(base.reference(case, caps, hi), caps, lag)[t]
                                        for lo, hi in active)))
            assert previous <= target <= min(n, previous + caps[t])
            previous = target
        cumulative.append(previous)
        snapshots.append({'t': t, 'boxes': len(active), 'cumulative_release': previous,
                          'known_response_coordinates': sorted(reports), 'certificate': certificate})
    result = base.evaluate(case, caps, cumulative, fallback, len(queries))
    if covered and result['actual_reference_clears']:
        assert result['completion_preserving'] and not result['fallback_used']
    result.update(radius=radius, response_width=width, method=method, lead=lead,
                  full_calendar_covered=covered, initial_completing_boxes=initial_count,
                  fixed_menu_fee=len(menu) if method == 'training_static_menu' else None,
                  queries=queries, inferred_responses=inferred, snapshots=snapshots)
    return result


def run_window(row):
    path = ROOT / row['input_path']
    assert base.digest(path) == row['input_sha256']
    case = base.read(path)
    problem = base.acquisition.prepare(case)
    menu = base.acquisition.menu_cover(problem, case['lag'] + 2)['selected_slots']
    details = []
    for radius in RADII:
        for width in WIDTHS:
            group = {}
            for method in METHODS:
                result = controller(case, radius, width, method, menu)
                result.update(case_id=row['case_id'], key=case['key'], cluster=row.get('cluster', '_'.join(case['key'].split('_')[:2])))
                details.append(result)
                group[method] = result
            selected, full = group['selective_available'], group['all_available']
            if selected['full_calendar_covered'] and selected['actual_reference_clears']:
                assert selected['cumulative_release'] == full['cumulative_release']
                assert selected['completed'] == full['completed']
    return details


def summarize(details):
    rows = []
    for radius in RADII:
        for width in WIDTHS:
            for method in METHODS:
                group = [r for r in details if r['radius'] == radius and r['response_width'] == width and r['method'] == method]
                eligible = [r for r in group if r['actual_reference_clears'] and r['completion_preserving']]
                rows.append({'radius': radius, 'response_width': width, 'method': method,
                             'windows': len(group), 'covered': sum(r['full_calendar_covered'] for r in group),
                             'preserving': sum(r['completion_preserving'] for r in group),
                             'fallback': sum(r['fallback_used'] for r in group),
                             'mean_queries': mean(r['query_count'] for r in group),
                             'mean_waiting_minutes_per_aircraft': mean(r['waiting_per_aircraft'] for r in group),
                             'mean_excess_minutes_per_aircraft_if_preserving': mean(r['excess_waiting_if_preserving'] / int(r['cumulative_release'][-1]) for r in eligible) if eligible else None})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'panel'], required=True)
    args = parser.parse_args()
    source = Path(__file__)
    parent = base.read(PARENT)
    source_hashes = {'source_sha256': base.digest(source), **base.hashes(),
                     'selective_source_sha256': base.digest(source.with_name('selective_interval.py'))}
    # base.hashes() has a source entry of its own; retain both explicitly.
    source_hashes['base_source_sha256'] = source_hashes.pop('source_sha256')
    source_hashes['source_sha256'] = base.digest(source)
    if args.phase == 'smoke':
        assert inferable_reports([([0], [1])], 2, 3, [0], {}) == {0: (0, 1)}
        assert inferable_reports([([1], [2])], 2, 3, [0], {}) == {}
        assert inferable_reports([([0], [1]), ([2], [2])], 2, 3, [0], {}) == {}
        assert inferable_reports([([0], [1])], 2, 3, [0], {0: (0, 1)}) == {}
        for n in range(1, 9):
            for width in WIDTHS:
                assert all(response(v, width, n)[0] <= v <= response(v, width, n)[1] for v in range(n + 1))
        smoke_keys = {r['case_id'] for r in base.read(base.DATA / 'smoke_design.json')['rows']}
        selected = [r for r in parent['rows'] if r['case_id'] in smoke_keys]
        assert len(selected) == 2
        details = [result for row in selected for result in run_window(row)]
        for result in details:
            if result['response_width'] != 1:
                continue
            row = next(r for r in selected if r['case_id'] == result['case_id'])
            case = base.read(ROOT / row['input_path'])
            problem = base.acquisition.prepare(case)
            menu = base.acquisition.menu_cover(problem, case['lag'] + 2)['selected_slots']
            if result['method'] == 'selective_available':
                from selective_interval import selective_controller
                old = selective_controller(case, case['caps'], result['radius'], case['lag'] + 2)
            else:
                old = base.box_controller(case, case['caps'], result['radius'], case['lag'] + 2, result['method'], menu)
            assert result['cumulative_release'] == old['cumulative_release']
            assert result['query_count'] == old['query_count']
        report = {'passed': True, 'phase': 'smoke', **source_hashes, 'runs': len(details), 'summary': summarize(details)}
        base.save(OUT / 'smoke.json', report)
    else:
        smoke = base.read(OUT / 'smoke.json')
        assert smoke['passed'] and all(smoke[k] == v for k, v in source_hashes.items())
        design = {'date': '2026-10-28', 'frozen_before_precision_panel': True, **source_hashes,
                  'parent_design_sha256': base.digest(PARENT), 'smoke_sha256': base.digest(OUT / 'smoke.json'),
                  'radii': RADII, 'response_widths': WIDTHS, 'methods': METHODS, 'workers': 2,
                  'responses': 'Fixed bins [w floor(c/w), min(N,w floor(c/w)+w-1)] containing the realized service count.',
                  'past': 'Always exact, including coordinates with prior coarse reports.',
                  'availability': 'L=lag+2 for all methods.', 'rows': parent['rows']}
        design['free_inference'] = 'An available response constant over all current box coordinates is deduced without issuing a request; closure is refreshed after every paid response.'
        design['supersedes_precision_design_sha256'] = base.digest(DATA / 'interval_precision_design.json')
        base.save(DATA / 'interval_precision_v2_design.json', design)
        OUT.mkdir(parents=True, exist_ok=True)
        details = []
        with ProcessPoolExecutor(max_workers=2) as pool:
            for row, results in zip(parent['rows'], pool.map(run_window, parent['rows'])):
                path = OUT / 'runs' / (row['case_id'] + '.json.gz')
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(gzip.compress(json.dumps(results, separators=(',', ':')).encode(), mtime=0))
                details.extend(results)
        summary = summarize(details)
        base.csv_write(OUT / 'summary.csv', summary)
        flat = [{k: v for k, v in r.items() if k not in ['queries', 'snapshots', 'cumulative_release', 'completed', 'oracle_cumulative_release', 'fallback']} for r in details]
        base.csv_write(OUT / 'windows.csv', flat)
        report = {'passed': True, 'phase': 'panel', **source_hashes,
                  'design_sha256': base.digest(DATA / 'interval_precision_v2_design.json'),
                  'windows': len(parent['rows']), 'runs': len(details), 'summary': summary}
        base.save(OUT / 'panel.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'summary'}, indent=2))
    for row in report['summary']:
        if row['radius'] == 4:
            print(row)


if __name__ == '__main__':
    main()
