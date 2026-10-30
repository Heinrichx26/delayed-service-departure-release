"""Summarize saved independent experiments without rerunning optimization."""
from statistics import mean
from independent_panel import ROOT, DATA, OUT, read, save, sha


def paired(rows, depth):
    values = [r['methods'][str(depth)] for r in rows]
    assert all(r['resolved'] for r in values)
    gaps = [60*(r['lookahead']['minutes']-r['exact']['minutes']) for r in values]
    assert min(gaps) >= -1e-8
    return dict(conditions=len(rows),
                lookahead_minutes=mean(r['lookahead']['minutes'] for r in values),
                exact_minutes=mean(r['exact']['minutes'] for r in values),
                saved_seconds=mean(gaps), maximum_saved_seconds=max(gaps),
                strictly_improved=sum(v>1e-8 for v in gaps),
                lookahead_max_requests=mean(r['lookahead']['max_requests'] for r in values),
                exact_max_requests=mean(r['exact']['max_requests'] for r in values))


def main():
    design = read(DATA/'design.json')
    full = read(OUT/'full.json')
    certificate = read(OUT/'certificate_full.json')
    trace = read(OUT/'decision_trace.json')
    assert full['passed'] and certificate['passed'] and trace['passed']
    rows = full['rows']; boundrows=certificate['rows']
    assert len(rows) == len(boundrows) == 70
    assert len({r['numerical_signature'] for r in rows}) == 70
    assert all(sha(ROOT/r['input_path']) == r['input_sha256'] for r in rows)
    assert sum(len(r['represented_windows']) for r in rows) == design['represented_window_conditions'] == 554
    assert len(design['sources']) == 95
    assert all(r['original_n']==r['donor_only_n'] for r in design['sources'])
    for row in rows:
        for method in row['methods'].values():
            assert method['resolved'] and method['lookahead']['histories']>0
            assert method['exact']['max_requests']<=method['lookahead']['max_requests']
            assert method['exact']['value']<=method['lookahead']['value']
    for row in boundrows:
        lower = [row['initial_lower']]+[v['lower'] for v in row['depths']]
        assert lower == sorted(lower) and lower[-1]<=row['optimum']<=row['upper']
        assert row['certified'] == (lower[-1]==row['upper'])
    summary=dict(conditions=70, distinct_physical_settings=35, source_windows=95,
        represented_window_conditions=554,
        sample_interpretation='Distinct numerical settings; repeated donor calendars are counted once, without date resampling',
        selection='Previously frozen source windows; donor-only group sizes; inherited completeness screening',
        donor_only_sizes_changed=0, error_allowance=1, requests_per_two_minute_slot=1,
        response_minutes=[2,4], lookahead={},airport_detail={},
        certified=sum(r['certified'] for r in boundrows),
        six_step_optimal=sum(r['true_gap']==0 for r in boundrows),
        exact_fallback=len(boundrows)-sum(r['certified'] for r in boundrows),
        certificate_seconds=mean(r['seconds'] for r in boundrows),
        certificates_by_response={str(d*2):sum(r['certified'] for r in boundrows if r['delay']==d) for d in (1,2)},
        all_bound_checks_resolved=all(r['resolved'] for r in boundrows),
        total_physical_report_histories=sum(r['methods'][str(depth)][method]['histories']
            for r in rows for depth in (2,4,6) for method in ('lookahead','exact')),
        source_file_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in
            [DATA/'design.json',OUT/'full.json',OUT/'certificate_full.json',OUT/'decision_trace.json']})
    for depth in (2,4,6):
        summary['lookahead'][str(depth)]={str(d*2):paired([r for r in rows if r['delay']==d],depth) for d in (1,2)}
    for airport in ('ATL','CLT','DFW'):
        summary['airport_detail'][airport]={str(depth):{str(d*2):paired(
            [r for r in rows if r['airport']==airport and r['delay']==d],depth) for d in (1,2)} for depth in (2,4,6)}
    summary['nonmonotone_example']=dict(
        condition=trace['condition'],six_worst_aircraftslots=trace['six_worst_value'],
        exact_worst_aircraftslots=trace['exact_worst_value'],
        exact_same_calendar_aircraftslots=trace['exact_same_calendar_value'],
        shared_request_counts=[6,8,9], six_request_order=[8,6,9],exact_request_order=[6,9,8],
        first_different_release_slot=3, release_gap_aircraftslots=21,
        group_saved_aircraftminutes=42, seconds_per_departure=42*60/19)
    save(OUT/'summary.json',summary)
    print({k:v for k,v in summary.items() if k not in ('airport_detail','source_file_sha256')})


if __name__=='__main__':
    main()
