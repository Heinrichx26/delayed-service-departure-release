"""Training-date range calibration and matched report-selection ablations."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from math import ceil
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/substantive_20261028'))
import holdout_box_smoke as physical
import selective_interval as selective

DATA = ROOT / 'data/trb_independent_20261028'
OUT = ROOT / 'results/trb_independent_20261028/calibration'
MANIFEST = ROOT / 'data/substantive_20261028/holdout_boxes/panel_design.json'
CAPS = [('N', 1), ('N4', 4), ('N8', 8)]
METHODS = ['past_only', 'all_available', 'selective_available', 'chronological', 'widest']

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def calibrate(case, split):
    # Calendar construction, cohort size and travel input are unchanged.
    seeds, calibrators = case['service'][:split], case['service'][split:]
    scores = [min(max(abs(x-y) for x, y in zip(c, s)) for s in seeds)
              for c in calibrators]
    radius = max(scores)
    reduced = dict(case, service=seeds, donor_days=case['donor_days'][:split],
                   probabilities=[f'1/{split}'] * split)
    return reduced, radius, scores

def independent_replay(case, caps, result):
    cumulative, actual = result['cumulative_release'], case['actual_service']
    released = done = 0
    trajectory = []
    for t, total in enumerate(cumulative):
        assert isinstance(total, int) and 0 <= total-released <= caps[t]
        released = total
        runway = cumulative[t-case['lag']] if t >= case['lag'] else 0
        done += min(actual[t], runway-done)
        trajectory.append(done)
    assert trajectory == result['completed']
    ground = case['delta'] * sum(case['n']-u for u in cumulative)
    after = case['delta'] * sum(u-d for u, d in zip(cumulative, trajectory))
    total = case['delta'] * sum(case['n']-d for d in trajectory)
    assert ground + after == total
    return ground/case['n'], total/case['n']

def one(row, splits=(8,12)):
    path = ROOT / row['input_path']
    assert digest(path) == row['input_sha256']
    original = read(path)
    output = []
    training = []
    for split in splits:
        case, radius, scores = calibrate(original, split)
        assert all(day < original['day'] for day in original['donor_days'])
        assert all(sum(s) == case['n'] for s in case['service'])
        training.append(dict(case_id=row['case_id'], split=split, radius=radius,
                             calibration_scores=scores, seed_dates=case['donor_days'],
                             calibration_dates=original['donor_days'][split:]))
        for label, divisor in CAPS:
            caps = [ceil(case['n']/divisor)] * len(case['caps'])
            lead = case['lag']+2
            runs = {}
            methods = METHODS if split == 8 else METHODS[:3]
            for method in methods:
                if method in ('chronological','widest'):
                    default = selective.choose_confirmation
                    def ordered(active, n, capacity, lag, t, previous, available_lead, known):
                        chosen, certificate = default(active, n, capacity, lag, t, previous,
                                                       available_lead, known)
                        if chosen is None:
                            return chosen, certificate
                        candidates = [j for j in range(t,min(len(capacity),t+available_lead+1))
                                      if j not in known and
                                      max(hi[j] for lo,hi in active) > min(lo[j] for lo,hi in active)]
                        if method == 'chronological':
                            return min(candidates), certificate
                        return max(candidates, key=lambda j:(max(hi[j] for lo,hi in active)-
                                    min(lo[j] for lo,hi in active),-j)), certificate
                    selective.choose_confirmation = ordered
                    try:
                        result = physical.box_controller(case,caps,radius,lead,'selective_available')
                    finally:
                        selective.choose_confirmation = default
                else:
                    result = physical.box_controller(case,caps,radius,lead,method)
                runs[method] = result
                gate, total = independent_replay(case,caps,result)
                output.append(dict(case_id=row['case_id'],airport=row['airport'],day=row['day'],
                                   cluster=row['cluster'],N=case['n'],split=split,radius=radius,
                                   capacity=label,method=method,covered=result['full_calendar_covered'],
                                   reference_complete=result['actual_reference_clears'],
                                   preserved=result['completion_preserving'],
                                   maximum_deficit=result['maximum_completion_deficit'],
                                   fallback=result['fallback_used'],requests=result['query_count'],
                                   post_release=result['waiting_per_aircraft'],gate_wait=gate,
                                   start_to_takeoff=total,
                                   oracle_post_release=result['oracle_waiting_aircraft_minutes']/case['n']
                                   if result['oracle_waiting_aircraft_minutes'] is not None else None))
            baseline = runs['all_available']
            for method in methods[2:]:
                if baseline['full_calendar_covered'] and baseline['actual_reference_clears']:
                    assert baseline['cumulative_release'] == runs[method]['cumulative_release']
    return dict(rows=output, training=training)

def protocol():
    manifest=read(MANIFEST)
    return dict(date='2026-10-28',purpose='Supplementary retrospective audit using existing data; no new airport dates.',
                main_split='Jan 3--10 centres; Jan 11--18 calibration; Jan 24--31 evaluation, separately by airport/year/start/group size.',
                alternative_split='Jan 3--14 centres; Jan 15--18 calibration.',
                radius='Maximum calibration distance to the nearest centre; distance is maximum absolute count difference across all service slots.',
                interpretation='Descriptive separate-date coverage; serial dependence and eight or four calibration dates preclude an asserted population coverage guarantee.',
                comparison='Identical radius, calendar centres, release limit, lead, report precision and takeoff reference for all methods.',
                ablations='Chronological and widest-report ordering retain the same stopping certificate and physical action.',
                source_sha256=digest(Path(__file__)),manifest_sha256=digest(MANIFEST),
                dependency_sha256=physical.hashes(),workers=2,rows=manifest['rows'])

def smoke():
    start=perf_counter()
    design=protocol()
    rows=design['rows']
    selected=[rows[0],next(r for r in rows if r['airport']=='PHX')]
    # Altering the evaluation calendar must leave training radius and centres unchanged.
    checks=[]
    for row in selected:
        case=read(ROOT/row['input_path'])
        for split in (8,12):
            a,r,s=calibrate(case,split)
            changed=dict(case,actual_service=[999]*len(case['caps']))
            b,q,v=calibrate(changed,split)
            assert (r,s,a['service'])==(q,v,b['service'])
        checks.append(one(row))
    save(DATA/'calibration_design.json',design)
    result=dict(passed=True,seconds=perf_counter()-start,
                design_sha256=digest(DATA/'calibration_design.json'),checks=checks)
    save(OUT/'smoke.json',result)
    print(json.dumps(dict(passed=True,seconds=result['seconds'],windows=2,
                          policy_runs=sum(len(x['rows']) for x in checks))),flush=True)

def summarize(rows):
    summary=[]
    for split in (8,12):
        for capacity,_ in CAPS:
            for method in METHODS:
                group=[r for r in rows if (r['split'],r['capacity'],r['method'])==(split,capacity,method)]
                if not group: continue
                complete=[r for r in group if r['reference_complete']]
                summary.append(dict(split=split,capacity=capacity,method=method,windows=len(group),
                                    covered=sum(r['covered'] for r in group),
                                    reference_complete=len(complete),
                                    preserved_complete=sum(r['preserved'] for r in complete),
                                    preserved_all=sum(r['preserved'] for r in group),
                                    losses=sum(not r['preserved'] for r in group),
                                    maximum_deficit=max(r['maximum_deficit'] for r in group),
                                    fallback=sum(r['fallback'] for r in group),
                                    mean_requests=mean(r['requests'] for r in group),
                                    mean_post_release_complete=mean(r['post_release'] for r in complete),
                                    mean_gate_wait_complete=mean(r['gate_wait'] for r in complete),
                                    mean_start_to_takeoff_complete=mean(r['start_to_takeoff'] for r in complete)))
    return summary

def run():
    start=perf_counter()
    frozen=read(DATA/'calibration_design.json')
    assert frozen==protocol()
    prior=read(OUT/'smoke.json')
    assert prior['passed'] and prior['design_sha256']==digest(DATA/'calibration_design.json')
    rows,training=[],[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for index,result in enumerate(pool.map(one,frozen['rows']),1):
            rows.extend(result['rows']);training.extend(result['training'])
            if index%10==0:print(index,flush=True)
    save(OUT/'rows.json',rows)
    save(OUT/'training_choices.json',training)
    save(OUT/'summary.json',summarize(rows))
    save(OUT/'audit.json',dict(passed=True,seconds=perf_counter()-start,windows=94,
                              policy_runs=len(rows),training_choices=len(training),
                              source_sha256=digest(Path(__file__)),
                              design_sha256=digest(DATA/'calibration_design.json'),
                              independent_replay=True,time_conservation=True,
                              covered_complete_action_equality=True))
    print(json.dumps(read(OUT/'audit.json')),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['smoke','run'])
    args=parser.parse_args()
    {'smoke':smoke,'run':run}[args.command]()
