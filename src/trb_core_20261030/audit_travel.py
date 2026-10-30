"""Audit saved paired travel results and export tables without rerunning solvers."""
from pathlib import Path
from statistics import mean, median
import gzip
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from variable_travel import read,save,simulate,prepare,profiles_for,sha
from math import ceil
OUT=ROOT/'results/trb_core_20261030/variable_travel'
full=read(OUT/'full.json')
assert full['passed'] and read(OUT/'public_smoke.json')['passed']
assert read(OUT/'theory_full.json')['passed']
jobs=read(ROOT/'data/trb_latency_20261028/panel_design.json')['jobs']
lookup={(j['case_id'],j['capacity']):j for j in jobs}
rows=[]
for row in full['rows']:
    if row['mode']!='uncertain_variable' or not row['eligible']:
        continue
    case_id,cap,d,b=row['case_id'],row['capacity'],row['delay'],row['budget']
    job=lookup[case_id,cap]
    case=read(ROOT/job['input_path'])
    case=dict(case,caps=[ceil(case['n']/job['divisor'])]*len(case['caps']))
    profiles=profiles_for(case)
    problem=prepare(case,profiles)
    original=ROOT/'results/trb_joint_20261028/replay'/f'{case_id}_{cap}_d{d}_b{b}.json.gz'
    with gzip.open(original,'rt',encoding='utf-8') as handle:
        saved=json.load(handle)
    with gzip.open(OUT/'replay'/f'{case_id}_{cap}_d{d}_b{b}_uncertain_variable.json.gz','rt',encoding='utf-8') as handle:
        generalized=json.load(handle)
    assert len(generalized['paths'])==len(case['service'])*3
    preserved=True
    deficits=[]
    for path in saved['paths']:
        s=path['s']
        for a,profile in enumerate(profiles):
            ref=problem['references'][3*s+a]
            # A common takeoff reference permits an unconfounded release comparison.
            assert ref==path['completed']
            done=simulate(path['release'],case['service'][s],profile)
            assert all(x<=y for x,y in zip(done,ref))
            deficits.append(sum(y-x for x,y in zip(done,ref))*case['delta'])
            preserved=preserved and done==ref
    added_wait=row['exact_waiting']-saved['result']['waiting']
    assert added_wait>=-1e-9
    rows.append(dict(row,original_preserved=preserved,
                     original_mean_shortfall_aircraft_min=mean(deficits),
                     additional_post_release_time=added_wait))
summary=[]
for d in [1,2]:
    for b in [1,2]:
        group=[r for r in rows if r['delay']==d and r['budget']==b]
        assert len(group)==163
        summary.append(dict(delay=d,budget=b,conditions=len(group),
          original_preserved=sum(r['original_preserved'] for r in group),generalized_preserved=len(group),
          extra_time=mean(r['additional_post_release_time'] for r in group),
          greedy_saving=mean(r['greedy_waiting']-r['exact_waiting'] for r in group),
          request_difference=mean(r['exact_requests']-r['greedy_requests'] for r in group),
          no_request_saving=mean(r['no_request_waiting']-r['exact_waiting'] for r in group)))
theory=read(OUT/'theory_full.json')
ratios=[r['raw_states']/r['states'] for r in theory['rows']]
known={(r['case_id'],r['capacity'],r['delay'],r['budget']):r for r in full['rows']
       if r['mode']=='known_variable' and r['eligible']}
assert all(r['exact_waiting']==known[r['case_id'],r['capacity'],r['delay'],r['budget']]['exact_waiting']
           and r['exact_requests']==known[r['case_id'],r['capacity'],r['delay'],r['budget']]['exact_requests'] for r in rows)
audit=dict(passed=True,full_conditions=len(full['rows']),complete_conditions=len(rows)*2,
           excluded_combinations=37,theory_comparisons=theory['comparisons'],
           feasibility_checks=theory['feasibility_checks'],median_raw_to_reduced_states=median(ratios),
           common_reference_verified=True,known_slowest_equals_uncertain_all=True,
           source_hashes={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('variable_travel.py')]},
           summary=summary)
save(OUT/'paired_audit.json',audit)
save(OUT/'paired_rows.json',rows)
print(json.dumps(audit,indent=2))
