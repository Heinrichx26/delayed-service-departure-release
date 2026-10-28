"""Matched ablation on the unchanged sixteen-calendar, radius-four inputs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from math import ceil
from statistics import mean
from time import perf_counter
import calibration_panel as audit
from calibration_panel import physical, selective

DATA=audit.DATA
OUT=audit.OUT.parent/'ablation'
METHODS=['all_available','travel_cutoff','chronological','widest','selective_available']

def cutoff_controller(case,caps,radius,lead):
    # Calendar counts beyond t+travel are omitted by a simple fixed rule.
    # Tight release limits can make those counts relevant; every loss is retained.
    return physical.box_controller(case,caps,radius,case['lag'],'all_available')

def one(row):
    case=audit.read(audit.ROOT/row['input_path'])
    assert audit.digest(audit.ROOT/row['input_path'])==row['input_sha256']
    result=[]
    for label,divisor in audit.CAPS:
        caps=[ceil(case['n']/divisor)]*len(case['caps'])
        runs={}
        for method in METHODS:
            if method=='travel_cutoff':
                run=cutoff_controller(case,caps,4,case['lag']+2)
            elif method in ('chronological','widest'):
                default=selective.choose_confirmation
                def ordered(active,n,capacity,lag,t,previous,lead,known):
                    chosen,certificate=default(active,n,capacity,lag,t,previous,lead,known)
                    if chosen is None:return chosen,certificate
                    candidates=[j for j in range(t,min(len(capacity),t+lead+1))
                                if j not in known and max(hi[j] for lo,hi in active)>min(lo[j] for lo,hi in active)]
                    if method=='chronological':return min(candidates),certificate
                    return max(candidates,key=lambda j:(max(hi[j] for lo,hi in active)-min(lo[j] for lo,hi in active),-j)),certificate
                selective.choose_confirmation=ordered
                try:run=physical.box_controller(case,caps,4,case['lag']+2,'selective_available')
                finally:selective.choose_confirmation=default
            else:run=physical.box_controller(case,caps,4,case['lag']+2,method)
            runs[method]=run
            gate,total=audit.independent_replay(case,caps,run)
            result.append(dict(case_id=row['case_id'],cluster=row['cluster'],airport=row['airport'],day=row['day'],N=case['n'],
                               capacity=label,method=method,reference_complete=run['actual_reference_clears'],
                               covered=run['full_calendar_covered'],preserved=run['completion_preserving'],
                               deficit=run['maximum_completion_deficit'],requests=run['query_count'],
                               post_release=run['waiting_per_aircraft'],gate_wait=gate,start_to_takeoff=total,
                               same_release_as_all=run['cumulative_release']==runs['all_available']['cumulative_release']))
        for method in ['chronological','widest','selective_available']:
            if runs['all_available']['actual_reference_clears']:
                assert runs[method]['cumulative_release']==runs['all_available']['cumulative_release']
    return result

def protocol():
    return dict(date='2026-10-28',source_sha256=audit.digest(audit.Path(__file__)),
                manifest_sha256=audit.digest(audit.MANIFEST),dependencies=physical.hashes(),
                radius=4,centres=16,release_limits=[list(x) for x in audit.CAPS],methods=METHODS,
                travel_cutoff='Same reports and immediate responses; restrict requested future counts to t+lag. No stopping certificate.',
                order_ablation='Same certified stopping test; chronological, widest range, or release-sensitivity ordering.',
                paired_comparison='Same physical inputs and exact service counts; all 94 windows and every completion loss retained.',
                rows=audit.read(audit.MANIFEST)['rows'])

def smoke():
    design=protocol();row=design['rows'][0]
    checks=one(row)
    audit.save(DATA/'ablation_design.json',design)
    audit.save(OUT/'smoke.json',dict(passed=True,checks=checks,
                                   design_sha256=audit.digest(DATA/'ablation_design.json')))
    print(dict(passed=True,policy_runs=len(checks)),flush=True)

def run():
    start=perf_counter();design=audit.read(DATA/'ablation_design.json')
    assert design==protocol()
    smoke=audit.read(OUT/'smoke.json')
    assert smoke['passed'] and smoke['design_sha256']==audit.digest(DATA/'ablation_design.json')
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(one,design['rows']),1):
            rows.extend(result)
            if i%20==0:print(i,flush=True)
    summaries=[]
    for label,_ in audit.CAPS:
        for method in METHODS:
            group=[r for r in rows if r['capacity']==label and r['method']==method]
            complete=[r for r in group if r['reference_complete']]
            summaries.append(dict(capacity=label,method=method,windows=len(group),reference_complete=len(complete),
                                  preserved=sum(r['preserved'] for r in complete),
                                  same_release=sum(r['same_release_as_all'] for r in complete),
                                  requests_complete=mean(r['requests'] for r in complete),
                                  post_release_complete=mean(r['post_release'] for r in complete)))
    audit.save(OUT/'rows.json',rows);audit.save(OUT/'summary.json',summaries)
    audit.save(OUT/'audit.json',dict(passed=True,seconds=perf_counter()-start,policy_runs=len(rows),
                                   independent_replay=True,source_sha256=audit.digest(audit.Path(__file__))))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['smoke','run']);args=p.parse_args()
    {'smoke':smoke,'run':run}[args.command]()
