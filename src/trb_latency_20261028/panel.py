"""Frozen delayed-acquisition comparison; export and figures run separately."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from math import ceil
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from delayed_acquisition import ROOT,DelayedDesign,StateLimit,delayed_targets,fixed_schedule,old
from validate import independent_replay

OUT=ROOT/'results/trb_latency_20261028'
DATA=ROOT/'data/trb_latency_20261028'
SOURCE=Path(__file__)

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def waiting(problem,target):
    return sum(w*problem['case']['delta']*sum(u-d for u,d in zip(target[s],problem['references'][s]))/problem['n']
               for s,w in enumerate(problem['weights']))/problem['denominator']

def prepare():
    theory=read(OUT/'theory_smoke.json');public=read(OUT/'public_smoke.json')
    assert theory['passed'] and public['passed']
    model=SOURCE.with_name('delayed_acquisition.py')
    assert model.stat().st_mtime<=(OUT/'public_smoke.json').stat().st_mtime
    manifest=ROOT/'data/substantive_20261028/holdout_boxes/panel_design.json'
    rows=read(manifest)['rows'];jobs=[];incomplete=[]
    for row in rows:
        path=ROOT/row['input_path'];assert sha(path)==row['input_sha256']
        case=read(path)
        for label,divisor in [('N',1),('N4',4),('N8',8)]:
            adjusted=dict(case,caps=[ceil(case['n']/divisor)]*len(case['caps']))
            if old.prepare(adjusted) is None:
                incomplete.append(dict(case=row['case_id'],capacity=label));continue
            jobs.append(dict(case_id=row['case_id'],input_path=row['input_path'],input_sha256=row['input_sha256'],
                             airport=row['airport'],day=row['day'],capacity=label,divisor=divisor))
    assert len(jobs)==200 and len(incomplete)==82
    protocol=dict(date='2026-10-28',model_sha256=sha(model),runner_sha256=sha(SOURCE),
                  independent_enumerator_sha256=sha(SOURCE.with_name('validate.py')),
                  theory_smoke_sha256=sha(OUT/'theory_smoke.json'),public_smoke_sha256=sha(OUT/'public_smoke.json'),
                  parent_manifest_sha256=sha(manifest),delays=[1,2],dispatch_limits=[1,2],
                  lead='travel slots + 2; dispatch begins at slot zero',
                  target='All-available delayed release; same target, physical inputs and request constraints for fixed and adaptive schedules.',
                  relaxation='When a target is infeasible, scan all smaller uniform report leads in descending order; retain the first feasible target. This optimizes waiting within this declared target family, not among all policies.',
                  interpretation='Finite-calendar decision-design evaluation on existing recorded inputs; delays and dispatch limits are stress settings, not measured tower values.',
                  states_limit=150000,seconds_limit=40,workers=2,jobs=jobs,incomplete=incomplete)
    save(DATA/'panel_design.json',protocol)
    print(json.dumps(dict(prepared=True,complete_jobs=len(jobs),incomplete_retained=len(incomplete))))

def one(job):
    path=ROOT/job['input_path'];assert sha(path)==job['input_sha256']
    case=read(path);case=dict(case,caps=[ceil(case['n']/job['divisor'])]*len(case['caps']))
    problem=old.prepare(case);assert problem is not None
    lead=case['lag']+2;instant=waiting(problem,delayed_targets(problem,lead,0));output=[]
    for delay in [1,2]:
        fulltarget=delayed_targets(problem,lead,delay);fullwaiting=waiting(problem,fulltarget)
        for budget in [1,2]:
            try:
                model=DelayedDesign(problem,lead,delay,budget,fulltarget)
                adaptive=model.solve();fixed=fixed_schedule(problem,lead,delay,budget,fulltarget)
                replay=model.replay() if adaptive['feasible'] else []
                if replay:assert independent_replay(problem,fulltarget,replay,lead,delay,budget)==adaptive['weighted_cost']
                if fixed['feasible']:
                    assert adaptive['feasible'] and adaptive['expected_queries']<=fixed['count']+1e-9
                best_a=dict(target_lead=lead,waiting=fullwaiting,**adaptive)
                best_f=dict(target_lead=lead,waiting=fullwaiting,**fixed)
                # Both use the same declared decreasing-lead target family.
                for targetlead in range(lead-1,-2,-1):
                    if best_a['feasible'] and best_f['feasible']:break
                    target=delayed_targets(problem,targetlead,delay)
                    cost=waiting(problem,target)
                    if not best_a['feasible']:
                        alternative=DelayedDesign(problem,lead,delay,budget,target)
                        ans=alternative.solve()
                        if ans['feasible']:
                            best_a=dict(target_lead=targetlead,waiting=cost,**ans)
                            alt_replay=alternative.replay()
                            assert independent_replay(problem,target,alt_replay,lead,delay,budget)==ans['weighted_cost']
                    if not best_f['feasible']:
                        alternative=fixed_schedule(problem,lead,delay,budget,target)
                        if alternative['feasible']:best_f=dict(target_lead=targetlead,waiting=cost,**alternative)
                assert best_a['feasible'] and best_f['feasible']
                assert best_a['waiting']<=best_f['waiting']+1e-9
                record=dict(case_id=job['case_id'],airport=job['airport'],day=job['day'],capacity=job['capacity'],
                            delay=delay,budget=budget,lead=lead,n=case['n'],support_size=len(case['service']),
                            immediate_waiting=instant,delayed_target_waiting=fullwaiting,adaptive=adaptive,fixed=fixed,
                            attainable_adaptive=best_a,attainable_fixed=best_f)
                audit=dict(job=job,delay=delay,budget=budget,target=fulltarget,replays=replay,primary=record)
                name=f"{job['case_id']}_{job['capacity']}_d{delay}_b{budget}.json.gz"
                folder=OUT/'replay';folder.mkdir(exist_ok=True)
                with gzip.open(folder/name,'wt',encoding='utf-8') as file:json.dump(audit,file)
                output.append(record)
            except StateLimit as error:
                output.append(dict(case_id=job['case_id'],capacity=job['capacity'],delay=delay,budget=budget,unresolved=str(error)))
    return output

def summarize(rows):
    result=[]
    for cap in ['all','N','N4','N8']:
        for delay in [1,2]:
            for budget in [1,2]:
                group=[r for r in rows if r['delay']==delay and r['budget']==budget and (cap=='all' or r['capacity']==cap)]
                known=[r for r in group if 'unresolved' not in r]
                common=[r for r in known if r['adaptive']['feasible'] and r['fixed']['feasible']]
                af=[r for r in known if r['adaptive']['feasible']]
                result.append(dict(capacity=cap,delay=delay,budget=budget,cases=len(group),unresolved=len(group)-len(known),
                                   adaptive_feasible=len(af),fixed_feasible=sum(r['fixed']['feasible'] for r in known),common=len(common),
                                   fixed_queries=mean(r['fixed']['count'] for r in common) if common else None,
                                   adaptive_queries=mean(r['adaptive']['expected_queries'] for r in common) if common else None,
                                   mean_within_case_request_reduction=mean(1-r['adaptive']['expected_queries']/r['fixed']['count'] for r in common if r['fixed']['count']) if any(r['fixed']['count'] for r in common) else None,
                                   delayed_target_penalty=mean(r['delayed_target_waiting']-r['immediate_waiting'] for r in known),
                                   attainable_adaptive_penalty=mean(r['attainable_adaptive']['waiting']-r['immediate_waiting'] for r in known),
                                   attainable_fixed_penalty=mean(r['attainable_fixed']['waiting']-r['immediate_waiting'] for r in known),
                                   max_states=max(r['adaptive']['states'] for r in known)))
    return result

def run():
    start=perf_counter();design=read(DATA/'panel_design.json')
    assert design['model_sha256']==sha(SOURCE.with_name('delayed_acquisition.py'))
    assert design['runner_sha256']==sha(SOURCE)
    rows=[]
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=design['workers']) as pool:
        for index,value in enumerate(pool.map(one,design['jobs']),1):
            rows.extend(value)
            if index%20==0:print(json.dumps(dict(jobs=index,rows=len(rows),seconds=perf_counter()-start)),flush=True)
    save(OUT/'rows.json',rows);summary=summarize(rows);save(OUT/'summary.json',summary)
    report=dict(passed=not any('unresolved' in r for r in rows),rows=len(rows),jobs=len(design['jobs']),
                model_sha256=design['model_sha256'],runner_sha256=design['runner_sha256'],
                design_sha256=sha(DATA/'panel_design.json'),independent_replay=True,seconds=perf_counter()-start)
    save(OUT/'audit.json',report);print(json.dumps(report));print(json.dumps(summary[:4]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run']);args=p.parse_args()
    {'prepare':prepare,'run':run}[args.phase]()
