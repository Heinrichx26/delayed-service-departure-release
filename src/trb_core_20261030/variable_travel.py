"""Time-varying FIFO travel: exact deadlines, independent oracles and panel.

One-slot perturbations are declared sensitivity inputs, not measured route times.
All aircraft released in a slot share its arrival time. No overtaking occurs.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
from itertools import combinations, product
import json
from math import ceil, lcm
import os
from pathlib import Path
import random
from statistics import mean
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/trb_joint_20261028'))
sys.path.insert(0, str(ROOT/'src/trb_six_20261028'))
from joint_design import JointDesign, old, StateLimit
from extensions import PairGreedy
OUT = ROOT/'results/trb_core_20261030/variable_travel'
DATA = ROOT/'data/trb_core_20261030/variable_travel'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def simulate(release, service, arrivals):
    """Propagate batches, using no deadline or optimization helper."""
    packets = defaultdict(int)
    for t, quantity in enumerate(old.base.increments(release)):
        packets[arrivals[t]] += quantity
    waiting = completed = 0
    result = []
    for t, capacity in enumerate(service[:len(release)]):
        waiting += packets[t]
        served = min(waiting, capacity)
        waiting -= served
        completed += served
        result.append(completed)
    return result

def prepare(case, profiles, prefix=()):
    h, n, caps = len(case['caps']), case['n'], case['caps']
    earliest = old.base.earliest(n, caps)
    paths, arrivals, probabilities = [], [], []
    for service, probability in zip(case['service'], case['probabilities']):
        for profile in profiles:
            assert len(profile)==h and all(profile[t]>=t for t in range(h))
            assert all(a<=b for a,b in zip(profile, profile[1:]))
            paths.append(service)
            arrivals.append(profile)
            probabilities.append(Fraction(probability)/len(profiles))
    references = [simulate(earliest, p, a) for p,a in zip(paths, arrivals)]
    if any(row[-1]!=n for row in references):
        return None
    required, bounds = [], []
    for ref, arrival in zip(references, arrivals):
        latest = [bisect_right(arrival,v)-1 for v in range(h)]
        deadlines = [max([0]+[ref[v] for v in range(h) if latest[v]<=t]) for t in range(h)]
        backward = deadlines[:]
        for t in range(h-2,-1,-1):
            backward[t] = max(backward[t],backward[t+1]-caps[t+1])
        assert all(backward[t]<=earliest[t] for t in range(h))
        required.append(deadlines)
        bounds.append(backward)
    denominator = lcm(*(p.denominator for p in probabilities))
    weights = [int(p*denominator) for p in probabilities]
    targets = []
    for bound in bounds:
        previous = prefix[-1] if prefix else 0
        target = list(prefix)
        for t in range(len(prefix),h):
            previous = max(previous,bound[t])
            target.append(previous)
        targets.append(target)
    return dict(case=case,paths=paths,arrivals=arrivals,required=required,bounds=bounds,
                references=references,caps=caps,n=n,lag=case['lag'],prefix=list(prefix),
                t0=len(prefix),horizon=h,weights=weights,denominator=denominator,target=targets)

def raw(problem,lead,delay,budget):
    """Enumerate every release and request; use physical simulation for safety."""
    paths,arrivals,refs = problem['paths'],problem['arrivals'],problem['references']
    weights,h,n = problem['weights'],problem['horizon'],problem['n']
    @lru_cache(None)
    def recurse(t,members,pending,prefix):
        if t==h:
            return (0,0) if prefix[-1]==n and all(simulate(prefix,paths[s],arrivals[s])==refs[s] for s in members) else None
        previous = prefix[-1] if prefix else 0
        existing = {j for _,j in pending}
        candidates = [j for j in range(t+delay,min(h,t+lead+1))
                      if j not in existing and len({paths[s][j] for s in members})>1]
        best = None
        for u in range(previous,min(n,previous+problem['caps'][t])+1):
            extended = prefix+(u,)
            if any(simulate(extended,paths[s],arrivals[s])!=refs[s][:t+1] for s in members):
                continue
            stage = sum(weights[s]*(u-refs[s][t]) for s in members)
            for size in range(min(budget,len(candidates))+1):
                for batch in combinations(candidates,size):
                    combined = pending+tuple((t+delay,j) for j in batch)
                    due = [j for d,j in combined if d==t+1]
                    future = tuple(sorted((d,j) for d,j in combined if d>t+1))
                    groups = defaultdict(list)
                    for s in members:
                        groups[tuple([paths[s][t]]+[paths[s][j] for j in due])].append(s)
                    value = [stage,size*sum(weights[s] for s in members)]
                    for group in groups.values():
                        child = recurse(t+1,tuple(group),future,extended)
                        if child is None:
                            break
                        value[0] += child[0]
                        value[1] += child[1]
                    else:
                        if best is None or tuple(value)<best:
                            best = tuple(value)
        return best
    groups = defaultdict(list)
    for s,path in enumerate(paths):
        groups[tuple(path[:problem['t0']])].append(s)
    values = [recurse(problem['t0'],tuple(group),(),tuple(problem['prefix'])) for group in groups.values()]
    assert all(v is not None for v in values)
    return tuple(sum(v[k] for v in values) for k in (0,1)),recurse.cache_info().currsize

def replay(model):
    problem = model.problem
    rows,histories = [],{}
    for s,path in enumerate(problem['paths']):
        mask = next(m for m in model.initial_groups() if m>>s&1)
        pending,queries,release = (),[],list(problem['prefix'])
        previous = release[-1] if release else 0
        for t in range(problem['t0'],model.h):
            model.value(t,mask,pending,previous)
            batch = model.choice[t,mask,pending,previous]
            u = max(previous,model.bound(mask,t))
            assert 0<=u-previous<=problem['caps'][t]
            history = (t,tuple(path[:t]),tuple((q['dispatch'],q['slot'],path[q['slot']])
                       for q in queries if q['arrival']<=t),
                       tuple((q['dispatch'],q['slot']) for q in queries))
            assert history not in histories or histories[history]==(u,batch)
            histories[history] = (u,batch)
            assert len(batch)<=model.budget
            release.append(u)
            for j in batch:
                assert t+model.delay<=j<=t+model.lead
                queries.append(dict(dispatch=t,arrival=t+model.delay,slot=j))
            mask,pending = next((m,p) for m,p in model.transition(t,mask,pending,batch) if m>>s&1)
            previous = u
        completed = simulate(release,path,problem['arrivals'][s])
        assert completed==problem['references'][s] and release[-1]==problem['n']
        rows.append(dict(s=s,release=release,completed=completed,queries=queries))
    result = model.solve()
    assert sum(problem['weights'][r['s']]*len(r['queries']) for r in rows)==result['weighted_queries']
    assert sum(problem['weights'][r['s']]*sum(r['release'][t]-r['completed'][t]
               for t in range(problem['t0'],model.h)) for r in rows)==result['weighted_waiting']
    return rows

def profiles_for(case):
    h,lag = len(case['caps']),case['lag']
    def ordered(offsets):
        result,previous = [],-1
        for t,offset in enumerate(offsets):
            previous = max(previous,t+lag+offset)
            result.append(previous)
        return result
    return [ordered([0]*h),
            ordered([int(2<=t<h//2) for t in range(h)]),
            ordered([-int(h//3<=t<2*h//3) for t in range(h)])]

def theory(count,output):
    rng = random.Random(2026103007)
    rows,feasibility_checks,attempts = [],0,0
    while len(rows)<count*4:
        attempts += 1
        h,n,m = 7,rng.choice([1,2,3]),rng.choice([2,3])
        paths = []
        for _ in range(m):
            path = [0]*h
            for _ in range(n):
                path[rng.randrange(2,h)] += 1
            paths.append(path)
        case = dict(service=paths,probabilities=[f'1/{m}']*m,
                    caps=[rng.choice([1,2]) for _ in range(h)],n=n,lag=1,delta=2)
        problem = prepare(case,profiles_for(case))
        if problem is None:
            continue
        for increments in product(*(range(k+1) for k in case['caps'])):
            if sum(increments)!=n:
                continue
            cumulative,u = [],0
            for amount in increments:
                u += amount
                cumulative.append(u)
            for s,path in enumerate(problem['paths']):
                physical = simulate(cumulative,path,problem['arrivals'][s])==problem['references'][s]
                deadlines = all(u>=d for u,d in zip(cumulative,problem['required'][s]))
                assert physical==deadlines
                feasibility_checks += 1
        for delay,budget in product([1,2],[1,2]):
            model = JointDesign(problem,3,delay,budget)
            answer = model.solve()
            oracle,raw_states = raw(problem,3,delay,budget)
            assert (answer['weighted_waiting'],answer['weighted_queries'])==oracle
            replay(model)
            rows.append(dict(attempt=attempts,delay=delay,budget=budget,
                             states=answer['states'],raw_states=raw_states))
    save(OUT/output,dict(passed=True,comparisons=len(rows),feasibility_checks=feasibility_checks,
                         seed=2026103007,rows=rows))
    return dict(passed=True,comparisons=len(rows),feasibility_checks=feasibility_checks)

def one(task):
    job,delay,budget = task
    source = ROOT/job['input_path']
    assert sha(source)==job['input_sha256']
    case = read(source)
    case = dict(case,caps=[ceil(case['n']/job['divisor'])]*len(case['caps']))
    profiles = profiles_for(case)
    results = []
    for mode,selected in [('known_variable',[profiles[1]]),('uncertain_variable',profiles)]:
        common = dict(case_id=job['case_id'],capacity=job['capacity'],airport=job['airport'],
                      day=job['day'],delay=delay,budget=budget,mode=mode,n=case['n'])
        problem = prepare(case,selected)
        if problem is None:
            results.append(dict(common,eligible=False,reason='Incomplete earliest-release reference.'))
            continue
        try:
            model = JointDesign(problem,case['lag']+2,delay,budget,seconds_limit=45)
            exact = model.solve()
            paths = replay(model)
            greedy = PairGreedy(problem,case['lag']+2,delay,budget)
            comparator = greedy.solve()
            replay(greedy)
            assert exact['weighted_waiting']<=comparator['weighted_waiting']
            free_release = [[] for _ in problem['paths']]
            for t in range(problem['horizon']):
                groups = defaultdict(list)
                for s,path in enumerate(problem['paths']):
                    groups[tuple(path[:t])].append(s)
                for members in groups.values():
                    before = free_release[members[0]][-1] if t else 0
                    u = max(before,max(problem['bounds'][s][t] for s in members))
                    for s in members:
                        free_release[s].append(u)
            free_value = sum(w*sum(u-d for u,d in zip(free_release[s],problem['references'][s]))
                             for s,w in enumerate(problem['weights']))*case['delta']/problem['denominator']/case['n']
            assert exact['waiting']<=free_value+1e-9
            result = dict(common,eligible=True,optimal=True,exact_waiting=exact['waiting'],
                          greedy_waiting=comparator['waiting'],no_request_waiting=free_value,
                          exact_requests=exact['expected_queries'],greedy_requests=comparator['expected_queries'],
                          states=exact['states'],seconds=exact['seconds'],support_size=len(problem['paths']))
            target = OUT/'replay'/f"{job['case_id']}_{job['capacity']}_d{delay}_b{budget}_{mode}.json.gz"
            target.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(target,'wt',encoding='utf-8') as handle:
                json.dump(dict(result=result,arrivals=problem['arrivals'],paths=paths),handle)
            results.append(result)
        except StateLimit as exc:
            results.append(dict(common,eligible=True,optimal=False,error=str(exc)))
    return results

def smoke():
    print(json.dumps(theory(6,'theory_smoke.json')),flush=True)
    jobs = read(ROOT/'data/trb_latency_20261028/panel_design.json')['jobs']
    selected = [next(j for j in jobs if j['airport']==a and j['capacity']==c)
                for a,c in product(['DEN','PHX'],['N','N4'])]
    rows = [r for job in selected for d in [1,2] for r in one((job,d,1))]
    passed = all(r.get('optimal',True) for r in rows) and any(r['eligible'] for r in rows)
    save(OUT/'public_smoke.json',dict(passed=passed,source_sha256=sha(__file__),rows=rows))
    print(json.dumps(dict(passed=passed,conditions=len(rows),eligible=sum(r['eligible'] for r in rows))),flush=True)
    assert passed

def full():
    smoke = read(OUT/'public_smoke.json')
    assert smoke['passed'] and smoke['source_sha256']==sha(__file__)
    print(json.dumps(theory(48,'theory_full.json')),flush=True)
    jobs = read(ROOT/'data/trb_latency_20261028/panel_design.json')['jobs']
    tasks = [(job,d,b) for job in jobs for d,b in product([1,2],[1,2])]
    save(DATA/'design.json',dict(source_sha256=sha(__file__),parent_sha256=sha(ROOT/'data/trb_latency_20261028/panel_design.json'),
         perturbation_slots=1,probability_rule='Equal profiles within each original calendar.',
         modes=['known_variable','uncertain_variable'],delays=[1,2],budgets=[1,2],jobs=jobs,workers=2))
    rows,start = [],perf_counter()
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,group in enumerate(pool.map(one,tasks),1):
            rows.extend(group)
            if i%100==0:
                print(json.dumps(dict(tasks=i,elapsed=perf_counter()-start)),flush=True)
                save(OUT/'progress.json',dict(rows=rows))
    save(OUT/'full.json',dict(passed=all(r.get('optimal',True) for r in rows),seconds=perf_counter()-start,rows=rows))
    assert all(r.get('optimal',True) for r in rows)
    summarize(rows)

def summarize(rows):
    summary = []
    for mode,d,b in product(['known_variable','uncertain_variable'],[1,2],[1,2]):
        group = [r for r in rows if r['mode']==mode and r['delay']==d and r['budget']==b and r['eligible'] and r.get('optimal')]
        summary.append(dict(mode=mode,delay=d,budget=b,eligible=len(group),excluded=200-len(group),
                            waiting=mean(r['exact_waiting'] for r in group),requests=mean(r['exact_requests'] for r in group),
                            greedy_saving=mean(r['greedy_waiting']-r['exact_waiting'] for r in group),
                            no_request_saving=mean(r['no_request_waiting']-r['exact_waiting'] for r in group)))
    save(OUT/'summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['smoke','full'])
    globals()[parser.parse_args().phase]()
