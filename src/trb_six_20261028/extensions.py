"""Transport-constrained information partitions, gate deadlines and recovery."""
from collections import defaultdict
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from time import perf_counter
import json, sys
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/trb_joint_20261028'))
from joint_design import JointDesign, old, StateLimit

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def save(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')

def backward(required,caps):
    b=list(required)
    for t in range(len(b)-2,-1,-1):
        b[t]=max(b[t],b[t+1]-caps[t+1])
    return b

def gate_problem(problem,required):
    """EDF rank identifies an aircraft; common travel and service remain fixed."""
    p=dict(problem)
    g=backward(required,p['caps'])
    assert all(g[t] <= old.base.earliest(p['n'],p['caps'])[t] for t in range(p['horizon']))
    p['bounds']=[[max(x,y) for x,y in zip(b,g)] for b in p['bounds']]
    p['gate_required']=required
    p['target']=[]
    for b in p['bounds']:
        previous=p['prefix'][-1] if p['prefix'] else 0
        row=list(p['prefix'])
        for t in range(p['t0'],p['horizon']):
            previous=max(previous,b[t]);row.append(previous)
        p['target'].append(row)
    return p

def reset_loss(case,caps,prefix,paths=None):
    """Earliest continuation attains maximum possible departures at every slot."""
    paths=case['service'] if paths is None else paths
    n=case['n'];lag=case['lag'];h=len(caps)
    continuation=list(prefix);previous=prefix[-1] if prefix else 0
    for t in range(len(prefix),h):
        previous=min(n,previous+caps[t]);continuation.append(previous)
    rows=[]
    for path in paths:
        ref=old.base.physical(old.base.earliest(n,caps),path,n,lag)[0]
        got=old.base.physical(continuation,path,n,lag)[0]
        deficit=[a-b for a,b in zip(ref,got)]
        assert min(deficit)>=0
        rows.append(dict(maximum=max(deficit),area_slots=sum(deficit),
                         first=next((i for i,v in enumerate(deficit) if v),None),
                         reference=ref,attainable=got))
    return dict(feasible=all(r['maximum']==0 for r in rows),
                continuation=continuation,paths=rows)

def partition_problem(problem,schedule,targets):
    paths=problem['paths'];h=problem['horizon'];m=len(paths)
    slots=sorted({q['slot'] for q in schedule})
    variables=[(j,k) for j in slots for k in sorted(set(p[j] for p in paths))[1:]]
    requirements=[]
    for s,q in combinations(range(m),2):
        d=next((t for t in range(problem['t0'],h) if targets[s][t]!=targets[q][t]),None)
        if d is None:continue
        f=next((t for t in range(h) if paths[s][t]!=paths[q][t]),h)
        if f<d:continue
        arrived={r['slot'] for r in schedule if r['arrival']<=d}
        cover=[v for v,(j,k) in enumerate(variables)
               if j in arrived and min(paths[s][j],paths[q][j])<k<=max(paths[s][j],paths[q][j])]
        if not cover:raise ValueError(('infeasible target schedule',s,q,d))
        requirements.append(dict(s=s,q=q,deadline=d,cover=cover))
    return variables,requirements

def compress_reports(problem,schedule,targets):
    variables,req=partition_problem(problem,schedule,targets)
    if not req:
        chosen=[]
    else:
        mat=lil_matrix((len(req),len(variables)),dtype=float)
        for i,r in enumerate(req):mat[i,r['cover']]=1.
        ans=milp(np.ones(len(variables)),integrality=np.ones(len(variables)),
                 bounds=Bounds(0,1),constraints=LinearConstraint(mat.tocsc(),1,np.inf),
                 options=dict(time_limit=15,mip_rel_gap=0))
        if ans.status!=0:
            return dict(optimal=False,status=int(ans.status))
        chosen=[variables[i] for i,x in enumerate(ans.x) if x>.5]
        assert len(chosen)==round(ans.fun)
    selected={v for v,x in enumerate(variables) if x in chosen}
    assert all(selected.intersection(r['cover']) for r in req)
    # Replay complete histories, not only the action at report arrival.
    def label(j,v):
        return sum(v>=k for jj,k in chosen if jj==j)
    states={}
    for s,path in enumerate(problem['paths']):
        for t in range(problem['t0'],problem['horizon']):
            history=(t,tuple(path[:t]),tuple((r['slot'],label(r['slot'],path[r['slot']]))
                      for r in schedule if r['arrival']<=t))
            if history in states:assert states[history]==targets[s][t]
            states[history]=targets[s][t]
    return dict(optimal=True,original_boundaries=len(variables),boundaries=len(chosen),
                original_labels=len(variables)+len(set(q['slot'] for q in schedule)),
                labels=len(chosen)+len(set(q['slot'] for q in schedule)),
                boundaries_by_slot={str(j):[k for jj,k in chosen if jj==j] for j in sorted(set(q['slot'] for q in schedule))},
                pairs=len(req),histories=len(states))

class PairGreedy(JointDesign):
    """Deadline-feasible greedy pair cover; identical reports and slot resources."""
    def value(self,t,mask,pending,previous):
        state=(t,mask,pending,previous)
        if t==self.h:
            self.values[state]=(0,0);self.choice[state]=();return 0,0
        release=max(previous,self.bound(mask,t))
        assert release-previous<=self.problem['caps'][t]
        remaining=self.outstanding(t,mask,pending)
        candidates=list(self.candidates(t,mask,pending))
        selected=[];uncovered=list(remaining)
        for _ in range(min(self.budget,len(candidates))):
            def score(j):
                pairs=[(s,q,d) for s,q,d in uncovered if t+self.delay<=d and self.paths[s][j]!=self.paths[q][j]]
                return (sum(self.weights[s]*self.weights[q] for s,q,d in pairs),
                        -min((d for s,q,d in pairs),default=self.h),-j)
            j=max(candidates,key=score)
            if score(j)[0]==0:break
            selected.append(j);candidates.remove(j)
            uncovered=[(s,q,d) for s,q,d in uncovered if not(t+self.delay<=d and self.paths[s][j]!=self.paths[q][j])]
        request=tuple(selected)
        value=[self.weight(mask)*release-self.ref(mask,t),self.weight(mask)*len(request)]
        for sub,future in self.transition(t,mask,pending,request):
            child=self.value(t+1,sub,future,release)
            value[0]+=child[0];value[1]+=child[1]
        self.choice[state]=request;self.values[state]=tuple(value)
        return tuple(value)

def smoke():
    from validate_joint import raw, physical
    rng=np.random.default_rng(2026102829);gate_checks=0;partition_checks=0;recovery_checks=0
    for i in range(20):
        n=2;h=6;m=3
        paths=[]
        for _ in range(m):
            p=[0]*h
            for j in rng.integers(2,h,size=n):p[int(j)]+=1
            paths.append(p)
        case=dict(n=n,lag=1,delta=2,caps=[1]*h,service=paths,probabilities=['1/3']*3)
        problem=old.prepare(case);assert problem is not None
        required=[0,0,1,1,2,2]
        gp=gate_problem(problem,required)
        # Unrestricted enumeration obtains original physical feasibility plus G.
        def brute_gate():
            @lru_cache(None)
            def visit(t,members,pending,prefix):
                if t==h:
                    return (0,0) if all(physical(prefix,paths[s],1)==problem['references'][s] for s in members) else None
                best=None;prev=prefix[-1] if prefix else 0
                candidates=[j for j in range(t+1,min(h,t+4)) if j not in {j for _,j in pending}
                            and len({paths[s][j] for s in members})>1]
                for release in range(max(prev,required[t]),min(n,prev+1)+1):
                    extended=prefix+(release,)
                    if any(physical(extended,paths[s],1)!=problem['references'][s][:t+1] for s in members):continue
                    for size in range(min(1,len(candidates))+1):
                        for batch in combinations(candidates,size):
                            allpending=pending+tuple((t+1,j) for j in batch)
                            due=[j for d,j in allpending if d==t+1]
                            future=tuple((d,j) for d,j in allpending if d>t+1)
                            groups=defaultdict(list)
                            for s in members:groups[(paths[s][t],)+tuple(paths[s][j] for j in due)].append(s)
                            val=[sum(release-problem['references'][s][t] for s in members),size*len(members)]
                            for group in groups.values():
                                child=visit(t+1,tuple(group),future,extended)
                                if child is None:break
                                val[0]+=child[0];val[1]+=child[1]
                            else:
                                if best is None or tuple(val)<best:best=tuple(val)
                return best
            return visit(0,tuple(range(m)),(),())
        model=JointDesign(gp,3,1,1);answer=model.solve();paths_replay=model.replay()
        assert (answer['weighted_waiting'],answer['weighted_queries'])==brute_gate()
        assert all(all(u>=g for u,g in zip(row['release'],required)) for row in paths_replay)
        gate_checks+=1
        # Fixed full allowed schedule yields a target for testing compression.
        schedule=[dict(dispatch=t,arrival=t+1,slot=t+2) for t in range(h-2)]
        targets=[]
        for s,p in enumerate(paths):
            prior=0;row=[]
            for t in range(h):
                compatible=[q for q,path in enumerate(paths) if path[:t]==p[:t] and
                            all(path[r['slot']]==p[r['slot']] for r in schedule if r['arrival']<=t)]
                prior=max(prior,max(problem['bounds'][q][t] for q in compatible));row.append(prior)
            targets.append(row)
        result=compress_reports(problem,schedule,targets)
        variables,req=partition_problem(problem,schedule,targets)
        best=min(sum(bits) for bits in product([0,1],repeat=len(variables))
                 if all(any(bits[v] for v in r['cover']) for r in req))
        assert result['boundaries']==best
        partition_checks+=1
        # Earliest continuation dominates every alternative with fixed prefix.
        for prefix in [(0,),(1,),(0,0),(1,1)]:
            loss=reset_loss(case,case['caps'],prefix)
            for increments in product([0,1],repeat=h-len(prefix)):
                cum=list(prefix);prev=cum[-1]
                for u in increments:prev=min(n,prev+u);cum.append(prev)
                for s,p in enumerate(paths):
                    done=physical(cum,p,1)
                    assert all(a<=b for a,b in zip(done,loss['paths'][s]['attainable']))
            recovery_checks+=1
    # Greedy same-information benchmark: report the optimality gap and cold
    # decision time on the same delayed state used by the exact recursion.
    greedy_checks=0; greedy_times=[]; greedy_gaps=[]
    for i in range(10):
        n=2;h=6;paths=[]
        for _ in range(3):
            p=[0]*h
            for j in rng.integers(2,h,size=n):p[int(j)]+=1
            paths.append(p)
        case=dict(n=n,lag=1,delta=2,caps=[1]*h,service=paths,probabilities=['1/3']*3)
        problem=old.prepare(case); exact=JointDesign(problem,3,1,1).solve()
        start=perf_counter(); greedy=PairGreedy(problem,3,1,1).solve(); greedy_times.append(perf_counter()-start)
        greedy_gaps.append((greedy['weighted_waiting']-exact['weighted_waiting'],
                            greedy['weighted_queries']-exact['weighted_queries']))
        assert greedy['weighted_waiting']>=exact['weighted_waiting']
        greedy_checks+=1
    out=dict(passed=True,gate_optima=gate_checks,partition_optima=partition_checks,
             recovery_prefixes=recovery_checks,seed=2026102829)
    out.update(greedy_checks=greedy_checks,greedy_max_waiting_gap=max(x[0] for x in greedy_gaps),
               greedy_max_query_gap=max(x[1] for x in greedy_gaps),
               greedy_max_seconds=max(greedy_times))
    save(ROOT/'results/trb_six_20261028/theory_smoke.json',out)
    print(json.dumps(out),flush=True)

if __name__=='__main__':smoke()
