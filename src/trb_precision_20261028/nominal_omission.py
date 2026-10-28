"""Optimize a nominal timetable with automatic omission of determined replies."""
from itertools import combinations
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import coo_matrix,vstack,csc_matrix
from band_design import old
from skip_known import skip_known


def optimize_omission(problem,lead,delay,budget,width,known_waiting=None,seconds=40):
    start=perf_counter()
    paths=problem['paths'];m=len(paths);h=problem['horizon'];t0=problem['t0'];n=problem['n']
    labels=[[v//width for v in p] for p in paths]
    first={(s,q):next((t for t in range(h) if paths[s][t]!=paths[q][t]),h)
           for s in range(m) for q in range(m)}
    requests=[(r,j) for r in range(t0,h-delay)
              for j in range(r+delay,min(h,r+lead+1))
              if any(first[s,q]>=r+delay and labels[s][j]!=labels[q][j]
                     for s,q in combinations(range(m),2))]
    k=len(requests)
    ykeys=[(s,a) for s in range(m) for a,(r,j) in enumerate(requests)
           if any(first[s,q]>=r and labels[s][j]!=labels[q][j] for q in range(m))]
    yidx={v:k+m*h+i for i,v in enumerate(ykeys)}
    nv=k+m*h+len(ykeys)
    rr=[];cc=[];vv=[];lower=[];upper=[]
    def constraint(terms,lo,hi):
        row=len(lower)
        for c,v in terms:rr.append(row);cc.append(c);vv.append(v)
        lower.append(lo);upper.append(hi)
    def uidx(s,t):return k+s*h+t
    for r in range(t0,h):
        constraint([(a,1) for a,(v,j) in enumerate(requests) if v==r],0,budget)
    for j in range(h):
        constraint([(a,1) for a,(r,v) in enumerate(requests) if v==j],0,1)
    for s in range(m):
        for t in range(h):
            constraint([(uidx(s,t),1)]+([(uidx(s,t-1),-1)] if t else []),
                       0,problem['caps'][t])
    for s,q in combinations(range(m),2):
        for t in range(t0,min(h,first[s,q]+1)):
            terms=[(a,-n) for a,(r,j) in enumerate(requests)
                   if r+delay<=t and labels[s][j]!=labels[q][j]]
            constraint([(uidx(s,t),1),(uidx(q,t),-1)]+terms,-np.inf,0)
            constraint([(uidx(q,t),1),(uidx(s,t),-1)]+terms,-np.inf,0)
    for s,a in ykeys:
        r,j=requests[a];yi=yidx[s,a]
        constraint([(yi,1),(a,-1)],-np.inf,0)
        for q in range(m):
            if first[s,q]<r or labels[s][j]==labels[q][j]:continue
            distinctions=[(v,1) for v,(dispatch,slot) in enumerate(requests)
                          if dispatch+delay<=r and labels[s][slot]!=labels[q][slot]]
            constraint([(yi,1),(a,-1)]+distinctions,0,np.inf)
    mat=coo_matrix((vv,(rr,cc)),shape=(len(lower),nv)).tocsc()
    lb=np.r_[np.zeros(k),np.array(problem['bounds']).reshape(-1),np.zeros(len(ykeys))]
    ub=np.r_[np.ones(k),np.full(m*h,n),np.ones(len(ykeys))]
    for s in range(m):
        for t,u in enumerate(problem['prefix']):lb[uidx(s,t)]=ub[uidx(s,t)]=u
    first_objective=np.r_[np.zeros(k),np.repeat(problem['weights'],h),np.zeros(len(ykeys))]
    for s in range(m):
        for t in range(t0):first_objective[uidx(s,t)]=0
    second=np.zeros(nv)
    for s,a in ykeys:second[yidx[s,a]]=problem['weights'][s]
    retries=[]
    def solve(objective,matrix,lo,hi,stage):
        ans=milp(objective,integrality=np.ones(nv),bounds=Bounds(lb,ub),
            constraints=LinearConstraint(matrix,lo,hi),
            options=dict(time_limit=seconds,mip_rel_gap=0))
        if ans.status==2:
            retries.append(dict(stage=stage,status=int(ans.status),message=ans.message))
            ans=milp(objective,integrality=np.ones(nv),bounds=Bounds(lb,ub),
                constraints=LinearConstraint(matrix,lo,hi),
                options=dict(time_limit=seconds,mip_rel_gap=0,presolve=False))
        return ans
    constant=sum(w*sum(ref[t0:]) for w,ref in zip(problem['weights'],problem['references']))
    if known_waiting is None:
        ans=solve(first_objective,mat,lower,upper,'waiting')
        if ans.status!=0:
            return dict(optimal=False,status=int(ans.status),message=ans.message,stage='waiting')
        first_value=int(round(ans.fun))
    else:first_value=known_waiting+constant
    mat2=vstack([mat,csc_matrix(first_objective.reshape(1,-1))],format='csc')
    lo,hi=np.r_[lower,first_value],np.r_[upper,first_value]
    ans=solve(second,mat2,lo,hi,'executed requests')
    if ans.status!=0:
        return dict(optimal=False,status=int(ans.status),message=ans.message,
                    stage='executed requests',seconds=perf_counter()-start)
    solution=np.rint(ans.x).astype(int)
    assert np.all(mat2@solution>=lo-1e-7) and np.all(mat2@solution<=hi+1e-7)
    schedule=[dict(dispatch=r,arrival=r+delay,slot=j)
              for a,(r,j) in enumerate(requests) if solution[a]]
    fixed_paths=[]
    for s,path in enumerate(paths):
        release=solution[k+s*h:k+(s+1)*h].tolist()
        done,inventory,_=old.base.physical(release,path,n,problem['lag'])
        assert done==problem['references'][s]
        fixed_paths.append(dict(s=s,queries=schedule,release=release,completed=done,
            post_release=problem['case']['delta']*sum(inventory[t0:])/n))
    count,executed_paths=skip_known(problem,fixed_paths,lead,delay,budget,width)
    expected=int(round(ans.fun))/problem['denominator']
    assert abs(count-expected)<1e-9
    for row in executed_paths:
        actual={(q['dispatch'],q['slot']) for q in row['queries']}
        for a,(r,j) in enumerate(requests):
            yy=solution[yidx[row['s'],a]] if (row['s'],a) in yidx else 0
            assert yy==int((r,j) in actual)
    return dict(optimal=True,weighted_waiting=first_value-constant,
                waiting=(first_value-constant)*problem['case']['delta']/problem['denominator']/n,
                weighted_queries=int(round(ans.fun)),expected_queries=expected,
                nominal_count=len(schedule),schedule=schedule,paths=executed_paths,
                seconds=perf_counter()-start,variables=nv,constraints=len(lower),
                retries=retries)
