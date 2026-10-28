"""Integer optimization of fixed requests and completion-preserving releases."""
from itertools import combinations
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import coo_matrix,vstack,csc_matrix
from joint_design import old

def fixed_joint(problem,lead,delay,budget,seconds=30):
    start=perf_counter();paths=problem['paths'];m=len(paths);h=problem['horizon'];t0=problem['t0'];n=problem['n']
    requests=[(r,j) for r in range(t0,h-delay) for j in range(r+delay,min(h,r+lead+1))
              if len({p[j] for p in paths})>1]
    k=len(requests);nv=k+m*h
    rr=[];cc=[];vv=[];lower=[];upper=[]
    def constraint(terms,lo,hi):
        row=len(lower)
        for c,v in terms:
            rr.append(row);cc.append(c);vv.append(v)
        lower.append(lo);upper.append(hi)
    idx=lambda s,t:k+s*h+t
    for r in range(t0,h):
        constraint([(a,1) for a,(v,j) in enumerate(requests) if r==v],0,budget)
    for j in range(h):
        constraint([(a,1) for a,(r,v) in enumerate(requests) if j==v],0,1)
    for s in range(m):
        for t in range(h):
            constraint([(idx(s,t),1)]+([(idx(s,t-1),-1)] if t else []),0,problem['caps'][t])
    for s,q in combinations(range(m),2):
        first=next((j for j in range(h) if paths[s][j]!=paths[q][j]),h)
        for t in range(t0,min(h,first+1)):
            distinguishes=[(a,-n) for a,(r,j) in enumerate(requests)
                           if r+delay<=t and paths[s][j]!=paths[q][j]]
            constraint([(idx(s,t),1),(idx(q,t),-1)]+distinguishes,-np.inf,0)
            constraint([(idx(q,t),1),(idx(s,t),-1)]+distinguishes,-np.inf,0)
    mat=coo_matrix((vv,(rr,cc)),shape=(len(lower),nv)).tocsc()
    lb=np.r_[np.zeros(k),np.array(problem['bounds']).reshape(-1)]
    ub=np.r_[np.ones(k),np.full(m*h,n)]
    for s in range(m):
        for t,u in enumerate(problem['prefix']):lb[idx(s,t)]=ub[idx(s,t)]=u
    objective=np.r_[np.zeros(k),np.repeat(problem['weights'],h)]
    for s in range(m):
        for t in range(t0):objective[idx(s,t)]=0
    feasible=np.r_[np.zeros(k),np.tile(old.base.earliest(n,problem['caps'],problem['prefix']),m)]
    assert np.all(feasible>=lb) and np.all(feasible<=ub)
    test=mat@feasible
    assert np.all(test>=np.array(lower)-1e-8) and np.all(test<=np.array(upper)+1e-8)
    result=milp(objective,integrality=np.ones(nv),bounds=Bounds(lb,ub),
                constraints=LinearConstraint(mat,lower,upper),options=dict(time_limit=seconds,mip_rel_gap=0))
    retries=[]
    if result.status==2:
        retries.append(dict(stage='waiting',status=int(result.status),message=result.message))
        result=milp(objective,integrality=np.ones(nv),bounds=Bounds(lb,ub),
                    constraints=LinearConstraint(mat,lower,upper),
                    options=dict(time_limit=seconds,mip_rel_gap=0,presolve=False))
    if result.status!=0:return dict(optimal=False,stage='waiting',status=int(result.status),message=result.message,seconds=perf_counter()-start,retries=retries)
    first_value=int(round(result.fun))
    second=np.r_[np.ones(k),np.zeros(m*h)]
    mat2=vstack([mat,csc_matrix(objective.reshape(1,-1))],format='csc')
    second_result=milp(second,integrality=np.ones(nv),bounds=Bounds(lb,ub),
                constraints=LinearConstraint(mat2,np.r_[lower,first_value],np.r_[upper,first_value]),
                options=dict(time_limit=seconds,mip_rel_gap=0))
    if second_result.status==2:
        retries.append(dict(stage='requests',status=int(second_result.status),message=second_result.message))
        second_result=milp(second,integrality=np.ones(nv),bounds=Bounds(lb,ub),
                    constraints=LinearConstraint(mat2,np.r_[lower,first_value],np.r_[upper,first_value]),
                    options=dict(time_limit=seconds,mip_rel_gap=0,presolve=False))
    if second_result.status!=0:return dict(optimal=False,stage='requests',status=int(second_result.status),message=second_result.message,seconds=perf_counter()-start,retries=retries)
    solution=np.rint(second_result.x).astype(int)
    check=mat2@solution
    assert np.all(check>=np.r_[lower,first_value]-1e-7) and np.all(check<=np.r_[upper,first_value]+1e-7)
    queries=[dict(dispatch=r,arrival=r+delay,slot=j) for a,(r,j) in enumerate(requests) if solution[a]]
    rows=[]
    for s,path in enumerate(paths):
        release=solution[k+s*h:k+(s+1)*h].tolist()
        done,inventory,_=old.base.physical(release,path,n,problem['lag'])
        assert done==problem['references'][s]
        rows.append(dict(s=s,queries=queries,release=release,completed=done,
                         post_release=problem['case']['delta']*sum(inventory[t0:])/n))
    constant=sum(w*sum(ref[t0:]) for w,ref in zip(problem['weights'],problem['references']))
    return dict(optimal=True,weighted_waiting=first_value-constant,
                waiting=(first_value-constant)*problem['case']['delta']/problem['denominator']/n,
                count=len(queries),schedule=queries,paths=rows,seconds=perf_counter()-start,
                variables=nv,constraints=len(lower),retries=retries)
