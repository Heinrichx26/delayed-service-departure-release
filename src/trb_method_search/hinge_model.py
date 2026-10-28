"""Exact shared-breakpoint formulation and a matching sparse full epigraph.

This is a candidate structural reformulation. It is not, by itself, a claim
of methodological novelty. Both alternatives use the same sparse builder,
solver, tolerances, physical replay, and timing definitions.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from trb_release.gate_model import replay,weighted_risk


def solve(panel,alpha=.9,shared=True,integral=True,time_limit=15,box=None):
    start=perf_counter()
    base,H,_,A,U,p,b,beta,gamma=panel.arrays(); S=len(p)
    assert 0<=alpha<1 and time_limit>0
    descriptors={};scenario_map=[]
    for s in range(S):
        key=(float(b[s]),tuple(int(v) for v in A[s])) if shared else s
        if key not in descriptors:
            descriptors[key]=[s,0.]
        descriptors[key][1]+=p[s]
        scenario_map.append(key)
    records=list(descriptors.values());K=len(records)
    objective=[0.]*H+[1.]+[q/(1-alpha) for s,q in records]
    eta=H; lower=[0.]*(H+K+1);upper=[float(u) for u in U]+[np.inf]*(K+1)
    if box is not None:
        low,high=box
        lower[:H]=np.maximum(0,low).tolist();upper[:H]=np.minimum(U,high).tolist()
    integer=[int(integral)]*H+[0]*(K+1)
    row_idx,col_idx,values,rhs=[],[],[],[]
    def constraint(coef,right):
        rid=len(rhs);rhs.append(float(right))
        for j,v in coef.items():
            if abs(v)>1e-15:
                row_idx.append(rid);col_idx.append(j);values.append(float(v))
    for t in range(H):
        coef={t:-1.}
        if t:
            coef[t-1]=1.
            constraint(coef,base.release_caps[t]);constraint({j:-v for j,v in coef.items()},0)
        else:
            constraint(coef,base.release_caps[t]-base.departures)
            constraint({0:1.},base.departures)
    hinges={}
    for k,(s,_) in enumerate(records):
        coef={t:float(beta) for t in range(H)};coef[eta]=-1.;coef[H+1+k]=-1.
        constant=float(b[s])
        for t in range(H):
            threshold=float(base.gates-A[s,t])
            if shared and threshold>=upper[t]-1e-12:
                continue
            if shared and threshold<=lower[t]+1e-12:
                coef[t]+=gamma;constant-=gamma*threshold;continue
            key=(t,threshold) if shared else (s,t)
            if key not in hinges:
                idx=len(objective);hinges[key]=idx
                objective.append(0.);lower.append(0.);upper.append(np.inf);integer.append(0)
                constraint({t:1.,idx:-1.},threshold)
            coef[hinges[key]]=coef.get(hinges[key],0.)+gamma
        constraint(coef,-constant)
    matrix=coo_matrix((values,(row_idx,col_idx)),shape=(len(rhs),len(objective))).tocsc()
    build_seconds=perf_counter()-start
    remaining=max(.001,time_limit-build_seconds)
    res=milp(np.array(objective),integrality=np.array(integer),bounds=Bounds(lower,upper),
             constraints=LinearConstraint(matrix,-np.inf,np.array(rhs)),
             options={'time_limit':remaining,'mip_rel_gap':0.})
    record=dict(shared=shared,integral=integral,status=int(res.status),message=res.message,
                variables=len(objective),constraints=len(rhs),matrix_nonzeros=len(values),
                hinges=len(hinges),loss_signatures=K,input_scenarios=S,
                build_seconds=build_seconds,total_seconds=perf_counter()-start,
                bound_scope='restricted box' if box is not None else 'global',alpha=alpha)
    if res.x is None:
        return record
    h=res.x[:H]
    holding=np.rint(h).astype(int) if integral else h
    losses=b+beta*sum(holding)+gamma*np.maximum(A-base.gates+holding,0).sum(axis=1)
    actual=weighted_risk(losses.tolist(),p.tolist(),'cvar',alpha)
    assert abs(actual-res.fun)<1e-5
    valid_bound=float(res.mip_dual_bound) if integral else (float(res.fun) if res.status==0 else None)
    record.update(objective=float(actual),holding=holding.tolist(),
                  bestbound=valid_bound,
                  nodes=int(res.mip_node_count) if integral else 0,
                  solver_gap=float(res.mip_gap) if integral else 0.)
    if integral:
        release=np.r_[base.departures,holding[:-1]]-holding
        physical=[replay(c,[int(u) for u in release]) for c in panel.cases]
        assert all(c['feasible'] for c in physical)
        for true,derived in zip(physical,losses): assert abs(true['loss']-derived)<1e-7
        record['physical_verified']=True
    return record
