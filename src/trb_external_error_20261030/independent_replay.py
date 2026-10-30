"""Independent physical replay, including whole-period request ceilings."""
from functools import lru_cache

def stress(model, problem, error_allowance=1):
    assert model.price==0
    paths,n,h,caps,lag=(problem[k] for k in ("paths","n","h","caps","lag"))
    records=[]
    for actual,path in enumerate(paths):
        @lru_cache(None)
        def walk(t,spent,pending,queried,previous,tail,done,error_used):
            if t==h:
                return (0,0,1,0,0,0)
            fallback=not any(e>=0 for e in spent)
            if fallback:
                batch=()
                z=min(n,previous+caps[t])
            else:
                state=(t,spent,pending,queried,previous)
                model.value(*state)
                batch=model.choice[state]
                z=model.release(t,spent,previous)
            assert len(batch)<=model.limit
            if hasattr(model,'total_requests'):
                assert (queried>>h)+len(batch)<=model.total_requests
            arrived=(tail[0] if len(tail)==lag else 0) if lag else z
            current_done=min(arrived,done+path[t])
            gap=problem["refs"][actual][t]-current_done
            assert gap>=0
            newtail=(tail+(z,))[-lag:] if lag else ()
            combined=pending+tuple((t+model.delay,j) for j in batch)
            due=tuple(j for d,j in combined if d==t+1)
            assert len(due)<=1
            future=tuple((d,j) for d,j in combined if d>t+1)
            q=queried|sum(1<<j for j in batch)
            q&=~((1<<min(h,t+1+model.delay))-1)
            if hasattr(model,'total_requests'):
                used=(queried>>h)+len(batch)
                q=(q&((1<<h)-1))|(used<<h)
            possibilities=[((),error_used)] if not due else [(tuple(path[j] for j in due),error_used)]
            if due and error_used<error_allowance:
                j=due[0]
                possibilities += [((path[j]+residual,),error_used+abs(residual))
                    for residual in range(-(error_allowance-error_used),error_allowance-error_used+1)
                    if residual and 0<=path[j]+residual<=n]
            outcomes=[]
            for y,used in possibilities:
                updated=[]
                for s,old in enumerate(spent):
                    residuals=[abs(paths[s][j]-v) for j,v in zip(due,y)]
                    error=(max(residuals,default=0) if model.independent_errors else old+sum(residuals))
                    updated.append((0 if model.independent_errors else error)
                                   if old>=0 and paths[s][t]==path[t] and error<=model.gamma else -1)
                e=tuple(updated)
                child=walk(t+1,e,future,q,z,newtail,current_done,used)
                outcomes.append((gap+child[0],max(gap,child[1]),child[2],
                                 child[2] if gap else child[3],z-current_done+child[4],len(batch)+child[5]))
            return (max(x[0] for x in outcomes),max(x[1] for x in outcomes),
                    sum(x[2] for x in outcomes),sum(x[3] for x in outcomes),
                    max(x[4] for x in outcomes),max(x[5] for x in outcomes))
        result=walk(0,(0,)*len(paths),(),0,0,(),0,0)
        records.append(dict(s=actual,worst_shortfall_aircraftslots=result[0],
                            worst_shortfall_count=result[1],error_histories=result[2],
                            histories_with_shortfall=result[3],worst_postrelease=result[4],worst_requests=result[5]))
    return dict(calendars=len(paths),calendars_with_loss=sum(x["worst_shortfall_count"]>0 for x in records),
                max_shortfall=max(x["worst_shortfall_count"] for x in records),
                histories=sum(x["error_histories"] for x in records),
                histories_with_shortfall=sum(x["histories_with_shortfall"] for x in records),records=records)
