"""Receding-horizon request selection with no-request continuation values."""
from functools import lru_cache
import json
from pathlib import Path

from robust_reports import RobustReports,prepare
from deadline_cover import solve,metric_smoke

class RolloutReports(RobustReports):
    def __init__(self,*args,depth=1,**kwargs):
        super().__init__(*args,mode="exact",**kwargs)
        self.depth=depth
        self.continuation=RobustReports(*args,mode="none",**kwargs)
        self.look=lru_cache(None)(self.look)

    def look(self,t,e,pending,queried,u,depth):
        if t==self.h:
            return 0
        if depth==0:
            return self.continuation.value(t,e,pending,queried,u)
        z=self.release(t,e,u)
        return min(max(z-r+self.look(t+1,en,pn,an,z,depth-1)
                       for _,_,en,pn,an,r in self.transitions(t,e,pending,queried,batch))
                   for batch in super().batches(t,e,pending,queried))

    def value(self,t,e,pending,queried,u):
        if t==self.h:
            return 0
        if self.value.cache_info().misses>self.state_limit:
            from robust_reports import SearchLimit
            raise SearchLimit("rollout state allowance reached")
        state=(t,e,pending,queried,u)
        z=self.release(t,e,u)
        if z==self.n:
            self.choice[state]=()
            return max(sum(self.n-self.p["refs"][s][k] for k in range(t,self.h))
                       for s in self.members(e))
        def score(batch):
            return (max(z-r+self.look(t+1,en,pn,an,z,self.depth-1)
                        for _,_,en,pn,an,r in self.transitions(t,e,pending,queried,batch)),
                    len(batch),batch)
        batch=min(super().batches(t,e,pending,queried),key=score)
        self.choice[state]=batch
        return max(z-r+self.value(t+1,en,pn,an,z)
                   for _,_,en,pn,an,r in self.transitions(t,e,pending,queried,batch))

def check_example():
    p=prepare(((0,0,0,0,2,0,0,0),(0,0,0,0,0,2,0,0)),2,(2,)*8,2)
    results={}
    for name,options in (("exact",{}),("independent",dict(independent_errors=True)),
                         ("none",dict(mode="none"))):
        m=RobustReports(p,5,1,1,1,**options)
        results[name]=m.solve()["value"]
        assert m.verify()["passed"]
    for k in (1,2):
        m=RolloutReports(p,5,1,1,1,depth=k)
        results["rollout"+str(k)]=m.solve()["value"]
        assert m.verify()["passed"]
    target=solve(p,5,1,1,1)
    assert results["exact"]==4 and results["none"]==6 and results["independent"]==6
    assert target["feasible"] and target["requests"]==2
    result=dict(passed=True,values=results,fixed_target=target,distance_checks=metric_smoke())
    out=Path(__file__).resolve().parents[2]/"results/trb_error_20261030/complementarity.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    check_example()
