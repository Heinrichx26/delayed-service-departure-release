"""Frozen complete public calendar panel; smoke precedes the full run."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from math import ceil
import os
from pathlib import Path
from time import perf_counter

from robust_reports import RobustReports, SearchLimit, prepare
from deadline_cover import solve as cover_solve, metric_smoke
from nominal_failure import stress

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/trb_error_20261030"
DATA=ROOT/"data/trb_error_20261030"

def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8")

def one(task):
    job,delay,budget=task
    source=ROOT/Path(job["input_path"].replace("\\", "/"))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==job["input_sha256"]
    c=read(source)
    caps=(ceil(c["n"]/job["divisor"]),)*len(c["caps"])
    p=prepare(c["service"],c["n"],caps,c["lag"])
    row=dict(case_id=job["case_id"],airport=job["airport"],day=job["day"],
             capacity=job["capacity"],n=c["n"],delay=delay,budget=budget,
             input_path=job["input_path"],input_sha256=job["input_sha256"])
    lead=c["lag"]+2
    for name,gamma,mode,independent in (("accurate",0,"exact",False),
            ("robust",1,"exact",False),("independent",1,"exact",True),
            ("greedy",1,"greedy",False),("none",1,"none",False)):
        try:
            m=RobustReports(p,lead,delay,budget,gamma,mode=mode,
                            independent_errors=independent,seconds_limit=8,state_limit=100000)
            value=m.solve();check=m.verify()
            row[name]=dict(optimal=mode=="exact",**value,checks=check["physical_checks"],
                           worst_minutes_per_departure=value["value"]*c["delta"]/c["n"],
                           nominal_mean_minutes=sum(x["waiting"] for x in check["paths"])/len(p["paths"])*c["delta"]/c["n"],
                           max_requests=max(x["worst_requests"] for x in check["paths"]),
                           nominal_mean_requests=sum(x["requests"] for x in check["paths"])/len(p["paths"]))
            if name=="accurate":
                failure=stress(m,p)
                row["accurate_deployed_with_error"]={k:v for k,v in failure.items() if k!="records"}
        except SearchLimit as exc:
            row[name]=dict(resolved=False,reason=str(exc))
    row["fixed_target"]={str(g):cover_solve(p,lead,delay,budget,g) for g in (0,1)}
    if all("value" in row[x] for x in ("robust","independent","greedy","none")):
        assert row["robust"]["value"]<=min(row[x]["value"] for x in ("independent","greedy","none"))
        assert row["accurate"]["value"]<=row["robust"]["value"]
    return row

def run(stage):
    OUT.mkdir(parents=True,exist_ok=True)
    assert read(OUT/"smoke.json")["passed"]
    jobs=read(ROOT/"data/trb_latency_20261028/panel_design.json")["jobs"]
    if stage=="public_smoke":
        jobs=[next(j for j in jobs if j["airport"]==a and j["capacity"]=="N4")
              for a in ("DEN","PHX")]
    else:
        assert read(OUT/"public_smoke.json")["passed"]
    tasks=[(j,d,1) for j in jobs for d in (1,2)]
    design=dict(stage=stage,jobs=jobs,delays=[1,2],request_limit=1,error_allowance=1,
                error_unit="sum of absolute errors in reported two-minute takeoff counts",
                operational_interpretation="Sensitivity setting, not a measured controller error distribution",
                workers=2,models=["accurate","robust","independent","greedy","none"])
    save(DATA/(stage+"_design.json"),design)
    start=perf_counter();rows=[]
    os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,r in enumerate(pool.map(one,tasks),1):
            rows.append(r)
            if stage=="public_smoke" or i%20==0:
                print(json.dumps(dict(completed=i,total=len(tasks),seconds=perf_counter()-start,
                    last={k:r[k].get("value",r[k].get("reason")) for k in design["models"]})),flush=True)
                save(OUT/(stage+"_progress.json"),rows)
    resolved=all(all("value" in r[x] for x in design["models"]) for r in rows)
    result=dict(passed=resolved,stage=stage,conditions=len(rows),seconds=perf_counter()-start,
                metric_checks=metric_smoke() if stage=="public_smoke" else None,rows=rows)
    save(OUT/(stage+".json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["public_smoke","full"])
    run(ap.parse_args().stage)
