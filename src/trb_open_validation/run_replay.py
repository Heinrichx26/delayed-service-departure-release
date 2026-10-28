"""Frozen-phase runner. Export and plotting live in a separate module."""
import argparse
from collections import Counter
import hashlib
import json
from time import perf_counter

from trb_open_validation.replay import (DATA,OUT,DESIGN,read_design,training,evaluate_window,save_record,summarize)


def configurations(design,inventory,phase):
    base=dict(fraction=1.,occupancy="prefix_residence",costs=(2,3,0),scope="lower",transit_mode="training_q10")
    values=[base]
    if phase=="test":
        values += [dict(base,costs=tuple(c)) for c in design["costs"] if c!=[2,3,0]]
        values += [dict(base,fraction=f) for f in design["capacity_fractions"] if f!=1.]
        values += [dict(base,occupancy="arrival_20min")]
        values += [dict(base,transit_mode="zero_transit")]
        if inventory["nominal_capacity_lower"] != inventory["nominal_capacity_upper"]:
            values += [dict(base,scope="upper")]
    return values


def run(phase):
    started=perf_counter();design=read_design()
    if phase!="smoke":
        smoke=json.loads((OUT/"smoke"/"summary.json").read_text(encoding="utf-8"))
        assert smoke["execution_passed"],"Smoke must pass before panel execution"
    if phase=="test":
        assert (OUT/"validation"/"summary.json").exists(),"Validation precedes test"
    inventory=json.loads((DATA/"public_inventory/inventory_counts.json").read_text(encoding="utf-8"))
    sources=[r for r in inventory["records"] if r["usable_for_replay"]]
    records=[];source_exclusions=[r for r in inventory["records"] if not r["usable_for_replay"]]
    for item in sources:
        airport,year=item["airport"],int(item["panel_month"][:4])
        for hour in ([12] if phase=="smoke" else design["local_hours"]):
            donors,q10,n,q10_out,out_n=training(airport,year,hour)
            days=[19] if phase=="smoke" else design[phase+"_days"]
            for day in days:
                for cfg in configurations(design,item,phase):
                    nominal=item["nominal_capacity_"+cfg["scope"]]
                    for mode in design["modes"]:
                        row=evaluate_window(airport,f"{year}-01-{day:02d}",hour,nominal,donors,q10,n,design,mode,
                                            q10_out=q10_out,out_n=out_n,**cfg)
                        row["inventory_source_ids"]=item["source_ids"]
                        row["inventory_date"]=item["inventory_date"]
                        save_record(row,phase);records.append(row)
            print(json.dumps(dict(phase=phase,airport=airport,year=year,hour=hour,
                                  completed=len(records),statuses=dict(Counter(r['status'] for r in records)))),flush=True)
    base=[r for r in records if r["fraction"]==1. and r["occupancy"]=="prefix_residence" and
          r["costs"]==[2,3,0] and r["inventory_scope"]=="lower" and r["transit_mode"]=="training_q10"]
    summary=dict(phase=phase,execution_passed=True,design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),
                 source_code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [__import__('pathlib').Path(__file__),__import__('pathlib').Path(__file__).with_name('replay.py')]},
                 inventory_sha256=hashlib.sha256((DATA/"public_inventory/inventory_counts.json").read_bytes()).hexdigest(),
                 record_count=len(records),all_status_counts=dict(Counter(r["status"] for r in records)),
                 source_exclusions=source_exclusions,primary=summarize(base),
                 total_seconds=perf_counter()-started,unfavorable_outcomes_retained=True)
    folder=OUT/phase;folder.mkdir(parents=True,exist_ok=True)
    (folder/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=["smoke","validation","test"],required=True)
    run(parser.parse_args().phase)
