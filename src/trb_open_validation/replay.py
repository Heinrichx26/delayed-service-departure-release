"""Counterfactual airport replay with explicit information and saved plans.

The physical evaluator here is independent of the optimization replay. It
also evaluates plans that miss service on a held-out calendar.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from functools import lru_cache
import hashlib
import gzip
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trb_open_validation.bts_events import ROOT, DATA, local_to_utc
from trb_release.gate_model import GateCase
from trb_release.risk_frontier import PreservingPanel
from trb_method_search.hinge_model import solve

OUT = ROOT / "results/revision_20261027/trb_strengthening/open_validation/replay"
DESIGN = DATA / "replay_design_v3.json"


def read_design():
    return json.loads(DESIGN.read_text(encoding="utf-8"))


@lru_cache(maxsize=6)
def day_data(airport, day):
    folder = DATA / "month_panel" / f"{airport}_{day}"
    if not folder.exists():
        folder = DATA / "smoke" / f"{airport}_{day}"
    required = {"record_id","tail","origin","destination","origin_timezone","destination_timezone",
                "actual_out_minute","actual_in_minute","wheels_on_minute","wheels_off_minute",
                "actual_out_local_date","actual_in_local_date","gate_return_source_flag","single_departure_proxy_excluded"}
    month = DATA / "month_panel/airport_month" / f"{airport}_{day[:4]}.csv.gz"
    events = month_events(airport,day[:4],tuple(sorted(required)))
    intervals = pd.read_csv(folder / "recorded_tail_intervals.csv.gz", keep_default_na=False).to_dict("records")
    return events, intervals


@lru_cache(maxsize=2)
def month_events(airport,year,required):
    path = DATA / "month_panel/airport_month" / f"{airport}_{year}.csv.gz"
    if not path.exists():
        raise FileNotFoundError("Whole-month prefix interface required; day buckets omit overnight residents")
    return pd.read_csv(path,keep_default_na=False,usecols=lambda c:c in required).to_dict("records")


def epoch_at(events, airport, day, hour):
    event = next(e for e in events if e["origin"] == airport or e["destination"] == airport)
    tz = event["origin_timezone"] if event["origin"] == airport else event["destination_timezone"]
    return int(local_to_utc(datetime.fromisoformat(day).replace(hour=hour), ZoneInfo(tz)).timestamp()) // 60


def flag(e):
    value = e.get("single_departure_proxy_excluded", e.get("gate_return_source_flag", False))
    return value is True or str(value).lower() in ("true", "1", "1.0")


def cohort(events, airport, tau):
    return [e for e in events if e["origin"] == airport and
            int(e["actual_out_minute"]) <= tau < int(e["wheels_off_minute"]) and not flag(e)]


def histogram(times, tau, H, delta):
    output = [0] * H
    for minute in times:
        offset = int(minute) - tau
        if 0 <= offset < H * delta:
            output[offset // delta] += 1
    return tuple(output)


def background(events, airport, tau, operator):
    # Reconstruct from the observed prefix, without future-tail matching.
    state = {}
    for event in events:
        for field, place, priority in (("actual_out_minute", None, 1), ("actual_in_minute", event["destination"], 0)):
            minute = int(event[field])
            if minute <= tau and (minute,priority) > state.get(event["tail"], (-10**20,-1,None))[:2]:
                state[event["tail"]] = (minute, priority, place)
    full = [(minute,place) for minute,priority,place in state.values() if place == airport]
    if operator == "prefix_residence":
        return len(full)
    if operator == "arrival_20min":
        return sum(tau < minute+20 for minute,place in full)
    raise ValueError(operator)


@lru_cache(maxsize=2)
def network_context(airport,year):
    path = DATA / "month_panel/airport_month" / f"{airport}_{year}_context.csv.gz"
    return pd.read_csv(path,usecols=["tail","destination","actual_out_minute","actual_in_minute"],
                       keep_default_na=False)


@lru_cache(maxsize=1024)
def context_background(airport,year,tau):
    frame = network_context(airport,year)
    outs = frame.loc[frame.actual_out_minute <= tau,["tail","actual_out_minute"]].copy()
    outs.columns = ["tail","minute"];outs["place"]="";outs["priority"]=1
    ins = frame.loc[frame.actual_in_minute <= tau,["tail","actual_in_minute","destination"]].copy()
    ins.columns=["tail","minute","place"];ins["priority"]=0
    prefix = pd.concat([outs,ins],ignore_index=True).sort_values(["minute","priority"])
    latest = prefix.drop_duplicates("tail",keep="last")
    residents = latest[latest.place == airport]
    return {"prefix_residence":len(residents),"arrival_20min":int((residents.minute > tau-20).sum())}


def selected_requests(events, airport, tau, H, delta, q10, operator):
    values = []
    for e in events:
        if e["destination"] != airport or flag(e):
            continue
        on, inn = int(e["wheels_on_minute"]), int(e["actual_in_minute"])
        if inn <= tau:
            continue
        value = {"wheels_on": on, "recorded_gate_in": inn,
                 "training_q10_clipped": min(inn, on + q10)}[operator]
        value = max(tau,value)
        if value < tau+H*delta:
            values.append((e["record_id"],value))
    return values


def requests(events, airport, tau, H, delta, q10, operator):
    return histogram([minute for rid,minute in selected_requests(events,airport,tau,H,delta,q10,operator)],tau,H,delta)


def independent_replay(N, G, arrivals, service, release, delta, costs, lag_slots=0):
    """Conserve individual counts, including all incomplete plans."""
    assert len(arrivals) == len(service) == len(release)
    initial = N
    waiting_gate, surface, transit, arrived, admitted, departed = N, 0, 0, 0, 0, 0
    gate_minutes = queue_minutes = transit_minutes = arrival_minutes = 0
    completed, rows = [], []
    for t, (a, c, u) in enumerate(zip(arrivals, service, release)):
        assert isinstance(u, int) and 0 <= u <= waiting_gate
        waiting_gate -= u
        matured = release[t-lag_slots] if t>=lag_slots else 0
        transit += u-matured
        surface += matured
        physical_capacity = service[t-lag_slots] if t>=lag_slots else 0
        served = min(surface, physical_capacity)
        surface -= served
        departed += served
        arrived += a
        enter = min(arrived - admitted, G - waiting_gate - admitted)
        assert enter >= 0
        admitted += enter
        waiting = arrived - admitted
        assert waiting_gate + admitted <= G
        assert initial == waiting_gate + surface + transit + departed
        gate_minutes += delta * waiting_gate
        queue_minutes += delta * surface
        transit_minutes += delta * transit
        arrival_minutes += delta * waiting
        completed.append(departed)
        rows.append(dict(slot=t, held=waiting_gate, surface=surface, in_transit=transit, arrival_wait=waiting,
                         admitted=admitted, completed=departed))
    cq, ca, cg = costs
    return dict(queue_minutes=queue_minutes, arrival_minutes=arrival_minutes,transit_minutes=transit_minutes,
                surface_minutes=queue_minutes+transit_minutes,gate_minutes=gate_minutes,
                total_minutes=queue_minutes+transit_minutes+arrival_minutes+gate_minutes,
                waiting_minutes=queue_minutes+arrival_minutes+gate_minutes,
                weighted_loss=cq*(queue_minutes+transit_minutes)+ca*arrival_minutes+cg*gate_minutes,
                completed=completed, terminal_unserved=waiting_gate+surface+transit,
                terminal_arrival_backlog=arrived-admitted, rows=rows)


@lru_cache(maxsize=8)
def training_quantile(airport, year):
    design = read_design()
    days = [f"{year}-01-{d:02d}" for d in design["training_days"]]
    taxi = [];taxi_out=[]
    for day in days:
        events, intervals = day_data(airport,day)
        taxi.extend(int(e["actual_in_minute"])-int(e["wheels_on_minute"])
                    for e in events if e["destination"] == airport and e["actual_in_local_date"] == day and not flag(e))
        taxi_out.extend(int(e["wheels_off_minute"])-int(e["actual_out_minute"])
                        for e in events if e["origin"] == airport and e["actual_out_local_date"] == day and not flag(e))
    q10 = int(np.quantile(taxi, .1, method="lower"))
    return q10,len(taxi),int(np.quantile(taxi_out,.1,method="lower")),len(taxi_out)


@lru_cache(maxsize=32)
def training(airport, year, hour):
    design = read_design()
    q10,taxi_n,q10_out,out_n = training_quantile(airport,year)
    lag=int(np.ceil(q10_out/design["slot_minutes"]))
    days = [f"{year}-01-{d:02d}" for d in design["training_days"]]
    donors = []
    for day in days:
        events, ints = day_data(airport,day)
        tau = epoch_at(events, airport, day, hour)
        departures = cohort(events, airport, tau)
        offsets = sorted(int(e["wheels_off_minute"]) - tau for e in departures)
        donors.append(dict(day=day, offsets=offsets,
                           arrivals=requests(events, airport, tau, design["horizon_minutes"]//design["slot_minutes"]+lag,
                                             design["slot_minutes"], q10, "training_q10_clipped")))
    return donors, q10, taxi_n, q10_out, out_n


def build_panel(N, G, test_service, donors, mode, costs, design):
    H, delta = len(test_service), design["slot_minutes"]
    virtual_H=design["horizon_minutes"]//delta
    lag=H-virtual_H
    cases, donor_counts = [], Counter()
    for donor in donors:
        if mode == "announced_calendar":
            service = test_service
        else:
            if not donor["offsets"]:
                donor_counts["empty_cohort"] += 1
                continue
            # Whole-curve empirical quantiles preserve donor temporal dependence.
            count = len(donor["offsets"])
            offsets = [donor["offsets"][min(count-1, int((i+.5)*count/N))] for i in range(N)]
            service = histogram(offsets, 0, virtual_H, delta)+(0,)*lag
            if sum(service) < N:
                donor_counts["incomplete_calendar"] += 1
                return None, dict(donor_counts)
        cq, ca, cg = costs
        cases.append(GateCase(N, G, (N,)*H, tuple(donor["arrivals"][:H]), tuple(service), cq, ca, cg, delta))
        donor_counts["used"] += 1
    if not cases:
        return None, dict(donor_counts)
    return PreservingPanel(tuple(cases), (1/len(cases),)*len(cases)), dict(donor_counts)


def plans(panel, design, lag=0):
    base, H, _, _, U, _, _, _, _ = panel.arrays()
    early = [base.departures]+[0]*(H-1)
    latest_h = np.asarray(U, dtype=int)
    latest = (np.r_[base.departures, latest_h[:-1]]-latest_h).astype(int).tolist()
    result = dict(earliest=dict(releases=early), latest_preserving=dict(releases=latest))
    S = len(panel.cases)
    for name, alpha in (("mean", 0.), ("cvar", design["alpha"]), ("worst", 1-1/S)):
        answer = solve(panel, alpha=alpha, shared=True, integral=True,
                       time_limit=design["solver_time_limit_seconds"])
        if "holding" in answer:
            h = np.asarray(answer["holding"], dtype=int)
            answer["releases"] = (np.r_[base.departures, h[:-1]]-h).astype(int).tolist()
        result[name] = answer
    # An independent lagged execution must agree with every training loss.
    from trb_release.gate_model import weighted_risk
    for name, answer in result.items():
        if "releases" not in answer:
            continue
        physical=[independent_replay(c.departures,c.gates,c.arrivals,c.service,answer["releases"],
                                    c.slot_minutes,(c.queue_cost,c.arrival_cost,c.gate_cost),lag)
                  for c in panel.cases]
        assert all(p["terminal_unserved"]==0 for p in physical)
        if "objective" in answer:
            criterion="mean" if name=="mean" else ("worst" if name=="worst" else "cvar")
            actual=weighted_risk([p["weighted_loss"] for p in physical],panel.probabilities,criterion,design["alpha"])
            shift=base.queue_cost*lag*base.slot_minutes*base.departures
            assert abs(actual-answer["objective"]-shift)<1e-5
            answer["physical_training_objective"]=actual
        answer["independent_lagged_training_verified"]=True
    return result


def evaluate_window(airport, day, hour, inventory, donors, q10, taxi_n, design,
                    mode, fraction=1., occupancy="prefix_residence", costs=(2,3,0), scope="lower",
                    q10_out=0,out_n=0,transit_mode="training_q10"):
    events, intervals = day_data(airport, day)
    tau = epoch_at(events, airport, day, hour)
    ready = cohort(events, airport, tau)
    N = len(ready)
    virtual_H, delta = design["horizon_minutes"]//design["slot_minutes"], design["slot_minutes"]
    lag=int(np.ceil(q10_out/delta)) if transit_mode=="training_q10" else 0
    H=virtual_H+lag
    B = context_background(airport,day[:4],tau)[occupancy]
    G = int(np.floor(inventory*fraction))-B
    key = f"{airport}_{day}_h{hour}_{mode}_f{fraction}_{occupancy}_{scope}_{transit_mode}_c{'-'.join(map(str,costs))}"
    record = dict(key=key, airport=airport, day=day, hour=hour, mode=mode,
                  fraction=fraction, occupancy=occupancy, costs=list(costs), inventory_scope=scope, N=N, G=G,
                  nominal_inventory=inventory, reserved_background=B, q10_taxi_in=q10,
                  q10_taxi_out=q10_out,training_taxi_out_sample=out_n,lag_slots=lag,transit_mode=transit_mode,
                  training_taxi_sample=taxi_n, cohort_record_ids=[e["record_id"] for e in ready])
    if not N:
        return dict(record, status="empty_cohort")
    if G < N:
        return dict(record, status="insufficient_constructed_capacity")
    service = histogram([int(e["wheels_off_minute"]) for e in ready], tau, virtual_H, delta)+(0,)*lag
    record["service"] = service
    record["recorded_calendar_clears"] = sum(service) == N
    if sum(service) < N and mode == "announced_calendar":
        arrival = requests(events,airport,tau,H,delta,q10,"training_q10_clipped")
        descriptive = independent_replay(N,G,arrival,service,[N]+[0]*(H-1),delta,costs,lag)
        return dict(record, status="incomplete_recorded_calendar", incomplete_count=N-sum(service),
                    descriptive_earliest=descriptive)
    panel, donor_counts = build_panel(N,G,service,donors,mode,costs,design)
    record["donor_counts"] = donor_counts
    if panel is None:
        return dict(record, status="unavailable_training_calendar")
    payload=json.dumps(dict(scenarios=[c.to_dict() for c in panel.cases],probabilities=panel.probabilities,
                            donor_days=[d["day"] for d in donors],mode=mode,lag_slots=lag),
                       sort_keys=True,separators=(",",":")).encode("utf-8")
    input_folder=DATA/"replay_inputs_v3";input_folder.mkdir(parents=True,exist_ok=True)
    input_path=input_folder/(key+".json.gz")
    if input_path.exists():
        assert gzip.decompress(input_path.read_bytes())==payload,"Frozen model input changed"
    else:
        input_path.write_bytes(gzip.compress(payload,mtime=0))
    record["planning_input_sha256"]=hashlib.sha256(payload).hexdigest()
    answers = plans(panel, design,lag)
    arrivals = {op: requests(events,airport,tau,H,delta,q10,op) for op in design["request_operators"]}
    primary = arrivals["training_q10_clipped"]
    # A separate audit measures conflict with the counterfactual stay rule.
    incoming_ids = {rid for rid,minute in selected_requests(events,airport,tau,H,delta,q10,"training_q10_clipped")}
    turnover = sum(r["incoming_record_id"] in incoming_ids and int(r["end_minute"]) < tau+H*delta for r in intervals)
    baseline = independent_replay(N,G,primary,service,answers["earliest"]["releases"],delta,costs,lag)
    evaluations = {}
    for name, answer in answers.items():
        if "releases" not in answer:
            evaluations[name] = dict(solver=answer, status="solver_no_plan")
            continue
        by_operator = {}
        for operator, arrival in arrivals.items():
            actual = independent_replay(N,G,arrival,service,answer["releases"],delta,costs,lag)
            deficits = [ref-got for ref,got in zip(baseline["completed"],actual["completed"])]
            assert min(deficits) >= 0
            actual.update(completion_preserving=all(v==0 for v in deficits),
                          completion_deficit_slots=sum(v>0 for v in deficits),
                          maximum_completion_deficit=max(deficits))
            by_operator[operator] = actual
        if mode == "announced_calendar":
            assert all(v["completion_preserving"] for v in by_operator.values())
        evaluations[name] = dict(status="evaluated", solver=answer, outcomes=by_operator)
    linked=sum(r["incoming_record_id"] in incoming_ids for r in intervals)
    return dict(record,status="evaluated",arrivals=arrivals,turnover_within_window=turnover,
                linked_request_count=linked,unresolved_next_gate_out=len(incoming_ids)-linked,
                arrival_count=sum(primary),evaluations=evaluations,
                trained_on=[d["day"] for d in donors],
                information_audit=dict(holdout_arrivals_used_in_planning=False,
                    holdout_service_used_in_planning=mode=="announced_calendar",
                    service_announced_by_experiment=mode=="announced_calendar",
                    retrospective_record_quality_screening=True))


def save_record(record, phase):
    folder = OUT/phase/"cases"
    folder.mkdir(parents=True,exist_ok=True)
    raw = json.dumps(record,indent=2,allow_nan=False)
    (folder/(record["key"]+".json")).write_text(raw,encoding="utf-8")


def summarize(records):
    statuses = dict(Counter(r["status"] for r in records))
    results = {}
    for mode in ("announced_calendar", "forecast_calendar"):
        selected = [r for r in records if r["mode"]==mode and r["status"]=="evaluated"]
        for method in ("earliest","latest_preserving","mean","cvar","worst"):
            rows = [(r,r["evaluations"][method]["outcomes"]["training_q10_clipped"]) for r in selected
                    if r["evaluations"][method]["status"]=="evaluated"]
            if not rows:
                continue
            metrics = {field:float(np.mean([out[field]/r["N"] for r,out in rows]))
                       for field in ("queue_minutes","arrival_minutes","gate_minutes","transit_minutes",
                                     "surface_minutes","total_minutes","weighted_loss")}
            metrics.update(windows=len(rows),completion_violations=sum(not o["completion_preserving"] for r,o in rows),
                           terminal_incomplete=sum(o["terminal_unserved"]>0 for r,o in rows))
            results[f"{mode}:{method}"] = metrics
    return dict(status_counts=statuses,methods=results,
                interpretation="Equal-weight windows, per reset departure. Capacity/cost sensitivity configurations are kept separate by the exporter.")
