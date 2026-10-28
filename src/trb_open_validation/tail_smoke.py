"""Two-airport-day public-data smoke before broader event reconstruction."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import pandas as pd

from trb_open_validation.bts_events import (DATA, ROOT, TARGETS, clock_minutes, digest, event_from_row,
                                          freeze_timezone_source, link_tail_intervals, local_to_utc,
                                          read_smoke_window)

OUT = ROOT / "results/revision_20261027/trb_strengthening/open_validation"
SMOKE = ((2024, "ATL", "2024-01-24"), (2025, "LAX", "2025-01-27"))


def unit_smoke(zones):
    row = dict(FlightDate="2024-01-24", Origin="LAX", Dest="JFK", Tail_Number="SYNTHETIC_TEST",
               Reporting_Airline="TEST", Flight_Number_Reporting_Airline="1", CRSDepTime="2300", CRSArrTime="700",
               DepTime="2330", ArrTime="715", WheelsOff="2350", WheelsOn="705", DepDelay="30", ArrDelay="15",
               CRSElapsedTime="300", ActualElapsedTime="285", TaxiOut="20", TaxiIn="10", AirTime="255",
               Cancelled="0", Diverted="0")
    value = event_from_row(row, 2024, -1, zones)
    assert value["status"] == "valid"
    assert value["scheduled_out_utc"] == "2024-01-25T07:00Z"
    assert value["scheduled_in_utc"] == "2024-01-25T12:00Z"
    assert value["actual_in_local_date"] == "2024-01-25"
    midnight = dict(row, Origin="ATL", Dest="ATL", CRSDepTime="2400", CRSArrTime="100", DepTime="2400", ArrTime="100",
                    WheelsOff="10", WheelsOn="50", DepDelay="0", ArrDelay="0", CRSElapsedTime="60", ActualElapsedTime="60",
                    TaxiOut="10", TaxiIn="10", AirTime="40")
    canonical = event_from_row(midnight, 2024, -2, zones)
    assert canonical["status"] == "valid" and canonical["scheduled_out_utc"] == "2024-01-25T05:00Z"
    assert clock_minutes("2400") == 1440 and clock_minutes("0") == 0
    assert event_from_row(dict(row, ArrTime="716"), 2024, -3, zones)["status"] == "clock_mismatch"
    assert event_from_row(dict(row, Cancelled="1"), 2024, -4, zones)["status"] == "cancelled"
    assert event_from_row(dict(row, Diverted="1"), 2024, -5, zones)["status"] == "diverted"
    for naive in (datetime(2024, 3, 10, 2, 30), datetime(2024, 11, 3, 1, 30)):
        try:
            local_to_utc(naive, ZoneInfo("America/New_York"))
        except ValueError:
            pass
        else:
            raise AssertionError("DST ambiguity was silently accepted")
    return dict(passed=True, checks=["cross-timezone overnight calendar", "2400 next-day normalization",
                                   "clock perturbation rejection", "cancelled/diverted separation", "DST gap/ambiguity rejection"],
                synthetic_unit_records_excluded_from_public_outputs=True)


def independent_check(raw, event):
    """Rebuild gate-in through actual elapsed time, independent of ArrDelay."""
    scheduled_local = datetime.fromisoformat(raw["FlightDate"][:10])+timedelta(minutes=clock_minutes(raw["CRSDepTime"]))
    origin = ZoneInfo(event["origin_timezone"])
    actual_out = scheduled_local.replace(tzinfo=origin)+timedelta(minutes=int(float(raw["DepDelay"])))
    actual_in_epoch = int(actual_out.timestamp())//60+int(float(raw["ActualElapsedTime"]))
    assert actual_in_epoch == event["actual_in_minute"]
    assert event["actual_out_minute"]+int(float(raw["TaxiOut"])) == event["wheels_off_minute"]
    assert event["actual_in_minute"]-int(float(raw["TaxiIn"])) == event["wheels_on_minute"]


def occupancy_trace(intervals, zones, airport, local_date):
    start = local_to_utc(datetime.fromisoformat(local_date), ZoneInfo(zones[airport]))
    end = local_to_utc(datetime.fromisoformat(local_date)+timedelta(days=1), ZoneInfo(zones[airport]))
    lower, upper = int(start.timestamp())//60, int(end.timestamp())//60
    rows = []
    for minute in range(lower, upper, 5):
        count = sum(row["start_minute"] <= minute < row["end_minute"] for row in intervals)
        rows.append(dict(utc_minute=minute, recorded_linked_tail_count=count))
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--unit-only", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    zones, source = freeze_timezone_source()
    unit = unit_smoke(zones)
    if args.unit_only:
        print(json.dumps(unit)); return
    DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    common_dates = [dict(airport=airport, local_date=f"{year}-01-{day:02d}") for year in (2024, 2025)
                    for day in (24, 27, 31) for airport in TARGETS]
    manifest = dict(version=1, created="2026-10-27", design="UTC-event and recorded-tail interval smoke only",
                    complete_candidate_airport_dates=common_dates,
                    smoke_cases=[dict(year=y, airport=a, local_date=d) for y, a, d in SMOKE],
                    pad_days=2, selected_fields_only=True,
                    raw_hashes_reused_from_prior_read_only_audit=True,
                    interpretation="Public reported events and consecutive-tail same-airport intervals; stand and ready unobserved")
    manifest_path = DATA / "smoke_manifest_v1.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cases = []
    for year, airport, day in SMOKE:
        key = f"{airport}_{day}"
        raw, missing, info = read_smoke_window(year, airport, day)
        events = [event_from_row(row, year, row["source_row_index"], zones) for row in raw]
        statuses = dict(Counter(row["status"] for row in events))
        valid = [event for event in events if event["status"] == "valid"]
        by_index = {int(row["source_row_index"]): row for row in raw}
        for event in valid:
            independent_check(by_index[event["source_row_index"]], event)
        intervals, coverage = link_tail_intervals(events, airport, day)
        # Each tail contributes disjoint known intervals; no double counting.
        by_tail = {}
        for row in sorted(intervals, key=lambda r: (r["tail"], r["start_minute"])):
            assert row["turnaround_minutes"] >= 0
            assert row["start_minute"] >= by_tail.get(row["tail"], row["start_minute"])
            by_tail[row["tail"]] = row["end_minute"]
        trace = occupancy_trace(intervals, zones, airport, day)
        folder = DATA / "smoke" / key
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(valid).to_csv(folder/"utc_events.csv.gz", index=False, compression="gzip")
        pd.DataFrame(intervals).to_csv(folder/"recorded_tail_intervals.csv.gz", index=False, compression="gzip")
        pd.DataFrame(trace).to_csv(folder/"recorded_linked_occupancy.csv", index=False)
        exclusions = [row for row in events if row["status"] != "valid"]
        (folder/"exclusions.json").write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
        target_events = [e for e in valid if (e["origin"] == airport and e["actual_out_local_date"] == day)
                         or (e["destination"] == airport and e["actual_in_local_date"] == day)]
        record = dict(key=key, source=info, event_status_counts=statuses, valid_network_events=len(valid),
                      independent_event_reconstructions=len(valid), all_event_reconstructions_match=True,
                      valid_target_local_date_events=len(target_events),
                      target_arrivals_overnight_from_source_date=sum(e["destination"] == airport and
                          e["actual_in_local_date"] == day and e["arrival_date_shift"] != 0 for e in valid),
                      gate_in_date_shift_counts=dict(Counter(str(e["arrival_date_shift"]) for e in valid)),
                      missing_tail_window_records=len(missing), linking=coverage,
                      interval_duration_summary=dict(minimum=min((r["turnaround_minutes"] for r in intervals), default=None),
                                                     maximum=max((r["turnaround_minutes"] for r in intervals), default=None),
                                                     median=float(pd.Series([r["turnaround_minutes"] for r in intervals]).median()) if intervals else None),
                      peak_linked_recorded_occupancy=max((r["recorded_linked_tail_count"] for r in trace), default=0),
                      output_paths={"events": str((folder/"utc_events.csv.gz").relative_to(ROOT)),
                                    "intervals": str((folder/"recorded_tail_intervals.csv.gz").relative_to(ROOT)),
                                    "occupancy": str((folder/"recorded_linked_occupancy.csv").relative_to(ROOT))})
        cases.append(record)
        print(json.dumps({"case": key, "statuses": statuses, "linking": coverage}), flush=True)
    result = dict(date="2026-10-27", unit=unit, smoke_only=True, timezone_source=source,
                  manifest_sha256=digest(manifest_path), cases=cases, elapsed_seconds=perf_counter()-started,
                  source_hashes={name: digest(Path(__file__).with_name(name)) for name in ("bts_events.py", "tail_smoke.py")},
                  usability="Clock and tail-link smoke; excluded clocks and missing links retained; not an operational-effect estimate",
                  next_stage="Review mismatches and linking coverage before the 72-date reconstruction")
    (OUT/"tail_smoke.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False))


if __name__ == "__main__": main()
