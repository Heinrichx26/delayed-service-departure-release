"""One-pass monthly UTC reconstruction, then indexed public-tail day panels.

Each source ZIP is streamed once in a phase. All network records are rebuilt
once, persisted in compressed JSON lines and a compact SQLite index. Per-day
tail linkage reuses the original five-source-day conservative linking rule.
Optional gate-return fields are preserved and flagged, without imputing events.
No optimizer is imported or invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import date, timedelta
import gzip
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import zipfile

import pandas as pd

from trb_open_validation.bts_events import (DATA, FIELDS, ROOT, SOURCE_HASHES, TARGETS, digest,
    event_from_row, freeze_timezone_source, link_tail_intervals)
from trb_open_validation.tail_smoke import SMOKE, independent_check, occupancy_trace, unit_smoke

OUT = ROOT / "results/revision_20261027/trb_strengthening/open_validation"
OPTIONAL = ("FirstDepTime", "TotalAddGTime", "LongestAddGTime")
BASE_HASH = "af133f772f8aca74235369c1c49114a91efdc3330ad2a0a2d4d1ed3fa12ffc6c"


def extra_fields(row, event, available):
    flags = {"raw_"+name: row.get(name, "") for name in OPTIONAL}
    flags["return_fields_available"] = all(name in available for name in OPTIONAL)
    positive = False
    for name in ("TotalAddGTime", "LongestAddGTime"):
        try:
            positive = positive or float(row.get(name, "") or 0) > 0
        except ValueError:
            pass
    flags["first_departure_clock_reported"] = bool(row.get("FirstDepTime", "").strip())
    flags["additional_ground_time_positive"] = positive
    flags["gate_return_source_flag"] = positive or flags["first_departure_clock_reported"]
    flags["clock_reconstruction_excluded"] = event["status"] != "valid"
    flags["single_departure_proxy_excluded"] = event["status"] != "valid" or flags["gate_return_source_flag"]
    reasons = []
    if event["status"] != "valid": reasons.append(event["status"])
    if flags["gate_return_source_flag"]: reasons.append("reported_first_departure_or_additional_ground_time")
    flags["single_departure_proxy_exclusion_reasons"] = "|".join(reasons)
    return flags


def dump_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    else:
        pd.DataFrame(columns=columns or ["record_id"]).to_csv(path, index=False, compression="gzip")


def candidates(phase):
    if phase == "smoke": return [(year, airport, day) for year, airport, day in SMOKE]
    original = [(year, airport, f"{year}-01-{day:02d}") for year in (2024, 2025)
                for day in (24, 27, 31) for airport in TARGETS]
    supplements = [(year, airport, f"{year}-01-{day:02d}") for year in (2024, 2025)
                   for day in range(3, 32) for airport in ("DEN", "PHX", "SEA")]
    return list(dict.fromkeys(original+supplements))


def freeze_scope(chosen, phase):
    """Version the explicit training supplement to the original 72 dates."""
    if phase == "smoke": return None
    scope = dict(version=2, date="2026-10-27", original_scope="12 airports x Jan24/27/31 x 2024/2025 = 72 days",
                 supplement="DEN/PHX/SEA x Jan03..31 x 2024/2025 = 174 days; 18 overlap original scope",
                 union_airport_days=len(chosen), original_airport_days=72, supplement_airport_days=174,
                 unique_additional_airport_days=156,
                 prespecified_roles=dict(training=list(range(3, 19)), validation=list(range(19, 24)),
                                         test=[24, 27, 31], supplemental_unassigned=[25, 26, 28, 29, 30]),
                 scope_statement="New version explicitly adds development dates; the original 72-date manifest remains unchanged",
                 panels=[dict(year=y, airport=a, local_date=d) for y, a, d in chosen])
    path = DATA/"month_panel_scope_v2.json"
    if path.exists(): assert json.loads(path.read_text(encoding="utf-8")) == scope
    else: path.write_text(json.dumps(scope, indent=2), encoding="utf-8")
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path), version=2)


def prepare_db(path):
    if path.exists(): raise RuntimeError(f"Refusing to overwrite existing phase index: {path}")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE events (record_id TEXT PRIMARY KEY, source_row_index INTEGER, flight_date TEXT, tail TEXT, origin TEXT, destination TEXT, status TEXT, out_min INTEGER, in_min INTEGER, out_local TEXT, in_local TEXT, out_utc TEXT, in_utc TEXT, return_flag INTEGER, raw_first TEXT, raw_total TEXT, raw_longest TEXT)")
    return connection


def reconstruct(year, chosen, phase_root, zones):
    path = ROOT / f"data/open_source/raw/bts_on_time_{year}_01.zip"
    cache = phase_root / "network"; cache.mkdir(parents=True, exist_ok=True)
    database = cache / f"{year}_01.sqlite"
    db = prepare_db(database)
    all_statuses, return_statuses, scanned, independently_checked = Counter(), Counter(), 0, 0
    day_buckets = {(a, d): [] for y, a, d in chosen if y == year}
    excluded_buckets = {(a, d): [] for y, a, d in chosen if y == year}
    valid_dest_counts = Counter(); started = perf_counter(); first_chunk = None
    network_path = cache / f"{year}_01_events.jsonl.gz"
    with zipfile.ZipFile(path) as archive, gzip.open(network_path, "wt", encoding="utf-8", newline="\n") as output:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(member) as raw:
            required = set(FIELDS)
            iterator = pd.read_csv(raw, usecols=lambda col: col in required or col in OPTIONAL,
                                   dtype=str, keep_default_na=False, chunksize=25000)
            for chunk_number, chunk in enumerate(iterator, 1):
                available = list(chunk.columns)
                if not required.issubset(available): raise ValueError("Required public fields are missing")
                records = chunk.to_dict("records"); compact = []
                for offset, raw_row in enumerate(records):
                    index = scanned+offset
                    event = event_from_row(raw_row, year, index, zones)
                    event.update(extra_fields(raw_row, event, available))
                    all_statuses[event["status"]] += 1
                    if event["gate_return_source_flag"]: return_statuses[event["status"]] += 1
                    if event["status"] == "valid":
                        independent_check(raw_row, event); independently_checked += 1
                        valid_dest_counts[(event["destination"], event["actual_in_local_date"])] += 1
                    output.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False)+"\n")
                    compact.append((event["record_id"], index, event["flight_date"][:10], event["tail"], event["origin"], event["destination"],
                                    event["status"], event.get("actual_out_minute"), event.get("actual_in_minute"),
                                    event.get("actual_out_local_date"), event.get("actual_in_local_date"),
                                    event.get("actual_out_utc"), event.get("actual_in_utc"), int(event["gate_return_source_flag"]),
                                    event["raw_FirstDepTime"], event["raw_TotalAddGTime"], event["raw_LongestAddGTime"]))
                    if event["status"] == "valid":
                        keys = {(event["origin"], event["actual_out_local_date"]), (event["destination"], event["actual_in_local_date"])}
                        for key in keys.intersection(day_buckets): day_buckets[key].append(event)
                    else:
                        for key in {(event["origin"], event["flight_date"][:10]), (event["destination"], event["flight_date"][:10])}.intersection(excluded_buckets):
                            excluded_buckets[key].append(event)
                db.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", compact); db.commit()
                scanned += len(chunk)
                elapsed = perf_counter()-started
                if first_chunk is None:
                    # Known prior audit row counts provide an explicit estimate.
                    expected = {2024: 547271, 2025: 539747}[year]
                    first_chunk = dict(rows=len(chunk), seconds=elapsed, projected_month_seconds=elapsed*expected/scanned)
                    print(json.dumps(dict(phase="first_chunk_estimate", year=year, **first_chunk)), flush=True)
                elif chunk_number % 4 == 0:
                    print(json.dumps(dict(phase="event_reconstruction", year=year, rows=scanned, seconds=elapsed)), flush=True)
                del records, compact
    db.execute("CREATE INDEX date_idx ON events(flight_date)")
    db.execute("CREATE INDEX tail_idx ON events(tail,flight_date)")
    db.execute("CREATE INDEX airport_idx ON events(origin,destination)"); db.commit()
    for (airport, day), rows in day_buckets.items():
        folder = phase_root / f"{airport}_{day}"
        dump_csv(folder/"utc_events.csv.gz", rows)
        dump_csv(folder/"exclusions_source_date.csv.gz", excluded_buckets[(airport, day)])
    info = dict(source=str(path.relative_to(ROOT)), source_sha256=SOURCE_HASHES[year],
                source_hash_basis="Prior independent raw ZIP SHA-256; no repeated raw-file hash", member=member,
                required_fields=list(FIELDS), optional_fields_present=[name for name in OPTIONAL if name in available],
                raw_rows=scanned, statuses=dict(all_statuses), independently_verified_valid_events=independently_checked,
                return_flag_statuses=dict(return_statuses),
                cache_paths=dict(events=str(network_path.relative_to(ROOT)), index=str(database.relative_to(ROOT))),
                event_output_sha256=digest(network_path), first_chunk_estimate=first_chunk,
                reconstruct_seconds=perf_counter()-started)
    return db, info


def reuse_reconstruction(year, chosen, phase_root, smoke):
    """Reuse the already audited one-pass monthly cache; no source ZIP reread."""
    info = next(item for item in smoke["month_sources"] if str(year) in item["source"])
    network_path, database = (ROOT/info["cache_paths"][key] for key in ("events", "index"))
    if digest(network_path) != info["event_output_sha256"]:
        raise ValueError("Audited monthly event cache hash changed")
    buckets = {(a, d): [] for y, a, d in chosen if y == year}
    excluded = {(a, d): [] for y, a, d in chosen if y == year}
    started, count = perf_counter(), 0
    with gzip.open(network_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line); count += 1
            if event["status"] == "valid":
                keys = {(event["origin"], event["actual_out_local_date"]), (event["destination"], event["actual_in_local_date"])}
                for key in keys.intersection(buckets): buckets[key].append(event)
            else:
                keys = {(event["origin"], event["flight_date"][:10]), (event["destination"], event["flight_date"][:10])}
                for key in keys.intersection(excluded): excluded[key].append(event)
    assert count == info["raw_rows"]
    for (airport, day), rows in buckets.items():
        folder = phase_root/f"{airport}_{day}"
        dump_csv(folder/"utc_events.csv.gz", rows)
        dump_csv(folder/"exclusions_source_date.csv.gz", excluded[(airport, day)])
    print(json.dumps(dict(phase="reuse_audited_network_cache", year=year, records=count, seconds=perf_counter()-started)), flush=True)
    return sqlite3.connect(database), dict(info, reused_from_smoke=True,
                                          full_panel_distribution_seconds=perf_counter()-started)


def indexed_window(db, zones, airport, day):
    central = date.fromisoformat(day)
    lower, upper = (central-timedelta(days=2)).isoformat(), (central+timedelta(days=2)).isoformat()
    statement = """SELECT e.record_id,e.source_row_index,e.flight_date,e.tail,e.origin,e.destination,e.status,
        e.out_min,e.in_min,e.out_local,e.in_local,e.out_utc,e.in_utc,e.return_flag,e.raw_first,e.raw_total,e.raw_longest
        FROM events e WHERE e.flight_date BETWEEN ? AND ? AND e.tail != '' AND e.tail IN
        (SELECT DISTINCT tail FROM events WHERE flight_date BETWEEN ? AND ? AND (origin=? OR destination=?))"""
    records, by_id = [], {}
    for row in db.execute(statement, (lower, upper, lower, upper, airport, airport)):
        key, index, flightdate, tail, origin, destination, status, outminute, inminute, outlocal, inlocal, oututc, inutc, returned, first, total, longest = row
        event = dict(record_id=key, source_row_index=index, flight_date=flightdate, tail=tail, origin=origin, destination=destination,
                     status=status, actual_out_minute=outminute, actual_in_minute=inminute,
                     actual_out_local_date=outlocal, actual_in_local_date=inlocal,
                     actual_out_utc=oututc, actual_in_utc=inutc, gate_return_source_flag=bool(returned),
                     raw_FirstDepTime=first, raw_TotalAddGTime=total, raw_LongestAddGTime=longest,
                     destination_timezone=zones.get(destination))
        records.append(event); by_id[key] = event
    missing = db.execute("SELECT COUNT(*) FROM events WHERE flight_date BETWEEN ? AND ? AND tail='' AND (origin=? OR destination=?)",
                         (lower, upper, airport, airport)).fetchone()[0]
    return records, by_id, dict(source_flightdate_window=[lower, upper], selected_tail_network_rows=len(records), missing_tail_window_records=missing)


def day_panel(db, zones, airport, day, phase_root):
    events, by_id, window = indexed_window(db, zones, airport, day)
    intervals, coverage = link_tail_intervals(events, airport, day)
    for row in intervals:
        incoming, outgoing = by_id[row["incoming_record_id"]], by_id[row["outgoing_record_id"]]
        row.update(incoming_gate_return_source_flag=incoming["gate_return_source_flag"],
                   outgoing_gate_return_source_flag=outgoing["gate_return_source_flag"],
                   single_departure_proxy_excluded=incoming["gate_return_source_flag"] or outgoing["gate_return_source_flag"])
    folder = phase_root / f"{airport}_{day}"
    dump_csv(folder/"recorded_tail_intervals.csv.gz", intervals)
    dump_csv(folder/"tail_link_window_audit.csv.gz", [dict(record_id=row["record_id"], tail=row["tail"], status=row["status"],
               flight_date=row["flight_date"], origin=row["origin"], destination=row["destination"],
               gate_return_source_flag=row["gate_return_source_flag"]) for row in events])
    trace = occupancy_trace(intervals, zones, airport, day)
    dump_csv(folder/"recorded_linked_occupancy.csv.gz", trace)
    linked_ids = {row["incoming_record_id"] for row in intervals}
    # Every target arrival receives an explicit linking status, including linked
    # records and the conservative whole-window barrier used by the smoke.
    blocked = {row["tail"] for row in events if row["status"] not in ("valid", "cancelled", "missing_tail")}
    tails = {}
    for row in events:
        if row["status"] == "valid": tails.setdefault(row["tail"], []).append(row)
    audit = []
    for tail, rows in tails.items():
        rows.sort(key=lambda row: (row["actual_out_minute"], row["record_id"]))
        for i, row in enumerate(rows):
            if row["destination"] != airport or row["actual_in_local_date"] != day: continue
            next_row = rows[i+1] if i+1 < len(rows) else None
            if tail in blocked: reason = "tail_has_unusable_record_in_window"
            elif next_row is None: reason = "no_next_record_in_window"
            elif next_row["origin"] != airport: reason = "next_record_origin_elsewhere"
            elif next_row["actual_out_minute"] < row["actual_in_minute"]: reason = "overlapping_clock_sequence"
            else: reason = "linked"
            audit.append(dict(record_id=row["record_id"], tail=tail, linking_status=reason,
                              next_record_id=next_row["record_id"] if next_row else None,
                              gate_return_source_flag=row["gate_return_source_flag"]))
    assert dict(Counter(row["linking_status"] for row in audit)) == {k: v for k, v in coverage["reasons"].items() if v}
    dump_csv(folder/"arrival_link_audit.csv.gz", audit)
    public = pd.read_csv(folder/"utc_events.csv.gz", keep_default_na=False)
    result = dict(key=f"{airport}_{day}", airport=airport, local_date=day, window=window,
                  utc_event_rows=len(public), event_return_flags=int(public["gate_return_source_flag"].sum()) if len(public) else 0,
                  single_departure_proxy_eligible_events=int((~public["single_departure_proxy_excluded"]).sum()) if len(public) else 0,
                  linking=coverage, interval_return_flags=sum(row["single_departure_proxy_excluded"] for row in intervals),
                  peak_recorded_linked_occupancy=max((row["recorded_linked_tail_count"] for row in trace), default=0),
                  interface_paths={name: str((folder/name).relative_to(ROOT)) for name in
                      ("utc_events.csv.gz", "recorded_tail_intervals.csv.gz", "arrival_link_audit.csv.gz", "tail_link_window_audit.csv.gz",
                       "exclusions_source_date.csv.gz", "recorded_linked_occupancy.csv.gz")},
                  file_sha256={name: digest(folder/name) for name in ("utc_events.csv.gz", "recorded_tail_intervals.csv.gz", "arrival_link_audit.csv.gz")})
    return result


def verify_original_smoke(results, phase_root):
    original = json.loads((OUT/"tail_smoke.json").read_text(encoding="utf-8"))
    previous = {row["key"]: row for row in original["cases"]}
    for result in results:
        if result["key"] not in previous: continue
        prior = previous[result["key"]]
        assert result["linking"] == prior["linking"]
        assert result["utc_event_rows"] == prior["valid_target_local_date_events"]
        assert result["peak_recorded_linked_occupancy"] == prior["peak_linked_recorded_occupancy"]
        old = pd.read_csv(ROOT/prior["output_paths"]["events"], keep_default_na=False)
        old = old[((old["origin"] == result["airport"]) & (old["actual_out_local_date"] == result["local_date"])) |
                  ((old["destination"] == result["airport"]) & (old["actual_in_local_date"] == result["local_date"]))]
        new = pd.read_csv(phase_root/result["key"]/"utc_events.csv.gz", keep_default_na=False)
        fields = ("record_id", "actual_out_minute", "actual_in_minute", "wheels_off_minute", "wheels_on_minute")
        assert old[list(fields)].sort_values("record_id").to_dict("records") == new[list(fields)].sort_values("record_id").to_dict("records")
    return True


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    if digest(Path(__file__).with_name("bts_events.py")) != BASE_HASH:
        raise RuntimeError("Base event semantics changed; review explicitly before rebuilding")
    if args.phase == "full":
        existing = OUT/"month_panel_smoke.json"
        if not existing.exists() or not json.loads(existing.read_text(encoding="utf-8"))["original_smoke_consistent"]:
            raise RuntimeError("Month-panel smoke must pass before full")
    zones, timezone_source = freeze_timezone_source(); unit = unit_smoke(zones)
    chosen = candidates(args.phase); scope = freeze_scope(chosen, args.phase)
    phase_root = DATA / ("month_panel_smoke" if args.phase == "smoke" else "month_panel")
    phase_root.mkdir(parents=True, exist_ok=True); started = perf_counter(); month_info, results = [], []
    print(json.dumps(dict(phase=args.phase, estimated_raw_records=1087018,
                         pre_run_estimate_seconds=[60, 180], method="First chunk measures the rate before the remaining monthly reconstruction")), flush=True)
    for year in (2024, 2025):
        if args.phase == "full":
            smoke = json.loads((OUT/"month_panel_smoke.json").read_text(encoding="utf-8"))
            db, info = reuse_reconstruction(year, chosen, phase_root, smoke)
        else:
            db, info = reconstruct(year, chosen, phase_root, zones)
        month_info.append(info)
        for y, airport, day in chosen:
            if y != year: continue
            result = day_panel(db, zones, airport, day, phase_root); results.append(result)
            print(json.dumps(dict(phase="day_panel", key=result["key"], events=result["utc_event_rows"],
                                  linking=result["linking"]["linked_fraction"])), flush=True)
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)"); db.close()
    consistent = verify_original_smoke(results, phase_root)
    manifest = dict(date="2026-10-27", phase=args.phase, original_smoke_consistent=consistent, unit_smoke=unit,
                    scope_version=scope,
                    timezone_source=timezone_source, month_sources=month_info, airport_days=len(results), panels=results,
                    source_hashes={name: digest(Path(__file__).with_name(name)) for name in ("month_panel.py", "bts_events.py")},
                    return_field_policy="Raw FirstDepTime/TotalAddGTime/LongestAddGTime retained; source gate-return flag marks a separate exclusion for a single-departure proxy, while valid UTC events remain available",
                    exclusion_date_scope="Invalid clocks lack certified actual dates; exclusions_source_date uses source FlightDate only",
                    january31_scope="Only January source months; right-edge next-leg censoring is retained explicitly",
                    total_seconds=perf_counter()-started)
    (phase_root/"interface_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (OUT/f"month_panel_{args.phase}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("panels", "month_sources", "timezone_source")}, ensure_ascii=False))


if __name__ == "__main__": main()
