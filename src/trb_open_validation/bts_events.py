"""Public BTS clock reconstruction and recorded-tail interval utilities.

This module builds UTC events from reported delays and scheduled elapsed
times. Every available local clock is checked independently modulo one day.
Recorded tail intervals do not identify a physical stand or true ready event.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
import zipfile

import airportsdata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/trb_open_validation"
TARGETS = ("ATL", "CLT", "DEN", "DFW", "JFK", "LAS", "LAX", "MCO", "MIA", "ORD", "PHX", "SEA")
SOURCE_HASHES = {2024: "fe089b45523f9d4ac0ccd0e176a543d274e914ee5ae5bd384dbaafc7dc06ebfd",
                 2025: "868387dcedaef1b8d8392e608e642219fde50d594c3cf2094aec858167629ccf"}
FIELDS = ("FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Tail_Number", "Origin", "Dest",
          "CRSDepTime", "CRSArrTime", "DepTime", "ArrTime", "WheelsOff", "WheelsOn",
          "DepDelay", "ArrDelay", "CRSElapsedTime", "ActualElapsedTime", "TaxiOut", "TaxiIn", "AirTime",
          "Cancelled", "Diverted")
UTC = timezone.utc


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def freeze_timezone_source():
    """Preserve the installed, public airportsdata snapshot and license."""
    folder = DATA / "sources/timezones"
    folder.mkdir(parents=True, exist_ok=True)
    distribution = importlib.metadata.distribution("airportsdata")
    source = Path(airportsdata.__file__).with_name("airports.csv")
    target = folder / f"airportsdata_{distribution.version}.csv"
    license_file = next(distribution.locate_file(p) for p in distribution.files if str(p).endswith("licenses/LICENSE"))
    license_target = folder / "airportsdata_LICENSE"
    for origin, dest in ((source, target), (Path(license_file), license_target)):
        if dest.exists():
            if dest.read_bytes() != origin.read_bytes():
                raise ValueError("Timezone snapshot already exists with different content")
        else:
            shutil.copyfile(origin, dest)
    tzdata_version = importlib.metadata.version("tzdata")
    metadata = dict(package="airportsdata", version=distribution.version,
                    source_repository="https://github.com/mborsetti/airportsdata",
                    package_registry="https://pypi.org/project/airportsdata/",
                    source_sha256=digest(target), license_sha256=digest(license_target),
                    tzdata_version=tzdata_version,
                    extraction="IATA airport identifier to IANA timezone; installed public package snapshot",
                    historical_use="Current mapping applied to January 2024/2025; reconstructed historical clocks independently checked")
    (folder / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    zones = {}
    with target.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("iata") and row.get("tz"):
                zones[row["iata"]] = row["tz"]
    return zones, metadata


def integer(value, field):
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing:{field}")
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"noninteger:{field}")
    return int(numeric)


def clock_minutes(value):
    clock = integer(value, "clock")
    if clock == 2400:
        return 1440
    hour, minute = divmod(clock, 100)
    if clock < 0 or hour > 23 or minute > 59:
        raise ValueError("invalid:clock")
    return 60*hour+minute


def local_to_utc(naive, zone):
    candidates = []
    for fold in (0, 1):
        localized = naive.replace(tzinfo=zone, fold=fold)
        value = localized.astimezone(UTC)
        if value.astimezone(zone).replace(tzinfo=None) == naive and value not in candidates:
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError("ambiguous_or_nonexistent:scheduled_origin_clock")
    return candidates[0]


def timestamp(value):
    return value.isoformat(timespec="minutes").replace("+00:00", "Z")


def epoch_minute(value):
    return int(value.timestamp())//60


def event_from_row(row, year, index, zones):
    """Return one certified clock record or an explicit exclusion reason."""
    base = dict(source_year=year, source_row_index=int(index), record_id=f"{year}:{index}",
                flight_date=row.get("FlightDate", ""), origin=row.get("Origin", ""), destination=row.get("Dest", ""),
                tail=row.get("Tail_Number", "").strip(), airline=row.get("Reporting_Airline", ""),
                flight_number=row.get("Flight_Number_Reporting_Airline", ""))
    try:
        if integer(row["Cancelled"], "Cancelled"):
            return dict(base, status="cancelled")
        if integer(row["Diverted"], "Diverted"):
            return dict(base, status="diverted")
        if not base["tail"]:
            return dict(base, status="missing_tail")
        if base["origin"] not in zones or base["destination"] not in zones:
            return dict(base, status="missing_timezone")
        origin_zone, destination_zone = ZoneInfo(zones[base["origin"]]), ZoneInfo(zones[base["destination"]])
        scheduled_local = datetime.combine(date.fromisoformat(base["flight_date"][:10]), datetime.min.time())
        scheduled_local += timedelta(minutes=clock_minutes(row["CRSDepTime"]))
        scheduled_out = local_to_utc(scheduled_local, origin_zone)
        crs_elapsed = integer(row["CRSElapsedTime"], "CRSElapsedTime")
        actual_elapsed = integer(row["ActualElapsedTime"], "ActualElapsedTime")
        taxi_out, taxi_in = integer(row["TaxiOut"], "TaxiOut"), integer(row["TaxiIn"], "TaxiIn")
        if min(crs_elapsed, actual_elapsed, taxi_out, taxi_in) < 0:
            return dict(base, status="negative_duration")
        scheduled_in = scheduled_out+timedelta(minutes=crs_elapsed)
        actual_out = scheduled_out+timedelta(minutes=integer(row["DepDelay"], "DepDelay"))
        actual_in = scheduled_in+timedelta(minutes=integer(row["ArrDelay"], "ArrDelay"))
        wheels_off, wheels_on = actual_out+timedelta(minutes=taxi_out), actual_in-timedelta(minutes=taxi_in)
        clocks = (("CRSDepTime", scheduled_out, origin_zone), ("CRSArrTime", scheduled_in, destination_zone),
                  ("DepTime", actual_out, origin_zone), ("ArrTime", actual_in, destination_zone),
                  ("WheelsOff", wheels_off, origin_zone), ("WheelsOn", wheels_on, destination_zone))
        failures = []
        for field, value, zone in clocks:
            local = value.astimezone(zone)
            if local.hour*60+local.minute != clock_minutes(row[field]) % 1440:
                failures.append(field)
        elapsed_check = int((actual_in-actual_out).total_seconds()//60) == actual_elapsed
        air_time = integer(row["AirTime"], "AirTime")
        airtime_check = int((wheels_on-wheels_off).total_seconds()//60) == air_time
        order_check = actual_out <= wheels_off <= wheels_on <= actual_in
        if failures or not elapsed_check or not airtime_check or not order_check:
            return dict(base, status="clock_mismatch", failed_clock_fields=failures,
                        elapsed_match=elapsed_check, airtime_match=airtime_check, event_order=order_check)
        return dict(base, status="valid", origin_timezone=zones[base["origin"]], destination_timezone=zones[base["destination"]],
                    scheduled_out_utc=timestamp(scheduled_out), scheduled_in_utc=timestamp(scheduled_in),
                    actual_out_utc=timestamp(actual_out), actual_in_utc=timestamp(actual_in),
                    wheels_off_utc=timestamp(wheels_off), wheels_on_utc=timestamp(wheels_on),
                    scheduled_out_minute=epoch_minute(scheduled_out), scheduled_in_minute=epoch_minute(scheduled_in),
                    actual_out_minute=epoch_minute(actual_out), actual_in_minute=epoch_minute(actual_in),
                    wheels_off_minute=epoch_minute(wheels_off), wheels_on_minute=epoch_minute(wheels_on),
                    actual_out_local_date=actual_out.astimezone(origin_zone).date().isoformat(),
                    actual_in_local_date=actual_in.astimezone(destination_zone).date().isoformat(),
                    scheduled_in_local_date=scheduled_in.astimezone(destination_zone).date().isoformat(),
                    arrival_date_shift=(actual_in.astimezone(destination_zone).date()-date.fromisoformat(base["flight_date"][:10])).days,
                    clock_fields_checked=6, actual_elapsed_minutes=actual_elapsed,
                    cancelled=0, diverted=0)
    except (ValueError, KeyError, OverflowError) as error:
        return dict(base, status="invalid_component", reason=str(error))


def read_smoke_window(year, airport, local_date, pad_days=2):
    """Stream only required fields; retain a short date window and its tails.

    All reported network legs of selected tails in the window are retained,
    including legs that do not visit the smoke airport, to audit off-airport
    breaks. A source row index supports exact raw-row reconnection.
    """
    path = ROOT / f"data/open_source/raw/bts_on_time_{year}_01.zip"
    day = date.fromisoformat(local_date)
    lower, upper = (day-timedelta(days=pad_days)).isoformat(), (day+timedelta(days=pad_days)).isoformat()
    retained, scanned, header = [], 0, None
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(member) as raw:
            for chunk in pd.read_csv(raw, usecols=list(FIELDS), dtype=str, keep_default_na=False, chunksize=60000):
                if header is None:
                    header = list(chunk.columns)
                chunk["source_row_index"] = range(scanned, scanned+len(chunk))
                scanned += len(chunk)
                selected = chunk[(chunk["FlightDate"].str[:10] >= lower) & (chunk["FlightDate"].str[:10] <= upper)]
                if len(selected):
                    retained.append(selected)
    window = pd.concat(retained, ignore_index=True) if retained else pd.DataFrame(columns=list(FIELDS)+["source_row_index"])
    touching = window[(window["Origin"] == airport) | (window["Dest"] == airport)]
    tails = set(touching["Tail_Number"])-{""}
    network = window[window["Tail_Number"].isin(tails)].copy()
    # Preserve unlinked missing-tail airport records separately.
    missing = touching[touching["Tail_Number"] == ""]
    info = dict(source=str(path.relative_to(ROOT)), source_sha256=SOURCE_HASHES[year],
                source_hash_basis="Previously independently hashed raw ZIP; no repeated full-file hash in this smoke",
                csv_member=member, required_fields=header, raw_rows_scanned=scanned,
                source_flightdate_window=[lower, upper], window_network_rows=len(window),
                touching_airport_rows=len(touching), selected_tails=len(tails),
                selected_tail_network_rows=len(network), missing_tail_airport_rows=len(missing))
    return network.to_dict("records"), missing.to_dict("records"), info


def link_tail_intervals(events, airport, local_date):
    """Link consecutive recorded legs; count every unlinked target arrival."""
    by_tail, blocked_tails = defaultdict(list), set()
    for event in events:
        if event["status"] == "valid":
            by_tail[event["tail"]].append(event)
        elif event["status"] not in ("cancelled", "missing_tail") and event["tail"]:
            blocked_tails.add(event["tail"])
    intervals, target_arrivals, reasons = [], [], Counter()
    for tail, records in by_tail.items():
        records.sort(key=lambda event: (event["actual_out_minute"], event["record_id"]))
        for i, incoming in enumerate(records):
            if incoming["destination"] != airport:
                continue
            day_start = local_to_utc(datetime.combine(date.fromisoformat(local_date), datetime.min.time()),
                                     ZoneInfo(incoming["destination_timezone"]))
            day_end = local_to_utc(datetime.combine(date.fromisoformat(local_date)+timedelta(days=1), datetime.min.time()),
                                   ZoneInfo(incoming["destination_timezone"]))
            target = incoming["actual_in_local_date"] == local_date
            if target:
                target_arrivals.append(incoming["record_id"])
            if tail in blocked_tails:
                if target: reasons["tail_has_unusable_record_in_window"] += 1
                continue
            if i+1 == len(records):
                if target: reasons["no_next_record_in_window"] += 1
                continue
            outgoing = records[i+1]
            if outgoing["origin"] != incoming["destination"]:
                if target: reasons["next_record_origin_elsewhere"] += 1
                continue
            if outgoing["actual_out_minute"] < incoming["actual_in_minute"]:
                if target: reasons["overlapping_clock_sequence"] += 1
                continue
            if target: reasons["linked"] += 1
            if outgoing["actual_out_minute"] <= epoch_minute(day_start) or incoming["actual_in_minute"] >= epoch_minute(day_end):
                continue
            interval = dict(tail=tail, airport=airport, incoming_record_id=incoming["record_id"],
                            outgoing_record_id=outgoing["record_id"], incoming_gate_in_utc=incoming["actual_in_utc"],
                            outgoing_gate_out_utc=outgoing["actual_out_utc"],
                            start_minute=incoming["actual_in_minute"], end_minute=outgoing["actual_out_minute"],
                            turnaround_minutes=outgoing["actual_out_minute"]-incoming["actual_in_minute"],
                            target_local_date=local_date, incoming_on_target_date=target,
                            carries_into_day=incoming["actual_in_minute"] < epoch_minute(day_start),
                            carries_out_of_day=outgoing["actual_out_minute"] > epoch_minute(day_end),
                            endpoint_scope="Consecutive reported-tail same-airport gate-in to gate-out interval",
                            stand_identity="unobserved", aircraft_ready_event="unobserved")
            intervals.append(interval)
    return intervals, dict(valid_arrivals_on_target_local_date=len(target_arrivals), reasons=dict(reasons),
                           linked_arrivals=reasons["linked"], overlapping_day_intervals=len(intervals),
                           recorded_carry_in_intervals=sum(row["carries_into_day"] for row in intervals),
                           recorded_carry_out_intervals=sum(row["carries_out_of_day"] for row in intervals),
                           linked_fraction=reasons["linked"]/len(target_arrivals) if target_arrivals else None)
