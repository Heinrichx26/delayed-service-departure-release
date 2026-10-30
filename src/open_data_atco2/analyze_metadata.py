"""Analyze public ATCO2 XML metadata without using audio content."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def read_info(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def read_xml(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    rows = []
    for seg in root.findall("segment"):
        rows.append({
            "start": float(seg.findtext("start", "nan")),
            "end": float(seg.findtext("end", "nan")),
            "speaker": seg.findtext("speaker", ""),
            "speaker_label": seg.findtext("speaker_label", ""),
            "text": seg.findtext("text", ""),
        })
    return rows


def analyze(root: Path, phase: str) -> dict:
    files = sorted(root.glob("*.xml"))
    if phase == "smoke":
        files = files[:3]
    records = []
    for xml_path in files:
        rows = read_xml(xml_path)
        info = read_info(xml_path.with_suffix(".info"))
        if not rows:
            continue
        durations = [r["end"] - r["start"] for r in rows if r["end"] >= r["start"]]
        gaps = [b["start"] - a["end"] for a, b in zip(rows, rows[1:])]
        turn_gaps = [g for g, a, b in zip(gaps, rows, rows[1:])
                     if a["speaker"] != b["speaker"] and g >= 0]
        controller = [r for r in rows if "ATCO" in r["speaker_label"].upper()]
        records.append({
            "file": xml_path.name,
            "airport": info.get("airport", ""),
            "channel": info.get("channel", ""),
            "utc_time": info.get("UTC time", ""),
            "segments": len(rows),
            "controller_segments": len(controller),
            "audio_seconds": max(r["end"] for r in rows),
            "mean_segment_seconds": statistics.mean(durations),
            "median_turn_gap_seconds": statistics.median(turn_gaps) if turn_gaps else None,
            "p90_turn_gap_seconds": sorted(turn_gaps)[max(0, int(.9*len(turn_gaps))-1)] if turn_gaps else None,
            "controller_speech_seconds": sum(r["end"]-r["start"] for r in controller),
        })
    all_durations = [r["mean_segment_seconds"] for r in records]
    return {
        "passed": True,
        "phase": phase,
        "files": len(records),
        "records": records,
        "mean_file_segment_seconds": statistics.mean(all_durations) if all_durations else None,
        "source": "ATCO2 XML metadata; audio not analyzed",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("phase", choices=("smoke", "full"))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = analyze(a.root, a.phase)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
