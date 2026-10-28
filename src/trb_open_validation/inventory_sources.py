"""Cache small, public primary documents used for the airport inventory audit.

This script downloads documents only; it does not fit or run an experiment.
Use --smoke for one document before downloading the curated collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "trb_open_validation" / "public_inventory"
MAX_BYTES = 16 * 1024 * 1024
SOURCES = [
    {"id": "den_2024_acfr", "file": "den_2024_acfr.pdf", "organization": "Denver International Airport", "title": "2024 Annual Comprehensive Financial Report", "publication_date": "2025-05-30", "publication_date_basis": "transmittal letter; uploaded June 2025", "inventory_date": "2024-12-31", "url": "https://cdn.flydenver.com/app/uploads/2025/06/04133645/FY2024-Annual-Comprehensive-Financial-Report-FINAL.pdf", "format": "pdf"},
    {"id": "den_2023_acfr", "file": "den_2023_acfr.pdf", "organization": "Denver International Airport", "title": "2023 Annual Comprehensive Financial Report", "publication_date": "2024-06-21", "publication_date_basis": "transmittal letter, PDF page 6", "inventory_date": "2023-12-31", "url": "https://cdn.flydenver.com/app/uploads/2024/06/14083346/den_Audited_Financial_Report_2023.pdf", "format": "pdf"},
    {"id": "phx_2024_annual", "file": "phx_2024_annual.pdf", "organization": "City of Phoenix Aviation Department", "title": "Annual Comprehensive Financial Report, fiscal year ended June 30, 2024", "publication_date": "2024-12-16", "publication_date_basis": "transmittal letter, PDF page 8", "inventory_date": "2024-06-30", "url": "https://www.skyharbor.com/media/rbwfcqrd/annual-report-2024.pdf", "format": "pdf"},
    {"id": "phx_2023_annual", "file": "phx_2023_annual.pdf", "organization": "City of Phoenix Aviation Department", "title": "Annual Comprehensive Financial Report, fiscal year ended June 30, 2023", "publication_date": "2023-12-12", "publication_date_basis": "transmittal letter, PDF page 8", "inventory_date": "2023-06-30", "url": "https://www.skyharbor.com/media/pqjdm4jf/annual-report-2023.pdf", "format": "pdf"},
    {"id": "phx_2022_eighth_concourse", "file": "phx_2022_eighth_concourse.pdf", "organization": "City of Phoenix Aviation Department", "title": "The Eighth Concourse media kit", "publication_date": "2022-06", "inventory_date": "2022-06", "url": "https://www.skyharbor.com/media/kxab2udk/eighth-concourse-media-kit.pdf", "format": "pdf"},
    {"id": "phx_2025_investor", "file": "phx_2025_investor.pdf", "organization": "City of Phoenix", "title": "2025 Investor Presentation, Airport Revenue Bonds", "publication_date": "2025", "inventory_date": "2025", "url": "https://www.phoenix.gov/content/dam/phoenix/financesite/documents/Phoenix-CIC-Airport-2025-Investor-Presentation.pdf", "format": "pdf"},
    {"id": "sea_2024_facilities", "file": "sea_2024_facilities.html", "organization": "Port of Seattle", "title": "Improvement Projects on the Horizon at SEA", "publication_date": "2024-07-31", "inventory_date": "2024-07-31", "url": "https://www.portseattle.org/blog/improvement-projects-horizon-sea", "format": "html"},
    {"id": "sea_2023_progress", "file": "sea_2023_progress.pdf", "organization": "Port of Seattle", "title": "SEA Airport Capital Program Progress report, Q4 2023", "publication_date": "2023-12-12", "inventory_date": "2023-12-12", "url": "https://meetings.portseattle.org/portmeetings/attachments/2023/2023_12_12_RM_10a_Presentation.pdf", "format": "pdf", "caution": "The hosted PDF says it was computer generated and is not the official meeting record; use as corroboration only."},
    {"id": "faa_registry_download", "file": "faa_registry_download.html", "organization": "Federal Aviation Administration", "title": "Releasable Aircraft Database Download", "publication_date": "2025-05-13", "publication_date_basis": "page last updated", "inventory_date": None, "url": "https://www.faa.gov/licenses_certificates/aircraft_certification/aircraft_registry/releasable_aircraft_download", "format": "html"},
    {"id": "faa_registry_layout", "file": "faa_registry_layout.pdf", "organization": "Federal Aviation Administration", "title": "Releasable Aircraft Database record layout", "publication_date": "2025-05-08", "publication_date_basis": "document page header", "inventory_date": None, "url": "https://registry.faa.gov/database/ardata.pdf", "format": "pdf"},
    {"id": "faa_acd_page", "file": "faa_acd_page.html", "organization": "Federal Aviation Administration", "title": "Aircraft Characteristics Database", "publication_date": "2024-10", "inventory_date": "2024-10", "url": "https://www.faa.gov/airports/engineering/aircraft_char_database", "format": "html"},
    {"id": "faa_acd_oct2024", "file": "faa_acd_oct2024.xlsx", "organization": "Federal Aviation Administration", "title": "Aircraft Characteristics Database, October 2024", "publication_date": "2024-10", "inventory_date": "2024-10", "url": "https://www.faa.gov/airports/engineering/aircraft_char_database/aircraft_data", "format": "xlsx"},
]


def fetch(spec: dict) -> dict:
    raw = DEST / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / spec["file"]
    result = dict(spec)
    result["fetched_utc"] = datetime.now(timezone.utc).isoformat()
    result["local_file"] = path.relative_to(ROOT).as_posix()
    try:
        if path.exists():
            content = path.read_bytes()
            result["cache_hit"] = True
        else:
            import requests
            with requests.get(spec["url"], headers={"User-Agent": "Academic public-document audit/1.0"}, timeout=45, stream=True) as response:
                response.raise_for_status()
                result["resolved_url"] = response.url
                result["content_type"] = response.headers.get("Content-Type", "")
                chunks = []
                size = 0
                for chunk in response.iter_content(65536):
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError(f"Document exceeds the {MAX_BYTES}-byte limit")
                content = b"".join(chunks)
            result["cache_hit"] = False
        if len(content) > MAX_BYTES:
            raise ValueError(f"Document exceeds the {MAX_BYTES}-byte limit")
        kind = spec["format"]
        if kind == "pdf" and not content.startswith(b"%PDF"):
            raise ValueError("Expected PDF signature")
        if kind == "xlsx" and not content.startswith(b"PK"):
            raise ValueError("Expected XLSX/ZIP signature")
        if kind == "html" and b"<html" not in content[:10000].lower():
            raise ValueError("Expected HTML document")
        if spec["id"] == "sea_2024_facilities" and b"89 gates" not in content:
            raise ValueError("Semantic validation failed: requested article was replaced by a blog index; do not use as inventory evidence")
        if not path.exists():
            path.write_bytes(content)
        result.update(status="ok", bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ids", nargs="+")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    selected = [row for row in SOURCES if not args.ids or row["id"] in args.ids]
    if args.smoke:
        selected = selected[:1]
    records = [fetch(spec) for spec in selected]
    manifest = {"purpose": "Dated nominal airport inventory and aircraft mapping primary sources", "maximum_bytes_per_document": MAX_BYTES, "sources": records}
    name = "source_smoke.json" if args.smoke else "source_manifest.json"
    (DEST / name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in records:
        print(row["id"], row["status"], row.get("bytes", row.get("error")))
    if any(row["status"] != "ok" for row in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
