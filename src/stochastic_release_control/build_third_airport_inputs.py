"""Reconstruct external-airport replication input cases from released BTS records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data' / 'third_airport_replication' / 'bts_calendar_corrected.csv.gz'
OUT = ROOT / 'data' / 'third_airport_replication'
HORIZON = 36
LAG = 6
START_HOURS = (10, 12, 14, 16, 18)
EVAL_DAYS = tuple(range(24, 32))
DONOR_DAYS = tuple(range(3, 19))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def profile(frame: pd.DataFrame, start_minute: int) -> list[int]:
    raw = np.zeros(HORIZON - LAG, dtype=int)
    if 'actual_takeoff' in frame.columns:
        relative = (pd.to_datetime(frame['actual_takeoff']) - frame['operation_date']).dt.total_seconds().to_numpy() / 60.0 - start_minute
    else:
        relative = frame['actual_takeoff_slot'].to_numpy(dtype=float) * 5.0 - start_minute
    for value in relative:
        slot = int(np.floor(value / 2.0))
        if 0 <= slot < HORIZON - LAG:
            raw[slot] += 1
    return [0] * LAG + raw.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--airport', choices=('ATL', 'CLT', 'DFW'), default='ATL')
    args = parser.parse_args()
    airport = args.airport
    if airport != 'ATL' and args.output == OUT:
        args.output = OUT / airport
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    frame = pd.read_csv(args.source, compression='gzip' if str(args.source).endswith('.gz') else None)
    frame['operation_date'] = pd.to_datetime(frame['operation_date'])
    frame['source_year'] = frame['operation_date'].dt.year
    frame = frame[(frame['airport'] == airport) & frame['source_year'].isin((2024, 2025))].copy()
    cases = []
    for year in (2024, 2025):
        year_frame = frame[frame['source_year'] == year]
        for day in EVAL_DAYS:
            for hour in START_HOURS:
                donors = [profile(year_frame[year_frame['operation_date'] == pd.Timestamp(f'{year}-01-{d:02d}')], hour * 60) for d in DONOR_DAYS]
                actual = profile(year_frame[year_frame['operation_date'] == pd.Timestamp(f'{year}-01-{day:02d}')], hour * 60)
                n = min(sum(path) for path in donors + [actual])
                if n < 2:
                    continue
                key = f'{airport}_{year:04d}-01-{day:02d}_h{hour:02d}'
                case_id = hashlib.sha256(key.encode()).hexdigest()[:12]
                case = {
                    'key': key, 'case_id': case_id, 'airport': airport,
                    'day': f'{year:04d}-01-{day:02d}', 'hour': hour, 'n': int(n),
                    'G': int(2 * n), 'lag': LAG, 'delta': 2, 'costs': [2, 3, 0],
                    'service': donors, 'probabilities': ['1/16'] * 16,
                    'donor_days': [f'{year:04d}-01-{d:02d}' for d in DONOR_DAYS],
                    'caps': [int(n)] * HORIZON, 'actual_service': actual,
                    'source': str(args.source.relative_to(ROOT)).replace('\\', '/'),
                }
                path = args.output / 'cases' / f'{case_id}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(case, indent=2), encoding='utf-8')
                cases.append({'case_id': case_id, 'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'airport': airport, 'day': case['day'], 'hour': hour, 'n': int(n)})
    manifest = {'airport': airport, 'source': str(args.source.relative_to(ROOT)).replace('\\', '/'), 'source_sha256': sha256(args.source), 'slot_minutes': 2, 'horizon_slots': HORIZON, 'lag_slots': LAG, 'donor_days': list(DONOR_DAYS), 'evaluation_days': list(EVAL_DAYS), 'start_hours': list(START_HOURS), 'cases': cases}
    (args.output / 'panel_design.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'cases': len(cases), 'airport': airport, 'source_sha256': manifest['source_sha256']}))


if __name__ == '__main__':
    main()
