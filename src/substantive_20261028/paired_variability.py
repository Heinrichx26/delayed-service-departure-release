"""Describe paired coordination savings with airport-day cluster resampling."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/revision_20261028/substantive_strengthening/paired_variability'
DATA = ROOT / 'data/substantive_20261028'
INPUT = ROOT / 'results/revision_20261028/problem_driven/greedy_comparison.csv'
SEED = 20261028


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def interval(records, field, repeats, rng):
    groups = defaultdict(list)
    for record in records:
        groups[record['cluster']].append(record[field])
    names = sorted(groups)
    sums = np.array([sum(groups[name]) for name in names])
    counts = np.array([len(groups[name]) for name in names])
    resamples = rng.multinomial(len(names), np.ones(len(names)) / len(names), size=repeats)
    estimates = (resamples @ sums) / (resamples @ counts)
    assert np.isfinite(estimates).all()
    return {
        'windows': len(records), 'airport_day_clusters': len(names),
        'mean': float(sum(sums) / sum(counts)),
        'percentile_95_interval': [float(v) for v in np.quantile(estimates, [.025, .975])],
        'bootstrap_standard_deviation': float(estimates.std(ddof=1)),
    }


def execute(repeats):
    with INPUT.open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream)
                if r['t0'] == '0' and r['lead_name'] == 'transport_plus_two']
    assert len(rows) == 200
    rng = np.random.default_rng(SEED)
    output = {}
    for label in ['original', 'ceil_N_over_4', 'ceil_N_over_8']:
        records = []
        for row in rows:
            if row['capacity_label'] != label:
                continue
            optimum, static, greedy = (float(row[name]) for name in
                                       ['optimal_queries', 'static_queries', 'greedy_queries'])
            assert 0 <= optimum <= greedy and optimum <= static
            airport, day = row['key'].split('_')[:2]
            records.append({
                'cluster': airport + '_' + day,
                'relative_static_saving': 1 - optimum / static,
                'relative_greedy_saving': 1 - optimum / greedy,
                'absolute_static_saving': static - optimum,
            })
        output[label] = {field: interval(records, field, repeats, rng) for field in
                         ['relative_static_saving', 'relative_greedy_saving', 'absolute_static_saving']}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'panel'], required=True)
    args = parser.parse_args()
    source = Path(__file__)
    if args.phase == 'smoke':
        control = [{'cluster': str(i // 2), 'x': .25} for i in range(20)]
        check = interval(control, 'x', 100, np.random.default_rng(SEED))
        assert check['percentile_95_interval'] == [.25, .25]
        results = execute(100)
        assert all(v['relative_static_saving']['mean'] > .35 for v in results.values())
        report = {'passed': True, 'phase': 'smoke', 'repeats': 100,
                  'source_sha256': sha(source), 'input_sha256': sha(INPUT), 'results': results}
        save(OUT / 'smoke.json', report)
    else:
        smoke = json.loads((OUT / 'smoke.json').read_text(encoding='utf-8'))
        assert smoke['passed'] and smoke['source_sha256'] == sha(source)
        assert smoke['input_sha256'] == sha(INPUT)
        design = {'date': '2026-10-28', 'frozen_before_resampling': True,
                  'source_sha256': sha(source), 'input_sha256': sha(INPUT),
                  'smoke_sha256': sha(OUT / 'smoke.json'), 'repeats': 5000, 'seed': SEED,
                  'cluster': 'airport-day; all windows within a sampled cluster travel together',
                  'estimand': 'window-average paired savings, separately by release capacity',
                  'scope': 'descriptive resampling of the record-derived panel'}
        save(DATA / 'paired_variability_design.json', design)
        report = {'passed': True, 'phase': 'panel', 'repeats': 5000,
                  'design_sha256': sha(DATA / 'paired_variability_design.json'),
                  'results': execute(5000)}
        save(OUT / 'panel.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
