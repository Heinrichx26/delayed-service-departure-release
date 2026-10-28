"""Download one public BTS archive and check its recorded SHA-256."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', required=True, type=int, choices=[2024, 2025])
    args = parser.parse_args()
    sources = json.loads((ROOT / 'data/public_sources.json').read_text())
    source = next(row for row in sources['flight_archives'] if row['year'] == args.year)
    target = ROOT / 'data/raw' / source['url'].rsplit('/', 1)[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(source['url'], target)
    digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
    if digest != source['sha256']:
        raise ValueError('The downloaded archive differs from the recorded source checksum.')
    print(json.dumps(dict(year=args.year, sha256=digest, verified=True)))


if __name__ == '__main__':
    main()
