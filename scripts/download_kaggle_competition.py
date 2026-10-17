"""Download competition files individually, bypassing a failing bulk archive endpoint.

Run with the Python interpreter from the pipx Kaggle environment.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import time

from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('competition')
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    files = []
    token = None
    while True:
        page = api.competition_list_files(args.competition, page_token=token, page_size=200)
        files.extend({'name': f.name, 'size': int(f.total_bytes)} for f in page.files)
        token = page.next_page_token
        if not token:
            break
    print(f"{len(files)} files, {sum(f['size'] for f in files):,} bytes", flush=True)
    args.path.mkdir(parents=True, exist_ok=True)
    (args.path / 'download_manifest.json').write_text(json.dumps(files, indent=2) + '\n')
    if args.list_only:
        return
    for index, item in enumerate(files, 1):
        relative = PurePosixPath(item['name'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe file path: {relative}')
        target = args.path.joinpath(*relative.parts)
        if target.is_file() and target.stat().st_size == item['size']:
            print(f'[{index}/{len(files)}] Verified {relative}', flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                print(f'[{index}/{len(files)}] Downloading {relative}', flush=True)
                api.competition_download_file(args.competition, item['name'], path=str(target.parent), quiet=True)
                if not target.is_file() or target.stat().st_size != item['size']:
                    raise RuntimeError(f'Size verification failed: {target}')
                break
            except Exception as exc:
                print(f'Attempt {attempt + 1} failed: {exc}', flush=True)
                if attempt == 2:
                    raise
                time.sleep(3)
    print(f'Complete: all {len(files)} files verified against listed byte sizes.', flush=True)


if __name__ == '__main__':
    main()
