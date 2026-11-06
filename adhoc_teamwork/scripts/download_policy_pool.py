#!/usr/bin/env python
"""Download one layout's checkpoints from ZSC-Eval's public policy pool, and
verify each against the SHA-256 the hub publishes for it. Safe to re-run.

Only ``.pt`` checkpoints are fetched, and ``load_zsceval_policy`` reads those with
``weights_only=True``. The pool's ``policy_config/*.pkl`` files are skipped on
purpose: unpickling a downloaded file can run arbitrary code. Generate them with
ZSC-Eval's ``shell/store_config.sh`` or copy them from a copy you already trust.

Usage:
    python -m adhoc_teamwork.scripts.download_policy_pool --layout random3 --out <pool root>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

REPO = "Leoxxxxh/ZSC-Eval-policy_pool"
USER_AGENT = {"User-Agent": "adhoc-teamwork-pool-downloader"}


def list_files(layout: str) -> List[Dict]:
    url = f"https://huggingface.co/api/models/{REPO}/tree/main/{layout}?recursive=1"
    files: List[Dict] = []
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=60) as response:
            files.extend(item for item in json.load(response) if item["type"] == "file")
            next_page = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("Link", ""))
        url = next_page.group(1) if next_page else None
    return files


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(item: Dict, root: str, retries: int = 3) -> str:
    """Returns "skipped" or "downloaded". Raises if the file cannot be verified."""
    path = os.path.join(root, item["path"])
    expected = item["lfs"]["oid"] if "lfs" in item else None
    if os.path.exists(path) and os.path.getsize(path) == item["size"] and (expected is None or sha256(path) == expected):
        return "skipped"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/main/{item['path']}"
    tmp = f"{path}.tmp{os.getpid()}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=120) as response, open(tmp, "wb") as out:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            if os.path.getsize(tmp) != item["size"]:
                raise IOError(f"size mismatch for {item['path']}")
            if expected is not None and sha256(tmp) != expected:
                raise IOError(f"checksum mismatch for {item['path']}")
            os.replace(tmp, path)
            return "downloaded"
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--out", required=True, help="pool root; files go in <out>/<layout>/...")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    files = list_files(args.layout)
    checkpoints = [f for f in files if f["path"].endswith(".pt")]
    print(f"{args.layout}: {len(files)} files listed, fetching {len(checkpoints)} checkpoints ({sum(f['size'] for f in checkpoints) / 1e9:.2f} GB)", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda f: fetch(f, args.out), checkpoints))
    print(f"downloaded {results.count('downloaded')}, already present {results.count('skipped')}; all verified", flush=True)


if __name__ == "__main__":
    main()
