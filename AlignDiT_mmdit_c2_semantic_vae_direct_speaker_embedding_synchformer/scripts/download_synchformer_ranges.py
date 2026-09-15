"""Resumable parallel HTTP fallback for the pinned public Synchformer weight.

Every range and the final official SHA256 are checked. Downloads remain outside
the source snapshot. The standard hf download route is also available in the
shell wrapper by setting DOWNLOAD_WORKERS=1.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import os
from pathlib import Path
import time

import requests

SIZE = 950058171
SHA256 = "8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c"
REVISION = "3abd4e833b95b8db0fc9c687afc52483a48e9a97"
URL = f"https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/{REVISION}/synchformer_state_dict.pth"
CHUNK = 8 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    parts = args.checkpoint.parent / ".synchformer_download_parts"
    parts.mkdir(parents=True, exist_ok=True)
    count = (SIZE + CHUNK - 1) // CHUNK

    def fetch(index):
        start = index * CHUNK
        end = min(SIZE, start + CHUNK) - 1
        target = parts / f"{index:04d}.part"
        if target.exists() and target.stat().st_size == end - start + 1:
            return index
        for attempt in range(5):
            try:
                with requests.get(URL, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    expected_range = f"bytes {start}-{end}/{SIZE}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError(f"Server did not honor requested range {index}")
                    temporary = target.with_suffix(".tmp")
                    with temporary.open("wb") as f:
                        for block in response.iter_content(1024 * 1024):
                            f.write(block)
                    if temporary.stat().st_size != end - start + 1:
                        raise RuntimeError(f"Truncated download range {index}")
                    temporary.replace(target)
                    return index
            except (requests.RequestException, RuntimeError) as error:
                if attempt == 4:
                    # Do not print signed redirect URLs or request headers.
                    raise RuntimeError(f"Range {index} failed after five attempts ({type(error).__name__})") from None
                time.sleep(min(2 ** attempt, 10))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for complete, future in enumerate(as_completed([executor.submit(fetch, i) for i in range(count)]), 1):
            future.result()
            print(f"Downloaded {complete}/{count} verified ranges", flush=True)
    temporary = args.checkpoint.with_suffix(".assembling")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for index in range(count):
            with (parts / f"{index:04d}.part").open("rb") as source:
                for block in iter(lambda: source.read(CHUNK), b""):
                    digest.update(block)
                    output.write(block)
        output.flush()
        os.fsync(output.fileno())
    if temporary.stat().st_size != SIZE or digest.hexdigest() != SHA256:
        raise RuntimeError("Downloaded checkpoint does not match the official size/SHA256")
    temporary.replace(args.checkpoint)
    print(f"Verified checkpoint: {args.checkpoint}; SHA256={SHA256}", flush=True)


if __name__ == "__main__":
    main()
