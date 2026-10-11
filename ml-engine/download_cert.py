"""Download the public CERT benchmark inputs with published checksum validation."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent / "benchmark/cert/raw"
# Checksums recorded from CMU's published CHECKSUMS-sha256.txt and Zenodo metadata.
FILES = {
    "answers.tar.bz2": ("https://ndownloader.figshare.com/files/24857828", "sha256",
        "ed21bf921def01e97326b314f2d1fa77651a945bd31bf513095f8ac7658c9ddd"),
    "CHECKSUMS-sha256.txt": ("https://ndownloader.figshare.com/files/24857831", "sha256",
        "3a03ec4e0a2cc39fa76d3c12f22323f3ac1ad1b0acc8a199a7b4a37566b245ca"),
    "master_behavioral_matrix.parquet": (
        "https://zenodo.org/api/records/19435851/files/master_behavioral_matrix.parquet/content", "md5",
        "3b473cfbab87de9d9c5dfce86f77ee28"),
    "r1.tar.bz2": ("https://ndownloader.figshare.com/files/24857825", "sha256",
        "8a70ad175df477dd5a35e8422ff236bc6a9ca76f48ae7e0d29c471c4efa52fa3"),
}


def download(root: Path = ROOT, *, include_background: bool = False) -> None:
    """Atomically save only checksum-verified inputs; reuse valid local files."""
    root.mkdir(parents=True, exist_ok=True)
    for name, (url, algorithm, expected) in FILES.items():
        if name == "r1.tar.bz2" and not include_background:
            continue
        target = root / name
        if target.exists() and hashlib.new(algorithm, target.read_bytes()).hexdigest() == expected:
            print(f"Verified existing {name}", flush=True)
            continue
        partial = root / (name + ".part")
        digest = hashlib.new(algorithm)
        size = 0
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 100 * 1024 * 1024:
                    raise ValueError("download exceeds the documented file-size bound")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"download checksum mismatch: {name}")
        partial.replace(target)
        print(f"Downloaded and verified {name}: {size} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--include-background", action="store_true", help="Also fetch the optional 88 MB r1 reference archive")
    args = parser.parse_args()
    download(args.root, include_background=args.include_background)
