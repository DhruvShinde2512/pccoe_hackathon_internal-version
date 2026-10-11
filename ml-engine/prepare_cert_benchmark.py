"""Build a reproducible employee-disjoint, chronological CERT r4.2 benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from employee_features import ACTIVITY_COLUMNS

ROOT = Path(__file__).resolve().parent / "benchmark" / "cert"
CERT_URL = "https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247"
MATRIX_URL = "https://zenodo.org/records/19435851"


def prepare_cert(root: Path = ROOT, seed: int = 20261010) -> dict[str, Any]:
    """Label only documented incident days; never label a user's entire history."""
    raw = root / "raw"
    matrix_path = raw / "master_behavioral_matrix.parquet"
    answers_path = raw / "answers.tar.bz2"
    expected = {line.split()[1]: line.split()[0]
                for line in (raw / "CHECKSUMS-sha256.txt").read_text().splitlines() if line.strip()}
    for name in ("answers.tar.bz2", "r1.tar.bz2"):
        if name == "r1.tar.bz2" and not (raw / name).exists():
            continue
        if hashlib.sha256((raw / name).read_bytes()).hexdigest() != expected[name]:
            raise ValueError(f"official CERT checksum mismatch: {name}")
    if hashlib.md5(matrix_path.read_bytes()).hexdigest() != "3b473cfbab87de9d9c5dfce86f77ee28":
        raise ValueError("published behavioral-matrix checksum mismatch")
    frame = pd.read_parquet(matrix_path)[["user", "day", *ACTIVITY_COLUMNS]].copy()
    frame["day"] = pd.to_datetime(frame.day, format="mixed").dt.normalize()
    if frame.duplicated(["user", "day"]).any():
        raise ValueError("curated matrix contains duplicate employee days")
    with tarfile.open(answers_path, "r:bz2") as archive:
        labels = pd.read_csv(archive.extractfile("answers/insiders.csv"), dtype={"dataset": str})
        for name in ("readme.txt", "scenarios.txt"):
            (raw / f"answers_{name}").write_bytes(archive.extractfile(f"answers/{name}").read())
    labels = labels.loc[labels.dataset.eq("4.2")]
    frame["label"] = 0
    frame["incident_id"] = ""
    for incident in labels.itertuples():
        start, end = pd.Timestamp(incident.start).normalize(), pd.Timestamp(incident.end).normalize()
        mask = frame.user.eq(incident.user) & frame.day.between(start, end)
        frame.loc[mask, "label"] = 1
        frame.loc[mask, "incident_id"] = incident.details
    users = np.sort(frame.user.unique())
    flags = pd.Series(users).isin(labels.user).astype(int).to_numpy()
    train, rest = train_test_split(users, test_size=.4, stratify=flags, random_state=seed)
    validation, test = train_test_split(rest, test_size=.5,
        stratify=pd.Series(rest).isin(labels.user), random_state=seed)
    periods = {"train": ("2010-01-01", "2010-09-01"),
               "validation": ("2010-09-01", "2011-01-01"),
               "test": ("2011-01-01", "2011-06-01")}
    groups = {"train": train, "validation": validation, "test": test}
    selected = []
    counts = {}
    for split, people in groups.items():
        start, end = periods[split]
        mask = frame.user.isin(people) & frame.day.ge(start) & frame.day.lt(end)
        subset = frame.loc[mask].copy()
        if subset.label.nunique() != 2:
            raise ValueError(f"{split} requires positive and negative labeled days")
        subset["split"] = split
        subset["day"] = subset.day.dt.strftime("%Y-%m-%d")
        subset["source_id"] = "CERT-DAILY:" + subset.user + ":" + subset.day
        selected.append(subset)
        counts[split] = {"rows": len(subset), "positive_days": int(subset.label.sum()),
                         "employees": int(subset.user.nunique()),
                         "positive_employees": int(subset.loc[subset.label.eq(1), "user"].nunique())}
    dataset = pd.concat(selected).sort_values(["user", "day"], kind="stable")
    target = root / "daily_observations.csv"
    dataset.to_csv(target, index=False)
    manifest = {"seed": seed, "periods_inclusive_exclusive": periods,
                "employees": {key: sorted(value.tolist()) for key, value in groups.items()},
                "counts": counts}
    (root / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    provenance = {"benchmark": "CERT r4.2 employee-day behavior, curated daily counts",
        "original_source": CERT_URL, "matrix_source": MATRIX_URL,
        "matrix_license": "CC-BY-NC-4.0; attribution to Antonio Barrera Mora",
        "raw_matrix_rows": len(frame), "incident_count": len(labels), "counts": counts,
        "dataset_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((raw / name).read_bytes()).hexdigest()
                          for name in ("answers.tar.bz2", "r1.tar.bz2", matrix_path.name) if (raw / name).exists()},
        "label_assumption": "A recorded employee-day is positive when it overlaps that employee's official r4.2 incident start/end interval. Other activity on the same day can be benign. Labels are not event-level guilt.",
        "split_policy": "Stratify employee IDs by ever-incident status with fixed seed; disjoint employees and strictly later target-day periods; no test tuning.",
        "limitations": ["Synthetic workplace activity, not bank employee-to-account links",
            "Daily aggregation supplied by a third-party curator; raw r4.2 logs not independently reconstructed",
            "Only two positive employees in this fixed held-out time/group split",
            "r1 contains background behavior and is retained as reference, not labeled malicious training data",
            "No personality, sentiment, identity or label fields used as model features"]}
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    print(json.dumps(prepare_cert(args.root, args.seed), indent=2))
