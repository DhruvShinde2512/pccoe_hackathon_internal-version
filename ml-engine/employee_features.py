"""Shared, causal employee-day features for published CERT activity aggregates.

The six input counts describe a complete UTC day. The seven-day baseline uses
strictly earlier calendar days; no identities, labels or personality traits enter X.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

EMPLOYEE_FEATURE_VERSION = "employee-daily-temporal-v1"
ACTIVITY_COLUMNS = ("total_logon_activity", "after_hours_activity", "http_activity_count",
                    "total_emails", "file_activity_count", "usb_activity_count")
EMPLOYEE_FEATURE_NAMES = (*ACTIVITY_COLUMNS, "after_hours_fraction",
    *(f"past_7d_mean_{name}" for name in ACTIVITY_COLUMNS),
    *(f"change_z_{name}" for name in ACTIVITY_COLUMNS),
    "history_days_7d", "history_missing", "missing_activity_fields")


def extract_employee_features(rows: pd.DataFrame | Sequence[dict[str, Any]]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return features in input order with prior-only calendar-window baselines."""
    frame = pd.DataFrame(rows).copy().reset_index(drop=True)
    required = {"user", "day", *ACTIVITY_COLUMNS}
    if not required.issubset(frame):
        raise ValueError(f"employee observations require columns: {sorted(required - set(frame))}")
    if frame.empty:
        return np.empty((0, len(EMPLOYEE_FEATURE_NAMES))), EMPLOYEE_FEATURE_NAMES
    frame["day"] = pd.to_datetime(frame.day, utc=True, format="mixed").dt.tz_localize(None).dt.normalize()
    if frame.user.isna().any() or frame.day.isna().any() or frame.duplicated(["user", "day"]).any():
        raise ValueError("employee-day observations require unique, non-null user/day pairs")
    counts = frame[list(ACTIVITY_COLUMNS)].apply(pd.to_numeric, errors="raise")
    missing = counts.isna().sum(axis=1)
    counts = counts.fillna(0).astype(float)
    if not np.isfinite(counts.to_numpy()).all() or (counts < 0).any().any():
        raise ValueError("employee activity counts must be finite and nonnegative")
    frame[list(ACTIVITY_COLUMNS)] = counts
    frame["_position"] = np.arange(len(frame))
    ordered = frame.sort_values(["user", "day"], kind="stable")
    rolling = ordered.set_index("day").groupby("user")[list(ACTIVITY_COLUMNS)].rolling(
        "7D", closed="left", min_periods=1)
    means = rolling.mean().reset_index()
    deviations = rolling.std(ddof=0).reset_index()
    days = rolling.count().reset_index()[["user", "day", ACTIVITY_COLUMNS[0]]]
    mean_names = {name: f"past_7d_mean_{name}" for name in ACTIVITY_COLUMNS}
    std_names = {name: f"_std_{name}" for name in ACTIVITY_COLUMNS}
    result = frame.merge(means.rename(columns=mean_names), on=["user", "day"], validate="one_to_one")
    result = result.merge(deviations.rename(columns=std_names), on=["user", "day"], validate="one_to_one")
    result = result.merge(days.rename(columns={ACTIVITY_COLUMNS[0]: "history_days_7d"}),
                          on=["user", "day"], validate="one_to_one").sort_values("_position")
    result["history_days_7d"] = result.history_days_7d.fillna(0)
    result["history_missing"] = (result.history_days_7d == 0).astype(float)
    result["missing_activity_fields"] = missing.to_numpy()
    result["after_hours_fraction"] = result.after_hours_activity / result.total_logon_activity.clip(lower=1)
    for name in ACTIVITY_COLUMNS:
        baseline = result[f"past_7d_mean_{name}"].fillna(0)
        deviation = result[f"_std_{name}"].fillna(0).clip(lower=1)
        result[f"past_7d_mean_{name}"] = baseline
        result[f"change_z_{name}"] = np.where(result.history_days_7d > 0,
            ((result[name] - baseline) / deviation).clip(-20, 20), 0)
    return result[list(EMPLOYEE_FEATURE_NAMES)].to_numpy(dtype=np.float32), EMPLOYEE_FEATURE_NAMES


def employee_autoencoder_inputs(X: np.ndarray) -> np.ndarray:
    """Log-transform nonnegative count features while retaining signed changes."""
    values = np.asarray(X, dtype=np.float32).copy()
    for index, name in enumerate(EMPLOYEE_FEATURE_NAMES):
        if name in ACTIVITY_COLUMNS or name.startswith("past_7d_mean_"):
            values[:, index] = np.log1p(values[:, index])
    return values


def snapshot_employee_features(emp_id: str, observations: Sequence[dict[str, Any]],
                               as_of: str, observed_until: str | None = None
                               ) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Select the latest complete observed day, deduplicate and attach provenance."""
    cutoff = pd.to_datetime(as_of, utc=True)
    seen = pd.to_datetime(observed_until or as_of, utc=True)
    unique: dict[str, dict[str, Any]] = {}
    for source in observations:
        if source.get("user", emp_id) != emp_id:
            raise ValueError("observation belongs to a different employee")
        day = pd.to_datetime(source["day"], utc=True).normalize()
        completed = day + pd.Timedelta(days=1)
        arrival = pd.to_datetime(source.get("arrival_time") or completed, utc=True)
        if completed > cutoff or arrival > seen:
            continue
        key = day.date().isoformat()
        counts = {name: source.get(name) for name in ACTIVITY_COLUMNS}
        if any(value is None for value in counts.values()):
            raise ValueError("supply all six daily counts; unknown values cannot be assumed zero")
        row = {"user": emp_id, "day": key, **counts}
        old = unique.get(key)
        if old and any(old[name] != row[name] for name in ACTIVITY_COLUMNS):
            raise ValueError(f"conflicting employee-day observation: {key}")
        ids = source.get("source_ids") or [f"CERT-DAILY:{emp_id}:{key}"]
        row["source_ids"] = sorted(set(ids) | set(old.get("source_ids", []) if old else []))
        unique[key] = row
    if not unique:
        raise ValueError("no complete employee day observed by the requested cutoff")
    current = max(unique)
    start = pd.Timestamp(current) - pd.Timedelta(days=7)
    selected = [unique[key] for key in sorted(unique) if pd.Timestamp(key) >= start]
    X, names = extract_employee_features(selected)
    canonical = json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    current_ids = selected[-1]["source_ids"]
    previous_ids = sorted({identifier for row in selected[:-1] for identifier in row["source_ids"]})
    evidence = {name: previous_ids if name.startswith("past_7d_") or name in {"history_days_7d", "history_missing"}
                else sorted(set(current_ids + previous_ids)) if name.startswith("change_z_")
                else current_ids for name in names}
    return X[-1:], names, {"day": current, "observation_count": len(selected),
        "evidence_hash": hashlib.sha256(canonical.encode()).hexdigest(),
        "feature_evidence": evidence, "source_kind": "DAILY_AGGREGATE",
        "history_days": len(selected) - 1}
