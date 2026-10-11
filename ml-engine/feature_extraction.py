"""Create numeric, transaction-level features from ledger and audit records.

All timestamps are interpreted as UTC when no timezone is supplied. Rolling
windows include transactions at the lower boundary and earlier rows at the
same timestamp. The returned matrix keeps the input transaction order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import deque
from bisect import bisect_left
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd


RecordSource: TypeAlias = (
    str | Path | pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]]
)

FEATURE_NAMES: tuple[str, ...] = (
    "amount",
    "amount_missing",
    "sender_missing",
    "timestamp_missing",
    "tx_count_1h",
    "tx_sum_1h",
    "tx_count_6h",
    "tx_sum_6h",
    "tx_count_24h",
    "tx_sum_24h",
    "near_reporting_limit",
    "below_reporting_limit",
    "seconds_since_access",
    "access_missing",
    "last_access_modify_phone",
    "last_access_override_alert",
    "is_dormant_sender",
    "account_missing",
    "closes_3hop_cycle_48h",
)


def _to_frame(source: RecordSource | None, label: str) -> pd.DataFrame:
    """Convert a CSV path, frame, row, or sequence of rows to a private frame."""
    if source is None:
        return pd.DataFrame()
    if isinstance(source, (str, Path)):
        return pd.read_csv(source, dtype=str)
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, Mapping):
        if all(not isinstance(value, (list, tuple, np.ndarray, pd.Series)) for value in source.values()):
            return pd.DataFrame([source])
        return pd.DataFrame(source)
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return pd.DataFrame(source)
    raise TypeError(f"{label} must be a CSV path, DataFrame, mapping, or sequence of mappings")


def _require(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    """Reject malformed nonempty inputs with an actionable error."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _clean_id(series: pd.Series) -> pd.Series:
    """Normalize account identifiers without merging null or blank accounts."""
    values = series.astype("string").str.strip()
    return values.mask(values.eq(""))


def _normalize_columns(frame: pd.DataFrame, aliases: dict[str, str], label: str) -> pd.DataFrame:
    """Accept teammate CSV names and API names without ambiguous duplicates."""
    rename = {old: new for old, new in aliases.items() if old in frame.columns and new not in frame.columns}
    for old, new in aliases.items():
        if old in frame.columns and new in frame.columns:
            raise ValueError(f"{label} contains both {old} and {new}")
    return frame.rename(columns=rename)


def _window_features(
    transactions: pd.DataFrame, amounts: pd.Series, timestamps: pd.Series
) -> dict[str, np.ndarray]:
    """Calculate sender windows with sorted timestamps and prefix sums."""
    size = len(transactions)
    result = {
        f"tx_{measure}_{hours}h": np.zeros(size, dtype=np.float64)
        for hours in (1, 6, 24)
        for measure in ("count", "sum")
    }
    sender = _clean_id(transactions["sender_account_id"])
    valid = pd.DataFrame(
        {"sender": sender, "time": timestamps, "amount": amounts, "position": np.arange(size)}
    ).dropna(subset=["time"])

    # Unknown senders cannot share account history with any other row.
    unknown = valid["sender"].isna().to_numpy()
    unknown_positions = valid.loc[unknown, "position"].to_numpy(dtype=np.int64)
    unknown_amounts = valid.loc[unknown, "amount"].to_numpy(dtype=np.float64)
    for hours in (1, 6, 24):
        result[f"tx_count_{hours}h"][unknown_positions] = 1.0
        result[f"tx_sum_{hours}h"][unknown_positions] = unknown_amounts

    for _, group in valid.loc[~unknown].groupby("sender", sort=False):
        ordered = group.sort_values(["time", "position"], kind="stable")
        times = ordered["time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        values = ordered["amount"].to_numpy(dtype=np.float64)
        positions = ordered["position"].to_numpy(dtype=np.int64)
        prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        ends = np.arange(1, len(times) + 1)
        for hours in (1, 6, 24):
            lower = times - np.int64(hours * 3_600_000_000_000)
            starts = np.searchsorted(times, lower, side="left")
            result[f"tx_count_{hours}h"][positions] = ends - starts
            result[f"tx_sum_{hours}h"][positions] = prefix[ends] - prefix[starts]
    return result


def _cycle_features(transactions: pd.DataFrame, timestamps: pd.Series) -> np.ndarray:
    """Flag a transfer that closes a two or three edge money flow within 48h."""
    result = np.zeros(len(transactions), dtype=np.float64)
    if "receiver_account_id" not in transactions:
        return result
    ordered = pd.DataFrame({
        "sender": _clean_id(transactions["sender_account_id"]),
        "receiver": _clean_id(transactions["receiver_account_id"]),
        "time": timestamps,
        "position": np.arange(len(transactions)),
    }).dropna(subset=["sender", "receiver", "time"]).sort_values(["time", "position"], kind="stable")
    edges: deque[tuple[int, str, str]] = deque()
    graph: dict[str, dict[str, deque[int]]] = {}
    horizon = 48 * 3_600_000_000_000
    for row in ordered.itertuples(index=False):
        current_ns = int(row.time.value)
        while edges and edges[0][0] < current_ns - horizon:
            _, old_sender, old_receiver = edges.popleft()
            neighbors = graph[old_sender]
            neighbors[old_receiver].popleft()
            if not neighbors[old_receiver]:
                del neighbors[old_receiver]
            if not neighbors:
                del graph[old_sender]
        # A -> B, B -> C, C -> A: before C -> A there is A -> B -> C.
        reverse_times = graph.get(row.receiver, {}).get(row.sender, ())
        direct = bool(reverse_times and reverse_times[0] < current_ns)
        two_hop = False
        for mid, first_times in graph.get(row.receiver, {}).items():
            if mid in {row.sender, row.receiver}:
                continue
            second_times = graph.get(mid, {}).get(row.sender, ())
            end = bisect_left(second_times, current_ns)
            if end and first_times[0] < second_times[end - 1]:
                two_hop = True
                break
        result[row.position] = float(row.sender != row.receiver and (direct or two_hop))
        graph.setdefault(row.sender, {}).setdefault(row.receiver, deque()).append(current_ns)
        edges.append((current_ns, row.sender, row.receiver))
    return result


def _access_features(
    transactions: pd.DataFrame, timestamps: pd.Series, access_logs: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Join only the latest access event at or before each transaction."""
    size = len(transactions)
    result = {
        "seconds_since_access": np.zeros(size, dtype=np.float64),
        "access_missing": np.ones(size, dtype=np.float64),
        "last_access_modify_phone": np.zeros(size, dtype=np.float64),
        "last_access_override_alert": np.zeros(size, dtype=np.float64),
    }
    if access_logs.empty:
        return result
    _require(access_logs, ("account_id", "timestamp", "action_type"), "access_logs")
    left = pd.DataFrame(
        {
            "account_id": _clean_id(transactions["sender_account_id"]),
            "tx_time": timestamps,
            "position": np.arange(size),
        }
    ).dropna(subset=["account_id", "tx_time"])
    right = pd.DataFrame(
        {
            "account_id": _clean_id(access_logs["account_id"]),
            "access_time": pd.to_datetime(access_logs["timestamp"], utc=True, errors="coerce"),
            "action": access_logs["action_type"].astype("string").str.strip().str.upper(),
        }
    ).dropna(subset=["account_id", "access_time"])
    if left.empty or right.empty:
        return result
    left = left.sort_values("tx_time", kind="stable")
    right = right.sort_values("access_time", kind="stable")
    matched = pd.merge_asof(
        left, right, by="account_id", left_on="tx_time", right_on="access_time", direction="backward"
    )
    found = matched["access_time"].notna().to_numpy()
    positions = matched.loc[found, "position"].to_numpy(dtype=np.int64)
    if positions.size:
        result["access_missing"][positions] = 0.0
        result["seconds_since_access"][positions] = (
            matched.loc[found, "tx_time"] - matched.loc[found, "access_time"]
        ).dt.total_seconds().to_numpy(dtype=np.float64)
        actions = matched.loc[found, "action"].to_numpy(dtype=str)
        result["last_access_modify_phone"][positions] = actions == "MODIFY_PHONE"
        result["last_access_override_alert"][positions] = actions == "OVERRIDE_ALERT"
    return result


def _account_features(transactions: pd.DataFrame, accounts: pd.DataFrame) -> dict[str, np.ndarray]:
    """Attach sender-account status while preserving missing-account evidence."""
    size = len(transactions)
    result = {
        "is_dormant_sender": np.zeros(size, dtype=np.float64),
        "account_missing": np.ones(size, dtype=np.float64),
    }
    if accounts.empty:
        return result
    _require(accounts, ("account_id", "status"), "accounts")
    lookup = pd.DataFrame(
        {"account_id": _clean_id(accounts["account_id"]), "status": accounts["status"]}
    ).dropna(subset=["account_id"])
    if lookup["account_id"].duplicated().any():
        raise ValueError("accounts contains duplicate account_id values")
    statuses = _clean_id(transactions["sender_account_id"]).map(
        lookup.set_index("account_id")["status"]
    )
    result["account_missing"] = statuses.isna().to_numpy(dtype=np.float64)
    result["is_dormant_sender"] = statuses.astype("string").str.strip().str.upper().eq("DORMANT").fillna(False).to_numpy(dtype=np.float64)
    return result


def extract_features(
    transactions: RecordSource,
    access_logs: RecordSource | None = None,
    accounts: RecordSource | None = None,
    *,
    reporting_limit: float = 10_000.0,
    limit_proximity: float = 1_000.0,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return ``(X, FEATURE_NAMES)`` in input transaction order.

    Sources may be CSV paths, DataFrames, a single dictionary, or a sequence
    of dictionaries. Windows include the current transaction and both time
    boundaries. Missing amounts contribute zero to sums; missing timestamps
    have zero window features. Missingness flags distinguish these values.
    ``seconds_since_access`` is zero when no prior access exists, with
    ``access_missing`` set to one. The reporting limit is illustrative and
    configurable; proximity is not itself a conclusion of wrongdoing.
    """
    if not np.isfinite(reporting_limit) or reporting_limit <= 0:
        raise ValueError("reporting_limit must be a positive finite number")
    if not np.isfinite(limit_proximity) or limit_proximity < 0:
        raise ValueError("limit_proximity must be a nonnegative finite number")

    tx = _to_frame(transactions, "transactions").reset_index(drop=True)
    if tx.empty:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64), FEATURE_NAMES
    tx = _normalize_columns(tx, {"sender_account": "sender_account_id", "receiver_account": "receiver_account_id", "type": "channel"}, "transactions")
    _require(tx, ("sender_account_id", "amount", "timestamp"), "transactions")
    logs = _normalize_columns(_to_frame(access_logs, "access_logs"), {"action": "action_type"}, "access_logs")
    account_rows = _to_frame(accounts, "accounts")

    amount = pd.to_numeric(tx["amount"], errors="coerce").astype(float)
    amount = amount.mask(~np.isfinite(amount))
    timestamps = pd.to_datetime(tx["timestamp"], utc=True, errors="coerce")
    safe_amount = amount.fillna(0.0)
    sender = _clean_id(tx["sender_account_id"])
    columns: dict[str, np.ndarray] = {
        "amount": safe_amount.to_numpy(dtype=np.float64),
        "amount_missing": amount.isna().to_numpy(dtype=np.float64),
        "sender_missing": sender.isna().to_numpy(dtype=np.float64),
        "timestamp_missing": timestamps.isna().to_numpy(dtype=np.float64),
    }
    columns.update(_window_features(tx, safe_amount, timestamps))
    columns["near_reporting_limit"] = amount.sub(reporting_limit).abs().le(limit_proximity).fillna(False).to_numpy(dtype=np.float64)
    columns["below_reporting_limit"] = amount.lt(reporting_limit).fillna(False).to_numpy(dtype=np.float64)
    columns.update(_access_features(tx, timestamps, logs))
    columns.update(_account_features(tx, account_rows))
    columns["closes_3hop_cycle_48h"] = _cycle_features(tx, timestamps)
    X = np.column_stack([columns[name] for name in FEATURE_NAMES]).astype(np.float64, copy=False)
    return X, FEATURE_NAMES


def extract_case_features(
    events: Sequence[Mapping[str, Any]], *, as_of: str,
    observed_until: str | None = None, accounts: Sequence[Mapping[str, Any]] = (),
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the separate case matrix registry without changing legacy models."""
    from graph_temporal_features import CASE_FEATURE_NAMES, build_case_features
    return build_case_features(events, as_of=as_of, observed_until=observed_until,
                               accounts=accounts).matrix(), CASE_FEATURE_NAMES


__all__ = ["FEATURE_NAMES", "extract_features", "extract_case_features"]
