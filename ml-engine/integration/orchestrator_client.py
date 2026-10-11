"""Typed integration helper for Dhruv's gateway and the local ML API on :8001.

Use the gateway's observed arrival timestamps and as-of account snapshots.
This module performs no graph writes, case discovery or policy decisions.
"""
from __future__ import annotations

from typing import Any, Literal, Sequence

import httpx


def case_payload(*, case_id: str, as_of: str, observed_until: str,
                 events: Sequence[dict[str, Any]], accounts: Sequence[dict[str, str]],
                 revision: int = 1) -> dict[str, Any]:
    """Normalize v2 gateway naming into the supported ML event snapshot contract."""
    normalized = []
    for event in events:
        kind = event["event_type"]
        if kind in {"ACCOUNT", "EMPLOYEE"}:
            continue  # Reference records belong in supplied historical snapshots.
        if kind not in {"TRANSACTION", "ACCESS_LOG", "EMPLOYEE_ACTION"}:
            raise ValueError(f"unsupported ML event type: {kind}")
        event_time = event.get("event_time") or event.get("event_timestamp")
        arrival_time = event.get("arrival_time") or event.get("arrival_timestamp") or event.get("first_arrival")
        if not event_time or not arrival_time:
            raise ValueError("supply both recorded event time and gateway arrival time")
        normalized.append({"event_id": event["event_id"],
            "event_type": "EMPLOYEE_ACTION" if kind == "ACCESS_LOG" else kind,
            "event_time": event_time, "arrival_time": arrival_time, "payload": event["payload"]})
    return {"case_id": case_id, "revision": revision, "as_of": as_of,
            "observed_until": observed_until, "events": normalized, "accounts": list(accounts)}


class MLClient:
    """Reusable HTTP connection pool with explicit errors and model profiles."""
    def __init__(self, base_url: str, timeout_seconds: float = 20) -> None:
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds)

    def close(self) -> None:
        """Release pooled connections at gateway shutdown."""
        self.http.close()

    def catalog(self) -> dict[str, Any]:
        """Read feature versions before enabling a gateway/model integration."""
        response = self.http.get("/models")
        response.raise_for_status()
        return response.json()

    def score_case(self, payload: dict[str, Any], *,
                   profile: Literal["integrated", "amlsim"] = "integrated") -> dict[str, Any]:
        """Return case priority plus separate novelty and TreeSHAP evidence."""
        response = self.http.post("/score-case", params={"profile": profile}, json=payload)
        response.raise_for_status()
        return response.json()

    def score_employee(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Score complete daily audit aggregates when the six signals are available."""
        response = self.http.post("/score-employee", json=payload)
        response.raise_for_status()
        return response.json()
