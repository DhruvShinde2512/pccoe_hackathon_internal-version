"""Event replay harness: stream a dataset split to the gateway as JSON events.

Simulates the messy delivery a real event bus produces:

* ``in-order``   events sorted by event time (equivalent to a clean batch load)
* ``shuffled``   random order
* ``late``       a fraction of events is held back and released at the end
* ``duplicates`` some events are delivered twice (immediately or much later)
* ``chaos``      shuffled + late + duplicates together

Scripted demo:

* ``--scenario late-link-demo`` sends the D1 case: the 11:00-11:30 wires first
  (case appears), waits, then the 10:00 OVERRIDE_ALERT arrives late and the case
  escalates. Load the rest of the data first with ``--exclude-demo``.

Invariance proof (judged deliverable):

* ``--verify N`` resets the gateway, replays the split N times in different
  chaotic orders and checks that the final state fingerprint is identical.

Examples::

    python replay_script.py --mode in-order --exclude-demo
    python replay_script.py --scenario late-link-demo
    python replay_script.py --verify 5
    python replay_script.py --mode chaos --seed 7 --target http://100.x.x.1:8000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from schemas import EVENT_ID_PREFIX, EventEnvelope

DATA_DIR = Path(__file__).resolve().parent / "data"
DEMO_MARKER = "-D1"  # every D1 entity id contains this (ACC-H-D1-SRC, TX-H-D1-1, EMP-H-D1, ...)


# ============================================================================ events

def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _event(event_type: str, record_id: str, ts: str, payload: dict[str, Any]) -> dict[str, Any]:
    envelope = EventEnvelope(
        event_id=f"{EVENT_ID_PREFIX[event_type]}:{record_id}",
        event_type=event_type,
        event_timestamp=datetime.fromisoformat(ts),
        payload=payload,
    )
    return envelope.model_dump(mode="json", exclude_none=True)


def load_events(split_dir: Path) -> list[dict[str, Any]]:
    """Convert a split's CSVs into envelopes, sorted by event time."""
    transactions = _read(split_dir / "transactions.csv")
    logs = _read(split_dir / "access_logs.csv")
    first_ts = min(r["timestamp"] for r in transactions + logs)
    events: list[dict[str, Any]] = []
    for row in _read(split_dir / "employees.csv"):
        row["is_privileged"] = row["is_privileged"].lower() == "true"
        events.append(_event("EMPLOYEE", row["emp_id"], first_ts, row))
    for row in _read(split_dir / "accounts.csv"):
        row["balance"] = float(row["balance"])
        events.append(_event("ACCOUNT", row["account_id"], first_ts, row))
    for row in transactions:
        row["amount"] = float(row["amount"])
        events.append(_event("TRANSACTION", row["tx_id"], row["timestamp"], row))
    for row in logs:
        events.append(_event("ACCESS_LOG", row["log_id"], row["timestamp"], row))
    events.sort(key=lambda e: (e["event_timestamp"], e["event_id"]))
    return events


def is_demo(event: dict[str, Any]) -> bool:
    return DEMO_MARKER in event["event_id"] or any(
        DEMO_MARKER in str(v) for k, v in event["payload"].items() if k.endswith(("_id", "_account"))
    )


# ============================================================================ disorder

def apply_mode(events: list[dict[str, Any]], mode: str, rng: random.Random, *, late_fraction: float,
               dup_rate: float) -> list[dict[str, Any]]:
    """Return a delivery sequence. Reference data (employees/accounts) may also arrive late."""
    stream = list(events)
    if mode in ("shuffled", "chaos"):
        rng.shuffle(stream)
    if mode in ("late", "chaos"):
        held = set(rng.sample(range(len(stream)), int(len(stream) * late_fraction)))
        stream = [e for i, e in enumerate(stream) if i not in held] + [stream[i] for i in sorted(held)]
    if mode in ("duplicates", "chaos"):
        out: list[dict[str, Any]] = []
        for event in stream:
            out.append(event)
            if rng.random() < dup_rate:
                out.append(event)  # immediate redelivery
        for event in rng.sample(stream, int(len(stream) * dup_rate)):
            out.insert(rng.randrange(len(out) + 1), event)  # redelivered much later
        stream = out
    return stream


# ============================================================================ HTTP

def _http(method: str, url: str, body: Any | None = None, timeout: float = 600) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {url} -> HTTP {exc.code}: {exc.read().decode()[:300]}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach {url}: {exc.reason}. Is the gateway running?") from exc


def send(target: str, stream: list[dict[str, Any]], batch_size: int, *, quiet: bool = False) -> dict[str, int]:
    """POST the stream in micro-batches, stamping sent_at. Returns outcome counts."""
    totals: dict[str, int] = {}
    started = time.perf_counter()
    for start in range(0, len(stream), batch_size):
        batch = [{**e, "sent_at": datetime.now(timezone.utc).isoformat()} for e in stream[start:start + batch_size]]
        result = _http("POST", f"{target}/api/v1/events/ingest", {"events": batch})
        for item in result["results"]:
            totals[item["outcome"]] = totals.get(item["outcome"], 0) + 1
            totals["late"] = totals.get("late", 0) + int(bool(item.get("late")))
        totals["_rev"] = totals.get("_rev", 0) + result.get("revisions_created", 0)
        if not quiet:
            done = min(start + batch_size, len(stream))
            print(f"\r  sent {done}/{len(stream)}  case revisions so far: {totals['_rev']}", end="", flush=True)
    if not quiet:
        print(f"\n  done in {time.perf_counter() - started:.1f}s")
    return totals


def print_totals(totals: dict[str, int]) -> None:
    keys = ["ACCEPTED", "DUPLICATE_IGNORED", "CONFLICT_REJECTED", "INVALID", "late", "_rev"]
    names = {"late": "late (accepted)", "_rev": "case revisions created"}
    for key in keys:
        if totals.get(key):
            print(f"  {names.get(key, key):<24} {totals[key]}")


# ============================================================================ commands

def late_link_demo(target: str, events: list[dict[str, Any]], pause: float) -> None:
    demo = [e for e in events if is_demo(e)]
    if not demo:
        raise SystemExit("No D1 events found in this split (use the held_out_test_set).")
    override = [e for e in demo if e["event_type"] == "ACCESS_LOG"]
    first = [e for e in demo if e["event_type"] != "ACCESS_LOG"]
    print("Step 1: reference data + the 11:00-11:30 wires (the override that preceded them is withheld)")
    print_totals(send(target, first, 50, quiet=True))
    if pause < 0:
        input("\n  Case is visible in the dashboard. Press Enter to deliver the late 10:00 OVERRIDE_ALERT... ")
    else:
        print(f"\n  waiting {pause:.0f}s (watch the dashboard)...")
        time.sleep(pause)
    print("Step 2: the 10:00 OVERRIDE_ALERT arrives late")
    print_totals(send(target, override, 50, quiet=True))
    print("\nOpen the case in the dashboard: the revision history shows the escalation and why.")


def verify(target: str, events: list[dict[str, Any]], runs: int, batch_size: int, args: argparse.Namespace) -> int:
    modes = ["in-order"] + ["chaos"] * max(runs - 1, 0)
    rows = []
    for run, mode in enumerate(modes[:runs], start=1):
        _http("POST", f"{target}/api/v1/admin/reset")
        stream = apply_mode(events, mode, random.Random(1000 + run), late_fraction=args.late_fraction,
                            dup_rate=args.dup_rate)
        print(f"Run {run}/{runs}: {mode} ({len(stream)} deliveries for {len(events)} unique events)")
        totals = send(target, stream, batch_size, quiet=True)
        fp = _http("GET", f"{target}/api/v1/admin/fingerprint")
        rows.append((run, mode, totals, fp))
        print(f"  accepted={totals.get('ACCEPTED', 0)} duplicates_ignored={totals.get('DUPLICATE_IGNORED', 0)} "
              f"late={totals.get('late', 0)} cases={fp['active_cases']} fingerprint={fp['fingerprint'][:16]}")
    full = _http("POST", f"{target}/api/v1/admin/recompute-all")
    fingerprints = {fp["fingerprint"] for *_, fp in rows} | {full["fingerprint"]}
    print("\n  run  mode       accepted  duplicates  late   cases  fingerprint")
    for run, mode, totals, fp in rows:
        print(f"  {run:<4} {mode:<10} {totals.get('ACCEPTED', 0):<9} {totals.get('DUPLICATE_IGNORED', 0):<11} "
              f"{totals.get('late', 0):<6} {fp['active_cases']:<6} {fp['fingerprint'][:24]}")
    print(f"  full recompute-from-scratch fingerprint:          {full['fingerprint'][:24]}")
    ok = len(fingerprints) == 1
    print("\nRESULT:", "PASS - identical final case state in every order" if ok else "FAIL - fingerprints differ")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="http://localhost:8000", help="gateway base URL")
    parser.add_argument("--split", default="held_out_test_set", choices=["held_out_test_set", "validation_set", "train_set"])
    parser.add_argument("--mode", default="in-order", choices=["in-order", "shuffled", "late", "duplicates", "chaos"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--late-fraction", type=float, default=0.10)
    parser.add_argument("--dup-rate", type=float, default=0.05)
    parser.add_argument("--exclude-demo", action="store_true", help="skip the D1 late-link demo case")
    parser.add_argument("--reset", action="store_true", help="POST /admin/reset before sending")
    parser.add_argument("--scenario", choices=["late-link-demo"])
    parser.add_argument("--pause", type=float, default=-1, help="seconds between demo steps (-1 = wait for Enter)")
    parser.add_argument("--verify", type=int, metavar="N", help="replay N permutations and compare fingerprints")
    args = parser.parse_args()

    events = load_events(DATA_DIR / args.split)
    target = args.target.rstrip("/")
    if args.scenario == "late-link-demo":
        late_link_demo(target, events, args.pause)
        return
    if args.verify:
        sys.exit(verify(target, events, args.verify, args.batch_size, args))
    if args.exclude_demo:
        events = [e for e in events if not is_demo(e)]
    if args.reset:
        _http("POST", f"{target}/api/v1/admin/reset")
    stream = apply_mode(events, args.mode, random.Random(args.seed), late_fraction=args.late_fraction,
                        dup_rate=args.dup_rate)
    print(f"Replaying {args.split}: {len(events)} events as {len(stream)} deliveries ({args.mode}) -> {target}")
    print_totals(send(target, stream, args.batch_size))
    fp = _http("GET", f"{target}/api/v1/admin/fingerprint")
    print(f"\nFinal state: {fp['active_cases']} active cases, fingerprint {fp['fingerprint'][:24]}")


if __name__ == "__main__":
    main()
