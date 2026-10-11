"""Verify ML snapshot replay invariants; this is not gateway ingestion/revision storage."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
from typing import Any

from case_inference import CaseScoreRequest, load_case_bundle, score_case
from evaluate import read_cases
from graph_temporal_features import utc


def verify(dataset: Path, artifact: Path, output: Path) -> dict[str, Any]:
    """Save measured before/after late evidence and delivery-order comparisons."""
    bundle = load_case_bundle(artifact)
    cases = [r for r in read_cases(dataset) if r["split"] == "test"]
    results = []
    late_example = None
    for case in cases:
        payload = {key: case[key] for key in ("case_id", "as_of", "observed_until", "events", "accounts")}
        final = score_case(CaseScoreRequest.model_validate(payload), bundle)
        shuffled = deepcopy(payload)
        shuffled["events"] = list(reversed(payload["events"])) + [payload["events"][0]]
        replayed = score_case(CaseScoreRequest.model_validate(shuffled), bundle)
        keys = ("feature_values", "evidence_hash", "evidence_count", "case_priority_score", "priority")
        if any(final[key] != replayed[key] for key in keys):
            raise AssertionError(f"replay invariant failed: {case['case_id']}")
        results.append({"case_id": case["case_id"], "final_priority": final["priority"], "duplicate_and_order_invariant": True})
        late = [e for e in case["events"] if (utc(e["arrival_time"]) - utc(e["event_time"])).total_seconds() >= 3600]
        if late:
            earlier = {**payload, "observed_until": (min(utc(e["arrival_time"]) for e in late) - timedelta(seconds=1)).isoformat()}
            before = score_case(CaseScoreRequest.model_validate(earlier), bundle)
            example = {"case_id": case["case_id"], "scope": "ML snapshots; gateway must persist actual case revisions",
                       "change_reason": "Previously unavailable employee evidence arrived before final observation cutoff",
                       "before": before, "after": final}
            if late_example is None or (before["priority"] != final["priority"] and late_example["before"]["priority"] == late_example["after"]["priority"]):
                late_example = example
    report = {"model_id": bundle["model_id"], "cases_checked": len(results), "cases": results,
              "late_evidence_example": late_example}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return {"cases_checked": len(results), "all_duplicate_and_order_invariant": True,
            "late_case_id": late_example["case_id"] if late_example else None,
            "late_priority_before": late_example["before"]["priority"] if late_example else None,
            "late_priority_after": late_example["after"]["priority"] if late_example else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dataset", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.case_dataset, args.artifact, args.output), indent=2))
