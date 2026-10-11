"""AMLSim import gates: label isolation, stable evidence, and grouped splitting."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amlsim_adapter import adapt_export
from graph_temporal_features import build_case_features


def export(tmp_path: Path, sar: str = "1") -> Path:
    source = tmp_path / "tx_log.csv"
    source.write_text("step,type,amount,nameOrig,nameDest,isSAR,alertID\n"
                      f"0,TRANSFER,100,A,B,{sar},12\n"
                      "1,TRANSFER,90,B,C,0,-1\n"
                      "2,CASH-IN,400,external,A,0,-1\n", encoding="utf-8")
    return source


def test_labels_do_not_change_features_or_event_ids(tmp_path):
    path = export(tmp_path)
    first = adapt_export(path, run_id="run-01", split="train", base_time="2026-01-01T00:00:00Z")
    second = adapt_export(export(tmp_path, "0"), run_id="run-01", split="train", base_time="2026-01-01T00:00:00Z")
    assert len(first) == len(second) == 1
    assert first[0]["label"] == "SUSPICIOUS"
    assert second[0]["label"] == "BENIGN"
    assert first[0]["events"] == second[0]["events"]
    assert first[0]["group_id"] == "run-01"
    assert all("isSAR" not in e["payload"] and "alertID" not in e["payload"] for e in first[0]["events"])
    assert build_case_features(first[0]["events"], as_of=first[0]["as_of"]).values["transaction_count"] == 2


def test_exports_namespace_accounts_and_keep_components_in_group(tmp_path):
    path = export(tmp_path)
    a = adapt_export(path, run_id="r1", split="train", base_time="2026-01-01T00:00:00Z")[0]
    b = adapt_export(path, run_id="r2", split="test", base_time="2026-03-01T00:00:00Z")[0]
    assert a["events"][0]["payload"]["sender_account_id"] != b["events"][0]["payload"]["sender_account_id"]
    with pytest.raises(ValueError, match="isSAR"):
        adapt_export(export(tmp_path, "unknown"), run_id="r1", split="train", base_time="2026-01-01T00:00:00Z")
