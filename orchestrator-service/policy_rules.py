"""
policy_rules.py

Established-wrongdoing rules (evidence_class = POLICY_BREACH).

These are deterministic checks over recorded facts: an employee action followed by a money
movement that bank policy forbids without an extra control. They are *not* predictions; the
ML model and the pattern detectors produce SUSPICION findings instead (see alert_engine.py).

| Rule | Policy breached |
|---|---|
| P1 OVERRIDE_THEN_TRANSFER     | Funds left an account within 2 h of an employee overriding its compliance alert |
| P2 UNFREEZE_THEN_TRANSFER     | Funds left a frozen account within 24 h of a manual unfreeze by an employee |
| P3 KYC_EDIT_THEN_LARGE_TRANSFER | A transfer >= 50,000 within 48 h of an employee changing the customer's phone number |
| P4 DORMANT_ACTIVATED_BY_STAFF | A DORMANT account sent funds within 48 h of a privileged employee action on it |

Each rule is evaluated on the insider-link rows of one case (risky access + following transfer),
so the result depends only on the evidence set, never on arrival order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

HOUR = 3_600


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    name: str
    title: str
    check: Callable[[dict[str, Any], float], bool]


def _gap(link: dict[str, Any]) -> float:
    return (datetime.fromisoformat(link["tx_time"]) - datetime.fromisoformat(link["access_time"])).total_seconds()


RULES: tuple[PolicyRule, ...] = (
    PolicyRule("P1", "OVERRIDE_THEN_TRANSFER", "Compliance alert overridden, funds moved within 2 h",
               lambda l, g: l["action_type"] == "OVERRIDE_ALERT" and g <= 2 * HOUR),
    PolicyRule("P2", "UNFREEZE_THEN_TRANSFER", "Manual unfreeze followed by outbound transfer within 24 h",
               lambda l, g: l["action_type"] == "MANUAL_UNFREEZE" and g <= 24 * HOUR),
    PolicyRule("P3", "KYC_EDIT_THEN_LARGE_TRANSFER", "Phone number changed, then >= $50,000 sent within 48 h",
               lambda l, g: l["action_type"] == "MODIFY_PHONE" and float(l["amount"]) >= 50_000 and g <= 48 * HOUR),
    PolicyRule("P4", "DORMANT_ACTIVATED_BY_STAFF", "Dormant account moved funds after a privileged staff action",
               lambda l, g: l["account_status"] == "DORMANT" and l["action_type"] in ("OVERRIDE_ALERT", "MANUAL_UNFREEZE")
               and bool(l.get("is_privileged")) and g <= 48 * HOUR),
)


def evaluate(links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one POLICY_BREACH finding per rule that fires, with the exact evidence ids."""
    findings = []
    for rule in RULES:
        hits = [l for l in links if rule.check(l, _gap(l))]
        if not hits:
            continue
        logs = sorted({h["log_id"] for h in hits})
        txs = sorted({h["tx_id"] for h in hits})
        accounts = sorted({h["account_id"] for h in hits})
        first = min(hits, key=lambda h: (h["access_time"], h["log_id"], h["tx_id"]))
        findings.append({
            "finding_id": rule.rule_id,
            "evidence_class": "POLICY_BREACH",
            "rule_id": rule.rule_id,
            "title": f"{rule.rule_id}: {rule.title}",
            "detail": (f"{first['employee_name']} ({first['emp_id']}) performed {first['action_type']} on "
                       f"{len(accounts)} account(s); {len(txs)} transfer(s) followed within the policy window. "
                       f"First: {first['log_id']} at {first['access_time'][:16]}Z, then {first['tx_id']} "
                       f"(${float(first['amount']):,.0f}) at {first['tx_time'][:16]}Z."),
            "evidence_refs": (logs + txs)[:16],
        })
    return findings


__all__ = ["RULES", "PolicyRule", "evaluate"]
