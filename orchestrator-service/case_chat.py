"""
case_chat.py

Case assistant ("chatbot") for one selected case. Deliberately narrow:

* Three intents: ``explain`` (Explain this alert), ``trace`` (Trace the money), ``changes``
  (What changed?). A free-text question is routed to the closest intent and answered from the
  same facts.
* Facts come only from controlled, read-only backend functions scoped to the case's own
  evidence (``fact_*`` below). The LLM never queries data and never writes anything.
* All amounts, totals and counts are calculated here in code. The LLM may only quote them.
* The answer is validated: every ID and every number it contains must appear in the facts, and it
  must not present suspicion as proven. Otherwise, or if no LLM is reachable, the reply is the
  deterministic fact-based answer built by ``render_*``.
* Suspicion (model and pattern signals) is always kept separate from established policy-rule
  matches.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Callable

from graph_queries import GraphQueries
from llm_explainer import ID_PATTERN, LlamaExplainer

INTENTS = ("explain", "trace", "changes")
NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
OVERCLAIM = re.compile(r"\b(proven|confirmed (?:fraud|laundering|wrongdoing)|definitely (?:guilty|fraud)|beyond doubt)\b", re.I)
DISCLAIMER = ("Answers use only this case's recorded evidence. Model scores and patterns are signals for review, "
              "not proof of wrongdoing.")

SYSTEM = (
    "You are the case assistant inside a financial-crime investigation tool. Answer ONLY from the FACTS given. "
    "Rules: (1) Never calculate or invent numbers; quote amounts, counts and scores exactly as written in FACTS. "
    "(2) Cite evidence IDs in square brackets, e.g. [TX-H-D1-1]. (3) Keep 'Established: policy rule matched' and "
    "'Suspicion: model or pattern signal' clearly separate, and never describe suspicion as proven. (4) If FACTS do "
    "not answer the question, say so. (5) At most 160 words, short paragraphs or '- ' bullets, no markdown headings. "
    'Respond with JSON only: {"answer": "...", "cited_ids": ["..."]}.'
)


def _money(x: float) -> str:
    return f"${x:,.2f}"


def route(question: str | None, intent: str | None) -> str:
    if intent in INTENTS:
        return intent
    q = (question or "").lower()
    if re.search(r"money|flow|trace|transfer|paid|sent|where.*(go|went)|amount|total|cycle|beneficiar", q):
        return "trace"
    if re.search(r"chang|revision|escalat|histor|before|late|version|why.*(now|up|priority)", q):
        return "changes"
    return "explain"


# ============================================================================ controlled fact functions

def fact_case(case: dict[str, Any]) -> dict[str, Any]:
    s = case["snapshot"]
    return {
        "case_id": case["case_id"], "title": s["title"], "risk_level": s["risk_level"],
        "score": f"{s['composite_score']}/100",
        "score_parts": f"graph & policy {s['graph_score']:.2f}, ML case model {s['ml_score']:.2f}",
        "workflow_status": case["workflow_status"], "assigned_to": case.get("assigned_to"),
        "employee": f"{s['employee_name']} ({s['emp_id']})" if s.get("emp_id") else None,
        "evidence_counts": s["counts"], "total_moved": _money(s["total_amount"]),
        "first_event": s.get("first_seen"), "last_event": s.get("last_seen"),
        "model_release": (s.get("model") or {}).get("release"),
    }


def fact_findings(case: dict[str, Any]) -> dict[str, Any]:
    findings = case["snapshot"]["findings"]
    pick = lambda cls: [{"rule": f["rule_id"], "title": f["title"], "detail": f["detail"],  # noqa: E731
                         "evidence_ids": f["evidence_refs"][:10]} for f in findings if f["evidence_class"] == cls]
    return {"established_policy_rule_matches": pick("POLICY_BREACH"), "suspicion_signals": pick("SUSPICION")}


def fact_model(case: dict[str, Any]) -> dict[str, Any]:
    cm = (case["snapshot"].get("ml") or {}).get("case_model") or {}
    drivers = [{"feature": e["feature"], "value": e["value"], "contribution": round(float(e["contribution_margin"]), 3),
                "supporting_ids": [i.split(":", 1)[-1] for i in (e.get("source_ids") or [])][:6]}
               for e in (cm.get("shap_evidence") or []) if e.get("direction") == "raises_class_margin"][:4]
    return {"case_model": {"predicted_scenario": cm.get("predicted_scenario"),
                           "priority_score": round(float(cm.get("case_priority_score") or 0), 3),
                           "threshold": round(float(cm.get("priority_threshold") or 0), 3),
                           "flagged": cm.get("is_flagged"), "novelty_method": cm.get("anomaly_method"),
                           "novelty_percentile": round(float(cm.get("anomaly_score") or 0), 3),
                           "top_drivers": drivers},
            "note": "Model scores are uncalibrated ranking scores, not probabilities of guilt."}


def fact_money(case: dict[str, Any], queries: GraphQueries) -> dict[str, Any]:
    """Every figure here is computed in code from the case's transfers."""
    txs = queries.transactions(case["snapshot"]["evidence"]["tx_ids"])
    if not txs:
        return {"transfers": 0}
    flows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    out_total: dict[str, float] = defaultdict(float)
    in_total: dict[str, float] = defaultdict(float)
    for t in txs:
        amount = float(t["amount"])
        flows[(t["sender_account"], t["receiver_account"])].append(t)
        out_total[t["sender_account"]] += amount
        in_total[t["receiver_account"]] += amount
    top_flows = sorted(flows.items(), key=lambda kv: -sum(float(t["amount"]) for t in kv[1]))[:6]
    largest = max(txs, key=lambda t: float(t["amount"]))
    accounts = sorted(set(out_total) | set(in_total))
    net = sorted(((a, in_total[a] - out_total[a]) for a in accounts), key=lambda x: x[1])
    near = [t for t in txs if 9_000 < float(t["amount"]) <= 10_000]
    external = {a: v for a, v in in_total.items() if a.startswith("EXT")}
    cycles = [{"path": " -> ".join(c["account_path"]), "transfer_ids": c["tx_path"],
               "hop_amounts": [_money(float(t["amount"])) for t in txs if t["tx_id"] in set(c["tx_path"])]}
              for c in case["snapshot"].get("cycles", [])]
    return {
        "transfers": len(txs), "total": _money(sum(float(t["amount"]) for t in txs)),
        "time_span": f"{txs[0]['timestamp'][:16]}Z to {txs[-1]['timestamp'][:16]}Z",
        "largest_transfer": {"id": largest["tx_id"], "amount": _money(float(largest["amount"])),
                             "from": largest["sender_account"], "to": largest["receiver_account"]},
        "top_flows": [{"from": s, "to": r, "transfers": len(ts), "total": _money(sum(float(t["amount"]) for t in ts)),
                       "transfer_ids": [t["tx_id"] for t in ts][:6]} for (s, r), ts in top_flows],
        "biggest_net_senders": [{"account": a, "net_outflow": _money(-v)} for a, v in net[:3] if v < 0],
        "biggest_net_receivers": [{"account": a, "net_inflow": _money(v)} for a, v in net[::-1][:3] if v > 0],
        "near_threshold_transfers": {"count": len(near), "total": _money(sum(float(t["amount"]) for t in near)),
                                     "ids": [t["tx_id"] for t in near][:8]},
        "to_external_beneficiaries": {"accounts": len(external), "total": _money(sum(external.values()))},
        "cycles": cycles,
        "accounts_involved": len(accounts),
    }


def fact_changes(history: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for rev in history[-8:]:
        prev = (rev.get("previous") or {}).get("risk_level")
        items.append({
            "case": rev["case_id"], "version": rev["version"], "at": rev["created_at"][:16] + "Z",
            "change": rev["change_type"], "risk": f"{prev} -> {rev['current']['risk_level']}" if prev else rev["current"]["risk_level"],
            "score": (f"{rev['previous']['composite_score']} -> {rev['current']['composite_score']}"
                      if rev.get("previous") else str(rev["current"]["composite_score"])),
            "new_patterns": rev["diff"].get("added_patterns", []), "new_policy_matches": rev["diff"].get("new_breaches", []),
            "trigger_events": [{"id": t["record_id"], "event_time": t["event_timestamp"][:16] + "Z", "late": t["late"],
                                "late_by_minutes": round(t["lateness_seconds"] / 60) if t.get("lateness_seconds") else None}
                               for t in rev["trigger"]["events"][:4]],
            "explanation": (rev.get("explanation") or {}).get("text"),
        })
    return {"revisions": items, "total_revisions": len(history)}


# ============================================================================ deterministic answers

def _ids(ids: list[str]) -> str:
    return " ".join(f"[{i}]" for i in ids[:6])


def render_explain(f: dict[str, Any]) -> str:
    c, fi, m = f["case"], f["findings"], f["model"]["case_model"]
    lines = [f"{c['case_id']} is {c['risk_level']} (score {c['score']}; {c['score_parts']})."]
    lines.append("Established: policy rule matched")
    lines += [f"- {x['rule']}: {x['title'].split(': ', 1)[-1]} {_ids(x['evidence_ids'])}" for x in fi["established_policy_rule_matches"]] \
        or ["- None. No policy rule is matched by the recorded facts."]
    lines.append("Suspicion: model and pattern signals (not proof)")
    lines += [f"- {x['title']} {_ids(x['evidence_ids'])}" for x in fi["suspicion_signals"]] or ["- None."]
    if m.get("top_drivers"):
        lines.append("Top case-model drivers")
        lines += [f"- {d['feature'].replace('_', ' ')}: {d['value']:g} {_ids(d['supporting_ids'][:3])}" for d in m["top_drivers"][:3]]
    return "\n".join(lines)


def render_trace(f: dict[str, Any]) -> str:
    m = f["money"]
    if not m.get("transfers"):
        return "This case has no transfers on record."
    lines = [f"{m['transfers']} transfers totalling {m['total']} ({m['time_span']}), across {m['accounts_involved']} accounts."]
    lines += [f"- {x['from']} -> {x['to']}: {x['total']} in {x['transfers']} transfer(s) {_ids(x['transfer_ids'][:3])}"
              for x in m["top_flows"][:4]]
    lg = m["largest_transfer"]
    lines.append(f"Largest single transfer: {lg['amount']} {lg['from']} -> {lg['to']} [{lg['id']}].")
    for cyc in m["cycles"]:
        lines.append(f"Funds returned to origin: {cyc['path']} {_ids(cyc['transfer_ids'])}.")
    if m["near_threshold_transfers"]["count"]:
        n = m["near_threshold_transfers"]
        lines.append(f"{n['count']} transfers ({n['total']}) sat just under the $10,000 reporting threshold.")
    if m["to_external_beneficiaries"]["accounts"]:
        e = m["to_external_beneficiaries"]
        lines.append(f"{e['total']} went to {e['accounts']} external beneficiary account(s).")
    return "\n".join(lines)


def render_changes(f: dict[str, Any]) -> str:
    revs = f["changes"]["revisions"]
    if not revs:
        return "No revisions are recorded for this case yet."
    lines = []
    for r in revs:
        late = [t for t in r["trigger_events"] if t["late"]]
        why = f" Trigger: {late[0]['id']} arrived {late[0]['late_by_minutes']} min late." if late else ""
        lines.append(f"- v{r['version']} ({r['case']}, {r['at']}): {r['change'].lower()}, {r['risk']}, score {r['score']}.{why}")
    last = revs[-1]
    if last.get("explanation"):
        lines.append(f"Latest change: {last['explanation']}")
    return "\n".join(lines)


RENDER: dict[str, Callable[[dict[str, Any]], str]] = {"explain": render_explain, "trace": render_trace,
                                                     "changes": render_changes}


# ============================================================================ validation

def _norm(token: str) -> str:
    return token.replace("$", "").replace(",", "").rstrip("%").rstrip(".")


def validate(answer: str, facts_text: str, allowed_ids: set[str], has_breach: bool) -> str | None:
    """Return a reason string if the answer is not grounded, else None."""
    if len(answer.strip()) < 20:
        return "answer too short"
    unknown = sorted({i for i in ID_PATTERN.findall(answer) if i not in allowed_ids and not i.startswith("CASE-")})
    if unknown:
        return f"cites ids not in this case: {unknown[:3]}"
    known_numbers = {_norm(t) for t in NUMBER.findall(facts_text)}
    known_numbers |= {n.split(".")[0] for n in known_numbers}
    for token in NUMBER.findall(answer):
        value = _norm(token)
        if not value or (value.isdigit() and int(value) <= 12):
            continue
        if value not in known_numbers and value.split(".")[0] not in known_numbers:
            return f"number not present in the facts: {token}"
    if OVERCLAIM.search(answer):
        return "presents suspicion as proven"
    if not has_breach and re.search(r"established (policy )?breach", answer, re.I) and "no " not in answer.lower():
        return "claims an established breach the facts do not contain"
    return None


# ============================================================================ entry point

async def answer(case: dict[str, Any], history: list[dict[str, Any]], queries: GraphQueries,
                 llm: LlamaExplainer | None, *, intent: str | None, question: str | None) -> dict[str, Any]:
    chosen = route(question, intent)
    facts: dict[str, Any] = {"case": fact_case(case)}
    if chosen == "explain":
        facts.update(findings=fact_findings(case), model=fact_model(case))
    elif chosen == "trace":
        facts["money"] = fact_money(case, queries)
    else:
        facts["changes"] = fact_changes(history)
    facts_text = json.dumps(facts, default=str)
    allowed_ids = set(ID_PATTERN.findall(facts_text)) | set(case["snapshot"]["evidence"]["account_ids"])
    has_breach = bool(case["snapshot"]["breaches"])
    fallback = RENDER[chosen](facts)
    base = {"case_id": case["case_id"], "intent": chosen, "disclaimer": DISCLAIMER,
            "facts_used": sorted(k for k in facts if k != "case")}

    if llm is not None:
        prompt = (f"QUESTION: {question or {'explain': 'Explain why this alert was raised.', 'trace': 'Trace the money in this case.', 'changes': 'What changed in this case and why?'}[chosen]}\n\n"
                  f"FACTS:\n{json.dumps(facts, indent=1, default=str)}\n\n"
                  "Return JSON exactly: {\"answer\": \"...\", \"cited_ids\": [\"...\"]}")
        try:
            raw, url = await llm._chat(SYSTEM, prompt, 500)
            text = str(raw.get("answer") or "").strip()
            problem = validate(text, facts_text, allowed_ids, has_breach)
            if problem is None:
                return {**base, "answer": text, "source": "llm", "model": llm.model, "llm_url": url,
                        "citations": sorted(set(ID_PATTERN.findall(text))), "validation": {"ok": True}}
            return {**base, "answer": fallback, "source": "facts", "model": None,
                    "citations": sorted(set(ID_PATTERN.findall(fallback))),
                    "validation": {"ok": False, "reason": f"LLM answer rejected ({problem}); showing computed facts"}}
        except Exception as exc:  # noqa: BLE001 - any LLM failure falls back to the computed facts
            reason = f"no chat LLM reachable ({type(exc).__name__}); showing computed facts"
            return {**base, "answer": fallback, "source": "facts", "model": None,
                    "citations": sorted(set(ID_PATTERN.findall(fallback))), "validation": {"ok": False, "reason": reason}}
    return {**base, "answer": fallback, "source": "facts", "model": None,
            "citations": sorted(set(ID_PATTERN.findall(fallback))), "validation": {"ok": True}}
