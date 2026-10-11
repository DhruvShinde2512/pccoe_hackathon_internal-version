"""
llm_explainer.py

Local Qwen3.5-9B (llama.cpp ``llama-server``, OpenAI-compatible API) explanations:

1. **Case narrative**: an audit note for one case (v1 behaviour), from the case's facts.
2. **Revision explanation** (FIN-04): why a case's priority changed between two versions. The
   model receives only the structured diff, the trigger events and both state summaries, never
   two full evidence dumps. Its answer is validated: every ID it cites must be in the diff or
   triggers, and the stated from/to risk levels must match. Otherwise the deterministic
   template (already stored with the revision) is kept.

Failover: ``LLM_BASE_URLS`` is tried in order (for example Windows GPU first, Mac second).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ID_PATTERN = re.compile(r"\b(?:TX|LOG|ACC|EXT|EMP)-[A-Z0-9-]+\b")

NARRATIVE_SYSTEM = (
    "You are a senior Financial Crime Compliance investigator writing an audit note for a human reviewer. "
    "Use ONLY the evidence provided; never invent names, amounts, dates or IDs. Cite evidence by the exact IDs given. "
    "Distinguish established policy breaches (facts, rule IDs P1-P4) from suspicion (model probabilities and "
    "patterns, which are not proof). Respond with a single JSON object and nothing else."
)
NARRATIVE_SHAPE = """{
  "summary": "2-3 sentences: who did what, to which accounts, and why it matters",
  "breach_tags": ["short policy / regulation labels"],
  "rationale": [{"title": "short step title", "detail": "one or two sentences", "evidence_refs": ["IDs cited"]}],
  "recommended_action": "one clear directive for the reviewer",
  "confidence": 0.0
}"""

REVISION_SYSTEM = (
    "You are an AI financial-crime investigator auditing a change in a case's priority. You are given the case state "
    "before (V_prev) and after (V_new), the structured difference, and the events that triggered the change, including "
    "whether they arrived late. Explain strictly WHY the priority changed, in one or two sentences, in the style: "
    "'Case escalated from MEDIUM to CRITICAL because newly arrived late evidence linked the sender to an unauthorized "
    "override at 10:00.' Use only IDs, times and facts present in the input. Respond with JSON only."
)
REVISION_SHAPE = """{"explanation": "one or two sentences", "cited_ids": ["IDs you mention"]}"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_from(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1] if start != -1 and end > start else text)


class LlamaExplainer:
    def __init__(self, base_urls: list[str], model: str, timeout: float = 180.0):
        self.base_urls = [u.rstrip("/") for u in base_urls]
        self.model = model
        self.timeout = timeout
        self._healthy: tuple[float, str | None] = (0.0, None)

    # ---------------------------------------------------------------- transport

    async def active_url(self, max_age: float = 10.0) -> str | None:
        """First healthy server in priority order (cached briefly)."""
        checked, url = self._healthy
        if time.monotonic() - checked < max_age:
            return url
        url = None
        async with httpx.AsyncClient(timeout=2.0) as client:
            for candidate in self.base_urls:
                try:
                    if (await client.get(f"{candidate}/health")).status_code == 200:
                        url = candidate
                        break
                except httpx.HTTPError:
                    continue
        self._healthy = (time.monotonic(), url)
        return url

    async def _chat(self, system: str, user: str, max_tokens: int) -> tuple[dict[str, Any], str]:
        last_error: Exception | None = None
        preferred = await self.active_url()
        for url in ([preferred] if preferred else []) + [u for u in self.base_urls if u != preferred]:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(f"{url}/v1/chat/completions", json={
                        "model": self.model,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        "temperature": 0.1, "top_p": 0.9, "max_tokens": max_tokens,
                        "response_format": {"type": "json_object"},
                        "chat_template_kwargs": {"enable_thinking": False},
                    })
                    response.raise_for_status()
                content = response.json()["choices"][0]["message"].get("content") or ""
                return _json_from(content), url
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                last_error = exc
                self._healthy = (0.0, None)
        raise RuntimeError(f"no LLM server answered: {last_error!r}")

    # ---------------------------------------------------------------- case narrative

    async def narrative(self, case: dict[str, Any], timeline: list[dict[str, Any]]) -> dict[str, Any]:
        brief = {k: case.get(k) for k in ("case_id", "title", "description", "risk_level", "composite_score",
                                          "graph_score", "ml_score", "patterns", "counts", "total_amount",
                                          "employee_name", "emp_id")}
        brief["findings"] = [{k: f[k] for k in ("evidence_class", "rule_id", "title", "detail", "evidence_refs")}
                             for f in case.get("findings", [])]
        brief["timeline"] = [{k: v for k, v in e.items() if k in ("event_id", "timestamp", "category", "description",
                                                                  "amount", "late") and v not in (None, "")}
                             for e in timeline[:40]]
        user = (f"Write the audit note for this case.\n\nEVIDENCE:\n{json.dumps(brief, indent=1, default=str)}\n\n"
                f"Return JSON exactly in this shape:\n{NARRATIVE_SHAPE}\nUse 3-5 rationale steps in time order.")
        raw, url = await self._chat(NARRATIVE_SYSTEM, user, 1200)
        summary = str(raw.get("summary") or "").strip()
        if not summary:
            raise ValueError("LLM narrative has no summary")
        steps = []
        for i, step in enumerate(raw.get("rationale") or []):
            step = step if isinstance(step, dict) else {"title": str(step)}
            steps.append({"step": i + 1, "title": str(step.get("title") or f"Step {i + 1}"),
                          "detail": str(step.get("detail") or ""),
                          "evidence_refs": [str(r) for r in step.get("evidence_refs") or []]})
        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence"))))
        except (TypeError, ValueError):
            confidence = None
        return {"summary": summary, "breach_tags": [str(t) for t in raw.get("breach_tags") or []], "rationale": steps,
                "recommended_action": str(raw.get("recommended_action") or ""), "model": self.model,
                "generated_at": _now(), "confidence": confidence, "llm_url": url}

    @staticmethod
    def rule_based(case: dict[str, Any], note: str) -> dict[str, Any]:
        steps = [{"step": i + 1, "title": f["title"], "detail": f["detail"], "evidence_refs": f["evidence_refs"][:8]}
                 for i, f in enumerate(case.get("findings", []))]
        action = ("Escalate to the MLRO, freeze the linked accounts and suspend the employee's privileged access "
                  "pending review." if case.get("risk_level") in ("CRITICAL", "HIGH")
                  else "Assign to an L1 analyst for manual review of the linked activity.")
        return {"summary": f"{case.get('title', 'Case')}. {case.get('description', '')} ({note})",
                "breach_tags": case.get("breach_tags", []), "rationale": steps, "recommended_action": action,
                "model": "rule-based", "generated_at": _now(), "confidence": None}

    # ---------------------------------------------------------------- revision explanation

    async def revision(self, rev: dict[str, Any]) -> dict[str, Any]:
        brief = {
            "case_id": rev["case_id"], "change_type": rev["change_type"],
            "V_prev": rev["previous"], "V_new": rev["current"], "difference": rev["diff"],
            "trigger_events": [{k: e[k] for k in ("record_id", "event_type", "event_timestamp", "late",
                                                  "lateness_seconds", "detail")} for e in rev["trigger"]["events"]],
            "merged_from": rev.get("merged_from") or None,
        }
        user = (f"INPUT:\n{json.dumps(brief, indent=1, default=str)}\n\n"
                f"Return JSON exactly in this shape:\n{REVISION_SHAPE}")
        raw, url = await self._chat(REVISION_SYSTEM, user, 300)
        text = str(raw.get("explanation") or "").strip()
        self.validate_revision(text, rev)
        return {"text": text, "source": "llm", "model": self.model, "generated_at": _now(), "llm_url": url}

    @staticmethod
    def validate_revision(text: str, rev: dict[str, Any]) -> None:
        """Reject explanations that cite unknown IDs or misstate the risk change."""
        if len(text) < 20:
            raise ValueError("explanation too short")
        allowed = {rev["case_id"], *(rev.get("merged_from") or []), *(filter(None, [rev.get("merged_into")]))}
        for event in rev["trigger"]["events"]:
            allowed |= {event["record_id"], *event["touched"]}
        for ids in rev["diff"].get("added_evidence", {}).values():
            allowed |= set(ids)
        unknown = [i for i in ID_PATTERN.findall(text) if i not in allowed and not i.startswith("CASE-")]
        if unknown:
            raise ValueError(f"cites ids not in the evidence: {unknown[:3]}")
        prev, cur = (rev.get("previous") or {}).get("risk_level"), rev["current"]["risk_level"]
        levels = re.findall(r"\b(LOW|MEDIUM|HIGH|CRITICAL)\b", text.upper())
        if prev and prev != cur and levels and (prev not in levels or cur not in levels):
            raise ValueError(f"misstates the change ({prev} -> {cur})")
