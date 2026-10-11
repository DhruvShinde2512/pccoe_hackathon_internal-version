"""
dossier_pdf.py

Renders a case dossier (as assembled by ``gateway.export_dossier``) to a PDF using fpdf2.
Core PDF fonts are Latin-1 only, so text is transliterated to keep rendering robust.
"""

from typing import Any, Dict

from fpdf import FPDF

MAX_TIMELINE_ROWS = 120
_REPLACEMENTS = {"→": "->", "—": "-", "–": "-", "’": "'", "“": '"', "”": '"', "•": "*"}


def _t(value: Any) -> str:
    text = "" if value is None else str(value)
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


class _Dossier(FPDF):
    def __init__(self, case_id: str):
        super().__init__(format="A4")
        self.case_id = case_id
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)

    def header(self) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(110)
        self.cell(0, 6, _t(f"CONFIDENTIAL - Evidence Dossier {self.case_id}"), align="L")
        self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)
        self.ln(2)

    def section(self, title: str) -> None:
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 7, _t(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def para(self, text: str, size: int = 10, style: str = "") -> None:
        self.set_font("Helvetica", style, size)
        self.multi_cell(0, 5, _t(text), new_x="LMARGIN", new_y="NEXT")

    def kv(self, key: str, value: Any) -> None:
        self.set_font("Helvetica", "B", 10)
        self.cell(45, 5, _t(key))
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, _t(value), new_x="LMARGIN", new_y="NEXT")


def render_pdf(dossier: Dict[str, Any]) -> bytes:
    alert = dossier["alert"]
    explanation = dossier["explanation"]
    evidence = dossier.get("evidence", {})
    pdf = _Dossier(dossier["case_id"])
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, _t(alert["title"]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.kv("Case ID", dossier["case_id"])
    pdf.kv("Risk level", f"{alert['risk_level']} (composite {alert['composite_score']}/100)")
    pdf.kv("Scores", f"graph {alert['graph_score']:.2f} | ML percentile {alert['ml_score']:.2f}")
    pdf.kv("Status", f"{alert['status']}" + (f" - assigned to {alert['assigned_to']}" if alert.get("assigned_to") else ""))
    pdf.kv("Patterns", ", ".join(alert.get("patterns", [])))
    if dossier.get("fraud_window"):
        pdf.kv("Fraud window", f"{dossier['fraud_window']['start']}  to  {dossier['fraud_window']['end']}")
    pdf.kv("Generated", dossier["generated_at"])

    pdf.section("Executive summary")
    pdf.para(explanation["summary"])
    pdf.set_font("Helvetica", "I", 8)
    confidence = explanation.get("confidence")
    pdf.cell(0, 5, _t(f"Narrative source: {explanation.get('model')}"
                      + (f", confidence {confidence:.2f}" if confidence is not None else "")),
             new_x="LMARGIN", new_y="NEXT")

    if explanation.get("breach_tags"):
        pdf.section("Policy / regulatory concerns")
        pdf.para(" | ".join(explanation["breach_tags"]))

    pdf.section("Audit rationale")
    for step in explanation.get("rationale", []):
        pdf.para(f"{step['step']}. {step['title']}", style="B")
        if step.get("detail"):
            pdf.para(step["detail"])
        if step.get("evidence_refs"):
            pdf.para("Evidence: " + ", ".join(step["evidence_refs"][:12]), size=8, style="I")
        pdf.ln(1)

    findings = dossier.get("findings", [])
    for cls, heading in (("POLICY_BREACH", "Established policy breaches (facts)"),
                         ("SUSPICION", "Suspicion (model and pattern signals, not proof)")):
        items = [f for f in findings if f.get("evidence_class") == cls]
        if items:
            pdf.section(heading)
            for f in items:
                pdf.para(f["title"], style="B")
                pdf.para(f["detail"], size=9)
                if f.get("evidence_refs"):
                    pdf.para("Evidence: " + ", ".join(f["evidence_refs"][:12]), size=8, style="I")
                pdf.ln(1)

    pdf.section("Recommended action")
    pdf.para(explanation.get("recommended_action") or evidence.get("recommended_action", ""))

    revisions = dossier.get("revisions", [])
    if revisions:
        pdf.section(f"Case revision history ({len(revisions)} versions)")
        for rev in revisions:
            prev = (rev.get("previous") or {}).get("risk_level")
            change = f"{prev} -> {rev['current']['risk_level']}" if prev else rev["current"]["risk_level"]
            pdf.para(f"v{rev['version']}  {rev['created_at'][:19]}  {rev['change_type']}  ({change})", style="B", size=9)
            explanation_rev = rev.get("explanation") or {}
            pdf.para(f"{explanation_rev.get('text', '')} [{explanation_rev.get('source', 'template')}]", size=9)
            pdf.ln(1)

    if evidence.get("employee"):
        pdf.section("Employee")
        for key, value in evidence["employee"].items():
            pdf.kv(key, value)

    pdf.section(f"Linked accounts ({len(dossier['associated_account_ids'])})")
    pdf.para(", ".join(dossier["associated_account_ids"]), size=8)

    timeline = dossier.get("timeline", [])
    pdf.section(f"Event timeline ({len(timeline)} events)")
    pdf.set_font("Courier", "", 7)
    for event in timeline[:MAX_TIMELINE_ROWS]:
        pdf.multi_cell(0, 3.6, _t(f"{event['timestamp'][:19]}  {event['category']:<13} {event['event_id']:<14} "
                                  f"{event.get('description', '')}"), new_x="LMARGIN", new_y="NEXT")
    if len(timeline) > MAX_TIMELINE_ROWS:
        pdf.para(f"... {len(timeline) - MAX_TIMELINE_ROWS} more events in the JSON export.", size=8, style="I")

    if dossier.get("case_notes"):
        pdf.section("Case activity")
        for note in dossier["case_notes"]:
            pdf.para(note, size=9)

    return bytes(pdf.output())
