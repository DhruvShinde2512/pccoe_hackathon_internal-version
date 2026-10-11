"""
gateway.py

FastAPI gateway (port 8000) for Drake&DonT v2. The dashboard and the replay harness talk only
to this service. Routes are under ``/api/v1``:

Streaming   POST /events/ingest, GET /events/stats
Cases       GET /alerts, GET /alerts/{id}/investigation, GET /cases/{id}/revisions,
            POST /cases/{id}/assign|freeze|escalate, GET /cases/{id}/dossier, GET /reviewers
Platform    GET /system/status
Admin       POST /admin/reset, POST /admin/recompute-all, GET /admin/fingerprint   (ALLOW_ADMIN)

LLM work (case narratives, revision explanations) runs on one background worker in priority
order, because llama-server serves one request at a time.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import httpx
from fastapi import APIRouter, Body, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import case_chat
from alert_engine import AlertEngine, MLUnavailable
from case_manager import CaseManager
from config import settings
from db_loader import Neo4jStore
from event_ingest import IngestPipeline
from graph_queries import GraphQueries
from llm_explainer import LlamaExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gateway")

REVISION_LLM_TYPES = {"ESCALATED", "DEESCALATED", "MERGED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================================ LLM job queue

class JobQueue:
    """Single consumer, priority ordered, one cached future per job key."""

    def __init__(self) -> None:
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.jobs: dict[str, tuple[asyncio.Future, Callable[[], Awaitable[Any]]]] = {}
        self.seq = itertools.count()
        self.worker: asyncio.Task | None = None

    def start(self) -> None:
        self.worker = asyncio.create_task(self._run())

    def clear(self) -> None:
        self.jobs.clear()

    def submit(self, key: str, factory: Callable[[], Awaitable[Any]], priority: int) -> asyncio.Future:
        existing = self.jobs.get(key)
        if existing and not (existing[0].done() and existing[0].exception()):
            if not existing[0].done():
                self.queue.put_nowait((priority, next(self.seq), key))  # re-prioritise; stale entries skipped
            return existing[0]
        future = asyncio.get_running_loop().create_future()
        self.jobs[key] = (future, factory)
        self.queue.put_nowait((priority, next(self.seq), key))
        return future

    def done(self, key: str) -> Any | None:
        job = self.jobs.get(key)
        return job[0].result() if job and job[0].done() and not job[0].exception() else None

    @property
    def pending(self) -> int:
        return sum(not f.done() for f, _ in self.jobs.values())

    async def _run(self) -> None:
        while True:
            _, _, key = await self.queue.get()
            job = self.jobs.get(key)
            if not job or job[0].done():
                continue
            future, factory = job
            started = time.perf_counter()
            try:
                future.set_result(await factory())
                logger.info(f"LLM job {key} done in {time.perf_counter() - started:.1f}s")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"LLM job {key} failed: {exc!r}")
                future.set_exception(exc)
                future.exception()


# ============================================================================ state

class State:
    store: Neo4jStore
    queries: GraphQueries
    engine: AlertEngine
    cases: CaseManager
    pipeline: IngestPipeline
    explainer: LlamaExplainer
    chat_llm: LlamaExplainer
    jobs: JobQueue
    investigations: dict[tuple[str, str], dict[str, Any]] = {}


state = State()


def _to_api(case: dict[str, Any]) -> dict[str, Any]:
    snap = case["snapshot"]
    return {
        "alert_id": case["case_id"], "case_id": case["case_id"], "title": snap["title"],
        "description": snap["description"], "risk_level": snap["risk_level"],
        "composite_score": snap["composite_score"], "graph_score": snap["graph_score"], "ml_score": snap["ml_score"],
        "status": case["workflow_status"], "emp_id": snap["emp_id"], "employee_name": snap["employee_name"],
        "created_at": snap["last_seen"], "opened_at": case.get("created_at"), "last_changed_at": case.get("updated_at"),
        "assigned_to": case.get("assigned_to"), "patterns": snap["patterns"], "breaches": snap["breaches"],
        "version": case.get("current_version", 1), "revision_count": case.get("current_version", 1),
        "evidence_counts": snap["counts"], "ml_available": snap.get("ml_available", True),
        "merged_from": sorted(c["case_id"] for c in state.cases.cases.values() if c.get("merged_into") == case["case_id"]),
    }


def _case(case_id: str) -> dict[str, Any]:
    case = state.cases.resolve(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
    return case


async def _investigation(case: dict[str, Any]) -> dict[str, Any]:
    key = (case["case_id"], case.get("updated_at") or "")
    if key not in state.investigations:
        ev = case["snapshot"]["evidence"]
        ledger = await run_in_threadpool(state.store.ledger_events, [f"TX:{t}" for t in ev["tx_ids"]]
                                         + [f"LOG:{l}" for l in ev["log_ids"]])
        state.investigations[key] = await run_in_threadpool(state.engine.investigation, case["snapshot"], ledger)
    return state.investigations[key]


def _history(case_id: str) -> list[dict[str, Any]]:
    """Revisions of a case plus those of every case merged into it (full lineage), oldest first."""
    lineage, frontier = [case_id], [case_id]
    while frontier:
        parent = frontier.pop()
        children = sorted(c["case_id"] for c in state.cases.cases.values() if c.get("merged_into") == parent)
        lineage += children
        frontier += children
    revisions = [r for cid in lineage for r in state.cases.revisions(cid)]
    # Same instant: lineage entries first, so this case's own revision reads as the latest.
    return sorted(revisions, key=lambda r: (r["created_at"], r["case_id"] == case_id, r["version"]))


def _queue_narrative(case: dict[str, Any], priority: int) -> asyncio.Future:
    """Queue the LLM audit narrative for the case's current version (built lazily in the worker)."""
    case_id, version = case["case_id"], case.get("current_version")

    async def job() -> dict[str, Any] | None:
        current = state.cases.cases.get(case_id)
        if not current or current.get("current_version") != version:
            return None  # superseded by a newer version before the worker got to it
        bundle = await _investigation(current)
        return await state.explainer.narrative(dict(current["snapshot"]), bundle["timeline"])

    return state.jobs.submit(f"narr:{case_id}:v{version}", job, priority)


async def _explanation(case: dict[str, Any], wait: float) -> dict[str, Any]:
    future = _queue_narrative(case, priority=0)
    try:
        result = await asyncio.wait_for(asyncio.shield(future), timeout=wait)
        if result:
            return result
        note = "AI summary is being updated for the latest version"
    except asyncio.TimeoutError:
        note = "AI summary is still being written; it will appear automatically"
    except Exception as exc:  # noqa: BLE001
        note = "AI summary unavailable: no LLM server reachable"
    return LlamaExplainer.rule_based(case["snapshot"], note)


def _queue_revision_explanations(revisions: list[dict[str, Any]]) -> None:
    """LLM 'why did priority change' for meaningful changes involving HIGH/CRITICAL."""
    for rev in revisions:
        prev = (rev.get("previous") or {}).get("risk_level")
        touches_high = {prev, rev["current"]["risk_level"]} & {"HIGH", "CRITICAL"}
        if rev["change_type"] not in REVISION_LLM_TYPES or not rev.get("previous") or not touches_high:
            continue

        async def job(rev: dict[str, Any] = rev) -> dict[str, Any]:
            explanation = await state.explainer.revision(rev)
            await run_in_threadpool(state.cases.set_explanation, rev["case_id"], rev["version"], explanation)
            return explanation

        state.jobs.submit(f"rev:{rev['case_id']}:v{rev['version']}", job, priority=1)


async def _prewarm() -> None:
    for case in sorted(state.cases.active().values(), key=lambda c: -c["snapshot"]["composite_score"]):
        if case["snapshot"]["risk_level"] in ("CRITICAL", "HIGH"):
            _queue_narrative(case, priority=3)


@asynccontextmanager
async def lifespan(application: FastAPI):
    state.store = Neo4jStore()
    state.queries = GraphQueries(state.store)
    state.engine = AlertEngine(state.queries, settings)
    state.cases = CaseManager(state.store)
    state.pipeline = IngestPipeline(state.store, state.queries, state.engine, state.cases, settings)
    state.explainer = LlamaExplainer(settings.LLM_BASE_URLS, settings.LLM_MODEL, settings.LLM_TIMEOUT_SECONDS)
    state.chat_llm = LlamaExplainer(settings.CHAT_LLM_URLS, settings.CHAT_LLM_MODEL, settings.CHAT_TIMEOUT_SECONDS)
    state.jobs = JobQueue()
    try:
        await run_in_threadpool(state.store.setup_schema)
        await run_in_threadpool(state.cases.load)
        events = (await run_in_threadpool(state.store.ledger_counts))["accepted"]
        if events and not state.cases.cases:
            logger.info("Ledger has events but no cases: rebuilding case state")
            await run_in_threadpool(state.pipeline.recompute_all)
        logger.info(f"Loaded {len(state.cases.active())} active cases from Neo4j ({events} events in ledger)")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Neo4j not ready at startup ({exc!r}); routes will retry")
    state.jobs.start()
    prewarm = asyncio.create_task(_prewarm())
    yield
    prewarm.cancel()
    state.store.close()


app = FastAPI(title="Drake&DonT Gateway", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"], expose_headers=["Content-Disposition", "X-Request-ID"])
api = APIRouter(prefix="/api/v1")


class AssignRequest(BaseModel):
    reviewer_id: str = Field(..., min_length=1)


class FreezeRequest(BaseModel):
    account_ids: list[str] = Field(..., min_length=1)


class EscalateRequest(BaseModel):
    note: str = ""


# ============================================================================ streaming

@api.post("/events/ingest")
async def ingest(body: Any = Body(...)) -> dict[str, Any]:
    """Accepts one envelope, a list of envelopes, or ``{"events": [...]}``."""
    events = body.get("events") if isinstance(body, dict) and "events" in body else body
    events = events if isinstance(events, list) else [events]
    if len(events) > settings.INGEST_MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"max {settings.INGEST_MAX_BATCH} events per request")
    try:
        result = await run_in_threadpool(state.pipeline.process, events)
    except MLUnavailable as exc:
        raise HTTPException(status_code=503, detail={
            "status": "ML_UNAVAILABLE", "message": str(exc),
            "action": "Batch rejected and not recorded; redeliver it once the ML node is back."}) from exc
    except Exception as exc:
        logger.exception("ingest failed")
        raise HTTPException(status_code=503, detail=f"ingest failed: {exc}") from exc
    _queue_revision_explanations(result["revisions"])
    for case_id in result["affected_case_ids"]:
        case = state.cases.cases.get(case_id)
        if case and case["workflow_status"] != "MERGED" and case["snapshot"]["risk_level"] in ("CRITICAL", "HIGH"):
            _queue_narrative(case, priority=3)
    return {k: v for k, v in result.items() if k != "revisions"}


@api.get("/events/stats")
async def event_stats() -> dict[str, Any]:
    ledger = await run_in_threadpool(state.store.ledger_counts)
    return {**state.pipeline.stats, "ledger_events": ledger["accepted"], "ledger_duplicates": ledger["duplicates"],
            "ledger_late": ledger["late"], "llm_jobs_pending": state.jobs.pending}


# ============================================================================ cases

@api.get("/alerts")
async def list_alerts(status: str | None = None, risk_level: str | None = None,
                      limit: int = Query(200, ge=1, le=2000)) -> list[dict[str, Any]]:
    rows = [_to_api(c) for c in state.cases.active().values()]
    rows = [r for r in rows if (not status or r["status"] == status.upper())
            and (not risk_level or r["risk_level"] == risk_level.upper())]
    rows.sort(key=lambda r: (-r["composite_score"], r["case_id"]))
    return rows[:limit]


@api.get("/alerts/{case_id}/investigation")
async def investigation(case_id: str) -> dict[str, Any]:
    case = _case(case_id)
    bundle = await _investigation(case)
    explanation = await _explanation(case, settings.LLM_INLINE_WAIT_SECONDS)
    revisions = await run_in_threadpool(_history, case["case_id"])
    return {
        "alert": _to_api(case), "graph": bundle["graph"], "timeline": bundle["timeline"],
        "explanation": explanation, "findings": case["snapshot"]["findings"],
        "associated_account_ids": bundle["associated_account_ids"], "fraud_window": bundle["fraud_window"],
        "revisions": revisions, "graph_truncated": bundle["truncated"],
        "redirected_from": case_id if case["case_id"] != case_id else None,
    }


@api.get("/cases/{case_id}/revisions")
async def revisions(case_id: str) -> dict[str, Any]:
    """A case's own revision log (merged cases keep theirs) plus the lineage merged into it."""
    case = state.cases.cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
    survivor = state.cases.resolve(case_id)
    return {"case_id": case_id, "status": case["workflow_status"], "merged_into": case.get("merged_into"),
            "surviving_case_id": survivor["case_id"] if survivor else None,
            "current_version": case.get("current_version"),
            "revisions": await run_in_threadpool(_history, case_id)}


@api.post("/cases/{case_id}/assign")
async def assign(case_id: str, request: AssignRequest) -> dict[str, Any]:
    case = _case(case_id)
    status = "ASSIGNED" if case["workflow_status"] == "OPEN" else None
    case = await run_in_threadpool(lambda: state.cases.workflow(case["case_id"], status=status,
                                                                 assigned_to=request.reviewer_id,
                                                                 note=f"assigned to {request.reviewer_id}"))
    return {"case_id": case["case_id"], "assigned_to": case["assigned_to"], "status": case["workflow_status"],
            "assigned_at": case["assigned_at"]}


@api.post("/cases/{case_id}/freeze")
async def freeze(case_id: str, request: FreezeRequest) -> dict[str, Any]:
    case = _case(case_id)
    unknown = set(request.account_ids) - set(case["snapshot"]["evidence"]["account_ids"])
    if unknown:
        raise HTTPException(status_code=400, detail=f"Accounts not linked to this case: {sorted(unknown)[:5]}")
    frozen = await run_in_threadpool(state.store.set_frozen_by_review, request.account_ids)
    await run_in_threadpool(lambda: state.cases.workflow(case["case_id"], status="FROZEN",
                                                         note=f"froze {len(frozen)} account(s)"))
    return {"case_id": case["case_id"], "status": "FROZEN", "affected_account_ids": frozen,
            "message": f"{len(frozen)} account(s) frozen by reviewer."}


@api.post("/cases/{case_id}/escalate")
async def escalate(case_id: str, request: EscalateRequest) -> dict[str, Any]:
    case = _case(case_id)
    await run_in_threadpool(lambda: state.cases.workflow(case["case_id"], status="ESCALATED",
                                                         note=f"escalated: {request.note}".rstrip(": ")))
    return {"case_id": case["case_id"], "status": "ESCALATED", "affected_account_ids": [],
            "message": "Case escalated to the compliance lead."}


@api.get("/cases/{case_id}/dossier")
async def dossier(case_id: str, format: str = Query("pdf", pattern="^(pdf|json)$")) -> Response:
    case = _case(case_id)
    bundle = await _investigation(case)
    explanation = await _explanation(case, wait=75.0)
    emp = (await run_in_threadpool(state.queries.employees, [case["snapshot"]["emp_id"]])).get(
        case["snapshot"]["emp_id"]) if case["snapshot"]["emp_id"] else None
    doc = {
        "case_id": case["case_id"], "generated_at": _now(), "alert": _to_api(case), "case_notes": case.get("notes", []),
        "explanation": explanation, "findings": case["snapshot"]["findings"],
        "associated_account_ids": bundle["associated_account_ids"], "fraud_window": bundle["fraud_window"],
        "timeline": bundle["timeline"], "revisions": await run_in_threadpool(_history, case["case_id"]),
        "evidence": {"employee": {k: emp.get(k) for k in ("emp_id", "name", "department", "role", "access_tier",
                                                          "is_privileged")} if emp else None},
    }
    if format == "json":
        body, media = json.dumps(doc, indent=2, default=str).encode(), "application/json"
    else:
        from dossier_pdf import render_pdf
        body, media = await run_in_threadpool(render_pdf, doc), "application/pdf"
    return Response(body, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="evidence-dossier-{case["case_id"]}.{format}"'})


class ChatRequest(BaseModel):
    intent: str | None = Field(default=None, description="explain | trace | changes")
    question: str | None = Field(default=None, max_length=500)


@api.post("/cases/{case_id}/chat")
async def chat(case_id: str, request: ChatRequest,
               x_reviewer_id: str | None = Header(default=None, alias="X-Reviewer-Id")) -> dict[str, Any]:
    """Read-only case assistant. Facts come from controlled functions scoped to this case only."""
    if not x_reviewer_id:
        raise HTTPException(status_code=401, detail="X-Reviewer-Id header is required")
    case = _case(case_id)
    if case.get("assigned_to") and x_reviewer_id not in (case["assigned_to"], "LEAD"):
        raise HTTPException(status_code=403, detail=f"{case['case_id']} is assigned to {case['assigned_to']}")
    if not (request.intent or (request.question or "").strip()):
        raise HTTPException(status_code=422, detail="provide an intent or a question")
    history = await run_in_threadpool(_history, case["case_id"])
    return await case_chat.answer(case, history, state.queries, state.chat_llm,
                                  intent=request.intent, question=request.question)


@api.get("/reviewers")
async def reviewers() -> list[dict[str, Any]]:
    flagged = {c["snapshot"]["emp_id"] for c in state.cases.active().values()
               if c["snapshot"]["emp_id"] and c["snapshot"]["risk_level"] in ("CRITICAL", "HIGH")}
    return await run_in_threadpool(state.queries.reviewers, flagged)


# ============================================================================ platform

async def _probe(url: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            response = await client.get(url)
        body = response.json() if "json" in response.headers.get("content-type", "") else {}
        release = (body.get("release") or {}).get("version") if isinstance(body.get("release"), dict) else None
        return {"online": response.status_code == 200, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "detail": (f"release {release}" if release else None) or body.get("model_version") or body.get("status")
                or f"HTTP {response.status_code}"}
    except httpx.HTTPError as exc:
        return {"online": False, "latency_ms": None, "detail": type(exc).__name__}


@api.get("/system/status")
async def system_status() -> dict[str, Any]:
    ml = await _probe(f"{settings.ML_SERVICE_URL}/health")
    llm_url = await state.explainer.active_url()
    llm = await _probe(f"{llm_url}/health") if llm_url else {"online": False, "latency_ms": None, "detail": "no server"}
    llm["detail"] = f"{llm_url} ({llm['detail']})" if llm_url else llm["detail"]
    chat_url = await state.chat_llm.active_url()
    chat = await _probe(f"{chat_url}/health") if chat_url else {"online": False, "latency_ms": None, "detail": "no server"}
    chat["detail"] = f"{chat_url} ({chat['detail']})" if chat_url else chat["detail"]
    chat["label"] = "Chat LLM (RTX first)"
    started = time.perf_counter()
    try:
        await run_in_threadpool(state.store.ping)
        graph = {"online": True, "latency_ms": round((time.perf_counter() - started) * 1000, 1), "detail": settings.NEO4J_URI}
    except Exception as exc:  # noqa: BLE001
        graph = {"online": False, "latency_ms": None, "detail": type(exc).__name__}
    ml["label"], llm["label"], graph["label"] = "ML Engine", f"LLM ({settings.LLM_MODEL})", "Neo4j Graph"
    active = [c for c in state.cases.active().values()]
    try:
        scanned = await run_in_threadpool(state.queries.count_transactions)
    except Exception:  # noqa: BLE001
        scanned = 0
    return {
        "nodes": {"ml": ml, "llm": llm, "graph": graph, "chat": chat},
        "metrics": {
            "open_alerts": sum(c["workflow_status"] == "OPEN" for c in active),
            "active_cases": sum(c["workflow_status"] in ("ASSIGNED", "ESCALATED", "FROZEN") for c in active),
            "critical_alerts": sum(c["snapshot"]["risk_level"] == "CRITICAL" for c in active),
            "transactions_scanned": scanned,
            "flagged_employees": len({c["snapshot"]["emp_id"] for c in active if c["snapshot"]["emp_id"]
                                      and c["snapshot"]["risk_level"] in ("CRITICAL", "HIGH")}),
        },
        "ingest": state.pipeline.stats,
        "checked_at": _now(),
    }


# ============================================================================ admin

def _admin() -> None:
    if not settings.ALLOW_ADMIN:
        raise HTTPException(status_code=403, detail="admin routes disabled (ALLOW_ADMIN=false)")


@api.post("/admin/reset")
async def admin_reset() -> dict[str, Any]:
    _admin()
    await run_in_threadpool(state.pipeline.reset)
    state.investigations.clear()
    state.jobs.clear()
    return {"status": "reset", "at": _now()}


@api.post("/admin/recompute-all")
async def admin_recompute() -> dict[str, Any]:
    _admin()
    try:
        result = await run_in_threadpool(state.pipeline.recompute_all)
    except MLUnavailable as exc:
        raise HTTPException(status_code=503, detail={"status": "ML_UNAVAILABLE", "message": str(exc)}) from exc
    state.investigations.clear()
    return result


@api.get("/admin/fingerprint")
async def admin_fingerprint() -> dict[str, Any]:
    snapshots = state.cases.active_snapshots()
    return {"fingerprint": state.engine.fingerprint(snapshots), "active_cases": len(snapshots),
            "by_risk": {lvl: sum(s["risk_level"] == lvl for s in snapshots.values())
                        for lvl in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}}


app.include_router(api)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "active_cases": len(state.cases.active()), "llm_jobs_pending": state.jobs.pending}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gateway:app", host=settings.API_HOST, port=settings.API_PORT)
