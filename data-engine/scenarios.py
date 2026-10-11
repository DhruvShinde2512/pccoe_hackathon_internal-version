"""Reproducible synthetic scenarios for the FIN-04 train / held-out datasets.

A :class:`World` accumulates employees, accounts, access logs, transactions and
ground-truth labels for one split. Every split has its own ID namespace
(``T`` for training, ``H`` for held-out) so no entity is shared between them.

Scenario catalogue (see ARCHITECTURE_V2_PLAN.md section 4.1):

Training split
  A   insider override on dormant accounts, then four $9,500 wires within 2 h   SUSPICIOUS
  B   MODIFY_PHONE then a 3-hop circular flow A->B->C->A                        SUSPICIOUS
  C_PAYROLL       recurring fortnightly payroll batches (some near $10k)        BENIGN
  C_CASH_POOL     daily corporate cash-pooling sweeps A->B->C->A                BENIGN
  C_DORMANT_LEGIT customer-initiated dormant reactivation                       BENIGN
  C_WEALTH        large but ordinary high-net-worth wires                       BENIGN

Held-out split (new variants, never seen in training)
  A2  insider override, then 3-5 wires of $9,700-$9,950 spread over ~6 h        SUSPICIOUS
  B2  MODIFY_PHONE then a 4-hop cycle spanning ~3 days                          SUSPICIOUS
  D1  late-link demo: OVERRIDE_ALERT at 10:00, three near-threshold wires
      from 11:00; the replay script delivers the override late                  SUSPICIOUS
  L1  payroll look-alike: 40 payments of $9,920-$9,990 within ~60 s, fortnightly BENIGN
  L2  customer-initiated dormant reactivation, no employee involvement          BENIGN
  L3  legitimate phone change followed by a small routine transfer              BENIGN
  L4  corporate cash-pooling cycles, daily, regular amounts                     BENIGN
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from schemas import (
    AccessLogSchema,
    AccountSchema,
    EmployeeSchema,
    GroundTruthLabelSchema,
    TransactionSchema,
)

Record = dict[str, Any]
Label = Literal["SUSPICIOUS", "BENIGN"]

DEPARTMENTS = [
    ("Retail Banking", ["Relationship Manager", "Branch Officer", "Teller"]),
    ("Finance", ["Analyst", "Controller", "Accountant"]),
    ("Compliance", ["Compliance Analyst", "AML Investigator", "Compliance Officer"]),
    ("Operations", ["Operations Specialist", "Payments Officer"]),
    ("Technology", ["Engineer", "Support Analyst"]),
    ("Client Due Diligence", ["KYC Specialist", "KYC Reviewer"]),
]
FIRST = ["Alex", "Priya", "Jordan", "Morgan", "Sam", "Riya", "Chris", "Neha", "Taylor", "Arjun",
         "Casey", "Isha", "Drew", "Kabir", "Jamie", "Ananya", "Robin", "Vikram", "Avery", "Meera"]
LAST = ["Lee", "Patel", "Shah", "Kim", "Rao", "Garcia", "Iyer", "Brown", "Singh", "Wong",
        "Mehta", "Clark", "Das", "Lopez", "Nair", "Young", "Joshi", "Hall", "Kapoor", "Reed"]
COMPANY = ["Northstar", "Bluefin", "Granite", "Aurora", "Keystone", "Silverline", "Redwood",
           "Harbor", "Summit", "Evergreen", "Orion", "Pioneer"]


@dataclass
class World:
    """Mutable container for one split's records."""

    prefix: str
    rng: random.Random
    start: datetime
    end: datetime
    employees: list[Record] = field(default_factory=list)
    accounts: list[Record] = field(default_factory=list)
    access_logs: list[Record] = field(default_factory=list)
    transactions: list[Record] = field(default_factory=list)
    labels: list[Record] = field(default_factory=list)
    _counters: dict[str, int] = field(default_factory=dict)
    _labelled: set[str] = field(default_factory=set)

    # ---------------------------------------------------------------- ids / time

    def next_id(self, kind: str, tag: str = "") -> str:
        key = f"{kind}{tag}"
        self._counters[key] = self._counters.get(key, 0) + 1
        middle = f"{self.prefix}-{tag}-" if tag else f"{self.prefix}-"
        width = 3 if kind == "EMP" else 5 if kind in ("TX", "LOG") else 4
        return f"{kind}-{middle}{self._counters[key]:0{width}d}"

    def random_time(self, start: datetime | None = None, end: datetime | None = None) -> datetime:
        lo, hi = start or self.start, end or self.end
        span = (hi - lo).total_seconds()
        # Business-hours bias: 70% of events between 08:00 and 19:00.
        moment = lo + timedelta(seconds=self.rng.uniform(0, span))
        if self.rng.random() < 0.7:
            moment = moment.replace(hour=self.rng.randint(8, 18), minute=self.rng.randint(0, 59))
            moment = min(max(moment, lo), hi - timedelta(seconds=1))
        return moment.replace(microsecond=0)

    def name(self) -> str:
        return f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"

    # ---------------------------------------------------------------- record builders

    def add_employee(self, *, emp_id: str | None = None, department: str | None = None, role: str | None = None,
                     privileged: bool | None = None, tag: str = "") -> str:
        dept, roles = (department, [role]) if department else self.rng.choice(DEPARTMENTS)
        emp_id = emp_id or self.next_id("EMP", tag)
        is_priv = privileged if privileged is not None else self.rng.random() < 0.18
        self.employees.append(EmployeeSchema(
            emp_id=emp_id, name=self.name(), department=dept, role=role or self.rng.choice(roles),
            access_tier="TIER_3" if is_priv else self.rng.choice(["TIER_1", "TIER_2"]), is_privileged=is_priv,
        ).model_dump(mode="python"))
        return emp_id

    def add_account(self, *, account_id: str | None = None, status: str = "ACTIVE", corporate: bool = False,
                    external: bool = False, balance: float | None = None, risk: str = "STANDARD", tag: str = "") -> str:
        if account_id is None:
            account_id = self.next_id("EXT" if external else "ACC", tag)
        holder = (f"{self.rng.choice(COMPANY)} {self.rng.choice(['Holdings', 'Trading', 'Logistics', 'Foods'])} Ltd"
                  if corporate else ("External Beneficiary" if external else self.name()))
        self.accounts.append(AccountSchema(
            account_id=account_id, customer_name=holder,
            balance=float(balance if balance is not None else round(self.rng.uniform(500, 90_000), 2)),
            status=status, risk_category=risk,
        ).model_dump(mode="python"))
        return account_id

    def add_log(self, emp_id: str, account_id: str, action: str, ts: datetime, *, log_id: str | None = None,
                tag: str = "") -> str:
        log_id = log_id or self.next_id("LOG", tag)
        self.access_logs.append(AccessLogSchema(
            log_id=log_id, emp_id=emp_id, account_id=account_id, action=action, timestamp=ts,
        ).model_dump(mode="python"))
        return log_id

    def add_tx(self, sender: str, receiver: str, amount: float, ts: datetime, *, tx_type: str = "INTERNAL_TRANSFER",
               tx_id: str | None = None, tag: str = "") -> str:
        tx_id = tx_id or self.next_id("TX", tag)
        self.transactions.append(TransactionSchema(
            tx_id=tx_id, sender_account=sender, receiver_account=receiver, amount=round(float(amount), 2),
            timestamp=ts, type=tx_type,
        ).model_dump(mode="python"))
        return tx_id

    def label(self, target_id: str, target_type: str, label: Label, scenario: str, case_group: str = "") -> None:
        if target_id in self._labelled:
            return
        self._labelled.add(target_id)
        self.labels.append(GroundTruthLabelSchema(
            target_id=target_id, target_type=target_type, label=label, scenario_tag=scenario, case_group=case_group,
        ).model_dump(mode="python"))


# ======================================================================== background

def build_background(w: World, *, n_employees: int, n_accounts: int, n_transactions: int, n_logs: int,
                     dormant_share: float = 0.08) -> dict[str, list[str]]:
    """Benign baseline traffic. Returns pools later scenarios draw from."""
    reviewers = [w.add_employee(department="Compliance", role="Compliance Analyst", privileged=False)
                 for _ in range(6)]
    staff = [w.add_employee() for _ in range(n_employees - len(reviewers))]
    active, dormant = [], []
    for _ in range(n_accounts):
        is_dormant = w.rng.random() < dormant_share
        acc = w.add_account(status="DORMANT" if is_dormant else "ACTIVE",
                            corporate=w.rng.random() < 0.15,
                            risk=w.rng.choices(["STANDARD", "ELEVATED", "HIGH"], [80, 16, 4])[0])
        (dormant if is_dormant else active).append(acc)
    externals = [w.add_account(external=True) for _ in range(max(40, n_accounts // 12))]

    # Each active account has a small set of habitual counterparties (realistic recurrence).
    habits = {acc: w.rng.sample(active + externals, k=w.rng.randint(2, 6)) for acc in active}
    for _ in range(n_transactions):
        sender = w.rng.choice(active)
        receiver = (w.rng.choice(habits[sender]) if w.rng.random() < 0.7
                    else w.rng.choice(active + externals))
        if receiver == sender:
            continue
        amount = min(max(w.rng.lognormvariate(7.3, 1.1), 20), 75_000)
        tx_type = w.rng.choices(["INTERNAL_TRANSFER", "WIRE", "ATM"], [68, 24, 8])[0]
        tx = w.add_tx(sender, receiver, amount, w.random_time(), tx_type=tx_type)
        w.label(tx, "TRANSACTION", "BENIGN", "BACKGROUND_BENIGN")

    actions = ["VIEW_PROFILE", "MODIFY_PHONE", "OVERRIDE_ALERT", "MANUAL_UNFREEZE"]
    for _ in range(n_logs):
        emp = w.rng.choice(staff + reviewers)
        acc = w.rng.choice(active + dormant)
        w.add_log(emp, acc, w.rng.choices(actions, [85, 10, 3.5, 1.5])[0], w.random_time())

    for emp in staff + reviewers:
        w.label(emp, "EMPLOYEE", "BENIGN", "BACKGROUND_BENIGN")
    for acc in active + dormant + externals:
        w.label(acc, "ACCOUNT", "BENIGN", "BACKGROUND_BENIGN")
    return {"active": active, "dormant": dormant, "externals": externals, "staff": staff, "reviewers": reviewers}


# ======================================================================== fraud scenarios

def _insider(w: World, tag: str) -> str:
    return w.add_employee(department="Compliance", role="Senior Compliance Analyst", privileged=True, tag=tag)


def scenario_insider_smurfing(w: World, *, tag: str, n_accounts: int, wires: tuple[int, int],
                              amount: tuple[float, float], spread_hours: tuple[float, float], scenario: str,
                              n_insiders: int = 1) -> None:
    """Insider overrides alerts on dormant accounts; each then sends near-threshold wires.

    ``spread_hours`` is a (min, max) range: each account's wires are spread over a random
    duration in that range, so the model can't learn one fixed cadence.
    """
    insiders = [_insider(w, tag) for _ in range(n_insiders)]
    for insider in insiders:
        w.label(insider, "EMPLOYEE", "SUSPICIOUS", scenario, f"{tag}-{insider}")
    for i in range(n_accounts):
        insider = insiders[i % n_insiders]
        group = f"{tag}-{i + 1:03d}"
        acc = w.add_account(status="DORMANT", balance=w.rng.uniform(40_000, 120_000), risk="ELEVATED", tag=tag)
        spread = w.rng.uniform(*spread_hours)
        t0 = w.random_time(w.start, w.end - timedelta(hours=spread + 1))
        w.add_log(insider, acc, "OVERRIDE_ALERT", t0, tag=tag)
        n = w.rng.randint(*wires)
        for k in range(n):
            offset = (spread * 60) * (k + 1) / (n + 0.5)
            ext = w.add_account(external=True, tag=tag)
            amt = amount[0] if amount[0] == amount[1] else w.rng.uniform(*amount)
            tx = w.add_tx(acc, ext, amt, t0 + timedelta(minutes=offset + w.rng.uniform(0, 4)), tx_type="WIRE", tag=tag)
            w.label(tx, "TRANSACTION", "SUSPICIOUS", scenario, group)
            w.label(ext, "ACCOUNT", "SUSPICIOUS", scenario, group)
        w.label(acc, "ACCOUNT", "SUSPICIOUS", scenario, group)


def scenario_circular(w: World, *, tag: str, n_cycles: int, hops: int, span_hours: float, scenario: str,
                      n_insiders: int = 1) -> None:
    """KYC phone change on the first account, then funds loop back to it."""
    insiders = [w.add_employee(department="Client Due Diligence", role="KYC Operations Specialist",
                               privileged=False, tag=tag) for _ in range(n_insiders)]
    for insider in insiders:
        w.label(insider, "EMPLOYEE", "SUSPICIOUS", scenario, f"{tag}-{insider}")
    for c in range(n_cycles):
        insider = insiders[c % n_insiders]
        group = f"{tag}-{c + 1:02d}"
        ring = [w.add_account(corporate=True, balance=w.rng.uniform(300_000, 900_000), risk="ELEVATED", tag=tag)
                for _ in range(hops)]
        t0 = w.random_time(w.start, w.end - timedelta(hours=span_hours + 2))
        w.add_log(insider, ring[0], "MODIFY_PHONE", t0, tag=tag)
        amount = w.rng.randrange(180_000, 260_001, 5_000)
        for k in range(hops):
            ts = t0 + timedelta(minutes=30) + timedelta(hours=span_hours * k / max(hops - 1, 1))
            tx = w.add_tx(ring[k], ring[(k + 1) % hops], amount, ts, tx_type="WIRE", tag=tag)
            amount = round(amount * w.rng.uniform(0.97, 0.995), -2)  # layering fees
            w.label(tx, "TRANSACTION", "SUSPICIOUS", scenario, group)
        for acc in ring:
            w.label(acc, "ACCOUNT", "SUSPICIOUS", scenario, group)


def scenario_late_link_demo(w: World, when: datetime) -> None:
    """D1: the case the demo escalates live. IDs are fixed so the replay script can find them."""
    insider = w.add_employee(emp_id=f"EMP-{w.prefix}-D1", department="Compliance",
                             role="Senior Compliance Analyst", privileged=True)
    acc = w.add_account(account_id=f"ACC-{w.prefix}-D1-SRC", status="ACTIVE", balance=64_000, risk="STANDARD")
    t_override = when.replace(hour=10, minute=0, second=0)
    w.add_log(insider, acc, "OVERRIDE_ALERT", t_override, log_id=f"LOG-{w.prefix}-D1-1")
    for k, (minutes, amount) in enumerate([(60, 9_850.0), (75, 9_900.0), (90, 9_875.0)], start=1):
        ext = w.add_account(account_id=f"EXT-{w.prefix}-D1-{k}", external=True)
        tx = w.add_tx(acc, ext, amount, t_override + timedelta(minutes=minutes), tx_type="WIRE",
                      tx_id=f"TX-{w.prefix}-D1-{k}")
        w.label(tx, "TRANSACTION", "SUSPICIOUS", "D1_LATE_LINK_DEMO", "D1")
        w.label(ext, "ACCOUNT", "SUSPICIOUS", "D1_LATE_LINK_DEMO", "D1")
    w.label(acc, "ACCOUNT", "SUSPICIOUS", "D1_LATE_LINK_DEMO", "D1")
    w.label(insider, "EMPLOYEE", "SUSPICIOUS", "D1_LATE_LINK_DEMO", "D1")


# ======================================================================== benign look-alikes

def scenario_payroll(w: World, *, tag: str, n_employers: int, staff_per_employer: int,
                     salary: tuple[float, float], spacing_seconds: float, scenario: str) -> None:
    """Fortnightly payroll: same employer, same employees, similar amounts, seconds apart."""
    for e in range(n_employers):
        group = f"{tag}-EMPLOYER-{e + 1}"
        employer = w.add_account(corporate=True, balance=2_500_000, tag=tag)
        payees = [w.add_account(tag=tag) for _ in range(staff_per_employer)]
        salaries = {p: round(w.rng.uniform(*salary), 2) for p in payees}
        run = w.start + timedelta(days=w.rng.randint(0, 6), hours=9)
        while run < w.end - timedelta(hours=1):
            for i, payee in enumerate(payees):
                ts = run + timedelta(seconds=i * spacing_seconds)
                tx = w.add_tx(employer, payee, salaries[payee] + w.rng.uniform(-3, 3), ts, tag=tag)
                w.label(tx, "TRANSACTION", "BENIGN", scenario, group)
            run += timedelta(days=14)
        for acc in [employer, *payees]:
            w.label(acc, "ACCOUNT", "BENIGN", scenario, group)


def scenario_cash_pool(w: World, *, tag: str, n_groups: int, scenario: str) -> None:
    """Daily sweeps between accounts of one corporate owner: a legitimate cycle."""
    for g in range(n_groups):
        group = f"{tag}-POOL-{g + 1}"
        ring = [w.add_account(corporate=True, balance=1_000_000, tag=tag) for _ in range(3)]
        base = w.rng.randrange(40_000, 120_000, 1_000)
        day = w.start.replace(hour=17, minute=30, second=0)
        while day < w.end - timedelta(hours=8):
            for k in range(3):
                tx = w.add_tx(ring[k], ring[(k + 1) % 3], base * w.rng.uniform(0.98, 1.02),
                              day + timedelta(minutes=10 * k), tag=tag)
                w.label(tx, "TRANSACTION", "BENIGN", scenario, group)
            day += timedelta(days=1)
        for acc in ring:
            w.label(acc, "ACCOUNT", "BENIGN", scenario, group)


def scenario_dormant_legit(w: World, *, tag: str, n: int, pool: list[str], scenario: str) -> None:
    """A dormant customer returns and makes one ordinary transfer; no employee involved."""
    for i in range(n):
        group = f"{tag}-{i + 1:02d}"
        acc = w.add_account(status="DORMANT", balance=w.rng.uniform(2_000, 30_000), tag=tag)
        tx = w.add_tx(acc, w.rng.choice(pool), w.rng.uniform(150, 3_500), w.random_time(), tag=tag)
        w.label(tx, "TRANSACTION", "BENIGN", scenario, group)
        w.label(acc, "ACCOUNT", "BENIGN", scenario, group)


def scenario_phone_change_legit(w: World, *, tag: str, n: int, staff: list[str], scenario: str) -> None:
    """Customer-requested phone update, then a small routine payment to a known payee."""
    for i in range(n):
        group = f"{tag}-{i + 1:02d}"
        acc = w.add_account(tag=tag)
        payee = w.add_account(tag=tag)
        span = w.end - w.start
        if span > timedelta(days=30):  # weekly routine fits (train / held-out windows)
            first, step, last_gap = w.random_time(w.start, w.start + span / 2), timedelta(days=7), 30
        else:  # short window (validation): compress the routine so every event stays inside it
            step = span / 5
            first, last_gap = w.random_time(w.start, w.start + step), min(30, step.total_seconds() / 3600 - 3)
        # Establish the routine before the phone change.
        for k in range(3):
            tx = w.add_tx(acc, payee, w.rng.uniform(200, 1_200), first + step * k, tag=tag)
            w.label(tx, "TRANSACTION", "BENIGN", scenario, group)
        change = first + step * 3 + timedelta(hours=2)
        w.add_log(w.rng.choice(staff), acc, "MODIFY_PHONE", change, tag=tag)
        tx = w.add_tx(acc, payee, w.rng.uniform(200, 1_200), change + timedelta(hours=w.rng.uniform(3, last_gap)), tag=tag)
        w.label(tx, "TRANSACTION", "BENIGN", scenario, group)
        for a in (acc, payee):
            w.label(a, "ACCOUNT", "BENIGN", scenario, group)


def scenario_wealth(w: World, *, tag: str, n: int, pool: list[str], scenario: str) -> None:
    """Large but ordinary high-net-worth wires."""
    for i in range(n):
        group = f"{tag}-{i + 1:02d}"
        acc = w.add_account(balance=5_000_000, risk="ELEVATED", tag=tag)
        tx = w.add_tx(acc, w.rng.choice(pool), w.rng.randrange(125_000, 1_250_001, 25_000), w.random_time(),
                      tx_type="WIRE", tag=tag)
        w.label(tx, "TRANSACTION", "BENIGN", scenario, group)
        w.label(acc, "ACCOUNT", "BENIGN", scenario, group)
