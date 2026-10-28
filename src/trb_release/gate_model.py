"""Shared-gate release model with an explicit, causal event order.

At each slot choose releases, reveal arrivals and service, admit arrivals,
then charge end-of-slot inventories. Inputs and all costs are integers.
This module contains no data fitting, experiment runner, or plotting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite
from typing import Iterator, Sequence


@dataclass(frozen=True)
class GateCase:
    departures: int
    gates: int
    release_caps: tuple[int, ...]
    arrivals: tuple[int, ...]
    service: tuple[int, ...]
    queue_cost: int = 1
    arrival_cost: int = 1
    gate_cost: int = 0
    slot_minutes: int = 1

    def __post_init__(self):
        values = (self.departures, self.gates, *self.release_caps,
                  *self.arrivals, *self.service, self.queue_cost,
                  self.arrival_cost, self.gate_cost, self.slot_minutes)
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("Counts, costs and duration must be nonnegative integers")
        if self.gates < self.departures or self.slot_minutes == 0:
            raise ValueError("Initial departures require gates; duration must be positive")
        if not self.release_caps or not (len(self.release_caps) == len(self.arrivals) == len(self.service)):
            raise ValueError("All paths require the same positive horizon")

    def to_dict(self):
        return asdict(self)


def replay(case: GateCase, releases: Sequence[int]) -> dict:
    if len(releases) != len(case.service):
        raise ValueError("Release horizon differs from service")
    h, q, cumulative_arrivals = case.departures, 0, 0
    previous_admitted = 0
    rows = []
    valid = True
    for t, (u, a, c, k) in enumerate(zip(releases, case.arrivals, case.service, case.release_caps)):
        if type(u) is not int or not 0 <= u <= min(k, h):
            valid = False
            break
        h -= u
        q = max(0, q + u - c)
        cumulative_arrivals += a
        admitted = min(cumulative_arrivals, case.gates - h)
        w = cumulative_arrivals - admitted
        assert admitted >= previous_admitted
        assert h + admitted <= case.gates
        previous_admitted = admitted
        rows.append({"slot": t, "release": u, "gate": h, "queue": q,
                     "arrival_wait": w, "admitted": admitted})
    valid = valid and h == 0 and q == 0
    quantities = {name: case.slot_minutes * sum(row[field] for row in rows)
                  for name, field in (("gate_wait", "gate"), ("queue_wait", "queue"),
                                      ("arrival_wait", "arrival_wait"))}
    loss = (case.gate_cost * quantities["gate_wait"]
            + case.queue_cost * quantities["queue_wait"]
            + case.arrival_cost * quantities["arrival_wait"])
    return {"feasible": valid, "loss": loss if valid else None,
            "total_wait": sum(quantities.values()), **quantities, "rows": rows,
            "terminal_arrival_backlog": rows[-1]["arrival_wait"] if rows else None}


def feasible_plans(case: GateCase) -> Iterator[tuple[int, ...]]:
    """Small-instance oracle; never used for production flow solving."""
    for plan in product(*(range(min(k, case.departures) + 1) for k in case.release_caps)):
        if sum(plan) == case.departures and replay(case, plan)["feasible"]:
            yield plan


def weighted_risk(values: Sequence[float], weights: Sequence[float],
                  criterion: str = "mean", alpha: float = 0.9) -> float:
    if not values or len(values) != len(weights):
        raise ValueError("Nonempty matching values and probabilities required")
    if not 0 <= alpha < 1 or any(p < 0 or not isfinite(p) for p in weights):
        raise ValueError("Invalid probability or CVaR confidence level")
    z = sum(weights)
    if z <= 0:
        raise ValueError("Positive probability mass required")
    pairs = [(float(v), p / z) for v, p in zip(values, weights) if p > 0]
    if criterion == "mean":
        return sum(v * p for v, p in pairs)
    if criterion == "worst":
        return max(v for v, p in pairs)
    if criterion != "cvar":
        raise ValueError(criterion)
    remaining, value = 1 - alpha, 0.0
    for v, p in sorted(pairs, reverse=True):
        take = min(remaining, p)
        value += take * v
        remaining -= take
        if remaining <= 1e-13:
            break
    return value / (1 - alpha)
