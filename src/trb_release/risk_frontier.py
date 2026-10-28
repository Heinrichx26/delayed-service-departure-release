"""Exact static-CVaR outer approximation on service-preserving schedules.

Preservation is an explicit feasible-set restriction, not a conclusion
about the unconstrained stochastic release problem. Two algorithms solve
the same restricted objective: a full hinge epigraph and cutting planes.
Neither solver observes a future outcome when choosing the common plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .gate_model import GateCase, replay, weighted_risk


@dataclass(frozen=True)
class PreservingPanel:
    cases: tuple[GateCase, ...]
    probabilities: tuple[float, ...]

    def arrays(self):
        base, horizon = self.cases[0], len(self.cases[0].service)
        assert len(self.cases) == len(self.probabilities) and min(self.probabilities) > 0
        common = (base.departures, base.gates, base.release_caps, base.queue_cost,
                  base.arrival_cost, base.gate_cost, base.slot_minutes)
        for case in self.cases:
            assert common == (case.departures, case.gates, case.release_caps, case.queue_cost,
                              case.arrival_cost, case.gate_cost, case.slot_minutes)
        h, earliest = base.departures, []
        for cap in base.release_caps:
            release = min(h, cap)
            h -= release
            earliest.append(release)
        completions, arrival_totals = [], []
        for case in self.cases:
            check = replay(case, earliest)
            if not check["feasible"]:
                raise ValueError("Reference cannot release and clear all departures")
            cumulative, ds = 0, []
            for u, row in zip(earliest, check["rows"]):
                cumulative += u
                ds.append(cumulative - row["queue"])
            completions.append(ds)
            arrival_totals.append(np.cumsum(case.arrivals))
        departures = np.asarray(completions, dtype=float)
        arrivals = np.asarray(arrival_totals, dtype=float)
        upper = base.departures - np.max(departures, axis=0)
        probability = np.asarray(self.probabilities) / sum(self.probabilities)
        b = base.slot_minutes * base.queue_cost * np.sum(base.departures - departures, axis=1)
        beta = base.slot_minutes * (base.gate_cost - base.queue_cost)
        gamma = base.slot_minutes * base.arrival_cost
        return base, horizon, departures, arrivals, upper, probability, b, beta, gamma


def _schedule_constraints(base, horizon, width):
    rows, lows, highs = [], [], []
    for t in range(horizon):
        row = np.zeros(width)
        row[t] = -1
        if t:
            row[t - 1] = 1
            lo, hi = 0, base.release_caps[t]
        else:
            lo, hi = -base.departures, base.release_caps[t] - base.departures
        rows.append(row)
        lows.append(lo)
        highs.append(hi)
    return rows, lows, highs


def _loss_gradient(arrivals, h, gates, b, beta, gamma):
    # At a hinge equality choose the lower one-sided supporting derivative.
    excess = arrivals - gates + h
    losses = b + beta * sum(h) + gamma * np.maximum(excess, 0).sum(axis=1)
    gradients = beta + gamma * (excess > 0)
    return losses, gradients


def _cvar_cut(losses, gradients, probabilities, alpha):
    remaining = 1 - alpha
    weights = np.zeros(len(losses))
    for idx in sorted(range(len(losses)), key=lambda i: (-losses[i], i)):
        take = min(remaining, probabilities[idx])
        weights[idx] = take / (1 - alpha)
        remaining -= take
        if remaining <= 1e-13:
            break
    return float(weights @ losses), weights @ gradients


def _verified_result(panel, holding, objective, extra):
    base = panel.cases[0]
    holding = [int(round(h)) for h in holding]
    previous, releases = base.departures, []
    for h in holding:
        releases.append(previous - h)
        previous = h
    physical = [replay(case, releases) for case in panel.cases]
    assert all(row["feasible"] for row in physical)
    actual = weighted_risk([r["loss"] for r in physical], panel.probabilities, "cvar", extra["alpha"])
    assert abs(actual - objective) < 1e-6
    _, _, ds, _, _, _, _, _, _ = panel.arrays()
    for idx, row in enumerate(physical):
        cumulative = 0
        for t, (u, state) in enumerate(zip(releases, row["rows"])):
            cumulative += u
            assert cumulative - state["queue"] == ds[idx, t]
    return {"objective": actual, "holding": holding, "releases": releases,
            "path_losses": [r["loss"] for r in physical],
            "mean_loss": weighted_risk([r["loss"] for r in physical], panel.probabilities),
            "terminal_arrival_backlogs": [r["terminal_arrival_backlog"] for r in physical], **extra}


def solve_epigraph(panel: PreservingPanel, alpha=.9, integral=True, time_limit=30):
    started = perf_counter()
    base, horizon, _, arrivals, upper, p, b, beta, gamma = panel.arrays()
    scenarios = len(panel.cases)
    eta = horizon
    z0 = horizon + 1
    w0 = z0 + scenarios
    width = w0 + scenarios * horizon
    costs = np.zeros(width)
    costs[eta] = 1
    costs[z0:w0] = p / (1 - alpha)
    lower, higher = np.zeros(width), np.full(width, inf)
    higher[:horizon] = upper
    integrality = np.zeros(width)
    if integral:
        integrality[:horizon] = 1
    rows, lows, highs = _schedule_constraints(base, horizon, width)
    for s in range(scenarios):
        # z_s >= b_s + beta sum h + gamma sum w_s - eta.
        row = np.zeros(width)
        row[:horizon] = beta
        row[eta], row[z0 + s] = -1, -1
        row[w0 + s * horizon:w0 + (s + 1) * horizon] = gamma
        rows.append(row); lows.append(-inf); highs.append(-b[s])
        for t in range(horizon):
            row = np.zeros(width)
            row[t], row[w0 + s * horizon + t] = 1, -1
            rows.append(row); lows.append(-inf); highs.append(base.gates - arrivals[s, t])
    res = milp(costs, integrality=integrality, bounds=Bounds(lower, higher),
               constraints=LinearConstraint(np.asarray(rows), lows, highs),
               options={"time_limit": time_limit, "mip_rel_gap": 0})
    if res.status != 0:
        raise RuntimeError({"status": res.status, "message": res.message})
    extra = {"alpha": alpha, "elapsed_seconds": perf_counter() - started,
             "variables": width, "rows": len(rows), "integral": integral}
    if integral:
        return _verified_result(panel, res.x[:horizon], res.fun, extra)
    losses, _ = _loss_gradient(arrivals, res.x[:horizon], base.gates, b, beta, gamma)
    actual = weighted_risk(losses.tolist(), p.tolist(), "cvar", alpha)
    assert abs(actual - res.fun) < 1e-6
    return {"objective": float(res.fun), "holding": res.x[:horizon].tolist(), **extra}


def round_relaxation(panel: PreservingPanel, relaxed: dict, alpha=.9):
    """Two feasible plans and a proved, solution-specific additive bound."""
    base = panel.cases[0]
    if abs(relaxed["alpha"] - alpha) > 1e-12:
        raise ValueError("Rounding and relaxation must use the same confidence level")
    h = np.asarray(relaxed["holding"], dtype=float)
    # Remove solver roundoff at exact integers before defining fractions.
    h = np.where(np.abs(h - np.rint(h)) < 1e-8, np.rint(h), h)
    floor, ceiling = np.floor(h), np.ceil(h)
    records = []
    for name, candidate in (("floor", floor), ("ceiling", ceiling)):
        releases = np.r_[base.departures, candidate[:-1]] - candidate
        physical = [replay(case, [int(v) for v in releases]) for case in panel.cases]
        assert all(p["feasible"] for p in physical)
        objective = weighted_risk([p["loss"] for p in physical], panel.probabilities, "cvar", alpha)
        records.append(_verified_result(panel, candidate, objective, {"alpha": alpha, "rounding": name}))
    a = max(base.queue_cost - base.gate_cost, 0)
    b = max(base.gate_cost + base.arrival_cost - base.queue_cost, 0)
    fractional = int(np.count_nonzero(ceiling - floor))
    specific_bound = base.slot_minutes * min(a * float(sum(h - floor)), b * float(sum(ceiling - h)))
    uniform_bound = base.slot_minutes * fractional * a * b / (a + b) if a + b else 0
    best = min(records, key=lambda r: r["objective"])
    assert best["objective"] - relaxed["objective"] <= specific_bound + 1e-6
    assert specific_bound <= uniform_bound + 1e-6
    return {"best": best, "candidates": records, "LP_lower_bound": relaxed["objective"],
            "solution_specific_bound": specific_bound, "fractional_coordinates": fractional,
            "uniform_additive_bound": uniform_bound,
            "certified_upper_minus_lower": max(0, best["objective"] - relaxed["objective"])}


def solve_cuts(panel: PreservingPanel, alpha=.9, time_limit=30, max_iterations=2000):
    started = perf_counter()
    base, horizon, _, arrivals, upper, p, b, beta, gamma = panel.arrays()
    width, cuts, trace = horizon + 1, [], []
    costs = np.zeros(width); costs[-1] = 1
    integrality = np.ones(width); integrality[-1] = 0
    lower, higher = np.zeros(width), np.r_[upper, inf]
    best, best_h = inf, None
    for iteration in range(max_iterations):
        remaining_time = time_limit - (perf_counter() - started)
        if remaining_time <= 0:
            raise TimeoutError("Cut generation did not close the exact gap")
        rows, lows, highs = _schedule_constraints(base, horizon, width)
        for point, objective, gradient in cuts:
            rows.append(np.r_[gradient, -1])
            lows.append(-inf)
            highs.append(float(gradient @ point - objective))
        res = milp(costs, integrality=integrality, bounds=Bounds(lower, higher),
                   constraints=LinearConstraint(np.asarray(rows), lows, highs),
                   options={"time_limit": remaining_time, "mip_rel_gap": 0})
        if res.status != 0:
            raise RuntimeError({"status": res.status, "message": res.message})
        holding = np.rint(res.x[:horizon])
        losses, gradients = _loss_gradient(arrivals, holding, base.gates, b, beta, gamma)
        objective, gradient = _cvar_cut(losses, gradients, p, alpha)
        if objective < best:
            best, best_h = objective, holding
        bound = float(res.fun)
        trace.append({"iteration": iteration + 1, "lower": bound, "upper": best,
                      "candidate": holding.tolist(), "candidate_loss": objective})
        if best - bound <= 1e-7:
            return _verified_result(panel, best_h, best, {"alpha": alpha,
                "elapsed_seconds": perf_counter() - started, "iterations": iteration + 1,
                "variables": width, "cuts": len(cuts), "trace": trace,
                "gap": max(0., best - bound)})
        cuts.append((holding, objective, gradient))
    raise RuntimeError("Maximum exact-cut iterations exceeded")
