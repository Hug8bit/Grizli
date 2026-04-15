"""
GRIZLI — Greedy Branch-Exchange Optimizer
Performs distribution network reconfiguration to minimize losses and violations.

Algorithm: Greedy forward search over switch candidates.
At each step, tries to open one additional switch (or redistribute load via tie-lines)
and keeps the action if it improves the objective function.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .opendss_loader import OpenDSSFeeder, FeederMetrics

logger = logging.getLogger("grizli.optimizer")


@dataclass
class OptimizationResult:
    feeder_id: str
    baseline: FeederMetrics
    optimized: FeederMetrics
    actions: list[str]           # Lines opened during optimization
    compute_time_s: float
    n_candidates_tested: int
    loss_reduction_pct: float
    violations_resolved: int
    overloads_resolved: int
    converged: bool = True

    def summary(self) -> dict:
        return {
            "feeder_id": self.feeder_id,
            "loss_reduction_pct": round(self.loss_reduction_pct, 3),
            "violations_resolved": self.violations_resolved,
            "overloads_resolved": self.overloads_resolved,
            "actions": self.actions,
            "compute_time_s": round(self.compute_time_s, 2),
            "candidates_tested": self.n_candidates_tested,
            "baseline_losses_kw": self.baseline.losses_kw,
            "optimized_losses_kw": self.optimized.losses_kw,
            "baseline_viol": self.baseline.v_violations,
            "optimized_viol": self.optimized.v_violations,
        }


class GrizliOptimizer:
    """
    GRIZLI greedy branch-exchange optimizer.
    
    Implements a forward greedy search over switch candidates.
    Each iteration tests opening one switch; keeps it if the
    objective score improves. Continues until no improvement found.
    
    Objective: minimize losses_kw + 5×voltage_violations + 2×overloaded_lines
    """

    def __init__(
        self,
        feeder: OpenDSSFeeder,
        load_mult: float = 0.75,
        w_loss: float = 1.0,
        w_viol: float = 5.0,
        w_ovl: float = 2.0,
        max_actions: int = 10,
        improvement_threshold: float = 0.5,
    ):
        self.feeder = feeder
        self.load_mult = load_mult
        self.w_loss = w_loss
        self.w_viol = w_viol
        self.w_ovl = w_ovl
        self.max_actions = max_actions
        self.improvement_threshold = improvement_threshold

    def _score(self, m: FeederMetrics) -> float:
        return m.losses_kw * self.w_loss + m.v_violations * self.w_viol + m.overloaded_lines * self.w_ovl

    def run(self, progress_callback=None) -> OptimizationResult:
        """
        Run the optimization. Returns an OptimizationResult.
        
        Args:
            progress_callback: optional callable(step, total, msg) for progress reporting
        """
        t0 = time.time()

        # Get baseline
        baseline = self.feeder.run(self.load_mult)
        if not baseline.converged:
            logger.error("Baseline did not converge — cannot optimize.")
            return OptimizationResult(
                feeder_id=self.feeder.master_path.stem,
                baseline=baseline, optimized=baseline,
                actions=[], compute_time_s=time.time()-t0,
                n_candidates_tested=0, loss_reduction_pct=0,
                violations_resolved=0, overloads_resolved=0,
                converged=False
            )

        # Discover switch candidates
        switches = self.feeder.list_switches()
        logger.info(f"Found {len(switches)} switch candidates")

        best_score = self._score(baseline)
        best_metrics = baseline
        open_set: set[str] = set()
        actions: list[str] = []
        tested = 0

        total_steps = len(switches)

        for i, sw in enumerate(switches):
            if len(actions) >= self.max_actions:
                break

            candidate = open_set | {sw}
            try:
                m = self.feeder.run(self.load_mult, list(candidate))
                tested += 1
            except Exception as e:
                logger.warning(f"Error testing switch {sw}: {e}")
                continue

            if not m.converged:
                continue

            sc = self._score(m)
            if sc < best_score - self.improvement_threshold:
                best_score = sc
                best_metrics = m
                open_set = candidate.copy()
                actions.append(sw)
                logger.info(f"✓ Open {sw}: loss={m.losses_kw}kW viol={m.v_violations} score={sc:.2f}")

            if progress_callback:
                progress_callback(i + 1, total_steps, f"Testing switch {sw}...")

        compute_time = time.time() - t0
        bl_loss = baseline.losses_kw
        opt_loss = best_metrics.losses_kw
        loss_red = (bl_loss - opt_loss) / bl_loss * 100 if bl_loss > 0 else 0

        return OptimizationResult(
            feeder_id=self.feeder.master_path.stem,
            baseline=baseline,
            optimized=best_metrics,
            actions=actions,
            compute_time_s=compute_time,
            n_candidates_tested=tested,
            loss_reduction_pct=round(loss_red, 3),
            violations_resolved=baseline.v_violations - best_metrics.v_violations,
            overloads_resolved=baseline.overloaded_lines - best_metrics.overloaded_lines,
        )


class MultiFeederBalancer:
    """
    GRIZLI load-balancing optimizer for multi-feeder substations.
    
    When feeders are interconnected via tie-lines, load can be
    transferred between them. This optimizer finds the best
    load distribution to minimize the global objective.
    
    Search space: delta ∈ [0.05, 0.20], split ∈ [0, 1]
    Constraint: total load conserved within ±10%
    """

    def __init__(self, feeders: dict[str, OpenDSSFeeder], base_mult: float = 0.75):
        self.feeders = feeders  # {feeder_id: OpenDSSFeeder}
        self.base_mult = base_mult

    def _agg_score(self, results: list[FeederMetrics]) -> float:
        return sum(m.losses_kw + m.v_violations * 5 + m.overloaded_lines * 2 for m in results)

    def run(self, progress_callback=None) -> dict:
        """
        Run multi-feeder load balancing optimization.
        Returns a dict with per-feeder results and optimization summary.
        """
        t0 = time.time()
        fids = list(self.feeders.keys())
        n = len(fids)

        # Baseline
        baseline = {fid: self.feeders[fid].run(self.base_mult) for fid in fids}
        base_score = self._agg_score(list(baseline.values()))
        base_total_mult = self.base_mult * n

        best_score = base_score
        best_mults = {fid: self.base_mult for fid in fids}
        best_results = baseline.copy()

        # For 3-feeder systems: transfer from overloaded feeder to others
        if n != 3:
            return {"baseline": baseline, "optimized": baseline,
                    "mults": best_mults, "note": "Multi-feeder balancing requires 3 feeders"}

        # Identify most overloaded feeder (most violations)
        sorted_fids = sorted(fids, key=lambda f: baseline[f].v_violations, reverse=True)
        stressed_fid = sorted_fids[0]   # most violations
        receiver_fids = sorted_fids[1:] # will absorb load

        step = 0
        total_steps = 5 * 5  # deltas × splits

        for delta in np.arange(0.05, 0.26, 0.05):
            for split in [0.0, 0.3, 0.5, 0.7, 1.0]:
                step += 1
                mults = {
                    fids[0]: self.base_mult,
                    fids[1]: self.base_mult,
                    fids[2]: self.base_mult,
                }
                # Transfer delta from stressed feeder
                mults[stressed_fid] = round(self.base_mult - delta, 2)
                mults[receiver_fids[0]] = round(self.base_mult + delta * split, 2)
                mults[receiver_fids[1]] = round(self.base_mult + delta * (1 - split), 2)

                # Constraint: total load conserved
                if abs(sum(mults.values()) - base_total_mult) > 0.3:
                    continue
                if any(m < 0.3 or m > 1.1 for m in mults.values()):
                    continue

                results = {}
                ok = True
                for fid in fids:
                    m = self.feeders[fid].run(mults[fid])
                    if not m.converged:
                        ok = False; break
                    results[fid] = m

                if not ok:
                    continue

                sc = self._agg_score(list(results.values()))
                if sc < best_score - 0.5:
                    best_score = sc
                    best_mults = mults.copy()
                    best_results = results.copy()

                if progress_callback:
                    progress_callback(step, total_steps, f"Testing δ={delta:.2f} split={split:.1f}")

        compute_time = time.time() - t0
        total_loss_before = sum(m.losses_kw for m in baseline.values())
        total_loss_after = sum(m.losses_kw for m in best_results.values())

        return {
            "baseline": baseline,
            "optimized": best_results,
            "mults": best_mults,
            "compute_time_s": round(compute_time, 2),
            "loss_reduction_pct": round((total_loss_before - total_loss_after) / total_loss_before * 100, 3) if total_loss_before > 0 else 0,
            "violations_resolved": sum(baseline[f].v_violations - best_results[f].v_violations for f in fids),
            "overloads_resolved": sum(baseline[f].overloaded_lines - best_results[f].overloaded_lines for f in fids),
        }
