"""
GRIZLI — OpenDSS Network Loader
Loads distribution feeders from OpenDSS files using dss-python.
"""

import dss
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeederMetrics:
    feeder_id: str
    load_kw: float
    losses_kw: float
    loss_pct: float
    vmin: float
    vmax: float
    v_violations: int  # nodes outside [0.95, 1.05] p.u.
    max_loading_pct: float
    overloaded_lines: int
    converged: bool
    n_buses: int = 0
    n_lines: int = 0
    n_loads: int = 0

    def obj_score(self, w_loss=1.0, w_viol=5.0, w_ovl=2.0) -> float:
        """Objective function for GRIZLI optimizer."""
        return self.losses_kw * w_loss + self.v_violations * w_viol + self.overloaded_lines * w_ovl


class OpenDSSFeeder:
    """Wraps a single OpenDSS feeder for simulation and metrics extraction."""

    def __init__(self, dss_master_path: str):
        self.master_path = Path(dss_master_path).resolve()
        if not self.master_path.exists():
            raise FileNotFoundError(f"Master DSS file not found: {self.master_path}")

    def run(self, load_mult: float = 1.0, open_lines: list[str] = None) -> FeederMetrics:
        """
        Simulate the feeder and return metrics.
        
        Args:
            load_mult: Load multiplier (e.g. 0.5=nominal, 0.75=stress, 1.0=peak)
            open_lines: List of line names to open (reconfiguration actions)
        Returns:
            FeederMetrics with all key performance indicators
        """
        d = dss.DSS
        d.Start(0)
        d.Text.Command = f"redirect \"{self.master_path}\""
        c = d.ActiveCircuit

        # Apply load multiplier
        c.Solution.LoadMult = load_mult

        # Open lines if specified (topology reconfiguration)
        if open_lines:
            for ln in open_lines:
                d.Text.Command = f"Open Line.{ln} 1"
                d.Text.Command = f"Open Line.{ln} 2"

        # Solve power flow
        c.Solution.Solve()
        converged = c.Solution.Converged

        if not converged:
            return FeederMetrics(
                feeder_id=self.master_path.stem,
                load_kw=0, losses_kw=0, loss_pct=0,
                vmin=0, vmax=0, v_violations=999,
                max_loading_pct=0, overloaded_lines=999,
                converged=False
            )

        # ── Voltage metrics ────────────────────────────────────────────────
        mv_voltages = []
        for i in range(c.NumBuses):
            b = c.Buses[i]
            if b.kVBase > 5:  # MV buses only
                v = list(b.puVmagAngle)
                if v and v[0] > 0.01:
                    mv_voltages.append(v[0])

        arr = np.array(mv_voltages) if mv_voltages else np.array([1.0])
        v_viol = int((arr < 0.95).sum() + (arr > 1.05).sum())

        # ── Line loading metrics ───────────────────────────────────────────
        loadings = []
        idx = c.Lines.First
        while idx > 0:
            norm_amps = c.Lines.NormAmps
            c.SetActiveElement(f"Line.{c.Lines.Name}")
            currents = list(c.ActiveCktElement.CurrentsMagAng)
            if currents and norm_amps > 0:
                loadings.append(currents[0] / norm_amps * 100)
            idx = c.Lines.Next

        la = np.array(loadings) if loadings else np.array([0.0])
        n_lines = len(loadings)

        # ── Load metrics ───────────────────────────────────────────────────
        total_kw = 0.0
        n_loads = 0
        idx = c.Loads.First
        while idx > 0:
            total_kw += c.Loads.kW
            n_loads += 1
            idx = c.Loads.Next

        losses_kw = c.Losses[0] / 1000.0
        load_kw = total_kw * load_mult

        return FeederMetrics(
            feeder_id=self.master_path.stem,
            load_kw=round(load_kw, 2),
            losses_kw=round(losses_kw, 3),
            loss_pct=round(losses_kw / load_kw * 100, 4) if load_kw > 0 else 0,
            vmin=round(float(arr.min()), 4),
            vmax=round(float(arr.max()), 4),
            v_violations=v_viol,
            max_loading_pct=round(float(la.max()), 1),
            overloaded_lines=int((la > 100).sum()),
            converged=True,
            n_buses=c.NumBuses,
            n_lines=n_lines,
            n_loads=n_loads,
        )

    def list_switches(self) -> list[str]:
        """Return all switchable elements (IsSwitch=True)."""
        d = dss.DSS
        d.Start(0)
        d.Text.Command = f"redirect \"{self.master_path}\""
        c = d.ActiveCircuit
        c.Solution.Solve()

        switches = []
        idx = c.Lines.First
        while idx > 0:
            if c.Lines.IsSwitch:
                switches.append(c.Lines.Name)
            idx = c.Lines.Next
        return switches
