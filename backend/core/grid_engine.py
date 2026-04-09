"""
GrizliV2 - Grid Engine
Power flow simulation using PandaPower.
Handles grid state, power flow computation, and metrics extraction.
"""
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class GridMetrics:
    """Snapshot of grid performance metrics."""
    total_losses_mw: float
    max_line_loading_pct: float
    min_voltage_pu: float
    max_voltage_pu: float
    voltage_violations: int
    overloaded_lines: int
    congestion_index: float  # 0-1, higher = worse
    renewable_curtailment_mw: float = 0.0
    timestamp: Optional[str] = None

    @property
    def is_feasible(self) -> bool:
        return self.voltage_violations == 0 and self.overloaded_lines == 0

    def to_dict(self) -> dict:
        return {
            "total_losses_mw": round(self.total_losses_mw, 4),
            "max_line_loading_pct": round(self.max_line_loading_pct, 2),
            "min_voltage_pu": round(self.min_voltage_pu, 4),
            "max_voltage_pu": round(self.max_voltage_pu, 4),
            "voltage_violations": self.voltage_violations,
            "overloaded_lines": self.overloaded_lines,
            "congestion_index": round(self.congestion_index, 4),
            "renewable_curtailment_mw": round(self.renewable_curtailment_mw, 4),
            "is_feasible": self.is_feasible,
        }


@dataclass
class GridState:
    """Full grid state including topology and power flow results."""
    net: pp.pandapowerNet
    metrics: Optional[GridMetrics] = None
    converged: bool = False

    def get_switch_states(self) -> dict[int, bool]:
        """Return {switch_idx: is_closed} for all switches."""
        if net_has_switches(self.net):
            return {i: bool(row["closed"]) for i, row in self.net.switch.iterrows()}
        return {}

    def to_dict(self) -> dict:
        buses = []
        if len(self.net.bus) > 0:
            for i, row in self.net.bus.iterrows():
                bus = {"id": int(i), "name": str(row.get("name", i)), "vn_kv": float(row["vn_kv"])}
                if self.converged and len(self.net.res_bus) > 0:
                    bus["vm_pu"] = round(float(self.net.res_bus.at[i, "vm_pu"]), 4)
                    bus["va_degree"] = round(float(self.net.res_bus.at[i, "va_degree"]), 4)
                buses.append(bus)

        lines = []
        if len(self.net.line) > 0:
            for i, row in self.net.line.iterrows():
                line = {
                    "id": int(i),
                    "from_bus": int(row["from_bus"]),
                    "to_bus": int(row["to_bus"]),
                    "in_service": bool(row["in_service"]),
                }
                if self.converged and len(self.net.res_line) > 0:
                    line["loading_pct"] = round(float(self.net.res_line.at[i, "loading_percent"]), 2)
                    line["p_mw"] = round(float(self.net.res_line.at[i, "p_from_mw"]), 4)
                lines.append(line)

        return {
            "buses": buses,
            "lines": lines,
            "n_buses": len(self.net.bus),
            "n_lines": len(self.net.line),
            "converged": self.converged,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


def net_has_switches(net: pp.pandapowerNet) -> bool:
    return hasattr(net, "switch") and len(net.switch) > 0


class GridEngine:
    """
    Core grid simulation engine.
    Wraps PandaPower for power flow computation and state management.
    """

    def __init__(self, net: Optional[pp.pandapowerNet] = None):
        self._net = net or self._default_network()

    def _default_network(self) -> pp.pandapowerNet:
        """IEEE 33-bus distribution network as default."""
        try:
            net = pn.case33bw()
            logger.info("Loaded IEEE 33-bus network")
            return net
        except Exception:
            net = pp.create_empty_network()
            logger.warning("Fallback: created empty network")
            return net

    def load_network(self, net: pp.pandapowerNet) -> None:
        self._net = net
        logger.info(f"Network loaded: {len(net.bus)} buses, {len(net.line)} lines")

    def run_power_flow(self, algorithm: str = "nr") -> GridState:
        """
        Run AC power flow (Newton-Raphson by default).
        Returns GridState with results and metrics.
        """
        net = self._net.deepcopy()
        try:
            pp.runpp(net, algorithm=algorithm, numba=False, verbose=False)
            converged = net.converged
        except pp.LoadflowNotConverged:
            logger.warning("Power flow did not converge")
            converged = False

        state = GridState(net=net, converged=converged)
        if converged:
            state.metrics = self._compute_metrics(net)
        return state

    def _compute_metrics(self, net: pp.pandapowerNet) -> GridMetrics:
        """Extract performance metrics from converged power flow results."""
        res_bus = net.res_bus
        res_line = net.res_line

        vm = res_bus["vm_pu"].values
        loading = res_line["loading_percent"].values

        total_losses = float(net.res_line["pl_mw"].sum()) if "pl_mw" in net.res_line else 0.0

        min_v = float(vm.min()) if len(vm) > 0 else 1.0
        max_v = float(vm.max()) if len(vm) > 0 else 1.0
        max_loading = float(loading.max()) if len(loading) > 0 else 0.0

        v_violations = int(np.sum((vm < 0.95) | (vm > 1.05)))
        overloaded = int(np.sum(loading > 80.0))

        # Congestion index: normalized mean loading
        congestion_idx = float(np.mean(loading) / 100.0) if len(loading) > 0 else 0.0

        return GridMetrics(
            total_losses_mw=total_losses,
            max_line_loading_pct=max_loading,
            min_voltage_pu=min_v,
            max_voltage_pu=max_v,
            voltage_violations=v_violations,
            overloaded_lines=overloaded,
            congestion_index=congestion_idx,
        )

    def apply_switch_action(self, switch_idx: int, close: bool) -> None:
        """Open or close a switch in the network."""
        if not net_has_switches(self._net):
            raise ValueError("Network has no switches")
        if switch_idx not in self._net.switch.index:
            raise ValueError(f"Switch {switch_idx} does not exist")
        self._net.switch.at[switch_idx, "closed"] = close

    def get_topology(self) -> dict:
        """Return lightweight topology dict for frontend rendering."""
        buses = [
            {"id": int(i), "name": str(row.get("name", i)), "vn_kv": float(row["vn_kv"])}
            for i, row in self._net.bus.iterrows()
        ]
        lines = [
            {
                "id": int(i),
                "from_bus": int(row["from_bus"]),
                "to_bus": int(row["to_bus"]),
                "length_km": float(row.get("length_km", 1.0)),
            }
            for i, row in self._net.line.iterrows()
        ]
        return {"buses": buses, "lines": lines}

    @property
    def net(self) -> pp.pandapowerNet:
        return self._net


def create_ieee33_engine() -> GridEngine:
    """Factory: IEEE 33-bus network."""
    return GridEngine()


def create_ieee14_engine() -> GridEngine:
    """Factory: IEEE 14-bus network."""
    net = pn.case14()
    return GridEngine(net)
