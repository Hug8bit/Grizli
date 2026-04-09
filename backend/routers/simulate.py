"""
Simulation router — POST /api/simulate, GET /api/network
"""
from __future__ import annotations

import time
import logging
from functools import lru_cache

import pandapower as pp
import numpy as np
from fastapi import APIRouter, HTTPException

from backend.models import (
    SimulateRequest, SimulateResponse,
    LineResult, NodeResult,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Simulation"])

# In-memory store for the last computed network (lightweight — single worker)
_last_net: dict = {}


def _build_network(req: SimulateRequest):
    """Instantiate a pandapower network from the request parameters."""
    from src.core.ieee_networks import (
        create_ieee33_with_renewables,
        create_congested_ieee33,
        create_atacama_scenario,
    )
    from src.core.network_generator import NetworkGenerator, NetworkType

    nt = req.network_type.lower()
    if nt == "ieee33":
        net = create_ieee33_with_renewables()
    elif nt in ("ieee33_congested", "congested"):
        net = create_congested_ieee33()
    elif nt == "atacama":
        net = create_atacama_scenario()
    else:
        gen = NetworkGenerator(num_buses=req.num_buses, network_type=NetworkType.MESHED, seed=42)
        net = gen.generate()
    return net


def _apply_factors(net, req: SimulateRequest) -> None:
    """Scale sgen (solar/wind) and loads by the request factors."""
    if len(net.sgen) > 0:
        for idx in net.sgen.index:
            gen_type = net.sgen.at[idx, "type"] if "type" in net.sgen.columns else "PV"
            factor = req.solar_factor if gen_type == "PV" else req.wind_factor
            net.sgen.at[idx, "p_mw"] *= factor
    if len(net.load) > 0:
        net.load.p_mw *= req.load_factor


def _extract_response(net, opt_result=None, t_ms: float = 0.0) -> SimulateResponse:
    """Build SimulateResponse from a solved pandapower net."""
    load_buses = set(net.load.bus.values.tolist()) if len(net.load) > 0 else set()
    gen_buses  = set(net.sgen.bus.values.tolist()) if len(net.sgen) > 0 else set()
    slack_buses = set(net.ext_grid.bus.values.tolist()) if len(net.ext_grid) > 0 else set()

    nodes = []
    for idx in net.bus.index:
        v = float(net.res_bus.at[idx, "vm_pu"]) if idx in net.res_bus.index else 1.0
        if idx in slack_buses:
            ntype = "slack"
        elif idx in gen_buses:
            ntype = "generator"
        elif idx in load_buses:
            ntype = "load"
        else:
            ntype = "junction"
        nodes.append(NodeResult(id=int(idx), voltage_pu=round(v, 4), node_type=ntype))

    lines = []
    for idx in net.line.index:
        loading = float(net.res_line.at[idx, "loading_percent"]) if idx in net.res_line.index else 0.0
        lines.append(LineResult(
            id=int(idx),
            from_bus=int(net.line.at[idx, "from_bus"]),
            to_bus=int(net.line.at[idx, "to_bus"]),
            loading_percent=round(loading, 2),
            in_service=bool(net.line.at[idx, "in_service"]),
        ))

    lines_opened = getattr(opt_result, "lines_to_open", []) or []
    lines_closed = getattr(opt_result, "lines_to_close", []) or []

    return SimulateResponse(
        converged=True,
        max_loading_pct=round(float(net.res_line.loading_percent.max()), 2),
        total_losses_mw=round(float(net.res_line.pl_mw.sum()), 4),
        num_overloaded=int((net.res_line.loading_percent > 100).sum()),
        lines=lines,
        nodes=nodes,
        optimization_applied=opt_result is not None,
        lines_opened=list(lines_opened),
        lines_closed=list(lines_closed),
        computation_ms=round(t_ms, 1),
    )


@router.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    """Run power flow simulation (+ optional fast topology optimisation)."""
    t0 = time.perf_counter()
    try:
        net = _build_network(req)
        _apply_factors(net, req)
        pp.runpp(net, numba=False, verbose=False)
    except Exception as exc:
        logger.exception("Power flow failed")
        raise HTTPException(status_code=500, detail=f"Power flow failed: {exc}")

    opt_result = None
    if req.optimize:
        try:
            from src.optimization.fast_optimizer import optimize_topology_fast, apply_optimization_result
            opt_result = optimize_topology_fast(net, max_iterations=5)
            if opt_result.lines_to_open or opt_result.lines_to_close:
                apply_optimization_result(net, opt_result)
                pp.runpp(net, numba=False, verbose=False)
        except Exception as exc:
            logger.warning("Optimisation failed (non-fatal): %s", exc)

    _last_net["net"] = net
    t_ms = (time.perf_counter() - t0) * 1000
    return _extract_response(net, opt_result, t_ms)


@router.get("/network", response_model=SimulateResponse)
def get_network():
    """Return the most recently computed network state."""
    if "net" not in _last_net:
        # Return a default IEEE33 on first call
        req = SimulateRequest()
        return simulate(req)
    net = _last_net["net"]
    return _extract_response(net)
