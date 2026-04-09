"""
GrizliV2 - API Routes
FastAPI router definitions for all endpoints.
"""
import logging
from typing import Optional, Literal
from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import asyncio
import json

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class GridNetworkResponse(BaseModel):
    buses: list[dict]
    lines: list[dict]
    n_buses: int
    n_lines: int
    converged: bool
    metrics: Optional[dict] = None


class PowerFlowRequest(BaseModel):
    algorithm: Literal["nr", "bfsw", "gs"] = "nr"
    scenario_id: Optional[int] = None


class SwitchActionRequest(BaseModel):
    switch_idx: int = Field(..., ge=0, description="Switch index to toggle")
    close: bool = Field(..., description="True=close, False=open")


class TrainRequest(BaseModel):
    total_timesteps: int = Field(default=100_000, ge=1_000, le=10_000_000)
    checkpoint_freq: int = Field(default=10_000, ge=1_000)


class AgentStepRequest(BaseModel):
    observation: Optional[list[float]] = None
    deterministic: bool = True


# ---------------------------------------------------------------------------
# Grid endpoints
# ---------------------------------------------------------------------------


@router.get("/grid/topology", response_model=GridNetworkResponse, tags=["grid"])
async def get_grid_topology():
    """Return current grid topology (buses + lines) without power flow."""
    from backend.main import app_state
    state = app_state["grid_engine"].run_power_flow()
    return GridNetworkResponse(**state.to_dict())


@router.post("/grid/power-flow", response_model=GridNetworkResponse, tags=["grid"])
async def run_power_flow(req: PowerFlowRequest):
    """Run AC power flow and return grid state with metrics."""
    from backend.main import app_state
    engine = app_state["grid_engine"]

    if req.scenario_id is not None:
        scenarios = app_state.get("scenarios", [])
        if req.scenario_id < len(scenarios):
            _apply_scenario(engine, scenarios[req.scenario_id], hour=0)

    state = engine.run_power_flow(algorithm=req.algorithm)
    if not state.converged:
        raise HTTPException(status_code=422, detail="Power flow did not converge")
    return GridNetworkResponse(**state.to_dict())


@router.post("/grid/switch", tags=["grid"])
async def apply_switch(req: SwitchActionRequest):
    """Open or close a switch in the network."""
    from backend.main import app_state
    try:
        app_state["grid_engine"].apply_switch_action(req.switch_idx, req.close)
        state = app_state["grid_engine"].run_power_flow()
        return {
            "success": True,
            "switch_idx": req.switch_idx,
            "closed": req.close,
            "metrics": state.metrics.to_dict() if state.metrics else None,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/grid/metrics", tags=["grid"])
async def get_metrics():
    """Run power flow and return only the metrics summary."""
    from backend.main import app_state
    state = app_state["grid_engine"].run_power_flow()
    if not state.converged:
        return {"converged": False, "metrics": None}
    return {"converged": True, "metrics": state.metrics.to_dict()}


@router.get("/grid/scenarios", tags=["grid"])
async def list_scenarios():
    """List available training scenarios."""
    from backend.main import app_state
    scenarios = app_state.get("scenarios", [])
    return {
        "count": len(scenarios),
        "scenarios": [
            {"id": i, "season": s.get("season"), "peak_load_mw": s.get("peak_load_mw")}
            for i, s in enumerate(scenarios[:20])  # preview first 20
        ],
    }


# ---------------------------------------------------------------------------
# RL Agent endpoints
# ---------------------------------------------------------------------------


@router.get("/agent/info", tags=["agent"])
async def agent_info():
    """Return current agent status and model info."""
    from backend.main import app_state
    agent = app_state.get("agent")
    if agent is None:
        return {"status": "not_initialized"}
    return agent.get_model_info()


@router.post("/agent/step", tags=["agent"])
async def agent_step(req: AgentStepRequest):
    """
    Run one agent inference step.
    If no observation provided, runs power flow to get current state.
    """
    from backend.main import app_state
    import numpy as np

    agent = app_state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    if not agent.is_trained:
        raise HTTPException(status_code=503, detail="Agent not trained yet")

    if req.observation is None:
        env = app_state.get("env")
        if env is None:
            raise HTTPException(status_code=503, detail="Environment not initialized")
        obs = env._get_observation()
    else:
        obs = np.array(req.observation, dtype=np.float32)

    action, info = agent.predict(obs, deterministic=req.deterministic)
    action_desc = "no-op" if action == 0 else f"toggle switch {action - 1}"

    return {"action": action, "action_description": action_desc, "info": info}


@router.post("/agent/train", tags=["agent"])
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Start RL training in the background."""
    from backend.main import app_state

    if app_state.get("training_active"):
        raise HTTPException(status_code=409, detail="Training already in progress")

    app_state["training_active"] = True
    background_tasks.add_task(
        _run_training,
        app_state,
        req.total_timesteps,
        req.checkpoint_freq,
    )
    return {"status": "started", "total_timesteps": req.total_timesteps}


@router.get("/agent/training-status", tags=["agent"])
async def training_status():
    """Check training progress."""
    from backend.main import app_state
    return {
        "active": app_state.get("training_active", False),
        "progress": app_state.get("training_progress", {}),
    }


@router.post("/agent/evaluate", tags=["agent"])
async def evaluate_agent(n_episodes: int = 10):
    """Evaluate trained agent over N episodes."""
    from backend.main import app_state
    agent = app_state.get("agent")
    if agent is None or not agent.is_trained:
        raise HTTPException(status_code=503, detail="Agent not trained")
    results = agent.evaluate(n_episodes=n_episodes)
    return results


# ---------------------------------------------------------------------------
# WebSocket for real-time updates
# ---------------------------------------------------------------------------


@router.websocket("/ws/grid")
async def websocket_grid(websocket: WebSocket):
    """
    WebSocket endpoint for real-time grid state streaming.
    Sends grid state every second while connected.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        from backend.main import app_state
        while True:
            state = app_state["grid_engine"].run_power_flow()
            data = {
                "type": "grid_state",
                "data": state.to_dict(),
                "training_active": app_state.get("training_active", False),
                "training_progress": app_state.get("training_progress", {}),
            }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@router.get("/health", tags=["system"])
async def health():
    """Health check endpoint."""
    from backend.main import app_state
    return {
        "status": "ok",
        "version": "2.0.0",
        "grid_loaded": app_state.get("grid_engine") is not None,
        "agent_ready": app_state.get("agent") is not None,
        "scenarios_loaded": len(app_state.get("scenarios", [])),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _run_training(app_state: dict, total_timesteps: int, checkpoint_freq: int) -> None:
    """Background training task."""
    try:
        agent = app_state.get("agent")
        if agent is None:
            logger.error("No agent to train")
            return

        def progress_cb(info: dict) -> None:
            app_state["training_progress"] = info

        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: agent.train(
                total_timesteps=total_timesteps,
                checkpoint_freq=checkpoint_freq,
                progress_callback=progress_cb,
            ),
        )
        app_state["training_progress"] = {"status": "completed", "timesteps": total_timesteps}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        app_state["training_progress"] = {"status": "error", "error": str(e)}
    finally:
        app_state["training_active"] = False


def _apply_scenario(engine, scenario: dict, hour: int) -> None:
    """Apply scenario load/generation profile to engine network."""
    import pandapower as pp
    net = engine.net
    load_factor = scenario["load_profile"][hour % 24]
    solar_factor = scenario["solar_profile"][hour % 24]
    if len(net.load) > 0:
        net.load["p_mw"] = net.load["p_mw"] * load_factor
    if len(net.sgen) > 0:
        net.sgen["p_mw"] = net.sgen["p_mw"] * solar_factor
