"""
GrizliV2 - FastAPI Application Entry Point
Grid optimization platform — "Waze for Electricity"
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import settings
from backend.core.grid_engine import GridEngine, create_ieee33_engine
from backend.data.smartds_loader import SmartDSLoader
from backend.ml.rl_environment import GrizliGridEnv
from backend.ml.rl_agent import GrizliAgent

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global application state
# Shared between routes via import — no singleton / DI framework needed yet.
# ---------------------------------------------------------------------------
app_state: dict = {
    "grid_engine": None,
    "env": None,
    "agent": None,
    "scenarios": [],
    "training_active": False,
    "training_progress": {},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # --- Startup ---
    logger.info("=== GrizliV2 starting up ===")

    # 1. Load grid network
    smartds_path = settings.smartds_dir
    loader = SmartDSLoader(smartds_path)
    net = loader.load_network()
    app_state["grid_engine"] = GridEngine(net)
    logger.info(f"Grid loaded: {len(net.bus)} buses, {len(net.line)} lines")

    # 2. Load scenarios for RL training
    scenarios = loader.load_scenarios(n_scenarios=200)
    app_state["scenarios"] = scenarios
    logger.info(f"Loaded {len(scenarios)} training scenarios")

    # 3. Initialize RL environment
    env = GrizliGridEnv(net=net, scenarios=scenarios, max_steps=96)
    app_state["env"] = env

    # 4. Initialize RL agent (load from disk if available)
    agent = GrizliAgent(env=env, models_dir=settings.models_dir)
    final_model = settings.models_dir / "grizli_ppo_final.zip"
    if final_model.exists():
        try:
            agent.build()
            agent.load(final_model)
            logger.info("Pre-trained model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load model: {e} — starting fresh")
            agent.build()
    else:
        agent.build()
        logger.info("No pre-trained model found — agent initialized untrained")
    app_state["agent"] = agent

    logger.info("=== GrizliV2 ready ===")
    yield

    # --- Shutdown ---
    logger.info("=== GrizliV2 shutting down ===")
    app_state["training_active"] = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GrizliV2",
    description=(
        "Grid optimization platform powered by Reinforcement Learning. "
        "Trains on NREL SMART-DS San Francisco data, "
        "targets LATAM distribution grids (Chile/Atacama)."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.api.routes import router  # noqa: E402
app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
