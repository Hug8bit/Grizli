# GrizliV2 — Grid Optimizer

> **"Waze for Electricity"** — An AI-powered distribution grid optimizer.
> Trains on NREL SMART-DS San Francisco data · Targets LATAM / Chile (Atacama)

---

## What is GrizliV2?

GrizliV2 optimizes electricity distribution grids using **Reinforcement Learning**.
Like Waze reroutes traffic in real-time, GrizliV2 continuously reconfigures the grid
topology to minimize losses, reduce congestion, and maintain voltage stability.

**Key differentiator for Chile/Atacama:**
- Extreme solar irradiation → high renewable penetration requiring smart grid management
- Long narrow grid (Chile's SEN) → topology optimization is critical
- Remote desert substations → autonomous AI control reduces operational costs

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    GrizliV2                          │
│                                                      │
│  ┌──────────────┐    ┌────────────────────────────┐  │
│  │  Frontend    │    │      Backend (FastAPI)     │  │
│  │  React/Vite  │◄──►│                            │  │
│  │  D3 viz      │    │  ┌──────────┐  ┌────────┐  │  │
│  │  Real-time   │    │  │  Grid    │  │  RL    │  │  │
│  └──────────────┘    │  │ Engine   │  │ Agent  │  │  │
│                      │  │(Pandap.) │  │ (PPO)  │  │  │
│  ┌──────────────┐    │  └────┬─────┘  └───┬────┘  │  │
│  │  SMART-DS    │    │       │            │       │  │
│  │  SF Bay Area │───►│  ┌────┴────────────┴────┐  │  │
│  │  (Training)  │    │  │  GrizliGridEnv       │  │  │
│  └──────────────┘    │  │  (Gymnasium)         │  │  │
│                      │  └──────────────────────┘  │  │
│                      └────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## Features

- **RL Agent (PPO)**: Learns optimal topology switching from thousands of grid scenarios
- **SMART-DS Integration**: Trains on real SF Bay Area distribution data (NREL/DOE)
- **FastAPI + WebSocket**: Real-time grid state streaming to frontend
- **D3 Visualization**: Interactive force-graph of buses and lines, colored by state
- **Scenario Engine**: 200+ synthetic 24h scenarios (load + solar + wind profiles)
- **Docker ready**: One-command deployment

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 20+

### 1. Install backend

```bash
pip install -e ".[rl]"
```

### 2. Start backend

```bash
uvicorn backend.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 3. Start frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### 4. (Optional) Train the RL agent

```bash
# Train on synthetic SF scenarios (no dataset download needed)
python scripts/train.py --timesteps 100000

# Train on actual SMART-DS data (see download instructions below)
python scripts/train.py --timesteps 1000000 --scenario-dir data/smartds/P1R
```

---

## SMART-DS Dataset (San Francisco Bay Area)

The NREL **SMART-DS** dataset provides realistic synthetic distribution grids
for the SF Bay Area. Publicly available under CC0 license.

**Download:**
1. Go to: https://data.openei.org/submissions/2981
2. Download `P1R` (residential, ~200MB) or `P4R` (larger area)
3. Extract to `data/smartds/`

Without downloading, GrizliV2 automatically uses a built-in synthetic 33-bus
network as a fallback.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/grid/topology` | GET | Grid buses + lines |
| `/api/v1/grid/power-flow` | POST | Run AC power flow |
| `/api/v1/grid/switch` | POST | Toggle a switch |
| `/api/v1/grid/metrics` | GET | Performance metrics |
| `/api/v1/agent/info` | GET | Agent status |
| `/api/v1/agent/train` | POST | Start RL training (background) |
| `/api/v1/agent/step` | POST | Run one inference step |
| `/api/v1/ws/grid` | WS | Real-time grid stream |

---

## Docker

```bash
docker compose up
# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

---

## Project Structure

```
GrizliV2/
├── backend/
│   ├── api/routes.py          # FastAPI endpoints + WebSocket
│   ├── core/grid_engine.py    # PandaPower simulation engine
│   ├── data/smartds_loader.py # SMART-DS parser + scenario generation
│   ├── ml/
│   │   ├── rl_environment.py  # Gymnasium env (topology control)
│   │   └── rl_agent.py        # PPO agent (Stable-Baselines3)
│   ├── config/settings.py     # Pydantic settings
│   └── main.py                # FastAPI app
├── frontend/
│   └── src/
│       ├── components/        # GridVisualization, MetricsPanel, AgentPanel
│       ├── hooks/             # useGridState (WebSocket + REST)
│       └── services/api.ts    # Typed API client
├── scripts/train.py           # Standalone training script
├── data/smartds/              # SMART-DS data (download separately)
├── models/                    # Saved RL models
├── archive/v1/                # V1 code (archived)
└── docker-compose.yml
```

---

## Roadmap

- [ ] Chilean grid data integration (CNE/SEN public data)
- [ ] Transfer learning: SF → Chile fine-tuning
- [ ] Geographic map view (Leaflet) with Chile topology overlay
- [ ] Multi-agent control for large-scale grids
- [ ] Renewable curtailment optimization (Atacama solar)
- [ ] SCADA integration via MQTT/OPC-UA

---

## Requirements

- Python 3.10+ · Node.js 20+
- 4 GB RAM minimum (8 GB for RL training)

## License

MIT License

## Acknowledgments

- [PandaPower](https://www.pandapower.org/) — Power system simulation
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — PPO implementation
- [NREL SMART-DS](https://data.openei.org/submissions/2981) — SF Bay Area grid data (CC0)
