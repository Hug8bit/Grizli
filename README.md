# GRIZLI — AI Grid Energy

**Grid Reconfiguration Intelligence for Zero-Loss Integration**

*"Waze for Electricity"* — Smart grid optimisation through dynamic network reconfiguration,
ML-based consumption prediction, and real-time San Francisco open-data integration.

---

## Features

| Module | Description |
|--------|-------------|
| **Network simulation** | IEEE 33-bus (+ renewables, congested, Atacama), custom synthetic networks |
| **Power flow** | AC Newton-Raphson via PandaPower |
| **Optimisation** | Greedy fast optimizer (<1 s) + MILP reconfiguration |
| **24 h time series** | Solar/wind/load profiles, seasonal scenarios |
| **SF data pipeline** | DataSF API with synthetic fallback (energy, SFMTA, weather) |
| **ML predictor** | Gradient Boosting — hourly consumption prediction + peak advice |
| **FastAPI backend** | REST API for simulation, ML, and SF data |
| **React dashboard** | Interactive network graph, ML curves, SF map (dark mode) |
| **Streamlit (legacy)** | Original interactive dashboard still functional |
| **RL agent** | Experimental PPO via Grid2Op (optional heavy deps) |

---

## Architecture

```
grizli/
├── src/
│   ├── core/           # Network builders (IEEE33, custom)
│   ├── simulation/     # Power flow + time series
│   ├── optimization/   # Fast greedy + MILP
│   ├── data/           # SF open-data loader (DataSF + synthetic fallback)
│   ├── ml/             # Energy predictor (sklearn GBM)
│   ├── rl_agent/       # PPO agent (optional)
│   └── visualization/  # Plot utilities
├── backend/            # FastAPI REST API
├── frontend/           # React 18 + Tailwind dashboard
├── tests/              # pytest test suite
├── models/             # Persisted ML models
├── data/               # Local data cache (populated at runtime)
└── app.py              # Legacy Streamlit dashboard
```

---

## Quick Start

### 1 — Prerequisites

```bash
python >= 3.11
node   >= 18     # for the React frontend
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Configure environment

```bash
cp .env.example .env
# Edit .env as needed (all defaults work out of the box)
```

### 4a — Run the FastAPI backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
# API docs: http://localhost:8000/docs
```

### 4b — Run the React frontend

```bash
cd frontend
npm install
npm run dev
# App: http://localhost:5173
```

### 4c — Run the legacy Streamlit dashboard

```bash
streamlit run app.py
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
# With coverage:
pytest tests/ --cov=src --cov-report=term-missing
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/simulate` | Run power flow with configurable params |
| `GET`  | `/api/network`  | Current network state (nodes, lines, flows) |
| `POST` | `/api/ml/train` | Train / retrain the ML predictor |
| `GET`  | `/api/ml/predict` | Predict energy consumption + optimisation advice |
| `GET`  | `/api/data/sf`  | SF processed data (energy, SFMTA stops, network nodes) |

Full interactive docs at `http://localhost:8000/docs` (Swagger UI).

---

## SF Data Pipeline

Data is fetched at runtime from the [DataSF API](https://data.sfgov.org/resource/).
If the API is unreachable, realistic synthetic data is generated automatically
(documented in `src/data/sf_data_loader.py`).

| Dataset | DataSF ID | Fallback |
|---------|-----------|---------|
| PG&E electric usage by zip | `h9km-ggyk` | Synthetic seasonal/diurnal profile |
| SFMTA bus stops | `i28k-bkz6` | Synthetic stops in SF bounding box |
| Network nodes (geo) | N/A | IEEE 33-bus mapped to SF coordinates |

---

## ML Model

`src/ml/energy_predictor.py` trains a `GradientBoostingRegressor` on:

- **Temporal features**: hour, day-of-week, month, is_weekend
- **Weather**: temperature (°C), humidity (%)
- **Urban activity**: SFMTA daily ridership

Typical metrics on synthetic data: **R² ≈ 0.97, MAE ≈ 2–3 MW**.

---

## License

MIT — see [LICENSE](LICENSE).
