# GRIZLI — Grid Reconfiguration Intelligence for Zero-Loss Integration

> Distribution network optimizer using real OpenDSS power flow data.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it does

GRIZLI is a distribution network reconfiguration tool. Given an OpenDSS network:
1. **Simulates** nominal and stress scenarios using full Newton-Raphson power flow
2. **Identifies** voltage violations, overloaded lines, and excess losses
3. **Optimizes** the network topology via greedy branch-exchange and tie-line load balancing
4. **Reports** KPIs: loss reduction %, violations resolved, compute time

## Validated Results (NREL SMART-DS v1.0 — Substation P1R)

| Metric | Stress ×0.75 | GRIZLI Optimized | Δ |
|--------|-------------|-----------------|---|
| Active losses (kW) | 511.1 | 423.9 | **−17.1%** |
| Voltage violations | 171 | 0 | **−100%** |
| V min (p.u.) | 0.934 | 0.964 | **+0.030** |
| Overloaded lines | 3 | 1 | **−67%** |

## Structure

```
grizli/
├── app.py                  # Streamlit interactive UI
├── requirements.txt
├── engine/
│   ├── opendss_loader.py   # OpenDSS feeder wrapper (dss-python)
│   └── optimizer.py        # Greedy branch-exchange + multi-feeder balancer
└── benchmark/
    ├── results.json         # Validated benchmark results
    └── case_study.html      # Interactive HTML report
```

## Algorithm

**Single-feeder mode**: Greedy forward search over switch candidates.
Objective: `minimize losses + 5×violations + 2×overloads`

**Multi-feeder mode**: Load balancing via tie-line reconfiguration.
Transfers load from overloaded feeders to feeders with headroom.
Constraint: total load conserved within ±10%.

## Input Format

Upload a `.zip` containing OpenDSS folder(s):
```
network.zip/
└── feeder1/
    ├── Master.dss
    ├── Lines.dss
    ├── Loads.dss
    ├── LineCodes.dss
    └── Transformers.dss
```

## License

MIT · Built on [NREL SMART-DS](https://data.openei.org/submissions/2671) open dataset.
