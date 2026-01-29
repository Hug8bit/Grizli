# GRIZLI

**Grid Reconfiguration Intelligence for Zero-Loss Integration**

*"Waze for Electricity"* - Smart grid optimization through dynamic network reconfiguration.

## Overview

GRIZLI optimizes power grid operations by dynamically reconfiguring network topology to:
- **Reduce congestion** on overloaded lines (30-70% improvement)
- **Minimize losses** in the distribution network (5-20% reduction)
- **Improve reliability** through proactive management
- **Enable higher renewable penetration** without infrastructure upgrades

## Features

- **Network Generation**: Create IEEE standard networks (14, 33 bus) or custom synthetic networks
- **Power Flow Simulation**: AC power flow using Newton-Raphson (PandaPower)
- **Fast Optimization**: Greedy topology optimization in <1 second
- **MILP Optimization**: Optimal reconfiguration using Mixed Integer Linear Programming
- **24h Time Series**: Simulate daily scenarios with solar/wind/load profiles
- **Interactive Dashboard**: Streamlit-based visualization and control
- **RL Agent**: Experimental PPO agent using Grid2Op (optional)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/grizli.git
cd grizli

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Run the Demo

```bash
python demo.py
```

### Run Tests

```bash
python test_optimization.py
```

## Project Structure

```
GRIZLI/
├── src/
│   ├── core/              # Network generation
│   │   ├── network_generator.py    # Synthetic network creation
│   │   ├── network_topology.py     # NetworkX-based topology
│   │   └── ieee_networks.py        # IEEE 14/33 bus networks
│   │
│   ├── simulation/        # Power flow simulation
│   │   ├── power_flow.py           # AC power flow (PandaPower)
│   │   └── time_series.py          # 24h scenario generation
│   │
│   ├── optimization/      # Reconfiguration algorithms
│   │   ├── reconfiguration.py      # MILP optimizer (PuLP)
│   │   ├── fast_optimizer.py       # Greedy optimizer (<1s)
│   │   └── benchmark.py            # Before/after comparison
│   │
│   ├── rl_agent/          # Reinforcement learning (optional)
│   │   ├── ppo_agent.py            # PPO with stable-baselines3
│   │   ├── grid2op_env_wrapper.py  # Gymnasium wrapper
│   │   ├── action_space.py         # Top-K action selection
│   │   └── ai_controller.py        # Dashboard integration
│   │
│   ├── grid2op_env/       # Grid2Op environment support
│   │   ├── grid2op_init.py         # Environment creation
│   │   └── scenario_player.py      # Chronic playback
│   │
│   ├── smartds/           # SMART-DS data support
│   │   ├── opendss_parser.py       # OpenDSS file parser
│   │   └── network_visualizer.py   # Network visualization
│   │
│   └── visualization/     # Plotting utilities
│       └── network_plot.py         # Matplotlib plots
│
├── app.py                 # Streamlit dashboard
├── demo.py                # Complete demonstration
├── train.py               # RL training script
├── test_optimization.py   # Optimization tests
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Usage Examples

### Create and Optimize a Congested Network

```python
from src.core.ieee_networks import create_congested_ieee33
from src.optimization.fast_optimizer import optimize_topology_fast, apply_optimization_result
import pandapower as pp

# Create congested network
net = create_congested_ieee33(load_increase_percent=50)

# Run initial power flow
pp.runpp(net)
print(f"Initial max loading: {net.res_line.loading_percent.max():.1f}%")

# Optimize
result = optimize_topology_fast(net, max_iterations=5)
print(f"Congestion reduction: {result.congestion_reduction_percent:.1f}%")

# Apply optimization
apply_optimization_result(net, result)
pp.runpp(net)
print(f"Final max loading: {net.res_line.loading_percent.max():.1f}%")
```

### Run 24-Hour Simulation

```python
from src.simulation.time_series import TimeSeriesSimulator, ScenarioGenerator, TimeSeriesConfig

config = TimeSeriesConfig(duration_hours=24, resolution_minutes=15)
scenario_gen = ScenarioGenerator(config, seed=42)
scenario = scenario_gen.generate_network_scenario(net)

ts_sim = TimeSeriesSimulator(net)
results = ts_sim.run_simulation(scenario['sgen'], scenario['load'])
stats = ts_sim.get_congestion_statistics()
print(f"Peak loading: {stats['max_loading_peak']:.1f}%")
```

### Train RL Agent (Optional)

```bash
# Install RL dependencies first
pip install grid2op stable-baselines3 torch gymnasium

# Train for 100k timesteps
python train.py --timesteps 100000 --env l2rpn_case14_sandbox

# Evaluate trained model
python train.py --eval-only --model-path models/ppo_grid/best_model
```

## How It Works

### The Problem

Modern power grids face increasing challenges:
- Variable renewable generation (solar peaks at noon, wind is unpredictable)
- Growing demand and aging infrastructure
- Lines operating near thermal limits

When lines become overloaded (>100% loading), they must be disconnected to prevent damage, potentially causing cascading failures.

### The Solution

GRIZLI reconfigures the network topology by opening and closing switches to redistribute power flow:

```
Before: Line A at 120% ──────────────> Overload!
        Line B at 40%

After:  Line A at 85%  ──────────────> Balanced
        Line B at 75%
```

### Optimization Approach

1. **Fast Greedy**: Branch exchange heuristic
   - Close a tie-line (normally open)
   - Open a line in the created loop
   - Keep the swap if it reduces congestion
   - Repeat until no improvement

2. **MILP**: Optimal solution using linearized DC power flow
   - Binary variables for switch states
   - Big-M formulation for on/off constraints
   - Minimize: congestion + losses + switching cost

## Requirements

- Python 3.9, 3.10, or 3.11
- 4 GB RAM minimum (8 GB for RL training)
- Windows, Linux, or macOS

## Dependencies

### Core (Required)
- `pandapower` - Power system simulation
- `networkx` - Graph algorithms
- `pulp` - MILP optimization
- `numpy`, `pandas` - Data processing
- `streamlit`, `plotly` - Dashboard
- `matplotlib` - Plotting

### Optional (for RL)
- `grid2op` - Grid control RL environment
- `stable-baselines3` - PPO implementation
- `torch` - Neural network backend
- `gymnasium` - RL interface

## Performance

| Metric | Typical Result |
|--------|---------------|
| Congestion reduction | 30-70% |
| Losses reduction | 5-20% |
| Fast optimizer time | <1 second |
| MILP optimizer time | ~10 seconds |

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [PandaPower](https://www.pandapower.org/) - Power system simulation
- [Grid2Op](https://grid2op.readthedocs.io/) - RL environment for power grids
- [NREL SMART-DS](https://data.nrel.gov/submissions/74) - Real distribution system data

## Contact

For questions or issues, please open a GitHub issue.
