#!/usr/bin/env python3
"""
GRIZLI Demo Script
Demonstrates network generation, power flow, optimization, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp

from src.core.ieee_networks import create_congested_ieee33, create_atacama_scenario
from src.core.network_generator import NetworkGenerator, NetworkType, GeneratorConfig, LoadConfig
from src.simulation.power_flow import PowerFlowSimulator
from src.simulation.time_series import TimeSeriesSimulator, ScenarioGenerator, TimeSeriesConfig, Season
from src.optimization.fast_optimizer import optimize_topology_fast, apply_optimization_result, get_reconfiguration_summary
from src.optimization.benchmark import Benchmark


def demo_network_generation():
    """Demo: Generate a synthetic network."""
    print("\n" + "="*60)
    print("DEMO 1: Network Generation")
    print("="*60)

    # Create a synthetic 25-bus meshed network
    generator = NetworkGenerator(
        num_buses=25,
        network_type=NetworkType.MESHED,
        voltage_kv=20.0,
        seed=42
    )

    gen_config = GeneratorConfig(num_solar=3, num_wind=2, solar_capacity_mw=2.0, wind_capacity_mw=3.0)
    load_config = LoadConfig(num_residential=8, num_industrial=2, num_commercial=3)

    net = generator.generate(gen_config, load_config)

    print(f"\nGenerated network: {net.name}")
    print(f"  Buses: {len(net.bus)}")
    print(f"  Lines: {len(net.line)}")
    print(f"  Generators: {len(net.sgen)}")
    print(f"  Loads: {len(net.load)}")
    print(f"  Total generation: {net.sgen.p_mw.sum():.2f} MW")
    print(f"  Total load: {net.load.p_mw.sum():.2f} MW")

    return net


def demo_power_flow(net: pp.pandapowerNet):
    """Demo: Run power flow simulation."""
    print("\n" + "="*60)
    print("DEMO 2: Power Flow Simulation")
    print("="*60)

    simulator = PowerFlowSimulator(net)
    result = simulator.run_power_flow()

    print(f"\nPower flow {'converged' if result.converged else 'FAILED'}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Total generation: {result.total_generation_mw:.2f} MW")
    print(f"  Total load: {result.total_load_mw:.2f} MW")
    print(f"  Total losses: {result.total_losses_mw*1000:.1f} kW ({result.total_losses_mw/result.total_generation_mw*100:.1f}%)")
    print(f"  Max line loading: {result.max_loading_percent:.1f}%")
    print(f"  Voltage range: {result.min_voltage_pu:.3f} - {result.max_voltage_pu:.3f} p.u.")
    print(f"  Overloaded lines: {result.num_overloaded_lines}")
    print(f"  Voltage violations: {result.num_voltage_violations}")
    print(f"  Congestion metric: {result.congestion_metric:.1f}")

    # Show congested lines
    congested = simulator.get_congested_lines(threshold=80)
    if congested:
        print(f"\nCongested lines (>80% loading):")
        for c in congested[:5]:
            print(f"  Line {c.line_idx}: {c.from_bus}->{c.to_bus}, {c.loading_percent:.1f}%")

    return result


def demo_optimization():
    """Demo: Network optimization."""
    print("\n" + "="*60)
    print("DEMO 3: Network Optimization")
    print("="*60)

    # Create a congested network
    print("\nCreating congested IEEE 33-bus network...")
    net = create_congested_ieee33(load_increase_percent=60, renewable_increase_percent=100)

    # Run initial power flow
    pp.runpp(net, numba=True)

    print(f"\nInitial state:")
    print(f"  Max loading: {net.res_line.loading_percent.max():.1f}%")
    print(f"  Overloaded lines: {(net.res_line.loading_percent > 100).sum()}")
    print(f"  Losses: {net.res_line.pl_mw.sum()*1000:.1f} kW")

    # Run optimization
    print("\nRunning fast optimization...")
    result = optimize_topology_fast(net, max_iterations=5)

    print(get_reconfiguration_summary(result))

    # Apply and verify
    if result.lines_to_open or result.lines_to_close:
        apply_optimization_result(net, result)
        pp.runpp(net, numba=True)

        print(f"\nFinal state after applying optimization:")
        print(f"  Max loading: {net.res_line.loading_percent.max():.1f}%")
        print(f"  Overloaded lines: {(net.res_line.loading_percent > 100).sum()}")
        print(f"  Losses: {net.res_line.pl_mw.sum()*1000:.1f} kW")

    return net, result


def demo_time_series():
    """Demo: 24-hour time series simulation."""
    print("\n" + "="*60)
    print("DEMO 4: 24-Hour Simulation")
    print("="*60)

    # Create network
    net = create_congested_ieee33()

    # Configure scenario
    config = TimeSeriesConfig(
        duration_hours=24,
        resolution_minutes=15,
        season=Season.SUMMER,
        cloud_variability=0.2,
        wind_variability=0.3
    )

    print(f"\nSimulation config:")
    print(f"  Duration: {config.duration_hours} hours")
    print(f"  Resolution: {config.resolution_minutes} minutes")
    print(f"  Time steps: {config.num_steps}")
    print(f"  Season: {config.season.value}")

    # Generate scenario
    scenario_gen = ScenarioGenerator(config, seed=42)
    scenario = scenario_gen.generate_network_scenario(net)

    print(f"\nGenerated profiles for:")
    print(f"  {len(scenario['sgen'].columns)} generators")
    print(f"  {len(scenario['load'].columns)} loads")

    # Run simulation
    print("\nRunning simulation...")
    ts_sim = TimeSeriesSimulator(net)
    results = ts_sim.run_simulation(scenario['sgen'], scenario['load'], verbose=False)

    stats = ts_sim.get_congestion_statistics()

    print(f"\nSimulation results:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Converged steps: {stats['converged_steps']} ({stats['convergence_rate']*100:.1f}%)")
    print(f"  Peak max loading: {stats['max_loading_peak']:.1f}%")
    print(f"  Mean max loading: {stats['max_loading_mean']:.1f}%")
    print(f"  Steps with overload: {stats['steps_with_overload']} ({stats['overload_rate']*100:.1f}%)")
    print(f"  Total energy losses: {stats['total_losses_mwh']:.2f} MWh")

    return results


def demo_visualization(net: pp.pandapowerNet, results):
    """Demo: Generate visualizations."""
    print("\n" + "="*60)
    print("DEMO 5: Visualization")
    print("="*60)

    try:
        import networkx as nx

        # Network graph
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Network topology
        ax = axes[0, 0]
        G = nx.Graph()
        for idx in net.bus.index:
            G.add_node(idx)
        for idx in net.line.index:
            if net.line.at[idx, 'in_service']:
                G.add_edge(int(net.line.at[idx, 'from_bus']),
                          int(net.line.at[idx, 'to_bus']))

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, node_size=100, node_color='lightblue',
               with_labels=False, edge_color='gray')
        ax.set_title("Network Topology")

        # 2. Line loading
        ax = axes[0, 1]
        if 'res_line' in net and len(net.res_line) > 0:
            loading = net.res_line.loading_percent.values
            colors = ['red' if l > 100 else 'orange' if l > 80 else 'green' for l in loading]
            ax.bar(range(len(loading)), loading, color=colors)
            ax.axhline(y=100, color='red', linestyle='--', label='Limit')
            ax.axhline(y=80, color='orange', linestyle='--', label='Warning')
            ax.set_xlabel('Line Index')
            ax.set_ylabel('Loading (%)')
            ax.set_title('Line Loading')
            ax.legend()

        # 3. Voltage profile
        ax = axes[1, 0]
        if 'res_bus' in net and len(net.res_bus) > 0:
            voltage = net.res_bus.vm_pu.values
            colors = ['red' if v < 0.95 or v > 1.05 else 'green' for v in voltage]
            ax.bar(range(len(voltage)), voltage, color=colors)
            ax.axhline(y=1.05, color='red', linestyle='--')
            ax.axhline(y=0.95, color='red', linestyle='--')
            ax.axhline(y=1.0, color='gray', linestyle=':')
            ax.set_xlabel('Bus Index')
            ax.set_ylabel('Voltage (p.u.)')
            ax.set_title('Voltage Profile')
            ax.set_ylim(0.9, 1.1)

        # 4. Time series results
        ax = axes[1, 1]
        if results is not None:
            ax.plot(results['max_loading_percent'], label='Max Loading', color='blue')
            ax.axhline(y=100, color='red', linestyle='--', label='Limit')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Loading (%)')
            ax.set_title('24h Max Loading Profile')
            ax.legend()

        plt.tight_layout()
        plt.savefig('demo_output.png', dpi=150)
        print("\nVisualization saved to: demo_output.png")
        plt.show()

    except ImportError as e:
        print(f"\nVisualization skipped (missing dependency: {e})")


def main():
    """Run all demos."""
    print("\n" + "#"*60)
    print("#" + " "*18 + "GRIZLI DEMO" + " "*18 + "#")
    print("#" + " "*14 + "Smart Grid Optimizer" + " "*14 + "#")
    print("#"*60)

    # Demo 1: Network generation
    net = demo_network_generation()

    # Demo 2: Power flow
    demo_power_flow(net)

    # Demo 3: Optimization
    opt_net, opt_result = demo_optimization()

    # Demo 4: Time series
    ts_results = demo_time_series()

    # Demo 5: Visualization
    demo_visualization(opt_net, ts_results)

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nTo run the interactive dashboard:")
    print("  streamlit run app.py")
    print("\nTo train the RL agent:")
    print("  python train.py --timesteps 10000")
    print()


if __name__ == "__main__":
    main()
