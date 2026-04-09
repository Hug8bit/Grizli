"""
GRIZLI - Streamlit Dashboard
Grid Reconfiguration Intelligence for Zero-Loss Integration

Interactive dashboard for network visualization and optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import time

import pandapower as pp

# Import GRIZLI modules
from src.core.ieee_networks import (
    create_ieee33_with_renewables,
    create_congested_ieee33,
    create_atacama_scenario,
)
from src.core.network_generator import NetworkGenerator, NetworkType
from src.simulation.power_flow import PowerFlowSimulator
from src.simulation.time_series import TimeSeriesSimulator, ScenarioGenerator, TimeSeriesConfig, Season
from src.optimization.fast_optimizer import optimize_topology_fast, apply_optimization_result
from src.optimization.benchmark import Benchmark, NetworkState


# Page configuration
st.set_page_config(
    page_title="GRIZLI - Smart Grid Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #856404;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


def create_network_graph(net: pp.pandapowerNet, highlight_overloaded: bool = True) -> go.Figure:
    """Create interactive Plotly network graph."""
    # Build NetworkX graph
    G = nx.Graph()

    for idx in net.bus.index:
        G.add_node(idx)

    for idx in net.line.index:
        if net.line.at[idx, 'in_service']:
            G.add_edge(
                int(net.line.at[idx, 'from_bus']),
                int(net.line.at[idx, 'to_bus']),
                idx=idx
            )

    # Layout
    if hasattr(net, 'bus_geodata') and len(net.bus_geodata) > 0:
        pos = {int(idx): (net.bus_geodata.at[idx, 'x'], net.bus_geodata.at[idx, 'y'])
               for idx in net.bus.index if idx in net.bus_geodata.index}
    else:
        pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(len(G.nodes())))

    # Edge traces
    edge_x = []
    edge_y = []
    edge_colors = []

    has_results = 'res_line' in net and len(net.res_line) > 0

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        if has_results and highlight_overloaded:
            idx = data.get('idx')
            if idx is not None and idx < len(net.res_line):
                loading = net.res_line.at[idx, 'loading_percent']
                if loading > 100:
                    edge_colors.extend(['red', 'red', 'red'])
                elif loading > 80:
                    edge_colors.extend(['orange', 'orange', 'orange'])
                else:
                    edge_colors.extend(['green', 'green', 'green'])
            else:
                edge_colors.extend(['gray', 'gray', 'gray'])
        else:
            edge_colors.extend(['gray', 'gray', 'gray'])

    # Node positions
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    # Node colors by type
    load_buses = set(net.load.bus.values) if len(net.load) > 0 else set()
    gen_buses = set(net.sgen.bus.values) if len(net.sgen) > 0 else set()
    slack_buses = set(net.ext_grid.bus.values) if len(net.ext_grid) > 0 else set()

    node_colors = []
    node_sizes = []
    node_text = []

    for n in G.nodes():
        if n in slack_buses:
            node_colors.append('red')
            node_sizes.append(20)
            node_text.append(f"Bus {n} (Slack)")
        elif n in gen_buses:
            node_colors.append('green')
            node_sizes.append(15)
            node_text.append(f"Bus {n} (Generator)")
        elif n in load_buses:
            node_colors.append('orange')
            node_sizes.append(12)
            node_text.append(f"Bus {n} (Load)")
        else:
            node_colors.append('lightblue')
            node_sizes.append(8)
            node_text.append(f"Bus {n}")

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        name='Lines'
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='black')),
        text=node_text,
        hoverinfo='text',
        name='Buses'
    ))

    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )

    return fig


def create_loading_bar_chart(net: pp.pandapowerNet) -> go.Figure:
    """Create line loading bar chart."""
    if 'res_line' not in net or len(net.res_line) == 0:
        return go.Figure()

    loading = net.res_line.loading_percent.values
    n_lines = len(loading)

    colors = ['red' if l > 100 else 'orange' if l > 80 else 'yellow' if l > 50 else 'green'
              for l in loading]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(n_lines)),
        y=loading,
        marker_color=colors,
        name='Loading'
    ))

    fig.add_hline(y=100, line_dash="dash", line_color="red",
                  annotation_text="Limit (100%)")
    fig.add_hline(y=80, line_dash="dash", line_color="orange",
                  annotation_text="Warning (80%)")

    fig.update_layout(
        title="Line Loading (%)",
        xaxis_title="Line Index",
        yaxis_title="Loading (%)",
        height=300,
        margin=dict(l=50, r=20, t=50, b=50)
    )

    return fig


def create_voltage_profile(net: pp.pandapowerNet) -> go.Figure:
    """Create voltage profile chart."""
    if 'res_bus' not in net or len(net.res_bus) == 0:
        return go.Figure()

    voltage = net.res_bus.vm_pu.values
    n_buses = len(voltage)

    colors = ['red' if v < 0.95 or v > 1.05 else 'orange' if v < 0.97 or v > 1.03 else 'green'
              for v in voltage]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(n_buses)),
        y=voltage,
        marker_color=colors,
        name='Voltage'
    ))

    fig.add_hline(y=1.05, line_dash="dash", line_color="red")
    fig.add_hline(y=0.95, line_dash="dash", line_color="red")
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray")

    fig.update_layout(
        title="Bus Voltage Profile (p.u.)",
        xaxis_title="Bus Index",
        yaxis_title="Voltage (p.u.)",
        yaxis=dict(range=[0.9, 1.1]),
        height=250,
        margin=dict(l=50, r=20, t=50, b=50)
    )

    return fig


def create_24h_simulation_chart(results: pd.DataFrame) -> go.Figure:
    """Create 24-hour simulation chart."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Max Line Loading (%)", "Network Losses (MW)", "Overloaded Lines"),
        vertical_spacing=0.1
    )

    # Max loading
    fig.add_trace(
        go.Scatter(x=list(range(len(results))), y=results['max_loading_percent'],
                  mode='lines', name='Max Loading', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_hline(y=100, line_dash="dash", line_color="red", row=1, col=1)

    # Losses
    fig.add_trace(
        go.Scatter(x=list(range(len(results))), y=results['total_losses_mw'],
                  mode='lines', name='Losses', line=dict(color='orange')),
        row=2, col=1
    )

    # Overloaded
    fig.add_trace(
        go.Bar(x=list(range(len(results))), y=results['num_overloaded'],
              name='Overloaded', marker_color='red'),
        row=3, col=1
    )

    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=50)
    )

    return fig


def create_comparison_chart(before: dict, after: dict) -> go.Figure:
    """Create before/after comparison chart."""
    metrics = ['Max Loading (%)', 'Losses (kW)', 'Overloaded Lines', 'Congestion Metric']
    before_vals = [
        before.get('max_loading', 0),
        before.get('losses_kw', 0),
        before.get('overloaded', 0),
        before.get('congestion', 0) / 100  # Scale down for display
    ]
    after_vals = [
        after.get('max_loading', 0),
        after.get('losses_kw', 0),
        after.get('overloaded', 0),
        after.get('congestion', 0) / 100
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Before', x=metrics, y=before_vals, marker_color='#ff6b6b'))
    fig.add_trace(go.Bar(name='After', x=metrics, y=after_vals, marker_color='#4ecdc4'))

    fig.update_layout(
        barmode='group',
        title="Before vs After Optimization",
        height=300,
        margin=dict(l=50, r=20, t=50, b=50)
    )

    return fig


def run_power_flow(net: pp.pandapowerNet) -> bool:
    """Run power flow and return success status."""
    try:
        pp.runpp(net, numba=True)
        return True
    except Exception as e:
        st.error(f"Power flow failed: {e}")
        return False


def page_simulation():
    """Main simulation page."""
    st.markdown('<h1 class="main-header">GRIZLI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Grid Reconfiguration Intelligence for Zero-Loss Integration</p>',
                unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Network Configuration")

    network_type = st.sidebar.selectbox(
        "Select Network",
        ["IEEE 33-bus", "IEEE 33-bus Congested", "Atacama Scenario", "Custom Synthetic"]
    )

    # Create network
    if 'net' not in st.session_state or st.sidebar.button("Reset Network"):
        with st.spinner("Creating network..."):
            if network_type == "IEEE 33-bus":
                st.session_state.net = create_ieee33_with_renewables()
            elif network_type == "IEEE 33-bus Congested":
                st.session_state.net = create_congested_ieee33()
            elif network_type == "Atacama Scenario":
                st.session_state.net = create_atacama_scenario()
            else:
                gen = NetworkGenerator(num_buses=30, network_type=NetworkType.MESHED, seed=42)
                st.session_state.net = gen.generate()

            st.session_state.optimized = False
            st.session_state.original_load = st.session_state.net.load.p_mw.copy()
            st.session_state.original_sgen = st.session_state.net.sgen.p_mw.copy() if len(st.session_state.net.sgen) > 0 else None

    net = st.session_state.net

    # Sliders for scenario adjustment
    st.sidebar.header("Scenario Adjustments")

    solar_factor = st.sidebar.slider("Solar Production", 0.0, 3.0, 1.0, 0.1)
    wind_factor = st.sidebar.slider("Wind Production", 0.0, 3.0, 1.0, 0.1)
    load_factor = st.sidebar.slider("Load Level", 0.5, 2.0, 1.0, 0.1)

    # Apply adjustments
    if len(net.sgen) > 0 and st.session_state.original_sgen is not None:
        for idx in net.sgen.index:
            gen_type = net.sgen.at[idx, 'type'] if 'type' in net.sgen.columns else 'PV'
            factor = solar_factor if gen_type == 'PV' else wind_factor
            net.sgen.at[idx, 'p_mw'] = st.session_state.original_sgen[idx] * factor

    net.load.p_mw = st.session_state.original_load * load_factor

    # Run power flow
    if not run_power_flow(net):
        st.stop()

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Network Topology")
        fig_network = create_network_graph(net)
        st.plotly_chart(fig_network, use_container_width=True)

    with col2:
        st.subheader("Network Status")

        max_loading = net.res_line.loading_percent.max()
        num_overloaded = (net.res_line.loading_percent > 100).sum()
        total_losses = net.res_line.pl_mw.sum() * 1000  # kW

        if num_overloaded > 0:
            st.markdown(f"""
            <div class="danger-box">
                <h3>CONGESTION DETECTED</h3>
                <p>Max Loading: {max_loading:.1f}%</p>
                <p>Overloaded Lines: {num_overloaded}</p>
                <p>Losses: {total_losses:.1f} kW</p>
            </div>
            """, unsafe_allow_html=True)
        elif max_loading > 80:
            st.markdown(f"""
            <div class="warning-box">
                <h3>HIGH LOADING</h3>
                <p>Max Loading: {max_loading:.1f}%</p>
                <p>Losses: {total_losses:.1f} kW</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <h3>NETWORK OK</h3>
                <p>Max Loading: {max_loading:.1f}%</p>
                <p>Losses: {total_losses:.1f} kW</p>
            </div>
            """, unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_loading_bar_chart(net), use_container_width=True)

    with col2:
        st.plotly_chart(create_voltage_profile(net), use_container_width=True)

    # Optimization section
    st.markdown("---")
    st.header("Optimization")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Optimize (FAST)", type="primary", use_container_width=True):
            with st.spinner("Optimizing topology..."):
                # Store before state
                before_max = net.res_line.loading_percent.max()
                before_losses = net.res_line.pl_mw.sum() * 1000
                before_overloaded = (net.res_line.loading_percent > 100).sum()
                before_congestion = np.sum(np.maximum(net.res_line.loading_percent.values - 100, 0) ** 2)

                # Run optimization
                result = optimize_topology_fast(net, max_iterations=5)

                # Apply result
                if result.lines_to_open or result.lines_to_close:
                    apply_optimization_result(net, result)
                    run_power_flow(net)
                    st.session_state.optimized = True
                    st.session_state.fast_result = result

                    # Store after state
                    st.session_state.before_metrics = {
                        'max_loading': before_max,
                        'losses_kw': before_losses,
                        'overloaded': before_overloaded,
                        'congestion': before_congestion
                    }
                    st.session_state.after_metrics = {
                        'max_loading': net.res_line.loading_percent.max(),
                        'losses_kw': net.res_line.pl_mw.sum() * 1000,
                        'overloaded': (net.res_line.loading_percent > 100).sum(),
                        'congestion': np.sum(np.maximum(net.res_line.loading_percent.values - 100, 0) ** 2)
                    }

                    st.success(f"Optimization complete in {result.computation_time_s*1000:.0f}ms")
                else:
                    st.info("No improvement found - network already optimal")

    with col2:
        if st.button("Reset Topology", use_container_width=True):
            # Reset all lines to original state
            for idx in net.line.index:
                net.line.at[idx, 'in_service'] = True
            run_power_flow(net)
            st.session_state.optimized = False
            st.rerun()

    # Show optimization results
    if st.session_state.get('optimized', False) and 'fast_result' in st.session_state:
        result = st.session_state.fast_result

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Switching Operations")
            if result.lines_to_open:
                st.write(f"**Lines Opened:** {result.lines_to_open}")
            if result.lines_to_close:
                st.write(f"**Lines Closed:** {result.lines_to_close}")
            st.write(f"**Total Switches:** {result.num_switches}")
            st.write(f"**Computation Time:** {result.computation_time_s*1000:.1f} ms")

        with col2:
            if 'before_metrics' in st.session_state and 'after_metrics' in st.session_state:
                fig = create_comparison_chart(
                    st.session_state.before_metrics,
                    st.session_state.after_metrics
                )
                st.plotly_chart(fig, use_container_width=True)

    # 24-hour simulation section
    st.markdown("---")
    st.header("24-Hour Simulation")

    if st.button("Run 24h Simulation"):
        with st.spinner("Running 24-hour simulation..."):
            config = TimeSeriesConfig(duration_hours=24, resolution_minutes=15, season=Season.SUMMER)
            scenario_gen = ScenarioGenerator(config, seed=42)
            scenario = scenario_gen.generate_network_scenario(net)

            ts_sim = TimeSeriesSimulator(net)
            results = ts_sim.run_simulation(scenario['sgen'], scenario['load'])

            st.session_state.ts_results = results

    if 'ts_results' in st.session_state:
        st.plotly_chart(create_24h_simulation_chart(st.session_state.ts_results), use_container_width=True)

        stats = st.session_state.ts_results
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Peak Loading", f"{stats['max_loading_percent'].max():.1f}%")
        col2.metric("Mean Loading", f"{stats['max_loading_percent'].mean():.1f}%")
        col3.metric("Overload Events", f"{(stats['num_overloaded'] > 0).sum()}")
        col4.metric("Total Losses", f"{stats['total_losses_mw'].sum():.1f} MWh")


def page_about():
    """About page."""
    st.markdown('<h1 class="main-header">About GRIZLI</h1>', unsafe_allow_html=True)

    st.markdown("""
    ## What is GRIZLI?

    **GRIZLI** (Grid Reconfiguration Intelligence for Zero-Loss Integration) is a smart grid
    optimization tool that dynamically reconfigures electrical networks to reduce congestion
    and losses.

    ### The Problem

    Modern power grids face increasing challenges from:
    - **Renewable energy variability** (solar, wind)
    - **Growing electricity demand**
    - **Aging infrastructure**
    - **Decentralized generation**

    These factors create **congestion** - situations where power lines operate near or above
    their thermal limits, potentially causing:
    - Equipment damage
    - Power outages
    - Increased losses
    - Safety hazards

    ### Our Solution: "Waze for Electricity"

    Just as Waze redirects cars to avoid traffic jams, GRIZLI redirects electricity to avoid
    congested lines by:

    1. **Monitoring** real-time network conditions
    2. **Detecting** congestion and potential overloads
    3. **Computing** optimal switching operations
    4. **Executing** topology changes (open/close switches)

    ### Technology Stack

    - **PandaPower**: Power flow simulation
    - **PuLP/CBC**: MILP optimization
    - **NetworkX**: Graph algorithms
    - **Streamlit**: Interactive dashboard
    - **Grid2Op**: Reinforcement learning (experimental)

    ### Results

    Typical improvements from GRIZLI optimization:
    - **30-70%** reduction in congestion metric
    - **5-20%** reduction in losses
    - **<1 second** computation time (greedy optimizer)

    ---

    **Author**: GRIZLI Team
    **Version**: 1.0.0
    **License**: MIT
    """)


def main():
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Simulation", "About"])

    if page == "Simulation":
        page_simulation()
    else:
        page_about()


if __name__ == "__main__":
    main()
