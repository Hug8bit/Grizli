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

        # Compute TVD and Voltage Stability Index D
        try:
            sim = PowerFlowSimulator(net)
            pf_result = sim.run_power_flow()
            tvd_value = pf_result.total_voltage_deviation
            if pf_result.voltage_stability is not None:
                d_margin = pf_result.voltage_stability.stability_margin
                weakest = pf_result.voltage_stability.weakest_bus
                v_stable = pf_result.voltage_stability.is_voltage_stable
            else:
                d_margin = None
                weakest = None
                v_stable = True
        except Exception:
            tvd_value = None
            d_margin = None
            weakest = None
            v_stable = True

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

        # Display TVD and Voltage Stability metrics
        st.markdown("##### Voltage Quality")
        tvd_col, d_col = st.columns(2)
        with tvd_col:
            if tvd_value is not None:
                tvd_color = "normal" if tvd_value < 0.5 else ("off" if tvd_value < 1.0 else "inverse")
                st.metric("TVD", f"{tvd_value:.4f} pu", delta_color=tvd_color)
            else:
                st.metric("TVD", "N/A")
        with d_col:
            if d_margin is not None:
                d_label = f"{d_margin:.3f}"
                st.metric("Stability D", d_label, help=f"Weakest bus: {weakest}")
                if not v_stable:
                    st.warning("Voltage collapse risk!")
            else:
                st.metric("Stability D", "N/A")

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


def page_fonctionnement():
    """Page Fonctionnement & Capacites - complete engineering documentation."""
    st.markdown('<h1 class="main-header">Fonctionnement & Capacites</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Documentation technique pour ingenieurs reseaux</p>',
                unsafe_allow_html=True)

    # ─── Section 1 : Vue d'ensemble ───────────────────────────────────────────
    st.header("1. Qu'est-ce que GRIZLI ?")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **GRIZLI** (**G**rid **R**econfiguration **I**ntelligence for **Z**ero-**L**oss
        **I**ntegration) est un outil d'optimisation de reseaux de distribution electrique.

        **Analogie** : c'est le **"Waze de l'electricite"**. De la meme facon que Waze
        redirige les voitures pour eviter les embouteillages, GRIZLI redirige les flux
        d'electricite en ouvrant/fermant des interrupteurs (switches) dans le reseau
        pour eviter les congestions sur les lignes.

        **Principe fondamental** : sans construire de nouvelle infrastructure, GRIZLI
        redistribue les flux de puissance en modifiant la topologie du reseau
        (quelles lignes sont connectees ou deconnectees) pour :
        - **Reduire la congestion** sur les lignes surchargees (30-70%)
        - **Minimiser les pertes** en ligne (5-20%)
        - **Ameliorer le profil de tension** sur tous les bus
        - **Integrer plus de renouvelables** sans risque de surcharge
        """)

    with col2:
        st.markdown("""
        <div class="success-box">
            <h4>Chiffres cles</h4>
            <p><strong>30-70%</strong> reduction congestion</p>
            <p><strong>5-20%</strong> reduction pertes</p>
            <p><strong>&lt;1 seconde</strong> optimisation rapide</p>
            <p><strong>~10 secondes</strong> optimisation MILP</p>
            <p><strong>33 bus</strong> reseau de reference</p>
        </div>
        """, unsafe_allow_html=True)

    # ─── Section 2 : Comment ca fonctionne ────────────────────────────────────
    st.markdown("---")
    st.header("2. Comment ca fonctionne")

    st.subheader("2.1 Pipeline de fonctionnement")
    st.markdown("""
    ```
    ┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
    │  1. MODELISATION │────>│  2. SIMULATION    │────>│  3. OPTIMISATION  │────>│  4. APPLICATION  │
    │                 │     │                  │     │                   │     │                  │
    │  Reseau IEEE    │     │  Power Flow AC   │     │  Greedy / MILP    │     │  Ouvrir/Fermer   │
    │  + Charges      │     │  Newton-Raphson   │     │  Topologie        │     │  les switches    │
    │  + Generateurs  │     │  + TVD + Indice D │     │  optimale         │     │  dans le reseau  │
    └─────────────────┘     └──────────────────┘     └───────────────────┘     └──────────────────┘
    ```
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **Etape 1 : Modelisation**
        - Reseau de distribution (IEEE 33-bus ou personnalise)
        - Charges (residentiel, industriel, commercial)
        - Generateurs distribues (PV, eolien)
        - Parametres des lignes (impedance, capacite thermique)
        """)
    with col2:
        st.markdown("""
        **Etape 2 : Simulation**
        - Power flow AC complet (Newton-Raphson)
        - Calcul des chargements par ligne (%)
        - Profil de tension par bus (p.u.)
        - Detection des violations (surcharges, sous-tensions)
        """)
    with col3:
        st.markdown("""
        **Etape 3 : Optimisation**
        - **Greedy** : echange de branches iteratif (<1s)
        - **MILP** : programmation lineaire mixte (~10s)
        - Minimise : congestion + pertes + nb de manoeuvres
        - Respecte : limites thermiques, radialite, connectivite
        """)
    with col4:
        st.markdown("""
        **Etape 4 : Application**
        - Liste des lignes a ouvrir / fermer
        - Validation par power flow apres manoeuvre
        - Comparaison avant/apres
        - Metriques de performance
        """)

    st.subheader("2.2 Algorithmes d'optimisation")

    tab_greedy, tab_milp, tab_rl = st.tabs(["Greedy (rapide)", "MILP (optimal)", "RL Agent (experimental)"])

    with tab_greedy:
        st.markdown("""
        ### Heuristique Greedy par echange de branches

        **Principe** : pour chaque ligne de couplage (tie-line, normalement ouverte),
        on la ferme, ce qui cree une boucle. On ouvre ensuite chaque ligne de cette
        boucle une par une et on garde la configuration qui reduit le plus la congestion.

        ```
        Pour chaque iteration (max 5-10) :
          Pour chaque tie-line T :
            1. Fermer T --> une boucle se forme
            2. Pour chaque ligne L de la boucle :
               a. Ouvrir L temporairement
               b. Executer un power flow
               c. Calculer la metrique de congestion
            3. Garder la meilleure ouverture
          Si amelioration > seuil : appliquer, sinon stop
        ```

        | Parametre | Valeur |
        |-----------|--------|
        | Temps typique | < 1 seconde |
        | Qualite solution | Bonne (heuristique locale) |
        | Garantie d'optimalite | Non |
        | Cas d'usage | Temps-reel, decisions rapides |
        """)

    with tab_milp:
        st.markdown("""
        ### MILP - Mixed Integer Linear Programming

        **Principe** : formulation mathematique exacte du probleme de reconfiguration
        comme un programme lineaire a variables mixtes (continues + binaires).

        **Variables de decision** :
        - `x_ij` binaire : etat de chaque switch (0=ouvert, 1=ferme)
        - `theta_i` continu : angle de tension au bus i
        - `P_ij` continu : puissance active sur la ligne ij

        **Fonction objectif** :
        ```
        min  w1 * sum(surcharge_ij^2) + w2 * sum(pertes_ij) + w3 * sum(manoeuvres)
        ```

        **Contraintes** :
        - Equilibre de puissance a chaque bus (KCL)
        - Limites thermiques des lignes
        - Radialite du reseau (optionnel)
        - Nombre max de manoeuvres

        | Parametre | Valeur |
        |-----------|--------|
        | Temps typique | ~10 secondes |
        | Qualite solution | Optimale (a l'ecart pres) |
        | Solveur | PuLP / CBC |
        | Cas d'usage | Planification, decisions strategiques |
        """)

    with tab_rl:
        st.markdown("""
        ### Agent RL (PPO) - Experimental

        **Principe** : un agent entraine par Proximal Policy Optimization (PPO)
        apprend a reconfigurer le reseau en interagissant avec un environnement Grid2Op.

        **Observation** : chargements des lignes, tensions, generation, charges
        **Action** : selection parmi les K meilleures topologies candidates
        **Recompense** : penalite pour surcharges + bonus pour stabilite

        | Parametre | Valeur |
        |-----------|--------|
        | Architecture | MLP [256, 256] |
        | Framework | Stable-Baselines3 + Grid2Op |
        | Statut | Experimental |
        """)

    # ─── Section 3 : Metriques avancees ───────────────────────────────────────
    st.markdown("---")
    st.header("3. Metriques et indicateurs")

    st.markdown("""
    GRIZLI calcule des indicateurs avances a chaque execution du power flow,
    au-dela du simple comptage de violations.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### Metrique de congestion
        ```
        C = sum( max(loading_i - 100, 0)^2 )
        ```
        Penalite quadratique : une ligne a 130% compte
        9x plus qu'une ligne a 110%.
        Utilisee comme objectif d'optimisation.
        """)

    with col2:
        st.markdown("""
        ### TVD (Total Voltage Deviation)
        ```
        TVD = sum( |V_i - 1.0| )  pour tous les bus
        ```
        Metrique **continue** de qualite de tension.
        Plus bas = meilleur (0 = tension parfaite partout).
        Fournit un gradient pour l'optimisation,
        contrairement au simple comptage binaire de violations.
        """)

    with col3:
        st.markdown("""
        ### Indice de stabilite D
        ```
        D = E / Delta_E
        ```
        - **-1 < D < 0** : stable
        - **D = -1** : seuil de destabilisation
        - **D < -1** : instable (effondrement)

        Identifie le **bus le plus faible** et quantifie
        la marge avant effondrement de tension.
        """)

    # Demo live des metriques
    st.subheader("3.1 Demonstration en direct")
    if st.button("Calculer les metriques sur IEEE 33-bus"):
        with st.spinner("Calcul en cours..."):
            import pandapower.networks as pn
            demo_net = pn.case33bw()
            pp.runpp(demo_net, numba=False)

            sim = PowerFlowSimulator(demo_net, algorithm='nr')
            result = sim.run_power_flow()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("TVD", f"{result.total_voltage_deviation:.4f} p.u.")
            col2.metric("Bus le plus faible",
                       f"Bus {result.voltage_stability.weakest_bus}")
            col3.metric("Marge de stabilite (D)",
                       f"{result.voltage_stability.stability_margin:.4f}")
            col4.metric("Stable ?",
                       "Oui" if result.voltage_stability.is_voltage_stable else "NON")

            # Afficher le profil de stabilite
            vs = result.voltage_stability
            fig_stab = go.Figure()
            fig_stab.add_trace(go.Bar(
                x=[f"Bus {b}" for b in vs.bus_indices],
                y=vs.stability_index_d,
                marker_color=['red' if d < -0.85 else 'orange' if d < -0.7 else 'green'
                             for d in vs.stability_index_d],
                name='Indice D'
            ))
            fig_stab.add_hline(y=-1.0, line_dash="dash", line_color="red",
                              annotation_text="Seuil de collapse (D = -1)")
            fig_stab.update_layout(
                title="Indice de stabilite D par bus (plus proche de -1 = plus vulnerable)",
                yaxis_title="Indice D",
                height=350
            )
            st.plotly_chart(fig_stab, use_container_width=True)

    # ─── Section 4 : Modele solaire ───────────────────────────────────────────
    st.markdown("---")
    st.header("4. Modele de generation solaire")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        Le profil solaire utilise une **distribution Beta** pour modeliser
        la variabilite de l'irradiance, au lieu du bruit gaussien traditionnel.

        **Pourquoi Beta plutot que Gaussien ?**
        - **Bornee [0, 1]** : l'irradiance ne peut pas etre negative
        - **Asymetrique** : correspond aux observations meteorologiques reelles
        - **Parametrable** : via `cloud_variability` (0 = ciel clair, 0.5 = tres nuageux)

        **Parametrisation** :
        ```
        mu = 1.0 - cloud_variability
        alpha = mu * kappa
        beta  = (1 - mu) * kappa
        irradiance ~ Beta(alpha, beta)
        ```

        Reference : Hachemi et al., Energy Reports 12 (2024), Eqs. 2-4.
        """)

    with col2:
        # Generate and display solar profiles
        from src.simulation.time_series import ScenarioGenerator, TimeSeriesConfig, Season
        config_demo = TimeSeriesConfig(duration_hours=24, resolution_minutes=15, season=Season.SUMMER)

        fig_solar = go.Figure()
        for cv, color, name in [(0.1, 'gold', 'Ciel clair (0.1)'),
                                (0.2, 'orange', 'Normal (0.2)'),
                                (0.4, 'gray', 'Tres nuageux (0.4)')]:
            config_cv = TimeSeriesConfig(
                duration_hours=24, resolution_minutes=15,
                season=Season.SUMMER, cloud_variability=cv
            )
            gen = ScenarioGenerator(config_cv, seed=42)
            p = gen.generate_solar_profile(capacity_mw=10.0)
            hours = np.arange(len(p)) * 15 / 60
            fig_solar.add_trace(go.Scatter(
                x=hours, y=p.values,
                mode='lines', name=name,
                line=dict(color=color)
            ))
        fig_solar.update_layout(
            title="Profils solaires (Beta) - 10 MW, Ete",
            xaxis_title="Heure",
            yaxis_title="Production (MW)",
            height=350
        )
        st.plotly_chart(fig_solar, use_container_width=True)

    # ─── Section 5 : Procedure d'utilisation ──────────────────────────────────
    st.markdown("---")
    st.header("5. Procedure d'utilisation")

    st.markdown("""
    ### 5.1 Utilisation via le Dashboard (onglet Simulation)

    | Etape | Action | Description |
    |-------|--------|-------------|
    | 1 | **Choisir un reseau** | Menu lateral : IEEE 33-bus, Congested, Atacama, ou Custom |
    | 2 | **Ajuster le scenario** | Curseurs : production solaire/eolienne, niveau de charge |
    | 3 | **Observer l'etat** | Topologie, statut reseau (OK / WARNING / CONGESTION) |
    | 4 | **Lancer l'optimisation** | Bouton "Optimize (FAST)" -- resultat en <1s |
    | 5 | **Comparer avant/apres** | Graphique comparatif automatique |
    | 6 | **Simuler sur 24h** | Bouton "Run 24h Simulation" -- profils journaliers |

    ### 5.2 Utilisation par le code Python

    ```python
    import pandapower as pp
    from src.core.ieee_networks import create_congested_ieee33
    from src.simulation.power_flow import PowerFlowSimulator
    from src.optimization.fast_optimizer import optimize_topology_fast, apply_optimization_result

    # 1. Creer un reseau congestionne
    net = create_congested_ieee33(load_increase_percent=60)

    # 2. Analyser l'etat initial
    sim = PowerFlowSimulator(net)
    result = sim.run_power_flow()
    print(f"TVD = {result.total_voltage_deviation:.4f}")
    print(f"Bus faible = {result.voltage_stability.weakest_bus}, D = {result.voltage_stability.stability_margin:.4f}")

    # 3. Optimiser la topologie
    optim = optimize_topology_fast(net, max_iterations=5)
    apply_optimization_result(net, optim)
    print(f"Reduction congestion : {optim.congestion_reduction_percent:.1f}%")

    # 4. Verifier l'amelioration
    result_after = sim.run_power_flow()
    print(f"TVD apres = {result_after.total_voltage_deviation:.4f}")
    ```

    ### 5.3 Simulation 24 heures

    ```python
    from src.simulation.time_series import TimeSeriesSimulator, ScenarioGenerator, TimeSeriesConfig

    config = TimeSeriesConfig(duration_hours=24, season=Season.SUMMER)
    gen = ScenarioGenerator(config, seed=42)
    scenario = gen.generate_network_scenario(net)

    sim_ts = TimeSeriesSimulator(net)
    results = sim_ts.run_simulation(scenario['sgen'], scenario['load'])
    stats = sim_ts.get_congestion_statistics()

    print(f"TVD moyen : {stats['tvd_mean']:.4f}")
    print(f"TVD max   : {stats['tvd_max']:.4f}")
    print(f"Taux de surcharge : {stats['overload_rate']*100:.1f}%")
    ```
    """)

    # ─── Section 6 : Secteurs d'application ───────────────────────────────────
    st.markdown("---")
    st.header("6. Secteurs d'application et efficacite")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Ou GRIZLI est efficace

        | Secteur | Applicabilite | Pourquoi |
        |---------|:---:|----------|
        | **Distribution MT (1-35 kV)** | **Eleve** | Coeur de cible : reseaux radiaux avec switches |
        | **Microgrids / ilotage** | **Eleve** | Reseaux petits/moyens, forte penetration ENR |
        | **Zones a forte penetration PV** | **Eleve** | Congestion inverse (production > consommation) |
        | **Reseaux ruraux etendus** | **Moyen** | Longues lignes = pertes elevees, bon potentiel |
        | **Distribution BT (<1 kV)** | **Moyen** | Moins de switches manoeuvrable en pratique |
        | **Transport HT (>63 kV)** | **Faible** | Topologie maillee fixe, autres contraintes |

        ### Cas d'usage principaux

        1. **Gestionnaire de reseau (GRD/DSO)** : outil d'aide a la decision en salle
           de conduite pour optimiser la topologie en temps reel
        2. **Bureau d'etudes** : planification de l'integration d'ENR -- tester
           differents scenarios de penetration sans etude complete
        3. **R&D / These** : plateforme experimentale pour tester de nouveaux
           algorithmes d'optimisation ou de reinforcement learning
        4. **Formation** : outil pedagogique pour comprendre l'impact de la
           reconfiguration sur les flux de puissance
        """)

    with col2:
        st.markdown("""
        ### Contextes problematiques resolus

        **Congestion par surproduction renouvelable**
        > Un parc PV de 5 MW injecte en milieu de journee sur un feeder
        > dimensionne pour 2 MW. GRIZLI redirige le surplus vers un feeder
        > adjacent moins charge en fermant un tie-switch.

        **Congestion par pointe de charge**
        > A 19h, la charge residentielle depasse 120% de la capacite
        > du feeder principal. GRIZLI ouvre une ligne et reroute via
        > un chemin alternatif, ramenant le chargement a 85%.

        **Pertes elevees sur reseau etendu**
        > Un reseau rural de 50 km a 8% de pertes. En reconfigurant
        > la topologie, GRIZLI reduit les distances electriques et
        > ramene les pertes a 5.5%.

        **Effondrement de tension en bout de ligne**
        > Le bus 18 est a 0.91 p.u. (sous le seuil 0.95). L'indice D
        > montre D = -0.92 (proche du collapse). Apres reconfiguration,
        > V = 0.97 p.u. et D = -0.65 (marge retrouvee).
        """)

    # ─── Section 7 : Architecture technique ───────────────────────────────────
    st.markdown("---")
    st.header("7. Architecture technique")

    st.markdown("""
    ```
    Grizli/
    ├── app.py                          # Dashboard Streamlit (cette interface)
    ├── demo.py                         # Script de demonstration complet
    ├── train.py                        # Entrainement agent RL (PPO)
    ├── test_optimization.py            # Tests de validation
    │
    ├── src/
    │   ├── core/                       # Generation de reseaux
    │   │   ├── network_generator.py    #   Reseaux synthetiques (radial, maille, anneau)
    │   │   ├── network_topology.py     #   Structures de graphe NetworkX
    │   │   └── ieee_networks.py        #   Reseaux IEEE 14/33 bus avec scenarios
    │   │
    │   ├── simulation/                 # Simulation electrique
    │   │   ├── power_flow.py           #   Power flow AC + TVD + Indice D + Congestion
    │   │   └── time_series.py          #   Simulation 24h avec profils Beta solaire
    │   │
    │   ├── optimization/               # Algorithmes d'optimisation
    │   │   ├── reconfiguration.py      #   Optimiseur MILP (PuLP/CBC)
    │   │   ├── fast_optimizer.py       #   Heuristique greedy rapide
    │   │   └── benchmark.py            #   Comparaison avant/apres
    │   │
    │   ├── rl_agent/                   # Reinforcement Learning (optionnel)
    │   │   ├── ppo_agent.py            #   Agent PPO (Stable-Baselines3)
    │   │   ├── grid2op_env_wrapper.py  #   Wrapper Gymnasium
    │   │   ├── action_space.py         #   Selection Top-K actions
    │   │   └── ai_controller.py        #   Integration dashboard
    │   │
    │   ├── grid2op_env/                # Environnement Grid2Op
    │   │   ├── grid2op_init.py         #   Creation environnement
    │   │   └── scenario_player.py      #   Rejeu de chroniques
    │   │
    │   └── visualization/              # Visualisation
    │       └── network_plot.py         #   Graphiques Matplotlib
    │
    └── cook/                           # Recherche et analyse
        └── ANALYSE_12_PAPERS_SYNTHESE.md  # Synthese de 12 papiers de recherche
    ```
    """)

    st.subheader("7.1 Stack technologique")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Simulation**
        - PandaPower >= 2.13
        - Newton-Raphson AC
        - Backward/Forward Sweep
        - Matrice d'admittance Ybus
        """)
    with col2:
        st.markdown("""
        **Optimisation**
        - PuLP >= 2.7 (MILP)
        - CBC solver
        - NetworkX >= 3.0 (graphes)
        - NumPy / Pandas
        """)
    with col3:
        st.markdown("""
        **RL / ML (optionnel)**
        - Grid2Op >= 1.9
        - Stable-Baselines3 >= 2.0
        - PyTorch >= 2.0
        - LightSim2Grid >= 0.7
        """)

    # ─── Section 8 : Limites connues ──────────────────────────────────────────
    st.markdown("---")
    st.header("8. Limites et evolutions prevues")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Limites actuelles
        - Approximation **DC** dans le MILP (pas de tension dans l'optimisation)
        - Pas de modele de **stockage / batteries**
        - Pas de **metriques economiques** (cout, emissions CO2)
        - Optimisation **deterministe** (pas d'incertitude)
        - Agent RL non transferable entre topologies
        - Teste sur reseaux **jusqu'a 33 bus** (scalabilite a valider)
        """)
    with col2:
        st.markdown("""
        ### Evolutions prevues (roadmap recherche)
        - **SOCP** au lieu de DC dans le MILP (tensions + reactif)
        - **DNN Surrogate** pour power flow en <10ms
        - **Stockage BESS** avec cout de degradation exponentiel
        - **Multi-objectif Pareto** (NNC + TOPSIS)
        - **Transfer Learning** (LSTM+TMMD) pour le PPO cross-topologie
        - **Optimisation stochastique** (LHS + scenarios)
        - **Dashboard Digital Twin 3D**

        *Source : Analyse de 12 papiers Energy Reports 2024*
        """)

    st.markdown("---")
    st.markdown("""
    **GRIZLI** v1.0 | Licence MIT |
    Documentation basee sur l'analyse de 12 publications Energy Reports 2024
    """)


def page_about():
    """About page - short version."""
    st.markdown('<h1 class="main-header">About GRIZLI</h1>', unsafe_allow_html=True)

    st.markdown("""
    **GRIZLI** (Grid Reconfiguration Intelligence for Zero-Loss Integration) est un
    outil d'optimisation de reseaux electriques de distribution.

    Pour la documentation technique complete, consultez l'onglet **Fonctionnement**.

    ### En bref

    | Aspect | Detail |
    |--------|--------|
    | **Objectif** | Reconfiguration topologique pour minimiser congestion et pertes |
    | **Methode** | Greedy (<1s) + MILP (~10s) + RL (experimental) |
    | **Reseaux** | Distribution MT, IEEE 14/33 bus |
    | **Metriques** | Congestion, TVD, Indice D, Pertes |
    | **Stack** | PandaPower, PuLP, NetworkX, Streamlit |
    | **Version** | 1.0.0 |
    | **Licence** | MIT |
    """)


def main():
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Simulation", "Fonctionnement", "About"])

    if page == "Simulation":
        page_simulation()
    elif page == "Fonctionnement":
        page_fonctionnement()
    else:
        page_about()


if __name__ == "__main__":
    main()
