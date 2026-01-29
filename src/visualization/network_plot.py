"""
Matplotlib-based visualization for power networks.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import pandapower as pp
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False


def _check_dependencies():
    """Check required dependencies."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required. Install with: pip install matplotlib")
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX required. Install with: pip install networkx")


def plot_network(
    net: "pp.pandapowerNet",
    figsize: Tuple[int, int] = (12, 8),
    show_labels: bool = False,
    highlight_lines: Optional[List[int]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot network topology.

    Args:
        net: PandaPower network
        figsize: Figure size
        show_labels: Show bus labels
        highlight_lines: List of line indices to highlight
        title: Plot title
        ax: Existing axes to plot on

    Returns:
        Matplotlib figure
    """
    _check_dependencies()
    if not PANDAPOWER_AVAILABLE:
        raise ImportError("PandaPower required")

    # Build graph
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

    # Get positions
    if hasattr(net, 'bus_geodata') and len(net.bus_geodata) > 0:
        pos = {int(idx): (net.bus_geodata.at[idx, 'x'], net.bus_geodata.at[idx, 'y'])
               for idx in net.bus.index if idx in net.bus_geodata.index}
        # Fill missing with spring layout
        missing = [n for n in G.nodes() if n not in pos]
        if missing:
            pos_spring = nx.spring_layout(G.subgraph(missing), seed=42)
            pos.update(pos_spring)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Prepare edge colors
    edge_colors = []
    edge_widths = []
    highlight_set = set(highlight_lines) if highlight_lines else set()

    for u, v, data in G.edges(data=True):
        if data.get('idx') in highlight_set:
            edge_colors.append('red')
            edge_widths.append(3)
        else:
            edge_colors.append('gray')
            edge_widths.append(1)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                          width=edge_widths, alpha=0.7)

    # Node types
    load_buses = set(net.load.bus.values) if len(net.load) > 0 else set()
    gen_buses = set(net.sgen.bus.values) if len(net.sgen) > 0 else set()
    slack_buses = set(net.ext_grid.bus.values) if len(net.ext_grid) > 0 else set()

    regular = [n for n in G.nodes()
              if n not in load_buses and n not in gen_buses and n not in slack_buses]

    nx.draw_networkx_nodes(G, pos, nodelist=regular, ax=ax,
                          node_color='lightblue', node_size=100)

    if load_buses:
        load_list = [n for n in G.nodes() if n in load_buses]
        nx.draw_networkx_nodes(G, pos, nodelist=load_list, ax=ax,
                              node_color='orange', node_size=150, node_shape='s')

    if gen_buses:
        gen_list = [n for n in G.nodes() if n in gen_buses]
        nx.draw_networkx_nodes(G, pos, nodelist=gen_list, ax=ax,
                              node_color='green', node_size=200, node_shape='^')

    if slack_buses:
        slack_list = [n for n in G.nodes() if n in slack_buses]
        nx.draw_networkx_nodes(G, pos, nodelist=slack_list, ax=ax,
                              node_color='red', node_size=250)

    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    # Title
    if title:
        ax.set_title(title)
    else:
        name = net.name if hasattr(net, 'name') and net.name else 'Network'
        ax.set_title(f"{name} ({len(net.bus)} buses, {len(net.line)} lines)")

    ax.axis('off')
    plt.tight_layout()

    return fig


def plot_loading_bars(
    net: "pp.pandapowerNet",
    figsize: Tuple[int, int] = (14, 6),
    threshold: float = 80.0,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot line loading as bar chart.

    Args:
        net: PandaPower network (with results)
        figsize: Figure size
        threshold: Warning threshold (%)
        title: Plot title
        ax: Existing axes

    Returns:
        Matplotlib figure
    """
    _check_dependencies()
    if not PANDAPOWER_AVAILABLE:
        raise ImportError("PandaPower required")

    if 'res_line' not in net or len(net.res_line) == 0:
        raise ValueError("No power flow results. Run pp.runpp() first.")

    loading = net.res_line.loading_percent.values
    n_lines = len(loading)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Colors based on loading
    colors = []
    for load in loading:
        if load > 100:
            colors.append('red')
        elif load > threshold:
            colors.append('orange')
        elif load > 50:
            colors.append('yellow')
        else:
            colors.append('green')

    # Bar plot
    x = np.arange(n_lines)
    bars = ax.bar(x, loading, color=colors, edgecolor='black', linewidth=0.5)

    # Threshold lines
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Limit (100%)')
    ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=1, label=f'Warning ({threshold}%)')

    # Labels
    ax.set_xlabel('Line Index')
    ax.set_ylabel('Loading (%)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Line Loading (max: {loading.max():.1f}%)')

    # X-axis
    if n_lines <= 40:
        ax.set_xticks(x)
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.set_xticks(x[::5])

    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, n_lines - 0.5)
    ax.set_ylim(0, max(120, loading.max() * 1.1))

    plt.tight_layout()
    return fig


def plot_voltage_profile(
    net: "pp.pandapowerNet",
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot bus voltage profile.

    Args:
        net: PandaPower network (with results)
        figsize: Figure size
        title: Plot title
        ax: Existing axes

    Returns:
        Matplotlib figure
    """
    _check_dependencies()
    if not PANDAPOWER_AVAILABLE:
        raise ImportError("PandaPower required")

    if 'res_bus' not in net or len(net.res_bus) == 0:
        raise ValueError("No power flow results. Run pp.runpp() first.")

    voltage = net.res_bus.vm_pu.values
    n_buses = len(voltage)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Colors based on voltage
    colors = []
    for v in voltage:
        if v < 0.95 or v > 1.05:
            colors.append('red')
        elif v < 0.97 or v > 1.03:
            colors.append('orange')
        else:
            colors.append('green')

    # Plot
    x = np.arange(n_buses)
    ax.bar(x, voltage, color=colors, edgecolor='black', linewidth=0.5)

    # Limit lines
    ax.axhline(y=1.05, color='red', linestyle='--', linewidth=1.5, label='Upper limit (1.05)')
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='Lower limit (0.95)')
    ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.5, label='Nominal (1.0)')

    ax.set_xlabel('Bus Index')
    ax.set_ylabel('Voltage (p.u.)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Voltage Profile (range: {voltage.min():.3f} - {voltage.max():.3f} p.u.)')

    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, n_buses - 0.5)
    ax.set_ylim(0.9, 1.1)

    plt.tight_layout()
    return fig


def plot_comparison(
    before: Dict[str, float],
    after: Dict[str, float],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Before vs After Optimization"
) -> plt.Figure:
    """
    Plot before/after comparison bar chart.

    Args:
        before: Dictionary of before metrics
        after: Dictionary of after metrics
        metrics: List of metrics to show (default: all)
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    _check_dependencies()

    if metrics is None:
        metrics = list(before.keys())

    metrics = [m for m in metrics if m in before and m in after]

    before_vals = [before[m] for m in metrics]
    after_vals = [after[m] for m in metrics]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, before_vals, width, label='Before', color='#ff6b6b')
    bars2 = ax.bar(x + width/2, after_vals, width, label='After', color='#4ecdc4')

    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()
    return fig


def plot_time_series(
    results: Any,  # DataFrame or dict with time series
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    title: str = "24-Hour Simulation"
) -> plt.Figure:
    """
    Plot time series results.

    Args:
        results: DataFrame with time series data
        metrics: Metrics to plot
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    _check_dependencies()

    import pandas as pd

    if isinstance(results, dict):
        results = pd.DataFrame(results)

    if metrics is None:
        metrics = ['max_loading_percent', 'total_losses_mw', 'num_overloaded']
        metrics = [m for m in metrics if m in results.columns]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(figsize[0], figsize[1] * n_metrics / 3),
                             sharex=True)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(results[metric], linewidth=1.5)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

        # Add threshold line for loading
        if 'loading' in metric.lower():
            ax.axhline(y=100, color='red', linestyle='--', alpha=0.7)

    axes[-1].set_xlabel('Time Step')
    axes[0].set_title(title)

    plt.tight_layout()
    return fig


def create_loading_colormap():
    """Create custom colormap for line loading."""
    colors = ['green', 'yellow', 'orange', 'red']
    return LinearSegmentedColormap.from_list('loading', colors)
