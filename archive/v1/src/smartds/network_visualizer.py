"""
Network visualization for SMART-DS and PandaPower networks.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
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

from .opendss_parser import DSSCircuit


class NetworkVisualizer:
    """
    Visualizer for electrical networks.

    Supports both DSSCircuit (from OpenDSS parsing) and
    PandaPower networks.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.

        Args:
            figsize: Figure size (width, height)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required. Install with: pip install matplotlib")
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required. Install with: pip install networkx")

        self.figsize = figsize
        self._graph: Optional[nx.Graph] = None

    def visualize_dss_circuit(
        self,
        circuit: DSSCircuit,
        show_loads: bool = True,
        show_generators: bool = True,
        highlight_overloaded: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize a DSSCircuit.

        Args:
            circuit: DSSCircuit to visualize
            show_loads: Show load nodes
            show_generators: Show generator nodes
            highlight_overloaded: List of overloaded line names
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Build NetworkX graph
        G = nx.Graph()

        # Add buses as nodes
        for name, bus in circuit.buses.items():
            G.add_node(name, node_type='bus', x=bus.x, y=bus.y)

        # Add lines as edges
        for name, line in circuit.lines.items():
            if line.bus1 and line.bus2 and line.enabled:
                G.add_edge(
                    line.bus1, line.bus2,
                    name=name,
                    line_type='line'
                )

        # Add transformer edges
        for name, xfmr in circuit.transformers.items():
            if len(xfmr.buses) >= 2 and xfmr.enabled:
                G.add_edge(
                    xfmr.buses[0], xfmr.buses[1],
                    name=name,
                    line_type='transformer'
                )

        self._graph = G

        # Create layout
        if all(circuit.buses[n].x != 0 or circuit.buses[n].y != 0
               for n in G.nodes() if n in circuit.buses):
            # Use geographic coordinates
            pos = {n: (circuit.buses[n].x, circuit.buses[n].y)
                   for n in G.nodes() if n in circuit.buses}
        else:
            # Use spring layout
            pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(len(G.nodes())))

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw edges
        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            if highlight_overloaded and data.get('name') in highlight_overloaded:
                edge_colors.append('red')
                edge_widths.append(3)
            elif data.get('line_type') == 'transformer':
                edge_colors.append('purple')
                edge_widths.append(2)
            else:
                edge_colors.append('gray')
                edge_widths.append(1)

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7
        )

        # Classify nodes
        load_buses = set(load.bus1 for load in circuit.loads.values())
        gen_buses = set(gen.bus1 for gen in circuit.generators.values())
        source_bus = {circuit.source_bus}

        # Draw nodes by type
        regular_nodes = [n for n in G.nodes()
                        if n not in load_buses and n not in gen_buses and n not in source_bus]
        nx.draw_networkx_nodes(
            G, pos, nodelist=regular_nodes, ax=ax,
            node_color='lightblue', node_size=50, alpha=0.8
        )

        if show_loads:
            load_nodes = [n for n in G.nodes() if n in load_buses]
            nx.draw_networkx_nodes(
                G, pos, nodelist=load_nodes, ax=ax,
                node_color='orange', node_size=100, alpha=0.8,
                node_shape='s'  # Square for loads
            )

        if show_generators:
            gen_nodes = [n for n in G.nodes() if n in gen_buses]
            nx.draw_networkx_nodes(
                G, pos, nodelist=gen_nodes, ax=ax,
                node_color='green', node_size=150, alpha=0.8,
                node_shape='^'  # Triangle for generators
            )

        # Draw source bus
        source_nodes = [n for n in G.nodes() if n in source_bus]
        nx.draw_networkx_nodes(
            G, pos, nodelist=source_nodes, ax=ax,
            node_color='red', node_size=200, alpha=0.8
        )

        # Legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Bus'),
            mpatches.Patch(color='red', label='Source'),
        ]
        if show_loads:
            legend_elements.append(mpatches.Patch(color='orange', label='Load'))
        if show_generators:
            legend_elements.append(mpatches.Patch(color='green', label='Generator'))
        if highlight_overloaded:
            legend_elements.append(mpatches.Patch(color='red', label='Overloaded'))

        ax.legend(handles=legend_elements, loc='upper right')

        # Title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{circuit.name}\n"
                        f"({circuit.n_buses} buses, {circuit.n_lines} lines, "
                        f"{circuit.n_loads} loads, {circuit.n_generators} gens)")

        ax.axis('off')
        plt.tight_layout()

        return fig

    def visualize_pandapower(
        self,
        net: "pp.pandapowerNet",
        show_loading: bool = True,
        loading_threshold: float = 80.0,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize a PandaPower network.

        Args:
            net: PandaPower network
            show_loading: Color lines by loading
            loading_threshold: Threshold for highlighting
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if not PANDAPOWER_AVAILABLE:
            raise ImportError("PandaPower required")

        # Build graph
        G = nx.Graph()

        # Add buses
        for idx in net.bus.index:
            G.add_node(idx, node_type='bus')

        # Add lines
        for idx in net.line.index:
            if net.line.at[idx, 'in_service']:
                G.add_edge(
                    int(net.line.at[idx, 'from_bus']),
                    int(net.line.at[idx, 'to_bus']),
                    idx=idx,
                    line_type='line'
                )

        self._graph = G

        # Get positions
        if hasattr(net, 'bus_geodata') and len(net.bus_geodata) > 0:
            pos = {idx: (net.bus_geodata.at[idx, 'x'], net.bus_geodata.at[idx, 'y'])
                   for idx in net.bus.index if idx in net.bus_geodata.index}
        else:
            pos = nx.spring_layout(G, seed=42)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Get loading if available
        loading = {}
        if show_loading and 'res_line' in net and len(net.res_line) > 0:
            for idx in net.line.index:
                if idx in net.res_line.index:
                    loading[idx] = net.res_line.at[idx, 'loading_percent']

        # Draw edges with loading colors
        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            idx = data.get('idx')
            if idx is not None and idx in loading:
                load_pct = loading[idx]
                if load_pct > 100:
                    edge_colors.append('red')
                    edge_widths.append(3)
                elif load_pct > loading_threshold:
                    edge_colors.append('orange')
                    edge_widths.append(2)
                elif load_pct > 50:
                    edge_colors.append('yellow')
                    edge_widths.append(1.5)
                else:
                    edge_colors.append('green')
                    edge_widths.append(1)
            else:
                edge_colors.append('gray')
                edge_widths.append(1)

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7
        )

        # Identify special buses
        load_buses = set(net.load.bus.values) if len(net.load) > 0 else set()
        gen_buses = set(net.sgen.bus.values) if len(net.sgen) > 0 else set()
        slack_buses = set(net.ext_grid.bus.values) if len(net.ext_grid) > 0 else set()

        # Draw nodes
        regular = [n for n in G.nodes()
                  if n not in load_buses and n not in gen_buses and n not in slack_buses]
        nx.draw_networkx_nodes(G, pos, nodelist=regular, ax=ax,
                              node_color='lightblue', node_size=50)

        if load_buses:
            nx.draw_networkx_nodes(G, pos, nodelist=list(load_buses), ax=ax,
                                  node_color='orange', node_size=80, node_shape='s')

        if gen_buses:
            nx.draw_networkx_nodes(G, pos, nodelist=list(gen_buses), ax=ax,
                                  node_color='green', node_size=100, node_shape='^')

        if slack_buses:
            nx.draw_networkx_nodes(G, pos, nodelist=list(slack_buses), ax=ax,
                                  node_color='red', node_size=150)

        # Legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Bus'),
            mpatches.Patch(color='orange', label='Load'),
            mpatches.Patch(color='green', label='Generator'),
            mpatches.Patch(color='red', label='Slack'),
        ]
        if show_loading:
            legend_elements.extend([
                mpatches.Patch(color='green', label='<50% loading'),
                mpatches.Patch(color='yellow', label='50-80% loading'),
                mpatches.Patch(color='orange', label='80-100% loading'),
                mpatches.Patch(color='red', label='>100% loading'),
            ])

        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{net.name if hasattr(net, 'name') else 'Network'}\n"
                        f"({len(net.bus)} buses, {len(net.line)} lines)")

        ax.axis('off')
        plt.tight_layout()

        return fig

    def get_graph(self) -> Optional[nx.Graph]:
        """Get the underlying NetworkX graph."""
        return self._graph


def quick_plot(net: Any, **kwargs) -> plt.Figure:
    """
    Quick visualization of any supported network.

    Args:
        net: DSSCircuit or PandaPower network
        **kwargs: Additional arguments for visualize methods

    Returns:
        Matplotlib figure
    """
    viz = NetworkVisualizer()

    if isinstance(net, DSSCircuit):
        return viz.visualize_dss_circuit(net, **kwargs)
    elif PANDAPOWER_AVAILABLE and isinstance(net, pp.pandapowerNet):
        return viz.visualize_pandapower(net, **kwargs)
    else:
        raise TypeError(f"Unsupported network type: {type(net)}")
