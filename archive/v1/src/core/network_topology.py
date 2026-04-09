"""
Network topology management using NetworkX.
Handles graph representation of electrical networks.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import networkx as nx
import numpy as np


class NodeType(Enum):
    """Type of network node."""
    SLACK = "slack"      # Reference bus (voltage source)
    PV = "pv"            # Generator bus (P and V specified)
    PQ = "pq"            # Load bus (P and Q specified)
    JUNCTION = "junction"  # Connection point only


class LineStatus(Enum):
    """Status of a network line/switch."""
    CLOSED = "closed"
    OPEN = "open"


@dataclass
class NodeConfig:
    """Configuration for a network node."""
    bus_id: int
    node_type: NodeType
    voltage_kv: float = 20.0
    p_mw: float = 0.0
    q_mvar: float = 0.0
    name: str = ""
    x: float = 0.0
    y: float = 0.0

    def __post_init__(self):
        if not self.name:
            self.name = f"bus_{self.bus_id}"


@dataclass
class LineConfig:
    """Configuration for a network line."""
    from_bus: int
    to_bus: int
    r_ohm_per_km: float = 0.1
    x_ohm_per_km: float = 0.1
    length_km: float = 1.0
    max_i_ka: float = 0.5
    name: str = ""
    is_switch: bool = False
    status: LineStatus = LineStatus.CLOSED

    def __post_init__(self):
        if not self.name:
            self.name = f"line_{self.from_bus}_{self.to_bus}"

    @property
    def r_ohm(self) -> float:
        """Total resistance in ohms."""
        return self.r_ohm_per_km * self.length_km

    @property
    def x_ohm(self) -> float:
        """Total reactance in ohms."""
        return self.x_ohm_per_km * self.length_km


class NetworkTopology:
    """
    Manages the topological structure of an electrical network.
    Uses NetworkX for graph operations.
    """

    def __init__(self):
        """Initialize empty network topology."""
        self._graph = nx.Graph()
        self._nodes: Dict[int, NodeConfig] = {}
        self._lines: Dict[Tuple[int, int], LineConfig] = {}
        self._slack_bus: Optional[int] = None

    def add_node(self, config: NodeConfig) -> None:
        """Add a node to the network."""
        self._nodes[config.bus_id] = config
        self._graph.add_node(
            config.bus_id,
            node_type=config.node_type,
            voltage_kv=config.voltage_kv,
            p_mw=config.p_mw,
            q_mvar=config.q_mvar,
            name=config.name,
            pos=(config.x, config.y)
        )

        if config.node_type == NodeType.SLACK:
            self._slack_bus = config.bus_id

    def add_line(self, config: LineConfig) -> None:
        """Add a line to the network."""
        key = (min(config.from_bus, config.to_bus),
               max(config.from_bus, config.to_bus))
        self._lines[key] = config

        if config.status == LineStatus.CLOSED:
            self._graph.add_edge(
                config.from_bus,
                config.to_bus,
                r_ohm=config.r_ohm,
                x_ohm=config.x_ohm,
                max_i_ka=config.max_i_ka,
                name=config.name,
                is_switch=config.is_switch
            )

    def set_line_status(self, from_bus: int, to_bus: int, status: LineStatus) -> bool:
        """
        Set the status of a line (open/closed).

        Returns True if the operation was successful.
        """
        key = (min(from_bus, to_bus), max(from_bus, to_bus))

        if key not in self._lines:
            return False

        self._lines[key].status = status

        if status == LineStatus.CLOSED:
            config = self._lines[key]
            self._graph.add_edge(
                from_bus, to_bus,
                r_ohm=config.r_ohm,
                x_ohm=config.x_ohm,
                max_i_ka=config.max_i_ka,
                name=config.name,
                is_switch=config.is_switch
            )
        else:
            if self._graph.has_edge(from_bus, to_bus):
                self._graph.remove_edge(from_bus, to_bus)

        return True

    def get_line_status(self, from_bus: int, to_bus: int) -> Optional[LineStatus]:
        """Get the status of a line."""
        key = (min(from_bus, to_bus), max(from_bus, to_bus))
        if key in self._lines:
            return self._lines[key].status
        return None

    def is_connected(self) -> bool:
        """Check if the network is fully connected."""
        if len(self._graph.nodes) == 0:
            return True
        return nx.is_connected(self._graph)

    def is_radial(self) -> bool:
        """Check if the network has a radial (tree) structure."""
        if not self.is_connected():
            return False
        return nx.is_tree(self._graph)

    def get_cycles(self) -> List[List[int]]:
        """Get all cycles in the network."""
        try:
            return list(nx.cycle_basis(self._graph))
        except nx.NetworkXError:
            return []

    def get_shortest_path(self, source: int, target: int) -> Optional[List[int]]:
        """Get shortest path between two nodes."""
        try:
            return nx.shortest_path(self._graph, source, target)
        except nx.NetworkXNoPath:
            return None

    def get_all_paths(self, source: int, target: int) -> List[List[int]]:
        """Get all simple paths between two nodes."""
        try:
            return list(nx.all_simple_paths(self._graph, source, target))
        except nx.NetworkXError:
            return []

    def get_neighbors(self, bus_id: int) -> List[int]:
        """Get neighboring buses."""
        return list(self._graph.neighbors(bus_id))

    def get_degree(self, bus_id: int) -> int:
        """Get the degree (number of connections) of a bus."""
        return self._graph.degree(bus_id)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the network."""
        return len(self._nodes)

    @property
    def num_lines(self) -> int:
        """Number of lines in the network."""
        return len(self._lines)

    @property
    def num_closed_lines(self) -> int:
        """Number of closed (active) lines."""
        return sum(1 for line in self._lines.values()
                   if line.status == LineStatus.CLOSED)

    @property
    def slack_bus(self) -> Optional[int]:
        """Get the slack bus ID."""
        return self._slack_bus

    @property
    def graph(self) -> nx.Graph:
        """Get the underlying NetworkX graph."""
        return self._graph.copy()

    def get_nodes(self) -> Dict[int, NodeConfig]:
        """Get all node configurations."""
        return self._nodes.copy()

    def get_lines(self) -> Dict[Tuple[int, int], LineConfig]:
        """Get all line configurations."""
        return self._lines.copy()

    def get_switchable_lines(self) -> List[Tuple[int, int]]:
        """Get all lines that can be switched."""
        return [key for key, line in self._lines.items() if line.is_switch]

    def get_tie_lines(self) -> List[Tuple[int, int]]:
        """Get all normally-open tie lines."""
        return [key for key, line in self._lines.items()
                if line.is_switch and line.status == LineStatus.OPEN]

    def get_closed_switches(self) -> List[Tuple[int, int]]:
        """Get all closed switchable lines."""
        return [key for key, line in self._lines.items()
                if line.is_switch and line.status == LineStatus.CLOSED]

    def copy(self) -> "NetworkTopology":
        """Create a deep copy of the topology."""
        new_topology = NetworkTopology()
        for node in self._nodes.values():
            new_topology.add_node(NodeConfig(
                bus_id=node.bus_id,
                node_type=node.node_type,
                voltage_kv=node.voltage_kv,
                p_mw=node.p_mw,
                q_mvar=node.q_mvar,
                name=node.name,
                x=node.x,
                y=node.y
            ))
        for line in self._lines.values():
            new_topology.add_line(LineConfig(
                from_bus=line.from_bus,
                to_bus=line.to_bus,
                r_ohm_per_km=line.r_ohm_per_km,
                x_ohm_per_km=line.x_ohm_per_km,
                length_km=line.length_km,
                max_i_ka=line.max_i_ka,
                name=line.name,
                is_switch=line.is_switch,
                status=line.status
            ))
        return new_topology

    def get_statistics(self) -> Dict:
        """Get network statistics."""
        return {
            "num_nodes": self.num_nodes,
            "num_lines": self.num_lines,
            "num_closed_lines": self.num_closed_lines,
            "is_connected": self.is_connected(),
            "is_radial": self.is_radial(),
            "num_cycles": len(self.get_cycles()),
            "num_switchable": len(self.get_switchable_lines()),
            "num_tie_lines": len(self.get_tie_lines()),
        }
