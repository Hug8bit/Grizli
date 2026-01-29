"""
Synthetic network generation for testing and simulation.
Generates realistic electrical networks with configurable topology.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandapower as pp
import networkx as nx

from .network_topology import NetworkTopology, NodeType, LineStatus, NodeConfig, LineConfig


class NetworkType(Enum):
    """Type of network topology."""
    RADIAL = "radial"       # Tree structure (typical distribution)
    MESHED = "meshed"       # With loops/cycles
    RING = "ring"           # Ring topology


@dataclass
class GeneratorConfig:
    """Configuration for distributed generators."""
    num_solar: int = 2
    num_wind: int = 1
    solar_capacity_mw: float = 2.0
    wind_capacity_mw: float = 3.0
    solar_variability: float = 0.3
    wind_variability: float = 0.4


@dataclass
class LoadConfig:
    """Configuration for network loads."""
    num_residential: int = 10
    num_industrial: int = 3
    num_commercial: int = 5
    residential_mean_mw: float = 0.5
    residential_std_mw: float = 0.2
    industrial_mean_mw: float = 2.0
    industrial_std_mw: float = 0.5
    commercial_mean_mw: float = 1.0
    commercial_std_mw: float = 0.3
    power_factor: float = 0.95


class NetworkGenerator:
    """
    Generates synthetic electrical networks with PandaPower.

    Supports radial, meshed, and ring topologies with configurable
    numbers of buses, generators, and loads.
    """

    def __init__(
        self,
        num_buses: int = 25,
        network_type: NetworkType = NetworkType.MESHED,
        voltage_kv: float = 20.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the network generator.

        Args:
            num_buses: Number of buses in the network
            network_type: Type of network topology
            voltage_kv: Nominal voltage level in kV
            seed: Random seed for reproducibility
        """
        self.num_buses = max(5, num_buses)
        self.network_type = network_type
        self.voltage_kv = voltage_kv
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._net: Optional[pp.pandapowerNet] = None
        self._topology: Optional[NetworkTopology] = None

    def generate(
        self,
        gen_config: Optional[GeneratorConfig] = None,
        load_config: Optional[LoadConfig] = None
    ) -> pp.pandapowerNet:
        """
        Generate a complete network.

        Args:
            gen_config: Generator configuration (uses defaults if None)
            load_config: Load configuration (uses defaults if None)

        Returns:
            PandaPower network object
        """
        if gen_config is None:
            gen_config = GeneratorConfig()
        if load_config is None:
            load_config = LoadConfig()

        # Create empty network
        self._net = pp.create_empty_network(name=f"synthetic_{self.num_buses}bus")
        self._topology = NetworkTopology()

        # Create buses
        self._create_buses()

        # Create lines based on topology type
        if self.network_type == NetworkType.RADIAL:
            self._create_radial_topology()
        elif self.network_type == NetworkType.RING:
            self._create_ring_topology()
        else:  # MESHED
            self._create_meshed_topology()

        # Add external grid (slack)
        self._add_external_grid()

        # Add generators
        self._add_generators(gen_config)

        # Add loads
        self._add_loads(load_config)

        # Add tie lines for switching
        self._add_tie_lines()

        return self._net

    def _create_buses(self) -> None:
        """Create all buses in the network."""
        # Generate bus positions in a grid-like pattern
        cols = int(np.ceil(np.sqrt(self.num_buses)))
        rows = int(np.ceil(self.num_buses / cols))

        for i in range(self.num_buses):
            row = i // cols
            col = i % cols
            x = col * 100 + self._rng.uniform(-20, 20)
            y = row * 100 + self._rng.uniform(-20, 20)

            pp.create_bus(
                self._net,
                vn_kv=self.voltage_kv,
                name=f"Bus_{i}",
                geodata=(x, y)
            )

            node_type = NodeType.SLACK if i == 0 else NodeType.PQ
            self._topology.add_node(NodeConfig(
                bus_id=i,
                node_type=node_type,
                voltage_kv=self.voltage_kv,
                name=f"Bus_{i}",
                x=x,
                y=y
            ))

    def _create_radial_topology(self) -> None:
        """Create a radial (tree) network topology."""
        # Use a spanning tree approach
        connected = {0}
        for i in range(1, self.num_buses):
            # Connect to a random already-connected bus
            parent = self._rng.choice(list(connected))
            self._add_line(parent, i)
            connected.add(i)

    def _create_ring_topology(self) -> None:
        """Create a ring network topology."""
        # Create main ring
        for i in range(self.num_buses - 1):
            self._add_line(i, i + 1)
        self._add_line(self.num_buses - 1, 0)  # Close the ring

    def _create_meshed_topology(self) -> None:
        """Create a meshed network topology with loops."""
        # First create a radial structure
        connected = {0}
        for i in range(1, self.num_buses):
            parent = self._rng.choice(list(connected))
            self._add_line(parent, i)
            connected.add(i)

        # Add extra lines to create meshes
        num_extra_lines = max(1, self.num_buses // 5)
        existing_lines = set()
        for idx in range(len(self._net.line)):
            f, t = self._net.line.at[idx, 'from_bus'], self._net.line.at[idx, 'to_bus']
            existing_lines.add((min(f, t), max(f, t)))

        added = 0
        attempts = 0
        while added < num_extra_lines and attempts < num_extra_lines * 10:
            i = self._rng.integers(0, self.num_buses)
            j = self._rng.integers(0, self.num_buses)
            if i != j:
                key = (min(i, j), max(i, j))
                if key not in existing_lines:
                    self._add_line(i, j, is_switch=True)
                    existing_lines.add(key)
                    added += 1
            attempts += 1

    def _add_line(
        self,
        from_bus: int,
        to_bus: int,
        is_switch: bool = False,
        in_service: bool = True
    ) -> int:
        """Add a line between two buses."""
        # Calculate distance-based length
        if self._net.bus_geodata is not None and len(self._net.bus_geodata) > max(from_bus, to_bus):
            x1, y1 = self._net.bus_geodata.at[from_bus, 'x'], self._net.bus_geodata.at[from_bus, 'y']
            x2, y2 = self._net.bus_geodata.at[to_bus, 'x'], self._net.bus_geodata.at[to_bus, 'y']
            length_km = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 1000
        else:
            length_km = self._rng.uniform(0.5, 2.0)

        length_km = max(0.1, length_km)

        # Line parameters (typical medium voltage)
        r_ohm_per_km = self._rng.uniform(0.1, 0.3)
        x_ohm_per_km = self._rng.uniform(0.1, 0.2)
        max_i_ka = self._rng.uniform(0.3, 0.6)

        idx = pp.create_line_from_parameters(
            self._net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=length_km,
            r_ohm_per_km=r_ohm_per_km,
            x_ohm_per_km=x_ohm_per_km,
            c_nf_per_km=10.0,
            max_i_ka=max_i_ka,
            name=f"Line_{from_bus}_{to_bus}",
            in_service=in_service
        )

        # Mark as switch in user data
        if 'is_switch' not in self._net.line.columns:
            self._net.line['is_switch'] = False
        self._net.line.at[idx, 'is_switch'] = is_switch

        # Add to topology
        self._topology.add_line(LineConfig(
            from_bus=from_bus,
            to_bus=to_bus,
            r_ohm_per_km=r_ohm_per_km,
            x_ohm_per_km=x_ohm_per_km,
            length_km=length_km,
            max_i_ka=max_i_ka,
            is_switch=is_switch,
            status=LineStatus.CLOSED if in_service else LineStatus.OPEN
        ))

        return idx

    def _add_external_grid(self) -> None:
        """Add external grid connection at bus 0."""
        pp.create_ext_grid(
            self._net,
            bus=0,
            vm_pu=1.02,
            va_degree=0,
            name="Grid Connection"
        )

    def _add_generators(self, config: GeneratorConfig) -> None:
        """Add distributed generators to the network."""
        available_buses = list(range(1, self.num_buses))
        self._rng.shuffle(available_buses)

        gen_buses = available_buses[:config.num_solar + config.num_wind]

        # Add solar generators
        for i in range(config.num_solar):
            if i >= len(gen_buses):
                break
            bus = gen_buses[i]
            capacity = config.solar_capacity_mw * (1 + self._rng.uniform(
                -config.solar_variability, config.solar_variability))
            pp.create_sgen(
                self._net,
                bus=bus,
                p_mw=capacity,
                q_mvar=0,
                name=f"Solar_{i}",
                type="PV"
            )

        # Add wind generators
        for i in range(config.num_wind):
            idx = config.num_solar + i
            if idx >= len(gen_buses):
                break
            bus = gen_buses[idx]
            capacity = config.wind_capacity_mw * (1 + self._rng.uniform(
                -config.wind_variability, config.wind_variability))
            pp.create_sgen(
                self._net,
                bus=bus,
                p_mw=capacity,
                q_mvar=0,
                name=f"Wind_{i}",
                type="WP"
            )

    def _add_loads(self, config: LoadConfig) -> None:
        """Add loads to the network."""
        available_buses = list(range(1, self.num_buses))
        self._rng.shuffle(available_buses)

        total_loads = config.num_residential + config.num_industrial + config.num_commercial
        load_buses = available_buses[:min(total_loads, len(available_buses))]

        idx = 0

        # Residential loads
        for i in range(config.num_residential):
            if idx >= len(load_buses):
                break
            bus = load_buses[idx]
            p_mw = max(0.1, self._rng.normal(
                config.residential_mean_mw, config.residential_std_mw))
            q_mvar = p_mw * np.tan(np.arccos(config.power_factor))
            pp.create_load(
                self._net,
                bus=bus,
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=f"Residential_{i}"
            )
            idx += 1

        # Industrial loads
        for i in range(config.num_industrial):
            if idx >= len(load_buses):
                break
            bus = load_buses[idx]
            p_mw = max(0.5, self._rng.normal(
                config.industrial_mean_mw, config.industrial_std_mw))
            q_mvar = p_mw * np.tan(np.arccos(config.power_factor - 0.05))
            pp.create_load(
                self._net,
                bus=bus,
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=f"Industrial_{i}"
            )
            idx += 1

        # Commercial loads
        for i in range(config.num_commercial):
            if idx >= len(load_buses):
                break
            bus = load_buses[idx]
            p_mw = max(0.2, self._rng.normal(
                config.commercial_mean_mw, config.commercial_std_mw))
            q_mvar = p_mw * np.tan(np.arccos(config.power_factor))
            pp.create_load(
                self._net,
                bus=bus,
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=f"Commercial_{i}"
            )
            idx += 1

    def _add_tie_lines(self) -> None:
        """Add normally-open tie lines for reconfiguration."""
        if self.network_type == NetworkType.RADIAL:
            # Add a few tie lines to enable reconfiguration
            num_ties = max(2, self.num_buses // 10)
            existing = set()
            for idx in range(len(self._net.line)):
                f, t = self._net.line.at[idx, 'from_bus'], self._net.line.at[idx, 'to_bus']
                existing.add((min(f, t), max(f, t)))

            added = 0
            attempts = 0
            while added < num_ties and attempts < num_ties * 20:
                i = self._rng.integers(1, self.num_buses)
                j = self._rng.integers(1, self.num_buses)
                if i != j and abs(i - j) > 2:
                    key = (min(i, j), max(i, j))
                    if key not in existing:
                        self._add_line(i, j, is_switch=True, in_service=False)
                        existing.add(key)
                        added += 1
                attempts += 1

    def get_topology(self) -> NetworkTopology:
        """Get the network topology object."""
        if self._topology is None:
            raise ValueError("Network not generated yet. Call generate() first.")
        return self._topology

    def get_network_summary(self) -> Dict:
        """Get a summary of the generated network."""
        if self._net is None:
            raise ValueError("Network not generated yet. Call generate() first.")

        return {
            "name": self._net.name,
            "num_buses": len(self._net.bus),
            "num_lines": len(self._net.line),
            "num_generators": len(self._net.sgen),
            "num_loads": len(self._net.load),
            "voltage_kv": self.voltage_kv,
            "network_type": self.network_type.value,
            "total_generation_mw": self._net.sgen.p_mw.sum() if len(self._net.sgen) > 0 else 0,
            "total_load_mw": self._net.load.p_mw.sum() if len(self._net.load) > 0 else 0,
            "num_switches": self._net.line['is_switch'].sum() if 'is_switch' in self._net.line.columns else 0,
        }


def create_synthetic_network(
    num_buses: int = 25,
    network_type: str = "meshed",
    seed: Optional[int] = None
) -> pp.pandapowerNet:
    """
    Convenience function to create a synthetic network.

    Args:
        num_buses: Number of buses
        network_type: "radial", "meshed", or "ring"
        seed: Random seed

    Returns:
        PandaPower network
    """
    type_map = {
        "radial": NetworkType.RADIAL,
        "meshed": NetworkType.MESHED,
        "ring": NetworkType.RING
    }
    ntype = type_map.get(network_type.lower(), NetworkType.MESHED)

    generator = NetworkGenerator(
        num_buses=num_buses,
        network_type=ntype,
        seed=seed
    )
    return generator.generate()
