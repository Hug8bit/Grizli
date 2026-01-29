"""
IEEE test networks with configurable congestion scenarios.
Provides standard test cases for optimization algorithms.
"""

from typing import Dict, Optional
import numpy as np
import pandapower as pp
import pandapower.networks as pn


def create_ieee33_with_renewables(
    num_pv: int = 3,
    num_wind: int = 2,
    pv_capacity_mw: float = 1.5,
    wind_capacity_mw: float = 2.0,
    add_tie_lines: bool = True,
    seed: Optional[int] = None
) -> pp.pandapowerNet:
    """
    Create IEEE 33-bus network with distributed renewables.

    The IEEE 33-bus is a standard radial distribution test system
    with 33 buses and 32 branches.

    Args:
        num_pv: Number of PV generators to add
        num_wind: Number of wind generators to add
        pv_capacity_mw: Capacity per PV generator
        wind_capacity_mw: Capacity per wind generator
        add_tie_lines: Whether to add normally-open tie lines
        seed: Random seed

    Returns:
        PandaPower network with renewables
    """
    rng = np.random.default_rng(seed)

    # Create base IEEE 33-bus network
    net = pn.case33bw()
    net.name = "IEEE 33-bus with Renewables"

    # Mark existing lines as switchable
    if 'is_switch' not in net.line.columns:
        net.line['is_switch'] = True

    # Available buses for generators (excluding slack bus 0)
    available_buses = list(range(1, len(net.bus)))
    rng.shuffle(available_buses)

    # Add PV generators
    pv_buses = available_buses[:num_pv]
    for i, bus in enumerate(pv_buses):
        pp.create_sgen(
            net,
            bus=bus,
            p_mw=pv_capacity_mw * (1 + rng.uniform(-0.2, 0.2)),
            q_mvar=0,
            name=f"PV_{i}",
            type="PV"
        )

    # Add wind generators
    wind_buses = available_buses[num_pv:num_pv + num_wind]
    for i, bus in enumerate(wind_buses):
        pp.create_sgen(
            net,
            bus=bus,
            p_mw=wind_capacity_mw * (1 + rng.uniform(-0.2, 0.2)),
            q_mvar=0,
            name=f"Wind_{i}",
            type="WP"
        )

    # Add tie lines (normally open) for reconfiguration
    if add_tie_lines:
        tie_line_pairs = [
            (7, 20),   # Tie line 1
            (8, 14),   # Tie line 2
            (11, 21),  # Tie line 3
            (17, 32),  # Tie line 4
            (24, 28),  # Tie line 5
        ]

        for from_bus, to_bus in tie_line_pairs:
            if from_bus < len(net.bus) and to_bus < len(net.bus):
                pp.create_line_from_parameters(
                    net,
                    from_bus=from_bus,
                    to_bus=to_bus,
                    length_km=1.0,
                    r_ohm_per_km=0.15,
                    x_ohm_per_km=0.1,
                    c_nf_per_km=10.0,
                    max_i_ka=0.4,
                    name=f"Tie_{from_bus}_{to_bus}",
                    in_service=False
                )
                net.line.at[len(net.line) - 1, 'is_switch'] = True

    return net


def create_congested_ieee33(
    load_increase_percent: float = 50.0,
    renewable_increase_percent: float = 80.0,
    seed: Optional[int] = None
) -> pp.pandapowerNet:
    """
    Create a heavily congested IEEE 33-bus scenario.

    This creates a stress-test scenario with high load and high
    renewable generation causing multiple line overloads.

    Args:
        load_increase_percent: Percentage increase in all loads
        renewable_increase_percent: Percentage increase in renewable output
        seed: Random seed

    Returns:
        Congested PandaPower network
    """
    net = create_ieee33_with_renewables(
        num_pv=4,
        num_wind=3,
        pv_capacity_mw=2.0,
        wind_capacity_mw=2.5,
        add_tie_lines=True,
        seed=seed
    )
    net.name = "IEEE 33-bus Congested"

    # Increase all loads
    load_factor = 1 + load_increase_percent / 100
    net.load.p_mw *= load_factor
    net.load.q_mvar *= load_factor

    # Increase renewable generation
    renewable_factor = 1 + renewable_increase_percent / 100
    net.sgen.p_mw *= renewable_factor

    # Reduce line capacities to simulate aging infrastructure
    net.line.max_i_ka *= 0.8

    return net


def create_atacama_scenario(seed: Optional[int] = None) -> pp.pandapowerNet:
    """
    Create an extreme solar scenario inspired by Atacama Desert.

    Simulates a network with massive solar penetration causing
    severe reverse power flow and congestion.

    Args:
        seed: Random seed

    Returns:
        Extremely congested PandaPower network
    """
    net = create_ieee33_with_renewables(
        num_pv=8,
        num_wind=2,
        pv_capacity_mw=4.0,
        wind_capacity_mw=3.0,
        add_tie_lines=True,
        seed=seed
    )
    net.name = "IEEE 33-bus Atacama Scenario"

    # Triple the solar output (peak desert conditions)
    for idx in net.sgen.index:
        if net.sgen.at[idx, 'type'] == 'PV':
            net.sgen.at[idx, 'p_mw'] *= 3.0

    # Reduce load (midday when people are at work)
    net.load.p_mw *= 0.7
    net.load.q_mvar *= 0.7

    # Critical line capacity constraints
    net.line.max_i_ka *= 0.7

    return net


def create_ieee14_with_renewables(
    num_renewables: int = 3,
    capacity_mw: float = 20.0,
    seed: Optional[int] = None
) -> pp.pandapowerNet:
    """
    Create IEEE 14-bus network with renewables.

    The IEEE 14-bus is a transmission-level test system.

    Args:
        num_renewables: Number of renewable generators
        capacity_mw: Capacity per generator
        seed: Random seed

    Returns:
        PandaPower network
    """
    rng = np.random.default_rng(seed)

    net = pn.case14()
    net.name = "IEEE 14-bus with Renewables"

    if 'is_switch' not in net.line.columns:
        net.line['is_switch'] = True

    # Available buses (excluding slack)
    available = list(range(1, len(net.bus)))
    rng.shuffle(available)

    for i in range(min(num_renewables, len(available))):
        bus = available[i]
        gen_type = "PV" if i % 2 == 0 else "WP"
        pp.create_sgen(
            net,
            bus=bus,
            p_mw=capacity_mw * (1 + rng.uniform(-0.2, 0.2)),
            q_mvar=0,
            name=f"Renewable_{i}",
            type=gen_type
        )

    return net


def get_ieee33_info() -> Dict:
    """
    Get information about the IEEE 33-bus network.

    Returns:
        Dictionary with network specifications
    """
    return {
        "name": "IEEE 33-bus Distribution System",
        "buses": 33,
        "lines": 32,
        "voltage_kv": 12.66,
        "total_load_mw": 3.72,
        "total_load_mvar": 2.30,
        "topology": "radial",
        "description": (
            "Standard radial distribution test feeder. "
            "Originally proposed by Baran and Wu (1989). "
            "Commonly used for distribution system optimization studies."
        ),
        "reference": "M.E. Baran and F.F. Wu, 'Network reconfiguration in distribution "
                     "systems for loss reduction and load balancing', IEEE Trans. Power "
                     "Delivery, vol. 4, no. 2, pp. 1401-1407, Apr. 1989."
    }


def get_ieee14_info() -> Dict:
    """
    Get information about the IEEE 14-bus network.

    Returns:
        Dictionary with network specifications
    """
    return {
        "name": "IEEE 14-bus Test System",
        "buses": 14,
        "lines": 20,
        "generators": 5,
        "voltage_kv": [69, 13.8],
        "description": (
            "Standard IEEE transmission test system. "
            "Represents a portion of the American Electric Power System. "
            "Commonly used for power flow and stability studies."
        ),
    }


def list_available_networks() -> Dict[str, Dict]:
    """
    List all available test networks.

    Returns:
        Dictionary mapping network names to their info
    """
    return {
        "ieee33": get_ieee33_info(),
        "ieee33_congested": {
            "name": "IEEE 33-bus Congested",
            "description": "IEEE 33-bus with increased load and renewables causing congestion"
        },
        "ieee33_atacama": {
            "name": "IEEE 33-bus Atacama",
            "description": "Extreme solar scenario with massive reverse power flow"
        },
        "ieee14": get_ieee14_info(),
    }
