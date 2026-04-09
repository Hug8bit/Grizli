"""
Unit tests — grid simulation core.
Tests: network creation, power flow, time-series simulation.
"""
import pytest
import numpy as np
import pandas as pd


# ── Network creation ──────────────────────────────────────────────────────────

class TestNetworkCreation:
    def test_ieee33_with_renewables(self):
        from src.core.ieee_networks import create_ieee33_with_renewables
        net = create_ieee33_with_renewables()
        assert len(net.bus) == 33
        assert len(net.line) >= 32
        assert len(net.ext_grid) >= 1

    def test_congested_ieee33(self):
        from src.core.ieee_networks import create_congested_ieee33
        net = create_congested_ieee33()
        assert len(net.bus) == 33

    def test_network_generator_radial(self):
        from src.core.network_generator import NetworkGenerator, NetworkType
        gen = NetworkGenerator(num_buses=10, network_type=NetworkType.RADIAL, seed=0)
        net = gen.generate()
        assert len(net.bus) == 10

    def test_network_generator_meshed(self):
        from src.core.network_generator import NetworkGenerator, NetworkType
        gen = NetworkGenerator(num_buses=15, network_type=NetworkType.MESHED, seed=1)
        net = gen.generate()
        assert len(net.bus) == 15


# ── Power flow ────────────────────────────────────────────────────────────────

class TestPowerFlow:
    @pytest.fixture
    def ieee33_net(self):
        from src.core.ieee_networks import create_ieee33_with_renewables
        return create_ieee33_with_renewables()

    def test_power_flow_runs(self, ieee33_net):
        from src.simulation.power_flow import PowerFlowSimulator
        sim = PowerFlowSimulator(ieee33_net)
        result = sim.run_power_flow()
        assert result.converged

    def test_power_flow_voltages_in_range(self, ieee33_net):
        from src.simulation.power_flow import PowerFlowSimulator
        sim = PowerFlowSimulator(ieee33_net)
        result = sim.run_power_flow()
        assert result.converged
        voltages = result.voltage_pu
        assert (voltages > 0.85).all()
        assert (voltages < 1.15).all()

    def test_power_flow_returns_losses(self, ieee33_net):
        from src.simulation.power_flow import PowerFlowSimulator
        sim = PowerFlowSimulator(ieee33_net)
        result = sim.run_power_flow()
        assert result.total_losses_mw >= 0

    def test_power_flow_loading_nonnegative(self, ieee33_net):
        from src.simulation.power_flow import PowerFlowSimulator
        sim = PowerFlowSimulator(ieee33_net)
        result = sim.run_power_flow()
        assert (result.line_loading_percent >= 0).all()


# ── Time-series simulation ────────────────────────────────────────────────────

class TestTimeSeriesSimulation:
    @pytest.fixture
    def net_and_sim(self):
        from src.core.ieee_networks import create_ieee33_with_renewables
        from src.simulation.time_series import (
            TimeSeriesSimulator, TimeSeriesConfig, ScenarioGenerator, Season,
        )
        net    = create_ieee33_with_renewables()
        config = TimeSeriesConfig(duration_hours=4, resolution_minutes=60, season=Season.SUMMER)
        gen    = ScenarioGenerator(config, seed=42)
        scenario = gen.generate_network_scenario(net)
        sim    = TimeSeriesSimulator(net)
        return sim, scenario

    def test_time_series_returns_dataframe(self, net_and_sim):
        sim, scenario = net_and_sim
        results = sim.run_simulation(scenario["sgen"], scenario["load"])
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

    def test_time_series_has_expected_columns(self, net_and_sim):
        sim, scenario = net_and_sim
        results = sim.run_simulation(scenario["sgen"], scenario["load"])
        for col in ("max_loading_percent", "total_losses_mw", "num_overloaded"):
            assert col in results.columns, f"Missing column: {col}"

    def test_time_series_losses_nonnegative(self, net_and_sim):
        sim, scenario = net_and_sim
        results = sim.run_simulation(scenario["sgen"], scenario["load"])
        assert (results["total_losses_mw"] >= 0).all()
