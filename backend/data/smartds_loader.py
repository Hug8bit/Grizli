"""
GrizliV2 - SMART-DS Data Loader
Loads San Francisco Bay Area distribution grid data from NREL SMART-DS dataset.

Dataset: https://data.openei.org/submissions/2981
Format: OpenDSS (.dss files) + geospatial metadata

The SMART-DS dataset provides synthetic but realistic distribution grid models
for the San Francisco Bay Area. We use it to train our RL agent on a real-world
scale grid before deploying to Chilean grid (LATAM transfer learning).
"""
import re
import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import pandapower as pp

logger = logging.getLogger(__name__)


class SmartDSLoader:
    """
    Loads and parses NREL SMART-DS dataset (San Francisco Bay Area).

    Usage:
        loader = SmartDSLoader("/path/to/smartds/P1R")
        net = loader.load_network()
        scenarios = loader.load_scenarios()
    """

    def __init__(self, dataset_path: Path):
        self.path = Path(dataset_path)
        self._bus_map: dict[str, int] = {}  # DSS name -> pandapower idx

    def load_network(self) -> pp.pandapowerNet:
        """
        Parse OpenDSS files and build a PandaPower network.
        Looks for Master.dss or equivalent entry point.
        """
        master = self._find_master_file()
        if master is None:
            logger.warning("No Master.dss found - loading synthetic fallback network")
            return self._synthetic_sf_network()

        logger.info(f"Loading SMART-DS from {master}")
        net = pp.create_empty_network(f_hz=60, sn_mva=1.0)

        lines_file = self.path / "Lines.dss"
        loads_file = self.path / "Loads.dss"
        pvsystems_file = self.path / "PVSystems.dss"

        if lines_file.exists():
            self._parse_lines(net, lines_file)
        if loads_file.exists():
            self._parse_loads(net, loads_file)
        if pvsystems_file.exists():
            self._parse_pv_systems(net, pvsystems_file)

        self._add_slack_bus(net)
        logger.info(f"Loaded: {len(net.bus)} buses, {len(net.line)} lines, {len(net.load)} loads")
        return net

    def load_scenarios(self, n_scenarios: int = 100) -> list[dict]:
        """
        Load or generate time-series scenarios for RL training.
        Each scenario = one 24h day with hourly load/gen profiles.
        """
        scenario_dir = self.path / "scenarios"
        if scenario_dir.exists():
            return self._load_scenario_files(scenario_dir, n_scenarios)
        return self._generate_synthetic_scenarios(n_scenarios)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_master_file(self) -> Optional[Path]:
        for name in ["Master.dss", "master.dss", "Master.DSS"]:
            p = self.path / name
            if p.exists():
                return p
        masters = list(self.path.rglob("Master.dss"))
        return masters[0] if masters else None

    def _parse_lines(self, net: pp.pandapowerNet, filepath: Path) -> None:
        """Parse OpenDSS Line definitions into PandaPower lines."""
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        pattern = re.compile(
            r"New\s+Line\.(\S+)\s+Bus1=(\S+)\s+Bus2=(\S+)"
            r"(?:.*?Length=([\d.]+))?(?:.*?R1=([\d.]+))?",
            re.IGNORECASE,
        )
        for m in pattern.finditer(content):
            name, bus1, bus2 = m.group(1), m.group(2).split(".")[0], m.group(3).split(".")[0]
            length_km = float(m.group(4)) * 0.001 if m.group(4) else 0.1
            r_ohm_per_km = float(m.group(5)) if m.group(5) else 0.1

            b1 = self._get_or_create_bus(net, bus1)
            b2 = self._get_or_create_bus(net, bus2)
            pp.create_line_from_parameters(
                net, from_bus=b1, to_bus=b2,
                length_km=length_km,
                r_ohm_per_km=r_ohm_per_km,
                x_ohm_per_km=0.1,
                c_nf_per_km=10,
                max_i_ka=0.3,
                name=name,
            )

    def _parse_loads(self, net: pp.pandapowerNet, filepath: Path) -> None:
        """Parse OpenDSS Load definitions."""
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        pattern = re.compile(
            r"New\s+Load\.(\S+)\s+Bus1=(\S+)(?:.*?kW=([\d.]+))?(?:.*?kvar=([\d.]+))?",
            re.IGNORECASE,
        )
        for m in pattern.finditer(content):
            name, bus = m.group(1), m.group(2).split(".")[0]
            p_kw = float(m.group(3)) if m.group(3) else 10.0
            q_kvar = float(m.group(4)) if m.group(4) else 2.0

            b = self._get_or_create_bus(net, bus)
            pp.create_load(net, bus=b, p_mw=p_kw / 1000.0, q_mvar=q_kvar / 1000.0, name=name)

    def _parse_pv_systems(self, net: pp.pandapowerNet, filepath: Path) -> None:
        """Parse OpenDSS PVSystem definitions as static generators."""
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        pattern = re.compile(
            r"New\s+PVSystem\.(\S+)\s+Bus1=(\S+)(?:.*?kVA=([\d.]+))?",
            re.IGNORECASE,
        )
        for m in pattern.finditer(content):
            name, bus = m.group(1), m.group(2).split(".")[0]
            kva = float(m.group(3)) if m.group(3) else 50.0
            b = self._get_or_create_bus(net, bus)
            pp.create_sgen(net, bus=b, p_mw=kva / 1000.0 * 0.9, q_mvar=0, name=name, type="PV")

    def _get_or_create_bus(self, net: pp.pandapowerNet, name: str) -> int:
        if name not in self._bus_map:
            idx = pp.create_bus(net, vn_kv=12.47, name=name)
            self._bus_map[name] = idx
        return self._bus_map[name]

    def _add_slack_bus(self, net: pp.pandapowerNet) -> None:
        if len(net.ext_grid) == 0 and len(net.bus) > 0:
            slack_bus = net.bus.index[0]
            pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0, va_degree=0)

    def _load_scenario_files(self, scenario_dir: Path, n: int) -> list[dict]:
        scenarios = []
        for f in sorted(scenario_dir.glob("*.json"))[:n]:
            with f.open() as fp:
                scenarios.append(json.load(fp))
        logger.info(f"Loaded {len(scenarios)} scenarios from {scenario_dir}")
        return scenarios

    def _generate_synthetic_scenarios(self, n: int) -> list[dict]:
        """Generate synthetic 24h load/solar scenarios for SF Bay Area climate."""
        logger.info(f"Generating {n} synthetic scenarios (SF Bay Area climate)")
        rng = np.random.default_rng(42)
        scenarios = []

        for i in range(n):
            hours = np.arange(24)
            # Residential load curve (morning + evening peaks)
            load_base = 0.4 + 0.3 * np.exp(-((hours - 8) ** 2) / 8) + 0.5 * np.exp(-((hours - 19) ** 2) / 6)
            load_noise = rng.normal(0, 0.03, 24)
            load_profile = np.clip(load_base + load_noise, 0.1, 1.0).tolist()

            # Solar generation (bell curve peaking at noon, ~5.5h peak sun SF)
            solar_base = np.maximum(0, np.exp(-((hours - 13) ** 2) / 10))
            cloud_factor = rng.uniform(0.6, 1.0)  # SF Bay fog variability
            solar_profile = (solar_base * cloud_factor).tolist()

            # Wind (Atacama-inspired: stronger at night/morning)
            wind_base = 0.3 + 0.2 * np.exp(-((hours - 4) ** 2) / 8)
            wind_noise = rng.normal(0, 0.05, 24)
            wind_profile = np.clip(wind_base + wind_noise, 0, 1.0).tolist()

            scenarios.append({
                "id": i,
                "season": ["summer", "fall", "winter", "spring"][i % 4],
                "load_profile": load_profile,    # normalized 0-1
                "solar_profile": solar_profile,   # normalized 0-1
                "wind_profile": wind_profile,     # normalized 0-1
                "peak_load_mw": float(rng.uniform(0.5, 2.0)),
                "solar_capacity_mw": float(rng.uniform(0.1, 0.8)),
            })

        return scenarios

    def _synthetic_sf_network(self) -> pp.pandapowerNet:
        """
        Fallback: create a synthetic 33-bus network representative of
        a San Francisco distribution feeder when SMART-DS files not found.
        """
        import pandapower.networks as pn
        net = pn.case33bw()
        net.name = "SF-synthetic-33bus"
        logger.info("Using synthetic 33-bus SF network (SMART-DS not downloaded)")
        return net


def download_smartds_instructions() -> str:
    """Return instructions for downloading the SMART-DS dataset."""
    return """
    SMART-DS Dataset Download Instructions
    =======================================
    1. Visit: https://data.openei.org/submissions/2981
    2. Download "P1R" (San Francisco Bay Area, residential)
       or "P4R" (larger coverage area)
    3. Extract to: data/smartds/

    File structure expected:
        data/smartds/
            Master.dss
            Lines.dss
            Loads.dss
            PVSystems.dss
            ...

    Total size: ~200MB for P1R region
    License: Creative Commons CC0 1.0

    Alternatively, use the synthetic fallback (already built-in).
    """
