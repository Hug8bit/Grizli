"""
OpenDSS → PandaPower converter for Grizli.

Converts a DSSCircuit (produced by OpenDSSParser) into a pandapowerNet
ready for AC power flow simulation and topology optimization.

Supported elements:
  - Buses (with voltage level assignment)
  - Lines (with linecode fallback and unit conversion)
  - Loads (3-phase wye/delta, single-phase)
  - Generators / PVSystems (as static generators)
  - Transformers (2-winding)
  - Capacitors (as shunt elements)

Design principles:
  - Lossless: no information silently dropped — unknown elements are logged
  - Defensive: missing parameters get physically reasonable defaults, not zeros
  - Compatible: output net works directly with PowerFlowSimulator and FastOptimizer

Author: FH — April 2026
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandapower as pp

# Parser types (same module, same package)
from src.smartds.opendss_parser import (
    DSSBus,
    DSSCapacitor,
    DSSCircuit,
    DSSGenerator,
    DSSLine,
    DSSLoad,
    DSSTransformer,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Default thermal rating when normamps is missing (A)
DEFAULT_NORMAMPS = 400.0

# Default X/R ratio used when reactance is missing but resistance is known
DEFAULT_XR_RATIO = 2.0

# Default line impedance (Ω/km) for 13.2 kV feeders when nothing is available
DEFAULT_R1_OHM_PER_KM = 0.306
DEFAULT_X1_OHM_PER_KM = 0.320

# Unit → km conversion factors (OpenDSS length units)
UNIT_TO_KM: Dict[str, float] = {
    "km":   1.0,
    "m":    1e-3,
    "mi":   1.60934,
    "kft":  0.3048,
    "ft":   3.048e-4,
    "in":   2.54e-5,
    "cm":   1e-5,
}

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversionResult:
    """Output of the OpenDSS → PandaPower conversion."""

    net: pp.pandapowerNet
    """The pandapower network, ready for runpp()."""

    bus_map: Dict[str, int]
    """Maps DSS bus name (lowercase) → pandapower bus index."""

    n_buses: int
    n_lines: int
    n_loads: int
    n_generators: int
    n_transformers: int
    n_capacitors: int

    warnings: List[str] = field(default_factory=list)
    """Non-fatal issues encountered during conversion."""

    skipped: List[str] = field(default_factory=list)
    """Elements that could not be converted (with reasons)."""

    @property
    def summary(self) -> str:
        lines = [
            f"Conversion summary",
            f"  Buses        : {self.n_buses}",
            f"  Lines        : {self.n_lines}",
            f"  Loads        : {self.n_loads}",
            f"  Generators   : {self.n_generators}",
            f"  Transformers : {self.n_transformers}",
            f"  Capacitors   : {self.n_capacitors}",
        ]
        if self.warnings:
            lines.append(f"  Warnings     : {len(self.warnings)}")
        if self.skipped:
            lines.append(f"  Skipped      : {len(self.skipped)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Converter
# ─────────────────────────────────────────────────────────────────────────────

class OpenDSSToPandaPower:
    """
    Converts a DSSCircuit to a pandapowerNet.

    Usage:
        from src.smartds.opendss_parser import parse_smartds
        from src.smartds.opendss_to_pandapower import OpenDSSToPandaPower

        circuit = parse_smartds("/path/to/smartds/P1R/")
        result  = OpenDSSToPandaPower().convert(circuit)
        net     = result.net

        import pandapower as pp
        pp.runpp(net)
        print(net.res_bus)
    """

    def __init__(self, *, freq_hz: float = 60.0, verbose: bool = False):
        """
        Args:
            freq_hz: System frequency in Hz (60 for US/Brazil, 50 for Chile/Europe).
            verbose: Emit INFO-level log messages during conversion.
        """
        self.freq_hz = freq_hz
        self.verbose = verbose

        # Internal state (reset on each convert() call)
        self._net: pp.pandapowerNet = None
        self._bus_map: Dict[str, int] = {}   # dss_name.lower() → pp bus index
        self._warnings: List[str] = []
        self._skipped: List[str] = []
        self._circuit: DSSCircuit = None

    # ── Public entry point ────────────────────────────────────────────────────

    def convert(self, circuit: DSSCircuit) -> ConversionResult:
        """
        Convert a DSSCircuit to a pandapowerNet.

        Args:
            circuit: Parsed OpenDSS circuit from OpenDSSParser.

        Returns:
            ConversionResult with the pandapower network and metadata.
        """
        self._reset(circuit)

        self._create_buses()
        self._create_external_grid()
        self._create_lines()
        self._create_loads()
        self._create_generators()
        self._create_transformers()
        self._create_capacitors()

        if self.verbose:
            logger.info(self._result().summary)

        return self._result()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset(self, circuit: DSSCircuit) -> None:
        self._circuit = circuit
        self._net = pp.create_empty_network(
            name=circuit.name,
            f_hz=self.freq_hz,
            sn_mva=100.0,
        )
        self._bus_map = {}
        self._warnings = []
        self._skipped = []

    # ── Buses ─────────────────────────────────────────────────────────────────

    def _create_buses(self) -> None:
        """Create one pandapower bus per DSS bus."""
        circuit = self._circuit

        # We need a voltage level for each bus.
        # Priority: explicit DSSBus.base_kv → circuit base_kv
        for name, dss_bus in circuit.buses.items():
            vn_kv = dss_bus.base_kv if dss_bus.base_kv > 0 else circuit.base_kv

            # Single-phase buses: voltage given as line-to-neutral
            # Normalise to 3-phase line-to-line equivalent
            if dss_bus.phases == 1 and vn_kv > 0:
                vn_kv = vn_kv * math.sqrt(3)

            idx = pp.create_bus(
                self._net,
                vn_kv=vn_kv,
                name=name,
                geodata=(dss_bus.x, dss_bus.y) if (dss_bus.x or dss_bus.y) else None,
            )
            self._bus_map[name.lower()] = idx

        if self.verbose:
            logger.info("Created %d buses", len(self._bus_map))

    def _bus_idx(self, dss_name: str) -> Optional[int]:
        """Look up pandapower bus index from DSS bus name. Returns None if missing."""
        key = dss_name.lower().split(".")[0]  # strip phase suffix e.g. "bus1.1.2.3"
        idx = self._bus_map.get(key)
        if idx is None:
            self._warn(f"Bus '{dss_name}' not found in bus_map")
        return idx

    # ── External grid (slack bus) ─────────────────────────────────────────────

    def _create_external_grid(self) -> None:
        """Attach an external grid (slack) to the source bus."""
        src = self._circuit.source_bus.lower()
        idx = self._bus_map.get(src)

        if idx is None:
            # Fallback: use bus 0
            if self._bus_map:
                idx = min(self._bus_map.values())
                self._warn(
                    f"Source bus '{src}' not found — using bus index {idx} as slack"
                )
            else:
                self._warn("No buses defined — cannot create external grid")
                return

        pp.create_ext_grid(self._net, bus=idx, vm_pu=1.0, name="SubstationGrid")

    # ── Lines ─────────────────────────────────────────────────────────────────

    def _create_lines(self) -> None:
        """Convert DSS lines to pandapower lines."""
        for name, dss_line in self._circuit.lines.items():
            if not dss_line.enabled:
                continue

            from_bus = self._bus_idx(dss_line.bus1)
            to_bus   = self._bus_idx(dss_line.bus2)
            if from_bus is None or to_bus is None:
                self._skip(f"Line '{name}'", "missing bus endpoint")
                continue

            length_km = self._length_to_km(dss_line.length, dss_line.units)
            if length_km <= 0:
                length_km = 0.001  # avoid division by zero
                self._warn(f"Line '{name}' has zero length — set to 1 m")

            r_ohm_km, x_ohm_km = self._line_impedance(dss_line, name)
            max_i_ka = (dss_line.normamps or DEFAULT_NORMAMPS) / 1000.0

            # Shunt capacitance: DSS gives c1 in nF/km, pandapower wants nF/km too
            c_nf_km = dss_line.c1 if dss_line.c1 > 0 else 0.0

            pp.create_line_from_parameters(
                self._net,
                from_bus=from_bus,
                to_bus=to_bus,
                length_km=length_km,
                r_ohm_per_km=r_ohm_km,
                x_ohm_per_km=x_ohm_km,
                c_nf_per_km=c_nf_km,
                max_i_ka=max_i_ka,
                name=name,
                in_service=dss_line.enabled,
            )

    def _length_to_km(self, length: float, units: str) -> float:
        factor = UNIT_TO_KM.get(units.lower(), None)
        if factor is None:
            self._warn(f"Unknown length unit '{units}' — assuming km")
            factor = 1.0
        return length * factor

    def _line_impedance(
        self, dss_line: DSSLine, name: str
    ) -> Tuple[float, float]:
        """Return (r_ohm_per_km, x_ohm_per_km), filling defaults if needed."""
        r = dss_line.r1
        x = dss_line.x1

        if r == 0 and x == 0:
            r = DEFAULT_R1_OHM_PER_KM
            x = DEFAULT_X1_OHM_PER_KM
            self._warn(
                f"Line '{name}' has no impedance — using defaults "
                f"({r} Ω/km, {x} Ω/km)"
            )
        elif r == 0 and x > 0:
            r = x / DEFAULT_XR_RATIO
            self._warn(f"Line '{name}' has R=0, inferred R={r:.4f} Ω/km from X/R={DEFAULT_XR_RATIO}")
        elif x == 0 and r > 0:
            x = r * DEFAULT_XR_RATIO
            self._warn(f"Line '{name}' has X=0, inferred X={x:.4f} Ω/km from X/R={DEFAULT_XR_RATIO}")

        return r, x

    # ── Loads ─────────────────────────────────────────────────────────────────

    def _create_loads(self) -> None:
        """Convert DSS loads to pandapower loads."""
        for name, dss_load in self._circuit.loads.items():
            if not dss_load.enabled:
                continue

            bus = self._bus_idx(dss_load.bus1)
            if bus is None:
                self._skip(f"Load '{name}'", "missing bus")
                continue

            p_mw   = dss_load.kw   / 1000.0
            q_mvar = dss_load.kvar / 1000.0

            # Derive Q from P and power factor if kvar not set
            if q_mvar == 0 and p_mw > 0 and dss_load.pf > 0:
                pf = min(abs(dss_load.pf), 1.0)
                if pf < 1.0:
                    q_mvar = p_mw * math.sqrt(1 - pf**2) / pf

            pp.create_load(
                self._net,
                bus=bus,
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=name,
                in_service=dss_load.enabled,
            )

    # ── Generators / PVSystems ────────────────────────────────────────────────

    def _create_generators(self) -> None:
        """Convert DSS generators and PV systems to pandapower static generators."""
        for name, dss_gen in self._circuit.generators.items():
            if not dss_gen.enabled:
                continue

            bus = self._bus_idx(dss_gen.bus1)
            if bus is None:
                self._skip(f"Generator '{name}'", "missing bus")
                continue

            p_mw   = dss_gen.kw   / 1000.0
            q_mvar = dss_gen.kvar / 1000.0

            # PV systems: reactive power from power factor
            if q_mvar == 0 and dss_gen.pf != 0 and abs(dss_gen.pf) < 1.0:
                pf = abs(dss_gen.pf)
                q_mvar = p_mw * math.sqrt(1 - pf**2) / pf

            pp.create_sgen(
                self._net,
                bus=bus,
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=name,
                type=dss_gen.gen_type,
                in_service=dss_gen.enabled,
            )

    # ── Transformers ──────────────────────────────────────────────────────────

    def _create_transformers(self) -> None:
        """Convert DSS 2-winding transformers to pandapower transformers."""
        for name, dss_xfmr in self._circuit.transformers.items():
            if not dss_xfmr.enabled:
                continue

            if len(dss_xfmr.buses) < 2:
                self._skip(f"Transformer '{name}'", "fewer than 2 buses")
                continue

            hv_bus = self._bus_idx(dss_xfmr.buses[0])
            lv_bus = self._bus_idx(dss_xfmr.buses[1])
            if hv_bus is None or lv_bus is None:
                self._skip(f"Transformer '{name}'", "missing bus")
                continue

            # Voltage levels
            hv_kv = dss_xfmr.kvs[0] if len(dss_xfmr.kvs) > 0 else self._circuit.base_kv
            lv_kv = dss_xfmr.kvs[1] if len(dss_xfmr.kvs) > 1 else self._circuit.base_kv

            # Rated power (use winding 1 rating)
            sn_mva = (dss_xfmr.kvas[0] / 1000.0) if dss_xfmr.kvas else 1.0

            # Short-circuit voltage (xhl is given in %, pandapower wants %)
            vk_percent  = dss_xfmr.xhl if dss_xfmr.xhl > 0 else 4.0
            vkr_percent = 1.0  # resistive part — typical distribution transformer

            try:
                pp.create_transformer_from_parameters(
                    self._net,
                    hv_bus=hv_bus,
                    lv_bus=lv_bus,
                    sn_mva=sn_mva,
                    vn_hv_kv=hv_kv,
                    vn_lv_kv=lv_kv,
                    vk_percent=vk_percent,
                    vkr_percent=vkr_percent,
                    pfe_kw=0.0,
                    i0_percent=0.0,
                    name=name,
                    in_service=dss_xfmr.enabled,
                )
            except Exception as exc:
                self._skip(f"Transformer '{name}'", str(exc))

    # ── Capacitors ────────────────────────────────────────────────────────────

    def _create_capacitors(self) -> None:
        """Convert DSS capacitors to pandapower shunt elements."""
        for name, dss_cap in self._circuit.capacitors.items():
            if not dss_cap.enabled:
                continue

            bus = self._bus_idx(dss_cap.bus1)
            if bus is None:
                self._skip(f"Capacitor '{name}'", "missing bus")
                continue

            q_mvar = dss_cap.kvar / 1000.0  # capacitive → positive in DSS convention

            pp.create_shunt(
                self._net,
                bus=bus,
                p_mw=0.0,
                q_mvar=-q_mvar,  # pandapower: negative = capacitive
                name=name,
                in_service=dss_cap.enabled,
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _warn(self, msg: str) -> None:
        self._warnings.append(msg)
        if self.verbose:
            logger.warning(msg)

    def _skip(self, element: str, reason: str) -> None:
        msg = f"{element}: {reason}"
        self._skipped.append(msg)
        if self.verbose:
            logger.warning("Skipped %s", msg)

    def _result(self) -> ConversionResult:
        net = self._net
        return ConversionResult(
            net=net,
            bus_map=dict(self._bus_map),
            n_buses=len(net.bus),
            n_lines=len(net.line),
            n_loads=len(net.load),
            n_generators=len(net.sgen),
            n_transformers=len(net.trafo),
            n_capacitors=len(net.shunt),
            warnings=list(self._warnings),
            skipped=list(self._skipped),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def convert_opendss(
    circuit: DSSCircuit,
    *,
    freq_hz: float = 60.0,
    verbose: bool = False,
) -> ConversionResult:
    """
    One-shot conversion from DSSCircuit to pandapowerNet.

    Args:
        circuit : DSSCircuit from OpenDSSParser.parse_file() or parse_smartds()
        freq_hz : System frequency — 60 Hz (US/Brazil) or 50 Hz (Chile/Europe)
        verbose : Log conversion details

    Returns:
        ConversionResult.net  →  pandapowerNet ready for pp.runpp()

    Example:
        from src.smartds.opendss_parser import parse_smartds
        from src.smartds.opendss_to_pandapower import convert_opendss

        circuit = parse_smartds("data/smartds/P1R/")
        result  = convert_opendss(circuit, freq_hz=60.0, verbose=True)

        print(result.summary)
        # → Conversion summary
        #     Buses        : 1284
        #     Lines        : 1296
        #     Loads        : 781
        #     Generators   : 47
        #     Transformers : 12
        #     Capacitors   : 8

        import pandapower as pp
        pp.runpp(result.net)
        print(result.net.res_bus.vm_pu.describe())
    """
    return OpenDSSToPandaPower(freq_hz=freq_hz, verbose=verbose).convert(circuit)
