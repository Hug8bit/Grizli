"""
Power flow simulation and congestion analysis.
Uses PandaPower for AC power flow calculations.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import pandapower as pp


# Voltage limits (per unit)
V_MIN_PU = 0.95
V_MAX_PU = 1.05

# Loading thresholds (percent)
LOADING_LOW = 50.0
LOADING_MEDIUM = 70.0
LOADING_HIGH = 90.0
LOADING_CRITICAL = 100.0


class CongestionLevel(Enum):
    """Congestion severity level for a line."""
    NONE = "none"           # < 50%
    LOW = "low"             # 50-70%
    MEDIUM = "medium"       # 70-90%
    HIGH = "high"           # 90-100%
    CRITICAL = "critical"   # > 100% (overloaded)


@dataclass
class CongestionInfo:
    """Detailed congestion information for a single line."""
    line_idx: int
    from_bus: int
    to_bus: int
    loading_percent: float
    current_ka: float
    max_current_ka: float
    level: CongestionLevel
    name: str = ""

    @property
    def is_overloaded(self) -> bool:
        """Check if line is overloaded (>100%)."""
        return self.loading_percent > LOADING_CRITICAL

    @property
    def margin_percent(self) -> float:
        """Available margin before overload."""
        return max(0, LOADING_CRITICAL - self.loading_percent)


@dataclass
class VoltageStabilityResult:
    """Voltage stability assessment results per bus."""
    bus_indices: np.ndarray
    stability_index_d: np.ndarray  # Energy-function-based D index per load bus
    weakest_bus: int               # Bus index with D closest to -1
    stability_margin: float        # Min D across all load buses (-1 = collapse)

    @property
    def is_voltage_stable(self) -> bool:
        """Check if all buses are voltage-stable (D > -1)."""
        return float(self.stability_margin) > -1.0


@dataclass
class PowerFlowResult:
    """Complete power flow calculation results."""
    converged: bool
    iterations: int
    voltage_pu: np.ndarray
    voltage_angle_deg: np.ndarray
    line_loading_percent: np.ndarray
    line_current_ka: np.ndarray
    line_losses_mw: np.ndarray
    total_losses_mw: float
    total_generation_mw: float
    total_load_mw: float
    max_loading_percent: float
    min_voltage_pu: float
    max_voltage_pu: float
    num_overloaded_lines: int
    num_voltage_violations: int
    congestion_metric: float
    # Total Voltage Deviation: continuous quality metric
    total_voltage_deviation: float = 0.0
    # Voltage stability assessment
    voltage_stability: Optional[VoltageStabilityResult] = None

    @property
    def is_secure(self) -> bool:
        """Check if system is in secure operating state."""
        return (
            self.converged and
            self.num_overloaded_lines == 0 and
            self.num_voltage_violations == 0
        )


class PowerFlowSimulator:
    """
    Power flow simulation and analysis engine.

    Provides AC power flow calculation, congestion detection,
    and various network analysis metrics.
    """

    def __init__(
        self,
        net: pp.pandapowerNet,
        algorithm: str = "nr"
    ):
        """
        Initialize the power flow simulator.

        Args:
            net: PandaPower network
            algorithm: Power flow algorithm
                      - "nr": Newton-Raphson (default, most robust)
                      - "bfsw": Backward-Forward Sweep (radial networks)
                      - "gs": Gauss-Seidel
                      - "fdbx": Fast-Decoupled (BX)
                      - "fdxb": Fast-Decoupled (XB)
        """
        self.net = net
        self.algorithm = algorithm
        self._last_result: Optional[PowerFlowResult] = None

    def run_power_flow(
        self,
        enforce_q_lims: bool = False,
        calculate_voltage_angles: bool = True
    ) -> PowerFlowResult:
        """
        Run AC power flow calculation.

        Args:
            enforce_q_lims: Enforce reactive power limits on generators
            calculate_voltage_angles: Calculate voltage angles

        Returns:
            PowerFlowResult with all metrics
        """
        try:
            pp.runpp(
                self.net,
                algorithm=self.algorithm,
                enforce_q_lims=enforce_q_lims,
                calculate_voltage_angles=calculate_voltage_angles,
                numba=True
            )
            converged = True
            iterations = self.net._ppc.get('iterations', 0) if hasattr(self.net, '_ppc') else 0
        except pp.LoadflowNotConverged:
            converged = False
            iterations = 0
        except Exception:
            converged = False
            iterations = 0

        # Extract results
        if converged and 'res_bus' in self.net and len(self.net.res_bus) > 0:
            voltage_pu = self.net.res_bus.vm_pu.values.copy()
            voltage_angle = self.net.res_bus.va_degree.values.copy()
        else:
            voltage_pu = np.ones(len(self.net.bus))
            voltage_angle = np.zeros(len(self.net.bus))

        if converged and 'res_line' in self.net and len(self.net.res_line) > 0:
            line_loading = self.net.res_line.loading_percent.values.copy()
            line_current = self.net.res_line.i_ka.values.copy() if 'i_ka' in self.net.res_line.columns else np.zeros(len(self.net.line))
            line_losses = self.net.res_line.pl_mw.values.copy()
        else:
            line_loading = np.zeros(len(self.net.line))
            line_current = np.zeros(len(self.net.line))
            line_losses = np.zeros(len(self.net.line))

        # Calculate metrics
        total_losses = float(line_losses.sum()) if converged else 0.0

        # Total generation (ext_grid + sgen + gen)
        total_gen = 0.0
        if converged:
            if 'res_ext_grid' in self.net and len(self.net.res_ext_grid) > 0:
                total_gen += self.net.res_ext_grid.p_mw.sum()
            if 'res_sgen' in self.net and len(self.net.res_sgen) > 0:
                total_gen += self.net.sgen.p_mw.sum()  # Use scheduled, not result
            if 'res_gen' in self.net and len(self.net.res_gen) > 0:
                total_gen += self.net.res_gen.p_mw.sum()

        total_load = float(self.net.load.p_mw.sum()) if len(self.net.load) > 0 else 0.0

        max_loading = float(line_loading.max()) if len(line_loading) > 0 else 0.0
        min_voltage = float(voltage_pu.min())
        max_voltage = float(voltage_pu.max())

        num_overloaded = int((line_loading > LOADING_CRITICAL).sum())
        num_voltage_violations = int(
            ((voltage_pu < V_MIN_PU) | (voltage_pu > V_MAX_PU)).sum()
        )

        congestion_metric = self._compute_congestion_metric(line_loading)

        # Total Voltage Deviation (continuous quality metric)
        tvd = self._compute_tvd(voltage_pu) if converged else 0.0

        # Voltage stability index D (energy-function-based)
        voltage_stability = None
        if converged:
            voltage_stability = self._compute_voltage_stability(
                voltage_pu, voltage_angle
            )

        result = PowerFlowResult(
            converged=converged,
            iterations=iterations,
            voltage_pu=voltage_pu,
            voltage_angle_deg=voltage_angle,
            line_loading_percent=line_loading,
            line_current_ka=line_current,
            line_losses_mw=line_losses,
            total_losses_mw=total_losses,
            total_generation_mw=total_gen,
            total_load_mw=total_load,
            max_loading_percent=max_loading,
            min_voltage_pu=min_voltage,
            max_voltage_pu=max_voltage,
            num_overloaded_lines=num_overloaded,
            num_voltage_violations=num_voltage_violations,
            congestion_metric=congestion_metric,
            total_voltage_deviation=tvd,
            voltage_stability=voltage_stability
        )

        self._last_result = result
        return result

    def _compute_congestion_metric(self, loading: np.ndarray) -> float:
        """
        Compute congestion metric (optimization objective).

        Uses quadratic penalty for overload: sum(max(loading - 100, 0)^2)
        This penalizes severe overloads more than minor ones.
        """
        overload = np.maximum(loading - LOADING_CRITICAL, 0)
        return float(np.sum(overload ** 2))

    @staticmethod
    def _compute_tvd(voltage_pu: np.ndarray) -> float:
        """
        Compute Total Voltage Deviation (TVD).

        Continuous metric: TVD = sum(|V_i - 1.0|) for all buses.
        Lower is better. 0.0 = all buses at nominal voltage.

        Reference: Hachemi et al., Energy Reports 12 (2024) 1623-1637.
        """
        return float(np.sum(np.abs(voltage_pu - 1.0)))

    def _compute_voltage_stability(
        self,
        voltage_pu: np.ndarray,
        voltage_angle_deg: np.ndarray
    ) -> Optional[VoltageStabilityResult]:
        """
        Compute energy-function-based voltage stability index D per load bus.

        D = E / Delta_E where:
        - E is the energy at current operating point
        - Delta_E is the energy distance to voltage collapse

        Interpretation:
        - -1 < D < 0 : stable
        - D = -1     : destabilization threshold
        - D < -1     : unstable (collapse imminent)

        Reference: Zhang et al., Energy Reports 12 (2024) 699-707.
        """
        net = self.net
        n_bus = len(net.bus)

        if n_bus == 0:
            return None

        # Identify load buses (non-slack, non-generator)
        # Slack buses are ext_grid buses
        slack_buses = set(net.ext_grid.bus.values) if len(net.ext_grid) > 0 else set()
        gen_buses = set(net.gen.bus.values) if len(net.gen) > 0 else set()
        all_gen_buses = slack_buses | gen_buses

        load_bus_mask = np.array([
            i not in all_gen_buses for i in range(n_bus)
        ])
        load_bus_indices = np.where(load_bus_mask)[0]

        if len(load_bus_indices) == 0:
            return VoltageStabilityResult(
                bus_indices=np.array([], dtype=int),
                stability_index_d=np.array([]),
                weakest_bus=0,
                stability_margin=0.0
            )

        # Get bus admittance matrix from pandapower internal data
        try:
            Ybus = net._ppc['internal']['Ybus'].toarray()
        except (KeyError, AttributeError):
            # Fallback: build approximate Ybus from line data
            Ybus = self._build_approx_ybus(net)

        V = voltage_pu.copy()
        theta = np.deg2rad(voltage_angle_deg.copy())

        # Reference operating point (flat start)
        V0 = np.ones(n_bus)
        theta0 = np.zeros(n_bus)

        G = Ybus.real
        B = Ybus.imag

        # Get load power at each bus
        P_load = np.zeros(n_bus)
        Q_load = np.zeros(n_bus)
        if len(net.load) > 0:
            for idx in net.load.index:
                bus = net.load.at[idx, 'bus']
                if bus < n_bus:
                    P_load[bus] += net.load.at[idx, 'p_mw']
                    Q_load[bus] += net.load.at[idx, 'q_mvar']

        d_indices = []
        d_values = []

        for bus_i in load_bus_indices:
            # Energy function at current operating point (Eq. 26 simplified)
            E_current = 0.0
            # Active power component
            E_current += P_load[bus_i] * (theta[bus_i] - theta0[bus_i])
            # Reactive power component (logarithmic voltage term)
            if V[bus_i] > 0 and V0[bus_i] > 0:
                E_current -= Q_load[bus_i] * np.log(V[bus_i] / V0[bus_i])

            # Admittance coupling terms
            for bus_j in range(n_bus):
                if bus_j == bus_i:
                    continue
                E_current += 0.5 * B[bus_i, bus_j] * V0[bus_i] * V0[bus_j] * \
                    np.cos(theta0[bus_i] - theta0[bus_j])
                E_current -= 0.5 * B[bus_i, bus_j] * V[bus_i] * V[bus_j] * \
                    np.cos(theta[bus_i] - theta[bus_j])

            # Critical energy (at voltage collapse point)
            # Approximate critical voltage: V_cr = V_source / sqrt(2*(1+cos(theta_z)))
            # For distribution networks, theta_z ~= pi/2, so V_cr ~= V_source / sqrt(2)
            V_cr = V0[bus_i] / np.sqrt(2.0)
            theta_cr = theta[bus_i] * 1.5  # Approximate critical angle

            E_critical = 0.0
            E_critical += P_load[bus_i] * (theta_cr - theta0[bus_i])
            if V_cr > 0 and V0[bus_i] > 0:
                E_critical -= Q_load[bus_i] * np.log(V_cr / V0[bus_i])

            for bus_j in range(n_bus):
                if bus_j == bus_i:
                    continue
                E_critical += 0.5 * B[bus_i, bus_j] * V0[bus_i] * V0[bus_j] * \
                    np.cos(theta0[bus_i] - theta0[bus_j])
                E_critical -= 0.5 * B[bus_i, bus_j] * V_cr * V[bus_j] * \
                    np.cos(theta_cr - theta[bus_j])

            # D = E / Delta_E
            delta_E = E_critical - E_current
            if abs(delta_E) > 1e-10:
                D = -abs(E_current / delta_E)
            else:
                D = 0.0  # At reference point, fully stable

            d_indices.append(bus_i)
            d_values.append(D)

        d_indices = np.array(d_indices, dtype=int)
        d_values = np.array(d_values)

        if len(d_values) > 0:
            weakest_idx = np.argmin(d_values)
            weakest_bus = int(d_indices[weakest_idx])
            stability_margin = float(d_values[weakest_idx])
        else:
            weakest_bus = 0
            stability_margin = 0.0

        return VoltageStabilityResult(
            bus_indices=d_indices,
            stability_index_d=d_values,
            weakest_bus=weakest_bus,
            stability_margin=stability_margin
        )

    @staticmethod
    def _build_approx_ybus(net: pp.pandapowerNet) -> np.ndarray:
        """Build approximate bus admittance matrix from line data."""
        n_bus = len(net.bus)
        Ybus = np.zeros((n_bus, n_bus), dtype=complex)

        for idx in net.line.index:
            if not net.line.at[idx, 'in_service']:
                continue
            fb = net.line.at[idx, 'from_bus']
            tb = net.line.at[idx, 'to_bus']
            r = net.line.at[idx, 'r_ohm_per_km'] * net.line.at[idx, 'length_km']
            x = net.line.at[idx, 'x_ohm_per_km'] * net.line.at[idx, 'length_km']
            if abs(r) + abs(x) < 1e-12:
                continue
            z = complex(r, x)
            y = 1.0 / z
            Ybus[fb, fb] += y
            Ybus[tb, tb] += y
            Ybus[fb, tb] -= y
            Ybus[tb, fb] -= y

        return Ybus

    def get_congestion_details(self) -> List[CongestionInfo]:
        """
        Get detailed congestion info for all lines.

        Returns:
            List of CongestionInfo for each line
        """
        result = self._last_result
        if result is None:
            result = self.run_power_flow()

        details = []
        for idx in range(len(self.net.line)):
            loading = result.line_loading_percent[idx]

            # Determine congestion level
            if loading < LOADING_LOW:
                level = CongestionLevel.NONE
            elif loading < LOADING_MEDIUM:
                level = CongestionLevel.LOW
            elif loading < LOADING_HIGH:
                level = CongestionLevel.MEDIUM
            elif loading < LOADING_CRITICAL:
                level = CongestionLevel.HIGH
            else:
                level = CongestionLevel.CRITICAL

            details.append(CongestionInfo(
                line_idx=idx,
                from_bus=int(self.net.line.at[idx, 'from_bus']),
                to_bus=int(self.net.line.at[idx, 'to_bus']),
                loading_percent=float(loading),
                current_ka=float(result.line_current_ka[idx]),
                max_current_ka=float(self.net.line.at[idx, 'max_i_ka']),
                level=level,
                name=str(self.net.line.at[idx, 'name']) if 'name' in self.net.line.columns else f"Line_{idx}"
            ))

        return details

    def get_overloaded_lines(self) -> List[CongestionInfo]:
        """Get only overloaded lines (>100% loading)."""
        return [c for c in self.get_congestion_details() if c.is_overloaded]

    def get_congested_lines(self, threshold: float = LOADING_HIGH) -> List[CongestionInfo]:
        """Get lines above specified loading threshold."""
        return [c for c in self.get_congestion_details()
                if c.loading_percent >= threshold]

    def compute_congestion_metric(self) -> float:
        """Compute and return the congestion metric."""
        if self._last_result is None:
            self.run_power_flow()
        return self._last_result.congestion_metric

    def get_losses_breakdown(self) -> Dict:
        """
        Get breakdown of network losses.

        Returns:
            Dictionary with loss statistics
        """
        result = self._last_result
        if result is None:
            result = self.run_power_flow()

        return {
            "total_losses_mw": result.total_losses_mw,
            "total_losses_kw": result.total_losses_mw * 1000,
            "losses_percent": (
                result.total_losses_mw / result.total_generation_mw * 100
                if result.total_generation_mw > 0 else 0
            ),
            "line_losses_mw": result.line_losses_mw.tolist(),
            "max_line_loss_mw": float(result.line_losses_mw.max()) if len(result.line_losses_mw) > 0 else 0,
        }

    def get_voltage_profile(self) -> pd.DataFrame:
        """
        Get voltage profile for all buses.

        Returns:
            DataFrame with bus voltage information
        """
        result = self._last_result
        if result is None:
            result = self.run_power_flow()

        df = pd.DataFrame({
            'bus_id': range(len(self.net.bus)),
            'voltage_pu': result.voltage_pu,
            'voltage_angle_deg': result.voltage_angle_deg,
            'voltage_kv': result.voltage_pu * self.net.bus.vn_kv.values,
            'voltage_deviation': np.abs(result.voltage_pu - 1.0),
            'under_voltage': result.voltage_pu < V_MIN_PU,
            'over_voltage': result.voltage_pu > V_MAX_PU,
        })

        # Add per-bus stability index if available
        if result.voltage_stability is not None:
            vs = result.voltage_stability
            d_map = dict(zip(vs.bus_indices, vs.stability_index_d))
            df['stability_index_d'] = [
                d_map.get(i, np.nan) for i in range(len(self.net.bus))
            ]

        if 'name' in self.net.bus.columns:
            df['name'] = self.net.bus.name.values

        return df

    def check_n1_contingency(self, line_idx: int) -> Tuple[bool, Optional[PowerFlowResult]]:
        """
        Check N-1 contingency for a specific line.

        Temporarily removes the line and runs power flow.

        Args:
            line_idx: Index of line to remove

        Returns:
            Tuple of (is_secure, PowerFlowResult or None if failed)
        """
        # Save original state
        original_status = self.net.line.at[line_idx, 'in_service']

        try:
            # Disable line
            self.net.line.at[line_idx, 'in_service'] = False

            # Run power flow
            result = self.run_power_flow()

            is_secure = result.is_secure

            return is_secure, result

        finally:
            # Restore original state
            self.net.line.at[line_idx, 'in_service'] = original_status
            # Re-run to restore last_result
            self._last_result = None

    def run_n1_analysis(self) -> Dict[int, Tuple[bool, Optional[PowerFlowResult]]]:
        """
        Run N-1 contingency analysis for all lines.

        Returns:
            Dictionary mapping line_idx to (is_secure, result)
        """
        results = {}
        for idx in range(len(self.net.line)):
            if self.net.line.at[idx, 'in_service']:
                results[idx] = self.check_n1_contingency(idx)
        return results

    def get_summary(self) -> Dict:
        """
        Get summary of power flow results.

        Returns:
            Dictionary with key metrics
        """
        result = self._last_result
        if result is None:
            result = self.run_power_flow()

        summary = {
            "converged": result.converged,
            "iterations": result.iterations,
            "is_secure": result.is_secure,
            "total_generation_mw": round(result.total_generation_mw, 2),
            "total_load_mw": round(result.total_load_mw, 2),
            "total_losses_mw": round(result.total_losses_mw, 3),
            "losses_percent": round(
                result.total_losses_mw / result.total_generation_mw * 100
                if result.total_generation_mw > 0 else 0, 2
            ),
            "max_loading_percent": round(result.max_loading_percent, 1),
            "min_voltage_pu": round(result.min_voltage_pu, 3),
            "max_voltage_pu": round(result.max_voltage_pu, 3),
            "num_overloaded_lines": result.num_overloaded_lines,
            "num_voltage_violations": result.num_voltage_violations,
            "congestion_metric": round(result.congestion_metric, 2),
            "total_voltage_deviation": round(result.total_voltage_deviation, 4),
        }

        if result.voltage_stability is not None:
            summary["voltage_stability_margin"] = round(
                result.voltage_stability.stability_margin, 4
            )
            summary["weakest_bus"] = result.voltage_stability.weakest_bus
            summary["voltage_stable"] = result.voltage_stability.is_voltage_stable

        return summary
