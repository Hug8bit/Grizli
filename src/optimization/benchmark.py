"""
Benchmarking tools for comparing network states before and after optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
import pandapower as pp

from .reconfiguration import ReconfigurationOptimizer, OptimizationResult


@dataclass
class NetworkState:
    """Snapshot of network state at a point in time."""
    timestamp: float
    converged: bool
    total_generation_mw: float
    total_load_mw: float
    total_losses_mw: float
    losses_percent: float
    min_voltage_pu: float
    max_voltage_pu: float
    max_loading_percent: float
    num_overloaded_lines: int
    num_congested_lines: int
    num_voltage_violations: int
    congestion_metric: float
    line_loading: List[float] = field(default_factory=list)
    bus_voltage: List[float] = field(default_factory=list)

    @classmethod
    def from_network(cls, net: pp.pandapowerNet) -> "NetworkState":
        """Create NetworkState from current network state."""
        timestamp = time.time()

        try:
            pp.runpp(net, numba=True)
            converged = True
        except Exception:
            converged = False

        if not converged or 'res_bus' not in net:
            return cls(
                timestamp=timestamp,
                converged=False,
                total_generation_mw=0,
                total_load_mw=0,
                total_losses_mw=0,
                losses_percent=0,
                min_voltage_pu=1,
                max_voltage_pu=1,
                max_loading_percent=0,
                num_overloaded_lines=0,
                num_congested_lines=0,
                num_voltage_violations=0,
                congestion_metric=float('inf')
            )

        # Calculate metrics
        total_gen = 0.0
        if 'res_ext_grid' in net and len(net.res_ext_grid) > 0:
            total_gen += net.res_ext_grid.p_mw.sum()
        if len(net.sgen) > 0:
            total_gen += net.sgen.p_mw.sum()
        if len(net.gen) > 0:
            total_gen += net.gen.p_mw.sum()

        total_load = net.load.p_mw.sum() if len(net.load) > 0 else 0
        total_losses = net.res_line.pl_mw.sum() if 'res_line' in net else 0
        losses_percent = total_losses / total_gen * 100 if total_gen > 0 else 0

        voltage = net.res_bus.vm_pu.values
        loading = net.res_line.loading_percent.values if 'res_line' in net else np.array([])

        min_v = float(voltage.min())
        max_v = float(voltage.max())
        max_loading = float(loading.max()) if len(loading) > 0 else 0

        num_overloaded = int((loading > 100).sum())
        num_congested = int((loading > 80).sum())
        num_v_violations = int(((voltage < 0.95) | (voltage > 1.05)).sum())

        # Congestion metric
        overload = np.maximum(loading - 100, 0)
        congestion_metric = float(np.sum(overload ** 2))

        return cls(
            timestamp=timestamp,
            converged=converged,
            total_generation_mw=float(total_gen),
            total_load_mw=float(total_load),
            total_losses_mw=float(total_losses),
            losses_percent=float(losses_percent),
            min_voltage_pu=min_v,
            max_voltage_pu=max_v,
            max_loading_percent=max_loading,
            num_overloaded_lines=num_overloaded,
            num_congested_lines=num_congested,
            num_voltage_violations=num_v_violations,
            congestion_metric=congestion_metric,
            line_loading=loading.tolist(),
            bus_voltage=voltage.tolist()
        )


@dataclass
class BenchmarkResult:
    """Result of benchmark comparison."""
    before: NetworkState
    after: NetworkState
    optimization_result: Optional[OptimizationResult]
    congestion_reduction_percent: float
    losses_reduction_mw: float
    losses_reduction_percent: float
    loading_reduction_percent: float
    overload_eliminated: int
    voltage_violations_change: int
    improvement_score: float

    @property
    def is_improvement(self) -> bool:
        """Check if optimization improved the network."""
        return (
            self.congestion_reduction_percent > 0 or
            self.overload_eliminated > 0
        )


class Benchmark:
    """
    Benchmarking tool for network optimization.

    Compares network state before and after optimization
    and computes improvement metrics.
    """

    def __init__(self, net: pp.pandapowerNet):
        """
        Initialize benchmark with network.

        Args:
            net: PandaPower network to benchmark
        """
        self.net = net
        self._baseline: Optional[NetworkState] = None

    def capture_baseline(self) -> NetworkState:
        """Capture current network state as baseline."""
        self._baseline = NetworkState.from_network(self.net)
        return self._baseline

    def compare_to_baseline(self) -> BenchmarkResult:
        """
        Compare current state to baseline.

        Returns:
            BenchmarkResult comparing current state to baseline
        """
        if self._baseline is None:
            raise ValueError("No baseline captured. Call capture_baseline() first.")

        current = NetworkState.from_network(self.net)
        return self._create_benchmark_result(self._baseline, current, None)

    def run_benchmark(
        self,
        optimizer: ReconfigurationOptimizer,
        time_limit_s: float = 10.0,
        max_switches: int = 5
    ) -> BenchmarkResult:
        """
        Run complete benchmark with optimization.

        Args:
            optimizer: Optimizer to use
            time_limit_s: Time limit for optimization
            max_switches: Maximum switching operations

        Returns:
            BenchmarkResult with full comparison
        """
        # Capture before state
        before = NetworkState.from_network(self.net)

        # Store original line status
        original_status = self.net.line.in_service.copy()

        # Run optimization
        optimizer.time_limit_s = time_limit_s
        opt_result = optimizer.optimize(max_switches=max_switches)

        # Apply optimization
        if opt_result.is_successful:
            for idx in opt_result.lines_to_open:
                self.net.line.at[idx, 'in_service'] = False
            for idx in opt_result.lines_to_close:
                self.net.line.at[idx, 'in_service'] = True

        # Capture after state
        after = NetworkState.from_network(self.net)

        # Restore original state
        self.net.line.in_service = original_status

        return self._create_benchmark_result(before, after, opt_result)

    def _create_benchmark_result(
        self,
        before: NetworkState,
        after: NetworkState,
        opt_result: Optional[OptimizationResult]
    ) -> BenchmarkResult:
        """Create BenchmarkResult from before/after states."""
        # Calculate improvements
        if before.congestion_metric > 0:
            congestion_reduction = (before.congestion_metric - after.congestion_metric) / before.congestion_metric * 100
        else:
            congestion_reduction = 0.0

        losses_reduction_mw = before.total_losses_mw - after.total_losses_mw
        if before.total_losses_mw > 0:
            losses_reduction_pct = losses_reduction_mw / before.total_losses_mw * 100
        else:
            losses_reduction_pct = 0.0

        loading_reduction = before.max_loading_percent - after.max_loading_percent
        overload_eliminated = before.num_overloaded_lines - after.num_overloaded_lines
        v_violations_change = after.num_voltage_violations - before.num_voltage_violations

        # Composite improvement score (0-100)
        score = 0.0
        if before.congestion_metric > 0:
            score += min(50, congestion_reduction * 0.5)
        if before.num_overloaded_lines > 0:
            score += overload_eliminated / before.num_overloaded_lines * 30
        if before.total_losses_mw > 0:
            score += min(20, losses_reduction_pct * 0.5)

        return BenchmarkResult(
            before=before,
            after=after,
            optimization_result=opt_result,
            congestion_reduction_percent=congestion_reduction,
            losses_reduction_mw=losses_reduction_mw,
            losses_reduction_percent=losses_reduction_pct,
            loading_reduction_percent=loading_reduction,
            overload_eliminated=overload_eliminated,
            voltage_violations_change=v_violations_change,
            improvement_score=min(100, max(0, score))
        )

    def get_kpi_dataframe(self, result: BenchmarkResult) -> pd.DataFrame:
        """
        Get KPIs as DataFrame for display.

        Args:
            result: BenchmarkResult to format

        Returns:
            DataFrame with before/after/change columns
        """
        metrics = {
            'Converged': (result.before.converged, result.after.converged, '-'),
            'Total Generation (MW)': (
                round(result.before.total_generation_mw, 2),
                round(result.after.total_generation_mw, 2),
                '-'
            ),
            'Total Load (MW)': (
                round(result.before.total_load_mw, 2),
                round(result.after.total_load_mw, 2),
                '-'
            ),
            'Total Losses (kW)': (
                round(result.before.total_losses_mw * 1000, 1),
                round(result.after.total_losses_mw * 1000, 1),
                f"{round(result.losses_reduction_mw * 1000, 1):+.1f}"
            ),
            'Losses (%)': (
                round(result.before.losses_percent, 2),
                round(result.after.losses_percent, 2),
                f"{round(-result.losses_reduction_percent, 1):+.1f}%"
            ),
            'Max Line Loading (%)': (
                round(result.before.max_loading_percent, 1),
                round(result.after.max_loading_percent, 1),
                f"{round(-result.loading_reduction_percent, 1):+.1f}"
            ),
            'Overloaded Lines': (
                result.before.num_overloaded_lines,
                result.after.num_overloaded_lines,
                f"{-result.overload_eliminated:+d}"
            ),
            'Congested Lines (>80%)': (
                result.before.num_congested_lines,
                result.after.num_congested_lines,
                f"{result.after.num_congested_lines - result.before.num_congested_lines:+d}"
            ),
            'Voltage Range (pu)': (
                f"{result.before.min_voltage_pu:.3f}-{result.before.max_voltage_pu:.3f}",
                f"{result.after.min_voltage_pu:.3f}-{result.after.max_voltage_pu:.3f}",
                '-'
            ),
            'Voltage Violations': (
                result.before.num_voltage_violations,
                result.after.num_voltage_violations,
                f"{result.voltage_violations_change:+d}"
            ),
            'Congestion Metric': (
                round(result.before.congestion_metric, 1),
                round(result.after.congestion_metric, 1),
                f"{round(-result.congestion_reduction_percent, 1):+.1f}%"
            ),
        }

        df = pd.DataFrame(
            [(k, v[0], v[1], v[2]) for k, v in metrics.items()],
            columns=['Metric', 'Before', 'After', 'Change']
        )
        return df

    def print_benchmark_report(self, result: BenchmarkResult) -> None:
        """Print formatted benchmark report to console."""
        print("\n" + "="*60)
        print("BENCHMARK REPORT")
        print("="*60)

        if result.optimization_result:
            opt = result.optimization_result
            print(f"\nOptimization Status: {opt.status.value}")
            print(f"Computation Time: {opt.computation_time_s:.2f} s")
            print(f"Switching Operations: {opt.num_switches}")
            if opt.lines_to_open:
                print(f"  Lines Opened: {opt.lines_to_open}")
            if opt.lines_to_close:
                print(f"  Lines Closed: {opt.lines_to_close}")

        print("\n" + "-"*60)
        print(f"{'Metric':<30} {'Before':>12} {'After':>12} {'Change':>12}")
        print("-"*60)

        df = self.get_kpi_dataframe(result)
        for _, row in df.iterrows():
            print(f"{row['Metric']:<30} {str(row['Before']):>12} {str(row['After']):>12} {str(row['Change']):>12}")

        print("-"*60)
        print(f"\nImprovement Score: {result.improvement_score:.1f}/100")

        if result.is_improvement:
            print("Status: IMPROVED")
        else:
            print("Status: NO IMPROVEMENT")

        print("="*60 + "\n")


def quick_benchmark(
    net: pp.pandapowerNet,
    max_switches: int = 3,
    time_limit_s: float = 5.0
) -> BenchmarkResult:
    """
    Quick benchmark with default settings.

    Args:
        net: Network to benchmark
        max_switches: Maximum switches
        time_limit_s: Time limit

    Returns:
        BenchmarkResult
    """
    benchmark = Benchmark(net)
    optimizer = ReconfigurationOptimizer(net, time_limit_s=time_limit_s)
    return benchmark.run_benchmark(optimizer, max_switches=max_switches)


def compare_optimizers(
    net: pp.pandapowerNet,
    optimizers: Dict[str, ReconfigurationOptimizer],
    max_switches: int = 3
) -> pd.DataFrame:
    """
    Compare multiple optimizers on the same network.

    Args:
        net: Network to test
        optimizers: Dictionary of name -> optimizer
        max_switches: Maximum switches for all

    Returns:
        DataFrame comparing results
    """
    results = []

    for name, optimizer in optimizers.items():
        benchmark = Benchmark(net)
        result = benchmark.run_benchmark(optimizer, max_switches=max_switches)

        results.append({
            'Optimizer': name,
            'Status': result.optimization_result.status.value if result.optimization_result else 'N/A',
            'Time (s)': round(result.optimization_result.computation_time_s, 3) if result.optimization_result else 0,
            'Switches': result.optimization_result.num_switches if result.optimization_result else 0,
            'Congestion Reduction (%)': round(result.congestion_reduction_percent, 1),
            'Losses Reduction (%)': round(result.losses_reduction_percent, 1),
            'Overloads Eliminated': result.overload_eliminated,
            'Score': round(result.improvement_score, 1)
        })

    return pd.DataFrame(results)
