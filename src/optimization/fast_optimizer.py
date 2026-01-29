"""
Fast greedy optimizer for real-time network reconfiguration.
Achieves sub-second optimization for interactive applications.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import time
import numpy as np
import pandapower as pp


@dataclass
class FastOptimizationResult:
    """Result from fast optimization."""
    lines_to_open: List[int]
    lines_to_close: List[int]
    num_switches: int
    max_loading_before: float
    max_loading_after: float
    num_overloaded_before: int
    num_overloaded_after: int
    congestion_metric_before: float
    congestion_metric_after: float
    congestion_reduction_percent: float
    losses_before_kw: float
    losses_after_kw: float
    computation_time_s: float
    iterations: int
    converged: bool

    @property
    def is_improvement(self) -> bool:
        """Check if optimization improved the network."""
        return self.congestion_metric_after < self.congestion_metric_before


def _compute_congestion_metric(net: pp.pandapowerNet) -> float:
    """
    Compute congestion metric: sum of squared overloads.

    This metric penalizes severe overloads more than minor ones,
    which is desirable for optimization.
    """
    if 'res_line' not in net or len(net.res_line) == 0:
        return float('inf')
    loading = net.res_line.loading_percent.values
    overload = np.maximum(loading - 100, 0)
    return float(np.sum(overload ** 2))


def _get_network_state(net: pp.pandapowerNet) -> Tuple[float, float, int, float]:
    """Get current network state metrics."""
    try:
        pp.runpp(net, numba=True)
        max_loading = float(net.res_line.loading_percent.max())
        losses = float(net.res_line.pl_mw.sum()) * 1000  # kW
        num_overloaded = int((net.res_line.loading_percent > 100).sum())
        congestion = _compute_congestion_metric(net)
        return max_loading, losses, num_overloaded, congestion
    except Exception:
        return 0.0, 0.0, 0, float('inf')


def _find_tie_lines(net: pp.pandapowerNet) -> List[int]:
    """Find normally-open tie lines (switchable and currently open)."""
    tie_lines = []
    for idx in range(len(net.line)):
        is_switch = net.line.at[idx, 'is_switch'] if 'is_switch' in net.line.columns else True
        in_service = net.line.at[idx, 'in_service']
        if is_switch and not in_service:
            tie_lines.append(idx)
    return tie_lines


def _find_switchable_closed_lines(net: pp.pandapowerNet) -> List[int]:
    """Find switchable lines that are currently closed."""
    closed_switches = []
    for idx in range(len(net.line)):
        is_switch = net.line.at[idx, 'is_switch'] if 'is_switch' in net.line.columns else True
        in_service = net.line.at[idx, 'in_service']
        if is_switch and in_service:
            closed_switches.append(idx)
    return closed_switches


def _find_loop_lines(net: pp.pandapowerNet, closed_tie: int) -> List[int]:
    """
    Find lines that form a loop with the closed tie line.

    When a tie line is closed, it creates a loop. We need to find
    which lines are part of this loop so we can open one to restore
    radiality (if desired).
    """
    import networkx as nx

    # Build graph from current topology
    G = nx.Graph()
    for idx in range(len(net.line)):
        if net.line.at[idx, 'in_service']:
            f = int(net.line.at[idx, 'from_bus'])
            t = int(net.line.at[idx, 'to_bus'])
            G.add_edge(f, t, line_idx=idx)

    # Find cycles containing the closed tie line
    from_bus = int(net.line.at[closed_tie, 'from_bus'])
    to_bus = int(net.line.at[closed_tie, 'to_bus'])

    try:
        # Find path without the tie line
        G_temp = G.copy()
        G_temp.remove_edge(from_bus, to_bus)
        path = nx.shortest_path(G_temp, from_bus, to_bus)

        # Get line indices along the path
        loop_lines = []
        for i in range(len(path) - 1):
            edge_data = G.get_edge_data(path[i], path[i+1])
            if edge_data and 'line_idx' in edge_data:
                loop_lines.append(edge_data['line_idx'])

        return loop_lines
    except (nx.NetworkXNoPath, nx.NetworkXError):
        return []


def optimize_topology_fast(
    net: pp.pandapowerNet,
    max_iterations: int = 5,
    max_switches: int = 4,
    time_limit_s: float = 1.0
) -> FastOptimizationResult:
    """
    Fast greedy topology optimization.

    Algorithm:
    1. For each available tie line (normally open):
       a. Close the tie line
       b. For each line in the created loop:
          - Try opening it
          - Evaluate congestion metric
          - Keep best swap if it improves the metric
    2. Repeat until no improvement or limits reached

    This is a "branch exchange" heuristic commonly used in
    distribution network reconfiguration.

    Args:
        net: PandaPower network
        max_iterations: Maximum optimization iterations
        max_switches: Maximum switching operations
        time_limit_s: Time limit in seconds

    Returns:
        FastOptimizationResult with switching decisions
    """
    start_time = time.time()

    # Get initial state
    max_loading_before, losses_before, num_overloaded_before, congestion_before = _get_network_state(net)

    # Store original line status
    original_status = net.line.in_service.copy()

    # Track best solution
    best_to_open: List[int] = []
    best_to_close: List[int] = []
    best_congestion = congestion_before
    current_congestion = congestion_before

    iterations = 0
    total_switches = 0

    while iterations < max_iterations and total_switches < max_switches:
        if time.time() - start_time > time_limit_s:
            break

        iterations += 1
        improved = False

        # Find available tie lines
        tie_lines = _find_tie_lines(net)

        for tie_idx in tie_lines:
            if time.time() - start_time > time_limit_s:
                break

            # Close the tie line
            net.line.at[tie_idx, 'in_service'] = True

            # Run power flow to check if network is still valid
            try:
                pp.runpp(net, numba=True)
            except Exception:
                net.line.at[tie_idx, 'in_service'] = False
                continue

            # Find lines in the created loop
            loop_lines = _find_loop_lines(net, tie_idx)

            # Also consider all switchable closed lines as candidates
            switchable_closed = _find_switchable_closed_lines(net)
            candidate_lines = list(set(loop_lines + switchable_closed))

            best_swap_metric = current_congestion
            best_line_to_open = None

            for line_idx in candidate_lines:
                if line_idx == tie_idx:
                    continue

                # Try opening this line
                net.line.at[line_idx, 'in_service'] = False

                try:
                    pp.runpp(net, numba=True)
                    metric = _compute_congestion_metric(net)

                    if metric < best_swap_metric:
                        best_swap_metric = metric
                        best_line_to_open = line_idx
                except Exception:
                    pass

                # Restore line
                net.line.at[line_idx, 'in_service'] = True

            # Apply best swap if found
            if best_line_to_open is not None and best_swap_metric < current_congestion:
                # Keep tie line closed
                best_to_close.append(tie_idx)
                # Open the best line
                net.line.at[best_line_to_open, 'in_service'] = False
                best_to_open.append(best_line_to_open)

                current_congestion = best_swap_metric
                best_congestion = best_swap_metric
                total_switches += 2  # One open, one close
                improved = True
                break  # Restart search
            else:
                # Revert tie line
                net.line.at[tie_idx, 'in_service'] = False

        if not improved:
            break

    # Get final state
    max_loading_after, losses_after, num_overloaded_after, congestion_after = _get_network_state(net)

    # Restore original status
    net.line.in_service = original_status

    # Calculate improvement
    congestion_reduction = 0.0
    if congestion_before > 0:
        congestion_reduction = (congestion_before - best_congestion) / congestion_before * 100

    computation_time = time.time() - start_time

    return FastOptimizationResult(
        lines_to_open=best_to_open,
        lines_to_close=best_to_close,
        num_switches=len(best_to_open) + len(best_to_close),
        max_loading_before=max_loading_before,
        max_loading_after=max_loading_after,
        num_overloaded_before=num_overloaded_before,
        num_overloaded_after=num_overloaded_after,
        congestion_metric_before=congestion_before,
        congestion_metric_after=best_congestion,
        congestion_reduction_percent=congestion_reduction,
        losses_before_kw=losses_before,
        losses_after_kw=losses_after,
        computation_time_s=computation_time,
        iterations=iterations,
        converged=iterations < max_iterations
    )


def apply_optimization_result(
    net: pp.pandapowerNet,
    result: FastOptimizationResult
) -> None:
    """
    Apply optimization result to network.

    Args:
        net: PandaPower network
        result: Optimization result to apply
    """
    for idx in result.lines_to_open:
        if idx < len(net.line):
            net.line.at[idx, 'in_service'] = False

    for idx in result.lines_to_close:
        if idx < len(net.line):
            net.line.at[idx, 'in_service'] = True


def get_reconfiguration_summary(result: FastOptimizationResult) -> str:
    """
    Get human-readable summary of optimization result.

    Args:
        result: Optimization result

    Returns:
        Formatted summary string
    """
    lines = [
        "=== Fast Optimization Result ===",
        f"Computation time: {result.computation_time_s*1000:.1f} ms",
        f"Iterations: {result.iterations}",
        f"",
        f"Switching operations: {result.num_switches}",
        f"  Lines to open: {result.lines_to_open}",
        f"  Lines to close: {result.lines_to_close}",
        f"",
        f"Before optimization:",
        f"  Max loading: {result.max_loading_before:.1f}%",
        f"  Overloaded lines: {result.num_overloaded_before}",
        f"  Congestion metric: {result.congestion_metric_before:.1f}",
        f"  Losses: {result.losses_before_kw:.1f} kW",
        f"",
        f"After optimization:",
        f"  Max loading: {result.max_loading_after:.1f}%",
        f"  Overloaded lines: {result.num_overloaded_after}",
        f"  Congestion metric: {result.congestion_metric_after:.1f}",
        f"  Losses: {result.losses_after_kw:.1f} kW",
        f"",
        f"Improvement: {result.congestion_reduction_percent:.1f}% congestion reduction",
    ]
    return "\n".join(lines)
