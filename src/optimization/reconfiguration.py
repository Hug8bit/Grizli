"""
MILP-based network reconfiguration optimizer.
Uses linearized DC power flow with Big-M formulation.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import pandas as pd
import pandapower as pp

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


class OptimizationStatus(Enum):
    """Status of optimization result."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class OptimizationResult:
    """Result of reconfiguration optimization."""
    status: OptimizationStatus
    objective_value: float
    computation_time_s: float
    lines_to_open: List[int]
    lines_to_close: List[int]
    num_switches: int
    congestion_before: float
    congestion_after: float
    congestion_reduction_percent: float
    max_loading_before: float
    max_loading_after: float
    losses_before_mw: float
    losses_after_mw: float
    message: str = ""

    @property
    def is_successful(self) -> bool:
        """Check if optimization found a solution."""
        return self.status in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE)


class ReconfigurationOptimizer:
    """
    MILP optimizer for network reconfiguration.

    Uses linearized DC power flow approximation with Big-M constraints
    to optimize switching decisions while minimizing congestion and losses.

    Mathematical Formulation:
    - Variables:
      * theta_i: Voltage angle at bus i
      * P_ij: Active power flow on line (i,j)
      * y_ij: Binary, 1 if line is closed
      * s_ij: Binary, 1 if line was switched

    - Objective:
      minimize: w1 * sum(overload) + w2 * sum(|P|) + w3 * sum(switches)

    - Constraints:
      * Power balance at each bus (KCL)
      * DC power flow: P_ij = (theta_i - theta_j) / X_ij (when closed)
      * Big-M for on/off: |P_ij| <= M * y_ij
      * Thermal limits: |P_ij| <= P_max
      * Connectivity: Network must remain connected
      * Radiality (optional): No loops
    """

    def __init__(
        self,
        net: pp.pandapowerNet,
        time_limit_s: float = 60.0,
        gap_tolerance: float = 0.05,
        enforce_radiality: bool = False,
        congestion_weight: float = 100.0,
        loss_weight: float = 1.0,
        switching_penalty: float = 10.0
    ):
        """
        Initialize the optimizer.

        Args:
            net: PandaPower network
            time_limit_s: Solver time limit in seconds
            gap_tolerance: MIP gap tolerance (0.05 = 5%)
            enforce_radiality: Require radial topology (no loops)
            congestion_weight: Weight for congestion in objective
            loss_weight: Weight for losses in objective
            switching_penalty: Penalty per switching operation
        """
        if not PULP_AVAILABLE:
            raise ImportError("PuLP is required for MILP optimization. "
                            "Install with: pip install pulp")

        self.net = net
        self.time_limit_s = time_limit_s
        self.gap_tolerance = gap_tolerance
        self.enforce_radiality = enforce_radiality
        self.congestion_weight = congestion_weight
        self.loss_weight = loss_weight
        self.switching_penalty = switching_penalty

        # Big-M value (large enough but not too large for numerical stability)
        self.BIG_M = 1000.0

        # Extract network data
        self._prepare_network_data()

    def _prepare_network_data(self) -> None:
        """Extract and prepare network data for optimization."""
        self.n_bus = len(self.net.bus)
        self.n_line = len(self.net.line)

        # Bus data
        self.bus_ids = list(range(self.n_bus))

        # Find slack bus
        if len(self.net.ext_grid) > 0:
            self.slack_bus = int(self.net.ext_grid.bus.iloc[0])
        else:
            self.slack_bus = 0

        # Line data
        self.lines = []
        for idx in range(self.n_line):
            from_bus = int(self.net.line.at[idx, 'from_bus'])
            to_bus = int(self.net.line.at[idx, 'to_bus'])

            # Calculate reactance (for DC power flow)
            x_ohm = self.net.line.at[idx, 'x_ohm_per_km'] * self.net.line.at[idx, 'length_km']
            vn_kv = self.net.bus.at[from_bus, 'vn_kv']
            x_pu = x_ohm / (vn_kv ** 2 / 100)  # Per-unit on 100 MVA base
            x_pu = max(x_pu, 0.001)  # Avoid division by zero

            # Thermal limit
            max_i_ka = self.net.line.at[idx, 'max_i_ka']
            p_max_mw = max_i_ka * vn_kv * np.sqrt(3)

            # Initial status
            in_service = self.net.line.at[idx, 'in_service']

            # Is switchable?
            is_switch = self.net.line.at[idx, 'is_switch'] if 'is_switch' in self.net.line.columns else True

            self.lines.append({
                'idx': idx,
                'from': from_bus,
                'to': to_bus,
                'x_pu': x_pu,
                'p_max': p_max_mw,
                'in_service': in_service,
                'is_switch': is_switch
            })

        # Power injections at each bus
        self.p_inject = np.zeros(self.n_bus)

        # Generation (sgen)
        for idx in self.net.sgen.index:
            if self.net.sgen.at[idx, 'in_service']:
                bus = int(self.net.sgen.at[idx, 'bus'])
                self.p_inject[bus] += self.net.sgen.at[idx, 'p_mw']

        # Generation (gen)
        for idx in self.net.gen.index if len(self.net.gen) > 0 else []:
            if self.net.gen.at[idx, 'in_service']:
                bus = int(self.net.gen.at[idx, 'bus'])
                self.p_inject[bus] += self.net.gen.at[idx, 'p_mw']

        # Load (negative injection)
        for idx in self.net.load.index:
            if self.net.load.at[idx, 'in_service']:
                bus = int(self.net.load.at[idx, 'bus'])
                self.p_inject[bus] -= self.net.load.at[idx, 'p_mw']

    def optimize(self, max_switches: int = 5) -> OptimizationResult:
        """
        Run the optimization.

        Args:
            max_switches: Maximum number of switching operations allowed

        Returns:
            OptimizationResult with optimal switching decisions
        """
        import time
        start_time = time.time()

        # Get initial state
        initial_congestion = self._compute_congestion_metric()
        initial_losses = self._compute_losses()
        initial_max_loading = self._compute_max_loading()

        # Create MILP model
        model = pulp.LpProblem("NetworkReconfiguration", pulp.LpMinimize)

        # Decision variables
        # Voltage angles
        theta = {i: pulp.LpVariable(f"theta_{i}", lowBound=-np.pi, upBound=np.pi)
                for i in self.bus_ids}

        # Line power flows
        P = {line['idx']: pulp.LpVariable(f"P_{line['idx']}", lowBound=-line['p_max'], upBound=line['p_max'])
             for line in self.lines}

        # Line status (binary): 1 = closed
        y = {line['idx']: pulp.LpVariable(f"y_{line['idx']}", cat='Binary')
             for line in self.lines}

        # Overload variables (for objective)
        overload = {line['idx']: pulp.LpVariable(f"overload_{line['idx']}", lowBound=0)
                   for line in self.lines}

        # Switching indicator: 1 if status changed
        switch = {line['idx']: pulp.LpVariable(f"switch_{line['idx']}", cat='Binary')
                 for line in self.lines}

        # Objective: minimize congestion + losses + switching
        model += (
            self.congestion_weight * pulp.lpSum(overload.values()) +
            self.loss_weight * pulp.lpSum(P[line['idx']] * P[line['idx']] * line['x_pu']
                                          for line in self.lines) +
            self.switching_penalty * pulp.lpSum(switch.values())
        )

        # Constraints

        # 1. Slack bus angle = 0
        model += theta[self.slack_bus] == 0

        # 2. Power balance at each bus (KCL)
        for bus in self.bus_ids:
            if bus == self.slack_bus:
                continue  # Slack absorbs imbalance

            # Sum of flows into bus = injection
            inflow = []
            outflow = []
            for line in self.lines:
                if line['to'] == bus:
                    inflow.append(P[line['idx']])
                if line['from'] == bus:
                    outflow.append(P[line['idx']])

            if inflow or outflow:
                model += (
                    pulp.lpSum(inflow) - pulp.lpSum(outflow) == self.p_inject[bus],
                    f"KCL_bus_{bus}"
                )

        # 3. DC power flow with Big-M (linearized)
        for line in self.lines:
            idx = line['idx']
            from_bus = line['from']
            to_bus = line['to']
            b = 1.0 / line['x_pu']  # Susceptance

            # P = b * (theta_from - theta_to) when closed
            # Using Big-M relaxation:
            # P - b*(theta_from - theta_to) <= M*(1-y)
            # P - b*(theta_from - theta_to) >= -M*(1-y)

            model += (
                P[idx] - b * (theta[from_bus] - theta[to_bus]) <= self.BIG_M * (1 - y[idx]),
                f"DC_flow_upper_{idx}"
            )
            model += (
                P[idx] - b * (theta[from_bus] - theta[to_bus]) >= -self.BIG_M * (1 - y[idx]),
                f"DC_flow_lower_{idx}"
            )

            # If line open, power = 0
            model += P[idx] <= line['p_max'] * y[idx], f"P_max_upper_{idx}"
            model += P[idx] >= -line['p_max'] * y[idx], f"P_max_lower_{idx}"

        # 4. Overload definition (soft constraint)
        for line in self.lines:
            idx = line['idx']
            # overload >= |P| - P_max
            model += overload[idx] >= P[idx] - line['p_max'], f"overload_pos_{idx}"
            model += overload[idx] >= -P[idx] - line['p_max'], f"overload_neg_{idx}"

        # 5. Switching indicator
        for line in self.lines:
            idx = line['idx']
            initial_status = 1 if line['in_service'] else 0

            if line['is_switch']:
                # switch = |y - initial_status|
                model += switch[idx] >= y[idx] - initial_status, f"switch_pos_{idx}"
                model += switch[idx] >= initial_status - y[idx], f"switch_neg_{idx}"
            else:
                # Non-switchable: keep status
                model += y[idx] == initial_status, f"fixed_line_{idx}"
                model += switch[idx] == 0, f"no_switch_{idx}"

        # 6. Maximum switches
        model += pulp.lpSum(switch.values()) <= max_switches, "max_switches"

        # 7. Minimum connectivity (at least n_bus - 1 lines closed)
        model += pulp.lpSum(y.values()) >= self.n_bus - 1, "min_connectivity"

        # 8. Radiality (optional)
        if self.enforce_radiality:
            # For radiality: exactly n_bus - 1 lines closed
            model += pulp.lpSum(y.values()) == self.n_bus - 1, "radiality"

        # Solve
        solver = pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=self.time_limit_s,
            gapRel=self.gap_tolerance
        )

        try:
            model.solve(solver)
        except Exception as e:
            return OptimizationResult(
                status=OptimizationStatus.ERROR,
                objective_value=float('inf'),
                computation_time_s=time.time() - start_time,
                lines_to_open=[],
                lines_to_close=[],
                num_switches=0,
                congestion_before=initial_congestion,
                congestion_after=initial_congestion,
                congestion_reduction_percent=0,
                max_loading_before=initial_max_loading,
                max_loading_after=initial_max_loading,
                losses_before_mw=initial_losses,
                losses_after_mw=initial_losses,
                message=str(e)
            )

        computation_time = time.time() - start_time

        # Parse solution
        if model.status == pulp.LpStatusOptimal:
            status = OptimizationStatus.OPTIMAL
        elif model.status == pulp.LpStatusNotSolved:
            status = OptimizationStatus.TIMEOUT
        elif model.status == pulp.LpStatusInfeasible:
            status = OptimizationStatus.INFEASIBLE
        else:
            status = OptimizationStatus.FEASIBLE if model.status == 1 else OptimizationStatus.ERROR

        lines_to_open = []
        lines_to_close = []

        if status in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            for line in self.lines:
                idx = line['idx']
                new_status = pulp.value(y[idx])
                old_status = 1 if line['in_service'] else 0

                if new_status is not None:
                    new_status = round(new_status)
                    if new_status == 0 and old_status == 1:
                        lines_to_open.append(idx)
                    elif new_status == 1 and old_status == 0:
                        lines_to_close.append(idx)

        # Apply solution and measure results
        final_congestion = initial_congestion
        final_max_loading = initial_max_loading
        final_losses = initial_losses

        if lines_to_open or lines_to_close:
            # Temporarily apply changes
            original_status = self.net.line.in_service.copy()

            for idx in lines_to_open:
                self.net.line.at[idx, 'in_service'] = False
            for idx in lines_to_close:
                self.net.line.at[idx, 'in_service'] = True

            final_congestion = self._compute_congestion_metric()
            final_max_loading = self._compute_max_loading()
            final_losses = self._compute_losses()

            # Restore
            self.net.line.in_service = original_status

        congestion_reduction = (
            (initial_congestion - final_congestion) / initial_congestion * 100
            if initial_congestion > 0 else 0
        )

        return OptimizationResult(
            status=status,
            objective_value=pulp.value(model.objective) if model.objective else float('inf'),
            computation_time_s=computation_time,
            lines_to_open=lines_to_open,
            lines_to_close=lines_to_close,
            num_switches=len(lines_to_open) + len(lines_to_close),
            congestion_before=initial_congestion,
            congestion_after=final_congestion,
            congestion_reduction_percent=congestion_reduction,
            max_loading_before=initial_max_loading,
            max_loading_after=final_max_loading,
            losses_before_mw=initial_losses,
            losses_after_mw=final_losses,
            message=pulp.LpStatus[model.status]
        )

    def _compute_congestion_metric(self) -> float:
        """Compute current congestion metric."""
        try:
            pp.runpp(self.net, numba=True)
            loading = self.net.res_line.loading_percent.values
            overload = np.maximum(loading - 100, 0)
            return float(np.sum(overload ** 2))
        except Exception:
            return float('inf')

    def _compute_max_loading(self) -> float:
        """Compute maximum line loading."""
        try:
            if 'res_line' in self.net and len(self.net.res_line) > 0:
                return float(self.net.res_line.loading_percent.max())
            pp.runpp(self.net, numba=True)
            return float(self.net.res_line.loading_percent.max())
        except Exception:
            return 0.0

    def _compute_losses(self) -> float:
        """Compute total network losses."""
        try:
            if 'res_line' in self.net and len(self.net.res_line) > 0:
                return float(self.net.res_line.pl_mw.sum())
            pp.runpp(self.net, numba=True)
            return float(self.net.res_line.pl_mw.sum())
        except Exception:
            return 0.0


class QuickReconfigurationHeuristic:
    """
    Fast heuristic for network reconfiguration.

    Uses greedy approach to find good solutions quickly
    when MILP is too slow.
    """

    def __init__(self, net: pp.pandapowerNet):
        """Initialize with network."""
        self.net = net

    def optimize(self, max_switches: int = 3) -> OptimizationResult:
        """
        Run greedy optimization.

        Args:
            max_switches: Maximum switching operations

        Returns:
            OptimizationResult
        """
        import time
        start_time = time.time()

        # Get initial metrics
        try:
            pp.runpp(self.net, numba=True)
            initial_congestion = self._get_congestion_metric()
            initial_max_loading = float(self.net.res_line.loading_percent.max())
            initial_losses = float(self.net.res_line.pl_mw.sum())
        except Exception:
            return OptimizationResult(
                status=OptimizationStatus.ERROR,
                objective_value=float('inf'),
                computation_time_s=time.time() - start_time,
                lines_to_open=[],
                lines_to_close=[],
                num_switches=0,
                congestion_before=0,
                congestion_after=0,
                congestion_reduction_percent=0,
                max_loading_before=0,
                max_loading_after=0,
                losses_before_mw=0,
                losses_after_mw=0,
                message="Initial power flow failed"
            )

        # Find switchable lines
        switchable = []
        for idx in range(len(self.net.line)):
            is_switch = self.net.line.at[idx, 'is_switch'] if 'is_switch' in self.net.line.columns else True
            if is_switch:
                switchable.append(idx)

        best_to_open = []
        best_to_close = []
        best_metric = initial_congestion

        # Try each possible switch
        original_status = self.net.line.in_service.copy()
        switches_made = 0

        while switches_made < max_switches:
            improved = False

            for idx in switchable:
                # Try toggling this line
                current = self.net.line.at[idx, 'in_service']
                self.net.line.at[idx, 'in_service'] = not current

                try:
                    pp.runpp(self.net, numba=True)
                    metric = self._get_congestion_metric()

                    if metric < best_metric:
                        best_metric = metric
                        if current:  # Was closed, now open
                            if idx not in best_to_open:
                                best_to_open.append(idx)
                            if idx in best_to_close:
                                best_to_close.remove(idx)
                        else:  # Was open, now closed
                            if idx not in best_to_close:
                                best_to_close.append(idx)
                            if idx in best_to_open:
                                best_to_open.remove(idx)
                        improved = True
                        switches_made += 1
                        break  # Start over with new best
                except Exception:
                    pass

                # Revert
                self.net.line.at[idx, 'in_service'] = current

            if not improved:
                break

        # Apply best solution
        self.net.line.in_service = original_status
        for idx in best_to_open:
            self.net.line.at[idx, 'in_service'] = False
        for idx in best_to_close:
            self.net.line.at[idx, 'in_service'] = True

        # Get final metrics
        try:
            pp.runpp(self.net, numba=True)
            final_congestion = self._get_congestion_metric()
            final_max_loading = float(self.net.res_line.loading_percent.max())
            final_losses = float(self.net.res_line.pl_mw.sum())
        except Exception:
            final_congestion = initial_congestion
            final_max_loading = initial_max_loading
            final_losses = initial_losses

        # Restore original
        self.net.line.in_service = original_status

        congestion_reduction = (
            (initial_congestion - final_congestion) / initial_congestion * 100
            if initial_congestion > 0 else 0
        )

        return OptimizationResult(
            status=OptimizationStatus.FEASIBLE if best_to_open or best_to_close else OptimizationStatus.OPTIMAL,
            objective_value=final_congestion,
            computation_time_s=time.time() - start_time,
            lines_to_open=best_to_open,
            lines_to_close=best_to_close,
            num_switches=len(best_to_open) + len(best_to_close),
            congestion_before=initial_congestion,
            congestion_after=final_congestion,
            congestion_reduction_percent=congestion_reduction,
            max_loading_before=initial_max_loading,
            max_loading_after=final_max_loading,
            losses_before_mw=initial_losses,
            losses_after_mw=final_losses,
            message="Greedy heuristic"
        )

    def _get_congestion_metric(self) -> float:
        """Compute congestion metric."""
        loading = self.net.res_line.loading_percent.values
        overload = np.maximum(loading - 100, 0)
        return float(np.sum(overload ** 2))
