"""
GrizliV2 - RL Environment
Gymnasium environment for grid optimization.

The agent controls topology switches to minimize:
- Line losses
- Voltage violations
- Line congestion

Analogy: Waze rerouting traffic — the agent finds the best current path
through the grid to minimize losses and avoid congestion.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandapower as pp
import pandapower.networks as pn
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class GrizliGridEnv(gym.Env):
    """
    RL environment for distribution grid optimization.

    Observation space:
        - Bus voltages (vm_pu) for each bus
        - Line loading (%) for each line
        - Switch states (binary) for each controllable switch
        - Renewable generation levels (normalized)

    Action space:
        - Toggle any one switch (discrete: 0 = do nothing, 1..N = toggle switch i)

    Reward:
        + Reduction in total losses
        + Reduction in congestion
        - Voltage violations
        - Switch operations (operational cost)
        - Episode ends if grid becomes infeasible for 3+ consecutive steps
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        net: Optional[pp.pandapowerNet] = None,
        scenarios: Optional[list[dict]] = None,
        max_steps: int = 96,  # 24h x 15-min intervals
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.base_net = net or pn.case33bw()
        self.scenarios = scenarios or []
        self.max_steps = max_steps
        self.render_mode = render_mode

        self._n_buses = len(self.base_net.bus)
        self._n_lines = len(self.base_net.line)
        self._n_switches = len(self.base_net.switch) if hasattr(self.base_net, "switch") else 0

        # Action: 0=noop, 1..N_switches = toggle switch i-1
        self.action_space = spaces.Discrete(self._n_switches + 1)

        # Observation: [vm_pu x n_buses] + [loading_pct x n_lines] + [switch_state x n_switches]
        obs_size = self._n_buses + self._n_lines + self._n_switches
        self.observation_space = spaces.Box(
            low=np.zeros(obs_size, dtype=np.float32),
            high=np.ones(obs_size, dtype=np.float32) * 2.0,
            dtype=np.float32,
        )

        self._net: Optional[pp.pandapowerNet] = None
        self._step_count = 0
        self._infeasible_count = 0
        self._current_scenario: Optional[dict] = None
        self._prev_losses: float = 0.0
        self._switch_ops: int = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._net = self.base_net.deepcopy()
        self._step_count = 0
        self._infeasible_count = 0
        self._switch_ops = 0

        # Pick a random scenario if available
        if self.scenarios:
            idx = self.np_random.integers(0, len(self.scenarios))
            self._current_scenario = self.scenarios[idx]
            self._apply_scenario_step(0)

        obs = self._get_observation()
        self._prev_losses = self._get_losses()
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._net is not None, "Call reset() before step()"

        # Apply action
        switched = False
        if action > 0:
            switch_idx = action - 1
            if switch_idx < self._n_switches:
                self._toggle_switch(switch_idx)
                self._switch_ops += 1
                switched = True

        # Advance scenario time step
        if self._current_scenario:
            self._apply_scenario_step(self._step_count % 24)

        # Run power flow
        try:
            pp.runpp(self._net, algorithm="nr", numba=False, verbose=False)
            converged = self._net.converged
        except pp.LoadflowNotConverged:
            converged = False

        # Compute reward
        reward = self._compute_reward(converged, switched)

        # Update counters
        self._step_count += 1
        if not converged:
            self._infeasible_count += 1
        else:
            self._infeasible_count = 0

        terminated = self._infeasible_count >= 3
        truncated = self._step_count >= self.max_steps

        obs = self._get_observation()
        info = {
            "converged": converged,
            "step": self._step_count,
            "switch_ops": self._switch_ops,
            "losses_mw": self._get_losses() if converged else None,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        if self._net is None:
            return None
        losses = self._get_losses()
        msg = (
            f"Step {self._step_count:3d} | "
            f"Losses: {losses:.4f} MW | "
            f"Switch ops: {self._switch_ops}"
        )
        if self.render_mode == "human":
            print(msg)
        return msg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        offset = 0

        if self._net is not None and self._net.converged and len(self._net.res_bus) > 0:
            vm = self._net.res_bus["vm_pu"].values[:self._n_buses]
            obs[offset: offset + len(vm)] = vm.astype(np.float32)
        offset += self._n_buses

        if self._net is not None and self._net.converged and len(self._net.res_line) > 0:
            loading = self._net.res_line["loading_percent"].values[:self._n_lines] / 100.0
            obs[offset: offset + len(loading)] = loading.astype(np.float32)
        offset += self._n_lines

        if self._n_switches > 0 and self._net is not None:
            sw = self._net.switch["closed"].values[:self._n_switches].astype(np.float32)
            obs[offset: offset + len(sw)] = sw
        return obs

    def _compute_reward(self, converged: bool, switched: bool) -> float:
        if not converged:
            return -10.0  # Heavy penalty for infeasible grid

        current_losses = self._get_losses()
        res_bus = self._net.res_bus
        res_line = self._net.res_line

        # 1. Loss reduction reward (core objective)
        loss_reward = (self._prev_losses - current_losses) * 100.0
        self._prev_losses = current_losses

        # 2. Voltage quality reward
        vm = res_bus["vm_pu"].values
        v_violations = np.sum((vm < 0.95) | (vm > 1.05))
        voltage_reward = -2.0 * v_violations

        # 3. Congestion reward
        loading = res_line["loading_percent"].values
        overloaded = np.sum(loading > 80.0)
        congestion_reward = -1.0 * overloaded

        # 4. Operational cost for switching
        switch_cost = -0.1 if switched else 0.0

        return float(loss_reward + voltage_reward + congestion_reward + switch_cost)

    def _get_losses(self) -> float:
        if (
            self._net is None
            or not self._net.converged
            or len(self._net.res_line) == 0
            or "pl_mw" not in self._net.res_line.columns
        ):
            return 0.0
        return float(self._net.res_line["pl_mw"].sum())

    def _toggle_switch(self, switch_idx: int) -> None:
        current = bool(self._net.switch.at[switch_idx, "closed"])
        self._net.switch.at[switch_idx, "closed"] = not current

    def _apply_scenario_step(self, hour: int) -> None:
        """Scale loads and generation according to scenario profiles."""
        scenario = self._current_scenario
        if scenario is None:
            return

        load_factor = scenario["load_profile"][hour % 24]
        solar_factor = scenario["solar_profile"][hour % 24]
        peak_load = scenario.get("peak_load_mw", 1.0)
        solar_cap = scenario.get("solar_capacity_mw", 0.3)

        # Scale loads
        if len(self._net.load) > 0:
            base_load = self.base_net.load["p_mw"].values
            self._net.load["p_mw"] = base_load * load_factor * peak_load

        # Scale solar generation
        if len(self._net.sgen) > 0:
            self._net.sgen["p_mw"] = solar_factor * solar_cap / max(len(self._net.sgen), 1)
