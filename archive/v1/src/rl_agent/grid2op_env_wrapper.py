"""
Gymnasium wrapper for Grid2Op environments.
Makes Grid2Op compatible with stable-baselines3.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    import grid2op
    from grid2op.Environment import Environment
    GRID2OP_AVAILABLE = True
except ImportError:
    GRID2OP_AVAILABLE = False

from .action_space import TopKActionSpace


class Grid2OpEnvWrapper(gym.Env):
    """
    Gymnasium wrapper for Grid2Op environments.

    Converts Grid2Op's complex observation/action spaces to
    flat vectors compatible with standard RL algorithms.

    Observation Vector:
    - Line loadings (rho): n_line values in [0, inf)
    - Line status: n_line binary values
    - Generator power (normalized): n_gen values
    - Load power (normalized): n_load values
    - Voltage magnitudes: n_sub values
    - Time features: 2 values (hour, day_of_week normalized)

    Action Space:
    - Discrete(k) using TopKActionSpace
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env: "Environment",
        k_actions: int = 10,
        reward_type: str = "stability",
        normalize_obs: bool = True
    ):
        """
        Initialize the wrapper.

        Args:
            env: Grid2Op environment
            k_actions: Number of actions in reduced action space
            reward_type: Reward function type
                        - "stability": +1 survival, penalty for overload
                        - "simple": Native Grid2Op reward
                        - "l2rpn": L2RPN competition style
            normalize_obs: Normalize observations to [0,1] range
        """
        if not GYM_AVAILABLE:
            raise ImportError("Gymnasium required. Install with: pip install gymnasium")
        if not GRID2OP_AVAILABLE:
            raise ImportError("Grid2Op required. Install with: pip install grid2op")

        super().__init__()

        self.env = env
        self.reward_type = reward_type
        self.normalize_obs = normalize_obs

        # Build action space
        self.action_space_handler = TopKActionSpace(env, k=k_actions)
        self.action_space = spaces.Discrete(self.action_space_handler.n_actions)

        # Build observation space
        self._obs_size = self._compute_obs_size()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_size,),
            dtype=np.float32
        )

        # State tracking
        self._current_obs = None
        self._step_count = 0

    def _compute_obs_size(self) -> int:
        """Compute size of flattened observation vector."""
        n_line = self.env.n_line
        n_gen = self.env.n_gen
        n_load = self.env.n_load
        n_sub = self.env.n_sub

        return (
            n_line +      # Line loadings (rho)
            n_line +      # Line status
            n_gen +       # Generator power
            n_load +      # Load power
            n_sub +       # Voltage magnitudes
            2             # Time features
        )

    def _obs_to_vector(self, obs: Any) -> np.ndarray:
        """Convert Grid2Op observation to flat vector."""
        parts = []

        # Line loadings (rho)
        rho = obs.rho.copy()
        if self.normalize_obs:
            rho = np.clip(rho, 0, 2)  # Clip to [0, 2]
        parts.append(rho)

        # Line status (binary)
        line_status = obs.line_status.astype(np.float32)
        parts.append(line_status)

        # Generator power (normalized by max)
        gen_p = obs.gen_p.copy()
        if self.normalize_obs:
            gen_pmax = self.env.gen_pmax
            gen_p = np.divide(gen_p, gen_pmax, where=gen_pmax > 0,
                            out=np.zeros_like(gen_p))
        parts.append(gen_p)

        # Load power (normalized)
        load_p = obs.load_p.copy()
        if self.normalize_obs:
            load_p = load_p / (load_p.max() + 1e-6)
        parts.append(load_p)

        # Voltage magnitudes
        v_or = obs.v_or.copy() if hasattr(obs, 'v_or') else np.ones(self.env.n_sub)
        if self.normalize_obs:
            v_or = (v_or - 0.9) / 0.2  # Normalize around 1.0 pu
        parts.append(v_or[:self.env.n_sub])

        # Time features
        hour = obs.hour_of_day / 24.0  # Normalize to [0, 1]
        day = obs.day_of_week / 7.0
        parts.append(np.array([hour, day], dtype=np.float32))

        return np.concatenate(parts).astype(np.float32)

    def _compute_reward(self, obs: Any, done: bool, info: Dict) -> float:
        """Compute reward based on reward type."""
        if self.reward_type == "simple":
            return float(info.get("reward", 0))

        elif self.reward_type == "stability":
            if done:
                return -100.0  # Large penalty for game over

            reward = 1.0  # Base survival reward

            # Penalty for high loading
            max_rho = obs.rho.max() if len(obs.rho) > 0 else 0
            if max_rho > 1.0:
                reward -= (max_rho - 1.0) * 10  # Penalty for overload
            elif max_rho > 0.8:
                reward -= (max_rho - 0.8) * 2  # Small penalty for high loading
            else:
                reward += 0.5  # Bonus for low loading

            return reward

        elif self.reward_type == "l2rpn":
            # L2RPN competition style: margin to thermal limits
            if done:
                return -1.0

            rho = obs.rho
            if len(rho) == 0:
                return 0.0

            # Margin: how far from thermal limit
            margin = 1.0 - rho.max()
            return float(np.clip(margin, -1, 1))

        else:
            return 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            self.env.seed(seed)

        obs = self.env.reset()
        self._current_obs = obs
        self._step_count = 0

        return self._obs_to_vector(obs), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Action ID from reduced action space

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action ID to Grid2Op action
        g2op_action = self.action_space_handler.get_action(action)

        # Take step
        obs, reward, done, info = self.env.step(g2op_action)
        self._current_obs = obs
        self._step_count += 1

        # Compute custom reward
        reward = self._compute_reward(obs, done, info)

        # Convert observation
        obs_vector = self._obs_to_vector(obs)

        # Build info dict
        info_dict = {
            "action_description": self.action_space_handler.get_description(action),
            "max_rho": float(obs.rho.max()) if len(obs.rho) > 0 else 0,
            "step": self._step_count,
            "is_illegal": info.get("is_illegal", False),
            "is_ambiguous": info.get("is_ambiguous", False),
        }

        return obs_vector, reward, done, False, info_dict

    def render(self, mode: str = "human") -> None:
        """Render the environment."""
        if self._current_obs is not None:
            print(f"Step {self._step_count}")
            print(f"Max rho: {self._current_obs.rho.max():.2f}")
            print(f"Lines in service: {self._current_obs.line_status.sum()}/{self.env.n_line}")

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    def get_action_meanings(self) -> List[str]:
        """Get descriptions of all actions."""
        return self.action_space_handler.get_all_descriptions()


def create_gym_env(
    env_name: str = "l2rpn_case14_sandbox",
    k_actions: int = 10,
    reward_type: str = "stability",
    use_lightsim: bool = True,
    difficulty: str = "1"
) -> Grid2OpEnvWrapper:
    """
    Create a Gymnasium-compatible Grid2Op environment.

    Args:
        env_name: Grid2Op environment name
        k_actions: Size of reduced action space
        reward_type: Reward function type
        use_lightsim: Use LightSim2Grid backend (faster)
        difficulty: Difficulty level ("0", "1", "2")

    Returns:
        Grid2OpEnvWrapper instance
    """
    # Try to use LightSim2Grid for speed
    backend = None
    if use_lightsim:
        try:
            from lightsim2grid import LightSimBackend
            backend = LightSimBackend()
        except ImportError:
            pass

    # Create Grid2Op environment
    env = grid2op.make(
        env_name,
        backend=backend,
        difficulty=difficulty
    )

    return Grid2OpEnvWrapper(
        env,
        k_actions=k_actions,
        reward_type=reward_type
    )
