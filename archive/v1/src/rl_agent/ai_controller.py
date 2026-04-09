"""
High-level AI controller for dashboard integration.
Provides simple interface for AI-based grid control.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import time
import numpy as np

try:
    import grid2op
    GRID2OP_AVAILABLE = True
except ImportError:
    GRID2OP_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from .grid2op_env_wrapper import Grid2OpEnvWrapper, create_gym_env
from .ppo_agent import load_agent


@dataclass
class AIDecision:
    """A single AI decision with metadata."""
    timestep: int
    action_id: int
    action_description: str
    confidence: float
    reward: float
    max_rho: float
    execution_time_ms: float


@dataclass
class SimulationResult:
    """Results from AI simulation run."""
    n_steps: int
    total_reward: float
    mean_reward: float
    survival_steps: int
    survival_rate: float
    mean_max_rho: float
    peak_max_rho: float
    decisions: List[AIDecision] = field(default_factory=list)
    computation_time_s: float = 0.0

    # Comparison with baseline (if available)
    baseline_survival: Optional[int] = None
    improvement_vs_baseline: Optional[float] = None


class AIController:
    """
    High-level controller for AI-based grid management.

    Provides a simple interface for:
    - Loading trained models
    - Getting AI decisions
    - Running simulations
    - Comparing to baselines
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        env_name: str = "l2rpn_case14_sandbox",
        k_actions: int = 10,
        use_lightsim: bool = True
    ):
        """
        Initialize AI controller.

        Args:
            model_path: Path to trained model (optional)
            env_name: Grid2Op environment name
            k_actions: Action space size
            use_lightsim: Use faster LightSim backend
        """
        self.model_path = model_path
        self.env_name = env_name
        self.k_actions = k_actions
        self.use_lightsim = use_lightsim

        self._env: Optional[Grid2OpEnvWrapper] = None
        self._agent: Optional["PPO"] = None
        self._is_ready = False

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load trained model and create environment.

        Args:
            model_path: Override model path

        Returns:
            True if successful
        """
        if not GRID2OP_AVAILABLE or not SB3_AVAILABLE:
            print("Warning: Grid2Op or stable-baselines3 not available")
            return False

        path = model_path or self.model_path

        try:
            # Create environment
            self._env = create_gym_env(
                env_name=self.env_name,
                k_actions=self.k_actions,
                reward_type="stability",
                use_lightsim=self.use_lightsim
            )

            # Load model if path provided
            if path:
                path = Path(path)
                if not path.suffix:
                    path = path.with_suffix(".zip")
                if path.exists():
                    self._agent = load_agent(str(path), env=self._env)
                    self._is_ready = True
                else:
                    print(f"Warning: Model not found at {path}")
                    self._is_ready = False
            else:
                self._is_ready = False

            return self._is_ready

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_ai_decision(
        self,
        obs: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, str, float]:
        """
        Get AI decision for current observation.

        Args:
            obs: Observation vector
            deterministic: Use deterministic policy

        Returns:
            Tuple of (action_id, description, confidence)
        """
        if not self._is_ready or self._agent is None:
            return 0, "Do Nothing (no model)", 0.0

        start = time.time()

        # Get action from agent
        action, _ = self._agent.predict(obs, deterministic=deterministic)
        action_id = int(action)

        # Get action description
        description = self._env.get_action_meanings()[action_id]

        # Estimate confidence (using action probabilities if available)
        # For PPO, we can use the policy's action probabilities
        try:
            obs_tensor = self._agent.policy.obs_to_tensor(obs.reshape(1, -1))[0]
            distribution = self._agent.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.detach().numpy()[0]
            confidence = float(probs[action_id])
        except Exception:
            confidence = 1.0 if deterministic else 0.5

        elapsed = (time.time() - start) * 1000

        return action_id, description, confidence

    def run_simulation(
        self,
        n_steps: int = 100,
        deterministic: bool = True,
        compare_baseline: bool = False,
        verbose: bool = False
    ) -> SimulationResult:
        """
        Run AI simulation for multiple steps.

        Args:
            n_steps: Number of steps to simulate
            deterministic: Use deterministic policy
            compare_baseline: Compare to do-nothing baseline
            verbose: Print progress

        Returns:
            SimulationResult with all metrics
        """
        if self._env is None:
            raise ValueError("Environment not loaded. Call load_model() first.")

        start_time = time.time()
        decisions = []

        # Reset environment
        obs, info = self._env.reset()
        total_reward = 0
        max_rhos = []

        for step in range(n_steps):
            step_start = time.time()

            if self._is_ready and self._agent is not None:
                action_id, desc, confidence = self.get_ai_decision(obs, deterministic)
            else:
                action_id, desc, confidence = 0, "Do Nothing", 1.0

            obs, reward, terminated, truncated, info = self._env.step(action_id)
            done = terminated or truncated

            step_time = (time.time() - step_start) * 1000

            decisions.append(AIDecision(
                timestep=step,
                action_id=action_id,
                action_description=desc,
                confidence=confidence,
                reward=reward,
                max_rho=info.get("max_rho", 0),
                execution_time_ms=step_time
            ))

            total_reward += reward
            max_rhos.append(info.get("max_rho", 0))

            if verbose and step % 10 == 0:
                print(f"Step {step}: Action={desc}, Reward={reward:.2f}, Max Rho={info.get('max_rho', 0):.2f}")

            if done:
                if verbose:
                    print(f"Episode ended at step {step}")
                break

        survival_steps = len(decisions)
        survival_rate = survival_steps / n_steps

        # Compare to baseline (do-nothing)
        baseline_survival = None
        improvement = None
        if compare_baseline:
            baseline_survival = self._run_baseline(n_steps)
            if baseline_survival > 0:
                improvement = (survival_steps - baseline_survival) / baseline_survival * 100

        computation_time = time.time() - start_time

        return SimulationResult(
            n_steps=n_steps,
            total_reward=total_reward,
            mean_reward=total_reward / survival_steps if survival_steps > 0 else 0,
            survival_steps=survival_steps,
            survival_rate=survival_rate,
            mean_max_rho=float(np.mean(max_rhos)) if max_rhos else 0,
            peak_max_rho=float(np.max(max_rhos)) if max_rhos else 0,
            decisions=decisions,
            computation_time_s=computation_time,
            baseline_survival=baseline_survival,
            improvement_vs_baseline=improvement
        )

    def _run_baseline(self, n_steps: int) -> int:
        """Run do-nothing baseline."""
        if self._env is None:
            return 0

        obs, info = self._env.reset()
        for step in range(n_steps):
            obs, reward, terminated, truncated, info = self._env.step(0)  # Do nothing
            if terminated or truncated:
                return step + 1
        return n_steps

    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            "is_ready": self._is_ready,
            "model_loaded": self._agent is not None,
            "model_path": self.model_path,
            "env_name": self.env_name,
            "k_actions": self.k_actions,
            "n_actions": self._env.action_space.n if self._env else 0,
            "obs_size": self._env.observation_space.shape[0] if self._env else 0,
        }

    def get_action_descriptions(self) -> List[str]:
        """Get all available action descriptions."""
        if self._env is None:
            return ["Do Nothing"]
        return self._env.get_action_meanings()

    def close(self) -> None:
        """Close environment and clean up."""
        if self._env is not None:
            self._env.close()
            self._env = None
        self._agent = None
        self._is_ready = False


def create_ai_controller(
    model_path: Optional[str] = None,
    env_name: str = "l2rpn_case14_sandbox",
    k_actions: int = 10
) -> AIController:
    """
    Convenience function to create and initialize AI controller.

    Args:
        model_path: Path to trained model
        env_name: Grid2Op environment name
        k_actions: Action space size

    Returns:
        Initialized AIController
    """
    controller = AIController(
        model_path=model_path,
        env_name=env_name,
        k_actions=k_actions
    )
    controller.load_model()
    return controller
