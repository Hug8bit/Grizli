"""
GrizliV2 - RL Agent
PPO agent for grid optimization using Stable-Baselines3.

Training pipeline:
    1. Load SMART-DS SF network + scenarios
    2. Wrap in GrizliGridEnv
    3. Train PPO agent
    4. Evaluate and save checkpoint
    5. (Future) Fine-tune on Chilean grid data
"""
import logging
from pathlib import Path
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class GrizliAgent:
    """
    PPO-based RL agent for grid topology optimization.
    Wraps Stable-Baselines3 PPO with Grizli-specific training logic.
    """

    def __init__(
        self,
        env,
        models_dir: Path = Path("models"),
        device: str = "auto",
    ):
        self.env = env
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._model = None

    def build(
        self,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        policy_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize PPO model."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv

            if policy_kwargs is None:
                policy_kwargs = {"net_arch": [256, 256]}

            vec_env = DummyVecEnv([lambda: self.env])
            self._model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=self.device,
            )
            logger.info("PPO agent built successfully")
        except ImportError:
            logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3")
            raise

    def train(
        self,
        total_timesteps: int = 1_000_000,
        checkpoint_freq: int = 50_000,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """
        Train the agent and save checkpoints.

        Returns training summary with final metrics.
        """
        if self._model is None:
            self.build()

        from stable_baselines3.common.callbacks import (
            CheckpointCallback,
            EvalCallback,
            BaseCallback,
        )

        callbacks = []

        # Checkpoint every N steps
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(self.models_dir / "checkpoints"),
            name_prefix="grizli_ppo",
        )
        callbacks.append(checkpoint_cb)

        # Progress reporting callback
        if progress_callback:
            callbacks.append(_ProgressCallback(progress_callback))

        logger.info(f"Training for {total_timesteps:,} timesteps...")
        self._model.learn(total_timesteps=total_timesteps, callback=callbacks)

        final_path = self.models_dir / "grizli_ppo_final"
        self._model.save(str(final_path))
        logger.info(f"Model saved to {final_path}")

        return {"status": "completed", "timesteps": total_timesteps, "path": str(final_path)}

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[int, dict]:
        """Run inference — return (action, info)."""
        if self._model is None:
            raise RuntimeError("Agent not built. Call build() or load().")
        action, _states = self._model.predict(observation, deterministic=deterministic)
        return int(action), {}

    def evaluate(self, n_episodes: int = 10) -> dict:
        """Evaluate agent over N episodes, return mean reward and metrics."""
        if self._model is None:
            raise RuntimeError("Agent not built. Call build() or load().")

        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, std_reward = evaluate_policy(
            self._model, self.env, n_eval_episodes=n_episodes, deterministic=True
        )
        logger.info(f"Eval: mean_reward={mean_reward:.2f} ± {std_reward:.2f}")
        return {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "n_episodes": n_episodes,
        }

    def save(self, name: str = "grizli_ppo") -> Path:
        if self._model is None:
            raise RuntimeError("No model to save.")
        path = self.models_dir / name
        self._model.save(str(path))
        logger.info(f"Saved to {path}")
        return path

    def load(self, path: Optional[Path] = None) -> None:
        """Load a saved model from disk."""
        try:
            from stable_baselines3 import PPO

            if path is None:
                path = self.models_dir / "grizli_ppo_final"
            self._model = PPO.load(str(path), env=self.env, device=self.device)
            logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def get_model_info(self) -> dict:
        if self._model is None:
            return {"status": "not_built"}
        return {
            "status": "ready",
            "policy": str(type(self._model.policy).__name__),
            "device": str(self._model.device),
            "n_timesteps": int(self._model.num_timesteps),
        }


class _ProgressCallback:
    """Thin wrapper to relay training progress to external callback."""

    def __init__(self, callback: Callable[[dict], None]):
        self._cb = callback
        self._step = 0

    def __call__(self, locals_: dict, globals_: dict) -> bool:
        self._step += 1
        if self._step % 1000 == 0:
            self._cb({"timestep": self._step, "reward": locals_.get("mean_reward")})
        return True  # continue training


def load_or_build_agent(env, models_dir: Path, train: bool = False) -> GrizliAgent:
    """
    Convenience factory: load existing model or build+train a new one.
    """
    agent = GrizliAgent(env, models_dir=models_dir)
    final_model = models_dir / "grizli_ppo_final.zip"

    if final_model.exists():
        agent.build()
        agent.load(final_model)
        logger.info("Loaded existing trained model")
    elif train:
        agent.build()
        agent.train()
    else:
        agent.build()
        logger.info("New agent built (untrained)")

    return agent
