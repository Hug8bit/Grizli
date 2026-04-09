"""
PPO agent for grid control using stable-baselines3.
"""

from typing import Dict, Optional, Any, Callable
from pathlib import Path
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False


class RewardTrackingCallback(BaseCallback):
    """
    Callback for tracking rewards and saving best model.

    Saves the model whenever mean reward improves.
    """

    def __init__(
        self,
        save_path: str = "models/ppo_grid",
        check_freq: int = 1000,
        verbose: int = 1
    ):
        """
        Initialize callback.

        Args:
            save_path: Directory to save models
            check_freq: Check frequency (in timesteps)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _init_callback(self) -> None:
        """Initialize callback."""
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Called after each step."""
        # Track rewards
        self.current_episode_reward += self.locals.get("rewards", [0])[0]

        # Check for episode end
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0

        # Periodic check
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])

            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Mean reward = {mean_reward:.2f}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path / "best_model")
                if self.verbose > 0:
                    print(f"  -> New best! Saved to {self.save_path / 'best_model'}")

        return True

    def _on_training_end(self) -> None:
        """Called at end of training."""
        # Save final model
        self.model.save(self.save_path / "final_model")
        if self.verbose > 0:
            print(f"Training complete. Final model saved to {self.save_path / 'final_model'}")


def create_ppo_agent(
    env: gym.Env,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    verbose: int = 1,
    tensorboard_log: Optional[str] = None,
    policy_kwargs: Optional[Dict] = None
) -> "PPO":
    """
    Create a PPO agent for grid control.

    Args:
        env: Gymnasium environment
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Minibatch size
        n_epochs: Epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clipping parameter
        ent_coef: Entropy coefficient
        verbose: Verbosity level
        tensorboard_log: Tensorboard log directory
        policy_kwargs: Additional policy network arguments

    Returns:
        PPO agent
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 required. Install with: pip install stable-baselines3")

    # Default policy architecture
    if policy_kwargs is None:
        policy_kwargs = {
            "net_arch": {
                "pi": [256, 256],  # Policy network
                "vf": [256, 256]   # Value network
            }
        }

    # Wrap in DummyVecEnv if needed
    if not isinstance(env, DummyVecEnv):
        env = DummyVecEnv([lambda: env])

    agent = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs
    )

    return agent


def load_agent(
    model_path: str,
    env: Optional[gym.Env] = None
) -> "PPO":
    """
    Load a saved PPO agent.

    Args:
        model_path: Path to saved model
        env: Optional environment (for continued training)

    Returns:
        Loaded PPO agent
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 required")

    path = Path(model_path)
    if not path.suffix:
        path = path.with_suffix(".zip")

    return PPO.load(str(path), env=env)


def evaluate_agent(
    agent: "PPO",
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a trained agent.

    Args:
        agent: PPO agent to evaluate
        env: Environment to evaluate in
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions
        verbose: Print progress

    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    max_rhos = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_max_rho = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            episode_max_rho = max(episode_max_rho, info.get("max_rho", 0))

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        max_rhos.append(episode_max_rho)

        if verbose:
            print(f"Episode {ep+1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"Max Rho={episode_max_rho:.2f}")

    results = {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_max_rho": float(np.mean(max_rhos)),
        "survival_rate": float(np.mean([l > 10 for l in episode_lengths])),
    }

    if verbose:
        print("\n=== Evaluation Summary ===")
        print(f"Mean Reward: {results['mean_reward']:.2f} (+/- {results['std_reward']:.2f})")
        print(f"Mean Episode Length: {results['mean_length']:.1f}")
        print(f"Mean Max Rho: {results['mean_max_rho']:.2f}")
        print(f"Survival Rate: {results['survival_rate']*100:.1f}%")

    return results


def train_agent(
    env: gym.Env,
    total_timesteps: int = 100000,
    save_path: str = "models/ppo_grid",
    learning_rate: float = 3e-4,
    verbose: int = 1
) -> "PPO":
    """
    Train a PPO agent from scratch.

    Args:
        env: Training environment
        total_timesteps: Total training steps
        save_path: Path to save models
        learning_rate: Learning rate
        verbose: Verbosity level

    Returns:
        Trained PPO agent
    """
    agent = create_ppo_agent(env, learning_rate=learning_rate, verbose=verbose)

    callback = RewardTrackingCallback(
        save_path=save_path,
        check_freq=1000,
        verbose=verbose
    )

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    return agent
