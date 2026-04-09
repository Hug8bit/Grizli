"""
Scenario player for Grid2Op environments.
Plays chronics and analyzes grid behavior.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

try:
    import grid2op
    from grid2op.Environment import Environment
    GRID2OP_AVAILABLE = True
except ImportError:
    GRID2OP_AVAILABLE = False


@dataclass
class StepResult:
    """Result from a single simulation step."""
    timestep: int
    reward: float
    done: bool
    max_rho: float
    mean_rho: float
    n_overloaded: int
    total_gen_mw: float
    total_load_mw: float
    n_lines_disconnected: int
    action_taken: str


def format_observation(obs: Any, verbose: bool = False) -> str:
    """
    Format Grid2Op observation as readable string.

    Args:
        obs: Grid2Op observation
        verbose: Include detailed information

    Returns:
        Formatted string
    """
    lines = []

    # Basic info
    lines.append(f"=== Observation ===")
    lines.append(f"Time: {obs.hour_of_day}:{obs.minute_of_hour:02d} (Day {obs.day_of_week})")

    # Power balance
    total_gen = obs.gen_p.sum()
    total_load = obs.load_p.sum()
    lines.append(f"Generation: {total_gen:.1f} MW")
    lines.append(f"Load: {total_load:.1f} MW")
    lines.append(f"Balance: {total_gen - total_load:.1f} MW")

    # Line status
    n_lines = len(obs.line_status)
    n_connected = obs.line_status.sum()
    lines.append(f"Lines: {n_connected}/{n_lines} connected")

    # Loading
    max_rho = obs.rho.max() if len(obs.rho) > 0 else 0
    mean_rho = obs.rho.mean() if len(obs.rho) > 0 else 0
    n_overloaded = (obs.rho > 1.0).sum()
    lines.append(f"Max loading (rho): {max_rho:.2f}")
    lines.append(f"Mean loading: {mean_rho:.2f}")
    lines.append(f"Overloaded lines: {n_overloaded}")

    if verbose:
        # Detailed line info
        lines.append("\n--- Line Details ---")
        for i in range(min(10, n_lines)):
            status = "ON" if obs.line_status[i] else "OFF"
            rho = obs.rho[i] if obs.line_status[i] else 0
            lines.append(f"  Line {i}: {status}, rho={rho:.2f}")
        if n_lines > 10:
            lines.append(f"  ... ({n_lines - 10} more lines)")

        # Generator info
        lines.append("\n--- Generators ---")
        for i in range(min(5, len(obs.gen_p))):
            lines.append(f"  Gen {i}: {obs.gen_p[i]:.1f} MW")

    return "\n".join(lines)


def play_scenario(
    env: "Environment",
    n_steps: int = 100,
    display_every: int = 10,
    verbose: bool = True
) -> List[StepResult]:
    """
    Play a scenario with do-nothing agent.

    Args:
        env: Grid2Op environment
        n_steps: Number of steps to play
        display_every: Display progress every N steps
        verbose: Print progress

    Returns:
        List of StepResult for each step
    """
    if not GRID2OP_AVAILABLE:
        raise ImportError("Grid2Op required")

    results = []
    obs = env.reset()

    for step in range(n_steps):
        # Do nothing action
        action = env.action_space({})
        obs, reward, done, info = env.step(action)

        result = StepResult(
            timestep=step,
            reward=float(reward),
            done=done,
            max_rho=float(obs.rho.max()) if len(obs.rho) > 0 else 0,
            mean_rho=float(obs.rho.mean()) if len(obs.rho) > 0 else 0,
            n_overloaded=int((obs.rho > 1.0).sum()),
            total_gen_mw=float(obs.gen_p.sum()),
            total_load_mw=float(obs.load_p.sum()),
            n_lines_disconnected=int((~obs.line_status).sum()),
            action_taken="Do Nothing"
        )
        results.append(result)

        if verbose and step % display_every == 0:
            print(f"Step {step}: max_rho={result.max_rho:.2f}, "
                  f"overloaded={result.n_overloaded}, reward={result.reward:.2f}")

        if done:
            if verbose:
                print(f"Episode ended at step {step}")
            break

    return results


def play_random_actions(
    env: "Environment",
    n_steps: int = 50,
    action_prob: float = 0.1,
    verbose: bool = True
) -> List[StepResult]:
    """
    Play scenario with random actions.

    Args:
        env: Grid2Op environment
        n_steps: Number of steps
        action_prob: Probability of taking random action
        verbose: Print progress

    Returns:
        List of StepResult
    """
    if not GRID2OP_AVAILABLE:
        raise ImportError("Grid2Op required")

    results = []
    obs = env.reset()

    for step in range(n_steps):
        # Randomly decide to act
        if np.random.random() < action_prob:
            action = env.action_space.sample()
            action_desc = "Random Action"
        else:
            action = env.action_space({})
            action_desc = "Do Nothing"

        obs, reward, done, info = env.step(action)

        result = StepResult(
            timestep=step,
            reward=float(reward),
            done=done,
            max_rho=float(obs.rho.max()) if len(obs.rho) > 0 else 0,
            mean_rho=float(obs.rho.mean()) if len(obs.rho) > 0 else 0,
            n_overloaded=int((obs.rho > 1.0).sum()),
            total_gen_mw=float(obs.gen_p.sum()),
            total_load_mw=float(obs.load_p.sum()),
            n_lines_disconnected=int((~obs.line_status).sum()),
            action_taken=action_desc
        )
        results.append(result)

        if verbose and step % 10 == 0:
            print(f"Step {step}: {action_desc}, max_rho={result.max_rho:.2f}")

        if done:
            if verbose:
                print(f"Episode ended at step {step}")
            break

    return results


def analyze_chronic(
    env: "Environment",
    chronic_id: int = 0,
    quick: bool = True
) -> Dict[str, Any]:
    """
    Analyze a specific chronic (scenario).

    Args:
        env: Grid2Op environment
        chronic_id: ID of chronic to analyze
        quick: Do quick analysis (fewer steps)

    Returns:
        Dictionary with analysis results
    """
    if not GRID2OP_AVAILABLE:
        raise ImportError("Grid2Op required")

    # Set chronic
    env.set_id(chronic_id)
    obs = env.reset()

    n_steps = 100 if quick else 1000
    results = play_scenario(env, n_steps=n_steps, verbose=False)

    if not results:
        return {"error": "No results generated"}

    # Compute statistics
    max_rhos = [r.max_rho for r in results]
    rewards = [r.reward for r in results]
    survival = len(results)

    analysis = {
        "chronic_id": chronic_id,
        "n_steps_played": len(results),
        "survival_steps": survival,
        "survival_rate": survival / n_steps,
        "episode_ended_early": results[-1].done,
        "total_reward": sum(rewards),
        "mean_reward": np.mean(rewards),
        "max_rho": {
            "peak": max(max_rhos),
            "mean": np.mean(max_rhos),
            "min": min(max_rhos),
        },
        "overload_events": sum(1 for r in results if r.n_overloaded > 0),
        "generation": {
            "mean": np.mean([r.total_gen_mw for r in results]),
            "peak": max(r.total_gen_mw for r in results),
        },
        "load": {
            "mean": np.mean([r.total_load_mw for r in results]),
            "peak": max(r.total_load_mw for r in results),
        },
    }

    return analysis


def compare_strategies(
    env: "Environment",
    strategies: Dict[str, callable],
    n_steps: int = 100,
    n_episodes: int = 5
) -> Dict[str, Dict]:
    """
    Compare different action strategies.

    Args:
        env: Grid2Op environment
        strategies: Dictionary mapping name to action function
        n_steps: Steps per episode
        n_episodes: Number of episodes per strategy

    Returns:
        Dictionary with comparison results
    """
    if not GRID2OP_AVAILABLE:
        raise ImportError("Grid2Op required")

    results = {}

    for name, strategy_fn in strategies.items():
        episode_results = []

        for ep in range(n_episodes):
            obs = env.reset()
            total_reward = 0
            max_rhos = []

            for step in range(n_steps):
                action = strategy_fn(env, obs)
                obs, reward, done, info = env.step(action)

                total_reward += reward
                max_rhos.append(obs.rho.max() if len(obs.rho) > 0 else 0)

                if done:
                    break

            episode_results.append({
                "survival": step + 1,
                "total_reward": total_reward,
                "mean_max_rho": np.mean(max_rhos),
                "peak_max_rho": max(max_rhos),
            })

        results[name] = {
            "mean_survival": np.mean([r["survival"] for r in episode_results]),
            "mean_reward": np.mean([r["total_reward"] for r in episode_results]),
            "mean_max_rho": np.mean([r["mean_max_rho"] for r in episode_results]),
            "survival_rate": np.mean([r["survival"] >= n_steps for r in episode_results]),
        }

    return results


def get_critical_lines(
    env: "Environment",
    obs: Any,
    threshold: float = 0.8
) -> List[int]:
    """
    Get lines that are critically loaded.

    Args:
        env: Grid2Op environment
        obs: Current observation
        threshold: Loading threshold

    Returns:
        List of critical line indices
    """
    critical = []
    for i, rho in enumerate(obs.rho):
        if rho >= threshold and obs.line_status[i]:
            critical.append(i)
    return critical


def get_disconnected_lines(obs: Any) -> List[int]:
    """
    Get disconnected lines.

    Args:
        obs: Grid2Op observation

    Returns:
        List of disconnected line indices
    """
    return [i for i, status in enumerate(obs.line_status) if not status]
