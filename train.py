#!/usr/bin/env python3
"""
GRIZLI Training Script
Train PPO agent for grid control using Grid2Op.
"""

import argparse
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    try:
        import grid2op
    except ImportError:
        missing.append("grid2op")

    try:
        import stable_baselines3
    except ImportError:
        missing.append("stable-baselines3")

    try:
        import gymnasium
    except ImportError:
        missing.append("gymnasium")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    if missing:
        print("Missing dependencies for RL training:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install grid2op stable-baselines3 gymnasium torch")
        return False

    return True


def train(args):
    """Train PPO agent."""
    from src.rl_agent.grid2op_env_wrapper import create_gym_env
    from src.rl_agent.ppo_agent import create_ppo_agent, RewardTrackingCallback

    print(f"\n{'='*60}")
    print("GRIZLI PPO Training")
    print(f"{'='*60}")

    # Create environment
    print(f"\nCreating environment: {args.env}")
    print(f"  K actions: {args.k_actions}")
    print(f"  Reward type: {args.reward_type}")

    env = create_gym_env(
        env_name=args.env,
        k_actions=args.k_actions,
        reward_type=args.reward_type,
        use_lightsim=not args.no_lightsim
    )

    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n} actions")

    # Create agent
    print(f"\nCreating PPO agent...")
    print(f"  Learning rate: {args.lr}")

    agent = create_ppo_agent(
        env,
        learning_rate=args.lr,
        verbose=1
    )

    # Setup callback
    save_path = Path(args.save_path)
    callback = RewardTrackingCallback(
        save_path=str(save_path),
        check_freq=1000,
        verbose=1
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Total timesteps: {args.timesteps}")
    print(f"  Save path: {save_path}")

    agent.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True
    )

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"  Best model: {save_path / 'best_model.zip'}")
    print(f"  Final model: {save_path / 'final_model.zip'}")
    print(f"{'='*60}")

    return agent


def evaluate(args):
    """Evaluate trained agent."""
    from src.rl_agent.grid2op_env_wrapper import create_gym_env
    from src.rl_agent.ppo_agent import load_agent, evaluate_agent

    print(f"\n{'='*60}")
    print("GRIZLI Agent Evaluation")
    print(f"{'='*60}")

    # Create environment
    print(f"\nCreating environment: {args.env}")
    env = create_gym_env(
        env_name=args.env,
        k_actions=args.k_actions,
        reward_type=args.reward_type,
        use_lightsim=not args.no_lightsim
    )

    # Load agent
    model_path = Path(args.model_path)
    if not model_path.exists() and not model_path.with_suffix('.zip').exists():
        print(f"Error: Model not found at {model_path}")
        return None

    print(f"\nLoading model from: {model_path}")
    agent = load_agent(str(model_path), env=env)

    # Evaluate
    print(f"\nEvaluating over {args.eval_episodes} episodes...")
    results = evaluate_agent(
        agent, env,
        n_episodes=args.eval_episodes,
        deterministic=True,
        verbose=True
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="GRIZLI PPO Training Script")

    # Mode selection
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate, do not train')

    # Environment settings
    parser.add_argument('--env', type=str, default='l2rpn_case14_sandbox',
                       help='Grid2Op environment name')
    parser.add_argument('--k-actions', type=int, default=10,
                       help='Number of actions in reduced space')
    parser.add_argument('--reward-type', type=str, default='stability',
                       choices=['stability', 'simple', 'l2rpn'],
                       help='Reward function type')
    parser.add_argument('--no-lightsim', action='store_true',
                       help='Disable LightSim2Grid backend')

    # Training settings
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--save-path', type=str, default='models/ppo_grid',
                       help='Path to save models')

    # Evaluation settings
    parser.add_argument('--model-path', type=str, default='models/ppo_grid/best_model',
                       help='Path to model for evaluation')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        return

    if args.eval_only:
        evaluate(args)
    else:
        train(args)
        # Also run evaluation after training
        args.model_path = str(Path(args.save_path) / 'best_model')
        if Path(args.model_path + '.zip').exists():
            print("\nRunning post-training evaluation...")
            evaluate(args)


if __name__ == "__main__":
    main()
