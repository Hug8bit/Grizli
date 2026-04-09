"""
GrizliV2 — Standalone training script
Trains PPO agent on SMART-DS San Francisco data.

Usage:
    python scripts/train.py --timesteps 1000000 --scenario-dir data/smartds/P1R
"""
import argparse
import logging
import sys
from pathlib import Path

# Make sure backend is importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config.settings import settings
from backend.core.grid_engine import GridEngine
from backend.data.smartds_loader import SmartDSLoader
from backend.ml.rl_environment import GrizliGridEnv
from backend.ml.rl_agent import GrizliAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GrizliV2 RL Training")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--scenario-dir", type=str, default=str(settings.smartds_dir))
    parser.add_argument("--models-dir", type=str, default=str(settings.models_dir))
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    logger.info("=== GrizliV2 Training ===")
    logger.info(f"Timesteps: {args.timesteps:,}")
    logger.info(f"SMART-DS path: {args.scenario_dir}")

    # Load data
    loader = SmartDSLoader(Path(args.scenario_dir))
    net = loader.load_network()
    scenarios = loader.load_scenarios(n_scenarios=500)
    logger.info(f"Loaded {len(scenarios)} scenarios")

    # Create environment
    env = GrizliGridEnv(net=net, scenarios=scenarios, max_steps=96)
    logger.info(f"Environment: {env.observation_space.shape[0]} obs, {env.action_space.n} actions")

    # Build and train agent
    agent = GrizliAgent(env=env, models_dir=Path(args.models_dir))
    agent.build()

    logger.info("Starting training...")
    result = agent.train(
        total_timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq,
    )
    logger.info(f"Training complete: {result}")

    # Evaluate
    logger.info(f"Evaluating over {args.eval_episodes} episodes...")
    eval_result = agent.evaluate(n_episodes=args.eval_episodes)
    logger.info(f"Evaluation: {eval_result}")


if __name__ == "__main__":
    main()
