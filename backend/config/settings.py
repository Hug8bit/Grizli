"""
GrizliV2 - Configuration settings
"""
from pydantic_settings import BaseSettings
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    # App
    app_name: str = "GrizliV2"
    app_version: str = "2.0.0"
    debug: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Paths
    data_dir: Path = ROOT_DIR / "data"
    models_dir: Path = ROOT_DIR / "models"
    smartds_dir: Path = ROOT_DIR / "data" / "smartds"

    # RL Training
    rl_total_timesteps: int = 1_000_000
    rl_n_steps: int = 2048
    rl_batch_size: int = 64
    rl_n_epochs: int = 10
    rl_learning_rate: float = 3e-4
    rl_gamma: float = 0.99
    rl_checkpoint_freq: int = 50_000

    # Grid simulation
    max_line_loading_percent: float = 80.0
    min_voltage_pu: float = 0.95
    max_voltage_pu: float = 1.05

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
