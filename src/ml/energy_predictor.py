"""
ML module — hourly SF energy consumption predictor + optimisation adviser.

Uses scikit-learn GradientBoostingRegressor by default.
Persists trained model to models/energy_predictor.joblib.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.sf_data_loader import SFDataLoader

logger = logging.getLogger(__name__)

MODEL_DIR  = Path(os.getenv("MODEL_DIR", "models"))
MODEL_PATH = MODEL_DIR / "energy_predictor.joblib"

FEATURE_COLS      = ["hour", "day_of_week", "month", "is_weekend",
                      "temperature_c", "humidity_pct", "sfmta_ridership"]
TARGET_COL        = "consumption_mw"
PEAK_THRESHOLD_MW = 70.0


class EnergyPredictor:
    """
    Gradient Boosting predictor for hourly SF grid energy consumption.

    Public interface:
      .train(df?)           → metrics dict
      .predict(features)    → prediction + optimisation advice
      .get_predictions_vs_actual(df?, n_samples) → list[dict] for dashboard
    """

    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type       = model_type
        self.pipeline: Optional[Pipeline] = None
        self.feature_cols: list[str]      = list(FEATURE_COLS)
        self.metrics: dict                = {}
        self._training_history: list[dict] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_pipeline(self) -> Pipeline:
        if self.model_type == "random_forest":
            reg = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
            )
        elif self.model_type == "ridge":
            reg = Ridge(alpha=1.0)
        else:
            reg = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, subsample=0.8, random_state=42,
            )
        return Pipeline([("scaler", StandardScaler()), ("reg", reg)])

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, df: Optional[pd.DataFrame] = None) -> dict:
        """Train on *df* or auto-load SF data. Returns evaluation metrics."""
        if df is None:
            df = SFDataLoader().get_processed_features()

        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].values
        y = df[TARGET_COL].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.pipeline     = self._build_pipeline()
        self.feature_cols = available
        self.pipeline.fit(X_tr, y_tr)

        y_pred = self.pipeline.predict(X_te)
        self.metrics = {
            "mae":           round(float(mean_absolute_error(y_te, y_pred)), 4),
            "rmse":          round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 4),
            "r2":            round(float(r2_score(y_te, y_pred)), 4),
            "train_samples": int(len(X_tr)),
            "test_samples":  int(len(X_te)),
            "model_type":    self.model_type,
        }
        self._training_history = self._build_history(X_tr, y_tr, X_te, y_te)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, MODEL_PATH)
        logger.info(
            "Model trained — R²=%.3f  MAE=%.2f MW",
            self.metrics["r2"], self.metrics["mae"],
        )
        return self.metrics

    def _build_history(self, X_tr, y_tr, X_te, y_te) -> list[dict]:
        """Staged MSE history for GB learning curves."""
        history: list[dict] = []
        if self.model_type != "gradient_boosting" or self.pipeline is None:
            return history
        reg = self.pipeline.named_steps["reg"]
        sc  = self.pipeline.named_steps["scaler"]
        X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
        n    = len(reg.estimators_)
        step = max(1, n // 20)
        tr_staged = list(reg.staged_predict(X_tr_s))
        te_staged = list(reg.staged_predict(X_te_s))
        for i in range(step - 1, n, step):
            history.append({
                "iteration":  i + 1,
                "train_loss": round(float(mean_squared_error(y_tr, tr_staged[i])), 4),
                "val_loss":   round(float(mean_squared_error(y_te, te_staged[i])), 4),
            })
        return history

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, features: dict | pd.DataFrame) -> dict:
        """Predict consumption MW and advise on peak optimisation."""
        self._ensure_trained()
        df_in    = pd.DataFrame([features]) if isinstance(features, dict) else features.copy()
        available = [c for c in self.feature_cols if c in df_in.columns]
        preds    = self.pipeline.predict(df_in[available].values)
        val      = float(preds[0]) if len(preds) == 1 else float(preds.mean())
        savings  = (
            max(0.0, (val - PEAK_THRESHOLD_MW) / val * 100)
            if val > PEAK_THRESHOLD_MW else 0.0
        )
        return {
            "prediction_mw":          round(val, 2),
            "peak_threshold_mw":      PEAK_THRESHOLD_MW,
            "estimated_savings_pct":  round(savings, 2),
            "optimization_recommended": val > PEAK_THRESHOLD_MW,
        }

    def get_predictions_vs_actual(
        self,
        df: Optional[pd.DataFrame] = None,
        n_samples: int = 96,
    ) -> list[dict]:
        """Last *n_samples* rows with actual vs predicted for the dashboard."""
        self._ensure_trained()
        if df is None:
            df = SFDataLoader().get_processed_features()
        sample    = df.tail(n_samples).reset_index(drop=True)
        available = [c for c in self.feature_cols if c in sample.columns]
        preds     = self.pipeline.predict(sample[available].values)
        return [
            {
                "index":     i,
                "actual":    round(float(sample.at[i, TARGET_COL]), 2),
                "predicted": round(float(preds[i]), 2),
            }
            for i in range(len(sample))
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ensure_trained(self) -> None:
        if self.pipeline is not None:
            return
        if MODEL_PATH.exists():
            loaded: EnergyPredictor = joblib.load(MODEL_PATH)
            self.pipeline          = loaded.pipeline
            self.metrics           = loaded.metrics
            self.feature_cols      = loaded.feature_cols
            self._training_history = loaded._training_history
            logger.info("Loaded pre-trained model from %s", MODEL_PATH)
        else:
            logger.info("No saved model found — training from scratch")
            self.train()

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    @property
    def training_history(self) -> list[dict]:
        return self._training_history
