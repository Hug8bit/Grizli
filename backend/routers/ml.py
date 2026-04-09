"""
ML router — POST /api/ml/train, GET /api/ml/predict, GET /api/ml/status
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException

from backend.models import (
    TrainRequest, TrainResponse,
    PredictRequest, PredictResponse,
    MLStatusResponse, TrainingCurvePoint,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Machine Learning"])

# Module-level singleton predictor (shared across requests in single worker)
_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from src.ml.energy_predictor import EnergyPredictor
        _predictor = EnergyPredictor()
    return _predictor


@router.post("/train", response_model=TrainResponse)
def train_model(req: TrainRequest):
    """Train (or retrain) the ML energy predictor on SF data."""
    try:
        from src.data.sf_data_loader import SFDataLoader
        from src.ml.energy_predictor import EnergyPredictor

        global _predictor
        _predictor = EnergyPredictor(model_type=req.model_type)
        loader = SFDataLoader()
        df = loader.get_processed_features()
        # Limit rows if requested
        if req.n_hours < len(df):
            df = df.head(req.n_hours)

        metrics = _predictor.train(df)
        return TrainResponse(**metrics)
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


@router.get("/predict", response_model=PredictResponse)
def predict(
    hour: int = 12,
    day_of_week: int = 1,
    month: int = 6,
    is_weekend: int = 0,
    temperature_c: float = 18.0,
    humidity_pct: float = 65.0,
    sfmta_ridership: int = 180_000,
):
    """Predict hourly energy consumption and return optimisation advice."""
    try:
        predictor = _get_predictor()
        result = predictor.predict({
            "hour": hour, "day_of_week": day_of_week, "month": month,
            "is_weekend": is_weekend, "temperature_c": temperature_c,
            "humidity_pct": humidity_pct, "sfmta_ridership": sfmta_ridership,
        })
        return PredictResponse(**result)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@router.get("/status", response_model=MLStatusResponse)
def ml_status(n_samples: int = 96):
    """Return training metrics, learning curves, and prediction samples."""
    try:
        predictor = _get_predictor()
        predictor._ensure_trained()
        history = [
            TrainingCurvePoint(**h) for h in predictor.training_history
        ]
        preds = predictor.get_predictions_vs_actual(n_samples=n_samples)
        return MLStatusResponse(
            is_trained=predictor.is_trained,
            metrics=predictor.metrics,
            training_history=history,
            predictions_vs_actual=preds,
        )
    except Exception as exc:
        logger.exception("ML status failed")
        raise HTTPException(status_code=500, detail=f"ML status failed: {exc}")
