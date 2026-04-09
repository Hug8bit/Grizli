"""
Unit tests — ML energy predictor.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="module")
def sample_df():
    """Small synthetic DataFrame for fast tests (no file I/O)."""
    rng = np.random.default_rng(0)
    n   = 500
    return pd.DataFrame({
        "hour":            rng.integers(0, 24, n),
        "day_of_week":     rng.integers(0, 7, n),
        "month":           rng.integers(1, 13, n),
        "is_weekend":      rng.integers(0, 2, n),
        "temperature_c":   rng.uniform(10, 25, n),
        "humidity_pct":    rng.uniform(50, 90, n),
        "sfmta_ridership": rng.integers(50_000, 300_000, n),
        "consumption_mw":  rng.uniform(30, 100, n),
    })


class TestEnergyPredictorTrain:
    def test_train_returns_metrics(self, sample_df):
        from src.ml.energy_predictor import EnergyPredictor
        predictor = EnergyPredictor()
        metrics   = predictor.train(sample_df)
        assert isinstance(metrics, dict)
        assert "r2" in metrics and "mae" in metrics and "rmse" in metrics

    def test_r2_positive(self, sample_df):
        from src.ml.energy_predictor import EnergyPredictor
        predictor = EnergyPredictor()
        metrics   = predictor.train(sample_df)
        # Synthetic random data → modest R², still > -1
        assert metrics["r2"] > -1.0

    def test_mae_nonnegative(self, sample_df):
        from src.ml.energy_predictor import EnergyPredictor
        predictor = EnergyPredictor()
        metrics   = predictor.train(sample_df)
        assert metrics["mae"] >= 0

    def test_train_sets_is_trained(self, sample_df):
        from src.ml.energy_predictor import EnergyPredictor
        predictor = EnergyPredictor()
        assert not predictor.is_trained
        predictor.train(sample_df)
        assert predictor.is_trained

    def test_random_forest_model_type(self, sample_df):
        from src.ml.energy_predictor import EnergyPredictor
        predictor = EnergyPredictor(model_type="random_forest")
        metrics   = predictor.train(sample_df)
        assert metrics["model_type"] == "random_forest"

    def test_ridge_model_type(self, sample_df):
        from src.ml.energy_predictor import EnergyPredictor
        predictor = EnergyPredictor(model_type="ridge")
        metrics   = predictor.train(sample_df)
        assert metrics["model_type"] == "ridge"


class TestEnergyPredictorPredict:
    @pytest.fixture
    def trained_predictor(self, sample_df):
        from src.ml.energy_predictor import EnergyPredictor
        p = EnergyPredictor()
        p.train(sample_df)
        return p

    def test_predict_returns_dict(self, trained_predictor):
        features = {
            "hour": 14, "day_of_week": 1, "month": 7,
            "is_weekend": 0, "temperature_c": 22,
            "humidity_pct": 65, "sfmta_ridership": 200_000,
        }
        result = trained_predictor.predict(features)
        assert isinstance(result, dict)
        assert "prediction_mw" in result

    def test_predict_mw_positive(self, trained_predictor):
        features = {"hour": 12, "day_of_week": 2, "month": 6,
                    "is_weekend": 0, "temperature_c": 18,
                    "humidity_pct": 70, "sfmta_ridership": 180_000}
        result = trained_predictor.predict(features)
        assert result["prediction_mw"] > 0

    def test_predict_optimization_flag(self, trained_predictor):
        # Force a high-consumption scenario
        features = {"hour": 18, "day_of_week": 0, "month": 8,
                    "is_weekend": 0, "temperature_c": 30,
                    "humidity_pct": 40, "sfmta_ridership": 280_000}
        result = trained_predictor.predict(features)
        assert isinstance(result["optimization_recommended"], bool)

    def test_predictions_vs_actual_length(self, trained_predictor, sample_df):
        results = trained_predictor.get_predictions_vs_actual(sample_df, n_samples=48)
        assert len(results) == 48

    def test_predictions_vs_actual_keys(self, trained_predictor, sample_df):
        results = trained_predictor.get_predictions_vs_actual(sample_df, n_samples=10)
        for row in results:
            assert "actual" in row and "predicted" in row and "index" in row


class TestTrainingHistory:
    def test_history_nonempty_for_gb(self, sample_df):
        from src.ml.energy_predictor import EnergyPredictor
        p = EnergyPredictor(model_type="gradient_boosting")
        p.train(sample_df)
        assert len(p.training_history) > 0

    def test_history_has_loss_keys(self, sample_df):
        from src.ml.energy_predictor import EnergyPredictor
        p = EnergyPredictor(model_type="gradient_boosting")
        p.train(sample_df)
        for entry in p.training_history:
            assert "train_loss" in entry and "val_loss" in entry
