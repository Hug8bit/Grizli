"""
Pydantic request / response models for the GRIZLI API.
"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Simulation ────────────────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    network_type:  str   = Field("ieee33", description="ieee33 | ieee33_congested | atacama | synthetic")
    solar_factor:  float = Field(1.0,  ge=0.0, le=3.0)
    wind_factor:   float = Field(1.0,  ge=0.0, le=3.0)
    load_factor:   float = Field(1.0,  ge=0.5, le=2.0)
    optimize:      bool  = Field(False, description="Run fast optimizer after power flow")
    num_buses:     int   = Field(30,   ge=5,   le=100, description="Only used for synthetic network")


class LineResult(BaseModel):
    id:               int
    from_bus:         int
    to_bus:           int
    loading_percent:  float
    in_service:       bool


class NodeResult(BaseModel):
    id:             int
    voltage_pu:     float
    node_type:      str    # slack | load | generator | junction


class SimulateResponse(BaseModel):
    converged:          bool
    max_loading_pct:    float
    total_losses_mw:    float
    num_overloaded:     int
    lines:              list[LineResult]
    nodes:              list[NodeResult]
    optimization_applied: bool
    lines_opened:       list[int]
    lines_closed:       list[int]
    computation_ms:     float


# ── ML ────────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    model_type: str = Field("gradient_boosting",
                            description="gradient_boosting | random_forest | ridge")
    n_hours:    int = Field(8760, ge=100, le=87600)


class TrainResponse(BaseModel):
    mae:           float
    rmse:          float
    r2:            float
    train_samples: int
    test_samples:  int
    model_type:    str


class PredictRequest(BaseModel):
    hour:            int   = Field(..., ge=0,  le=23)
    day_of_week:     int   = Field(..., ge=0,  le=6)
    month:           int   = Field(..., ge=1,  le=12)
    is_weekend:      int   = Field(..., ge=0,  le=1)
    temperature_c:   float = Field(15.0)
    humidity_pct:    float = Field(70.0, ge=0.0, le=100.0)
    sfmta_ridership: int   = Field(150_000, ge=0)


class PredictResponse(BaseModel):
    prediction_mw:            float
    peak_threshold_mw:        float
    estimated_savings_pct:    float
    optimization_recommended: bool


class TrainingCurvePoint(BaseModel):
    iteration:  int
    train_loss: float
    val_loss:   float


class MLStatusResponse(BaseModel):
    is_trained:      bool
    metrics:         dict[str, Any]
    training_history: list[TrainingCurvePoint]
    predictions_vs_actual: list[dict[str, Any]]


# ── SF Data ───────────────────────────────────────────────────────────────────

class SFEnergyPoint(BaseModel):
    hour:            int
    day_of_week:     int
    month:           int
    consumption_mw:  float
    temperature_c:   float
    humidity_pct:    float
    sfmta_ridership: int
    source:          str


class SFStopPoint(BaseModel):
    stop_id:   str
    stop_name: str
    latitude:  float
    longitude: float
    routes:    Optional[str] = None


class NetworkNode(BaseModel):
    node_id:             int
    latitude:            float
    longitude:           float
    node_type:           str
    nominal_voltage_kv:  float
    zone:                str


class SFDataResponse(BaseModel):
    energy_sample:   list[SFEnergyPoint]
    sfmta_stops:     list[SFStopPoint]
    network_nodes:   list[NetworkNode]
    data_source:     str
