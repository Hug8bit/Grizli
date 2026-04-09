"""
San Francisco Open Data loader for GRIZLI.

Tries to fetch live data from DataSF API (https://data.sfgov.org/resource/).
Falls back to realistic synthetic data if the API is unavailable or returns
insufficient rows.

Decision (audit Phase 1): /data directory was empty.  Rather than shipping
large CSV files in the repo, data is fetched at runtime with a synthetic
fallback so the app always works offline.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATASF_BASE_URL = os.getenv("DATASF_BASE_URL", "https://data.sfgov.org/resource")

_DATASETS = {
    "electric_usage": "h9km-ggyk",   # PG&E electric usage by zip code
    "sfmta_stops":    "i28k-bkz6",   # SFMTA bus stop locations
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_datasf(dataset_id: str, limit: int = 1000, **params) -> Optional[pd.DataFrame]:
    """Call DataSF REST API.  Returns None on any network / HTTP error."""
    url = f"{DATASF_BASE_URL}/{dataset_id}.json"
    try:
        resp = requests.get(url, params={"$limit": limit, **params}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            return pd.DataFrame(data)
    except Exception as exc:
        logger.warning("DataSF fetch failed for %s: %s", dataset_id, exc)
    return None


def _synthetic_energy_data(n_hours: int = 8760) -> pd.DataFrame:
    """Realistic synthetic hourly SF electricity consumption (1 year default)."""
    rng = np.random.default_rng(42)
    ts = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    hours  = ts.hour.to_numpy()
    months = ts.month.to_numpy()
    dow    = ts.dayofweek.to_numpy()

    seasonal = 10.0 * np.sin(2 * np.pi * (months - 1) / 12)
    morning  = 15.0 * np.sin(2 * np.pi * (hours - 6)  / 24)
    evening  =  8.0 * np.sin(2 * np.pi * (hours - 18) / 24)
    noise    = rng.normal(0, 3, n_hours)
    consumption = np.clip(50.0 + seasonal + morning + evening + noise, 20, 120)

    temp     = 15.0 + 5 * np.sin(2 * np.pi * (months - 1) / 12) + rng.normal(0, 2, n_hours)
    humidity = np.clip(
        70.0 + 10 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 5, n_hours), 40, 100
    )

    is_weekday = (dow < 5).astype(float)
    peak_shape = np.maximum(np.sin(2 * np.pi * (hours - 8) / 24), 0)
    ridership = np.clip(
        is_weekday * (200_000 + 80_000 * peak_shape + rng.normal(0, 5_000, n_hours))
        + (1 - is_weekday) * (80_000 + rng.normal(0, 5_000, n_hours)),
        0, None,
    ).astype(int)

    return pd.DataFrame({
        "timestamp":       ts,
        "hour":            hours,
        "day_of_week":     dow,
        "month":           months,
        "is_weekend":      (dow >= 5).astype(int),
        "consumption_mw":  consumption.round(2),
        "temperature_c":   temp.round(1),
        "humidity_pct":    humidity.round(1),
        "sfmta_ridership": ridership,
        "source":          "synthetic",
    })


def _synthetic_sfmta_stops(n: int = 200) -> pd.DataFrame:
    """Synthetic SFMTA bus stop locations inside the SF bounding box."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "stop_id":   [f"SF{i:04d}" for i in range(n)],
        "stop_name": [f"Stop {i}" for i in range(n)],
        "latitude":  rng.uniform(37.70, 37.83, n).round(5),
        "longitude": rng.uniform(-122.52, -122.35, n).round(5),
        "routes":    [str(rng.integers(1, 60)) for _ in range(n)],
    })


def _synthetic_network_nodes(n: int = 33) -> pd.DataFrame:
    """IEEE 33-bus nodes mapped onto SF geography."""
    rng   = np.random.default_rng(7)
    types = (["substation"] + ["load"] * 20 + ["generator"] * 6 + ["junction"] * 6)[:n]
    zones = ["Mission", "SoMa", "Castro", "Richmond", "Sunset", "Bayview"]
    return pd.DataFrame({
        "node_id":           list(range(n)),
        "latitude":          (37.7749  + rng.uniform(-0.10, 0.10, n)).round(5),
        "longitude":         (-122.4194 + rng.uniform(-0.15, 0.15, n)).round(5),
        "node_type":         types,
        "nominal_voltage_kv": rng.choice([11.0, 22.0, 33.0], n).tolist(),
        "zone":              rng.choice(zones, n).tolist(),
    })


# ── Public API ────────────────────────────────────────────────────────────────

class SFDataLoader:
    """
    Loads San Francisco energy / SFMTA / geo data for GRIZLI.

    Attempts live DataSF API calls; falls back to synthetic data transparently.
    Results are cached in-memory for the lifetime of the instance.
    """

    def __init__(self, use_cache: bool = True):
        self._cache: dict[str, pd.DataFrame] = {}
        self._use_cache = use_cache

    def load_energy_data(self, n_hours: int = 8760) -> pd.DataFrame:
        """Hourly energy consumption with weather + SFMTA ridership features."""
        key = "energy"
        if self._use_cache and key in self._cache:
            return self._cache[key]

        df = _fetch_datasf(_DATASETS["electric_usage"], limit=5_000)
        if df is not None and len(df) > 100:
            logger.info("Loaded PG&E electric usage data from DataSF (%d rows)", len(df))
            df["source"] = "datasf"
        else:
            logger.info("DataSF unavailable — using synthetic SF energy data")
            df = _synthetic_energy_data(n_hours)

        if self._use_cache:
            self._cache[key] = df
        return df

    def load_sfmta_stops(self) -> pd.DataFrame:
        """SFMTA stop locations (lat/lon)."""
        key = "sfmta"
        if self._use_cache and key in self._cache:
            return self._cache[key]

        df = _fetch_datasf(_DATASETS["sfmta_stops"], limit=2_000)
        if df is not None and {"stop_lat", "stop_lon"}.issubset(df.columns):
            df = df.rename(columns={"stop_lat": "latitude", "stop_lon": "longitude"})
            logger.info("Loaded SFMTA stops from DataSF (%d rows)", len(df))
        else:
            logger.info("DataSF unavailable — using synthetic SFMTA stop data")
            df = _synthetic_sfmta_stops()

        if self._use_cache:
            self._cache[key] = df
        return df

    def load_network_nodes(self) -> pd.DataFrame:
        """Electrical network nodes with SF geo-coordinates."""
        key = "nodes"
        if self._use_cache and key in self._cache:
            return self._cache[key]
        df = _synthetic_network_nodes()
        if self._use_cache:
            self._cache[key] = df
        return df

    def get_processed_features(self) -> pd.DataFrame:
        """Clean feature matrix (no NaNs) ready for ML training."""
        df = self.load_energy_data()
        wanted = [
            "hour", "day_of_week", "month", "is_weekend",
            "temperature_c", "humidity_pct", "sfmta_ridership",
            "consumption_mw",
        ]
        available = [c for c in wanted if c in df.columns]
        return df[available].dropna().reset_index(drop=True)
