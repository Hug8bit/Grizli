"""
SF Data router — GET /api/data/sf
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException

from backend.models import SFDataResponse, SFEnergyPoint, SFStopPoint, NetworkNode

logger = logging.getLogger(__name__)
router = APIRouter(tags=["SF Data"])


@router.get("/sf", response_model=SFDataResponse)
def get_sf_data(energy_sample_size: int = 168):
    """
    Return processed SF data for the map and ML dashboard.
    - energy_sample_size: number of hourly rows to return (default 1 week = 168)
    """
    try:
        from src.data.sf_data_loader import SFDataLoader
        loader = SFDataLoader()

        # Energy data
        energy_df = loader.get_processed_features().tail(energy_sample_size)
        energy_points = []
        for _, row in energy_df.iterrows():
            energy_points.append(SFEnergyPoint(
                hour=int(row.get("hour", 0)),
                day_of_week=int(row.get("day_of_week", 0)),
                month=int(row.get("month", 1)),
                consumption_mw=float(row.get("consumption_mw", 0)),
                temperature_c=float(row.get("temperature_c", 15)),
                humidity_pct=float(row.get("humidity_pct", 70)),
                sfmta_ridership=int(row.get("sfmta_ridership", 100_000)),
                source=str(row.get("source", "synthetic")),
            ))

        # SFMTA stops
        stops_df = loader.load_sfmta_stops().head(200)
        stops = []
        for _, row in stops_df.iterrows():
            stops.append(SFStopPoint(
                stop_id=str(row.get("stop_id", "")),
                stop_name=str(row.get("stop_name", "")),
                latitude=float(row.get("latitude", 37.77)),
                longitude=float(row.get("longitude", -122.42)),
                routes=str(row.get("routes", "")) or None,
            ))

        # Network nodes
        nodes_df = loader.load_network_nodes()
        nodes = []
        for _, row in nodes_df.iterrows():
            nodes.append(NetworkNode(
                node_id=int(row["node_id"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                node_type=str(row["node_type"]),
                nominal_voltage_kv=float(row["nominal_voltage_kv"]),
                zone=str(row["zone"]),
            ))

        # Determine data source
        source = "datasf" if any(p.source == "datasf" for p in energy_points) else "synthetic"

        return SFDataResponse(
            energy_sample=energy_points,
            sfmta_stops=stops,
            network_nodes=nodes,
            data_source=source,
        )

    except Exception as exc:
        logger.exception("SF data fetch failed")
        raise HTTPException(status_code=500, detail=f"SF data fetch failed: {exc}")
