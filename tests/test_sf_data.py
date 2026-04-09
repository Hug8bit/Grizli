"""
Unit tests — SF data loader.
All tests use the synthetic fallback (no network required).
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="module")
def loader():
    from src.data.sf_data_loader import SFDataLoader
    # Patch requests so we always hit the synthetic path
    import unittest.mock as mock
    with mock.patch("src.data.sf_data_loader.requests.get", side_effect=ConnectionError):
        yield SFDataLoader(use_cache=False)


class TestEnergyData:
    def test_returns_dataframe(self, loader):
        df = loader.load_energy_data(n_hours=200)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, loader):
        df = loader.load_energy_data(n_hours=200)
        for col in ("hour", "day_of_week", "month", "consumption_mw",
                    "temperature_c", "humidity_pct", "sfmta_ridership"):
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nans(self, loader):
        df = loader.load_energy_data(n_hours=200)
        assert df[["consumption_mw", "temperature_c", "humidity_pct"]].isna().sum().sum() == 0

    def test_consumption_positive(self, loader):
        df = loader.load_energy_data(n_hours=200)
        assert (df["consumption_mw"] > 0).all()

    def test_hours_in_range(self, loader):
        df = loader.load_energy_data(n_hours=200)
        assert df["hour"].between(0, 23).all()

    def test_correct_length(self, loader):
        df = loader.load_energy_data(n_hours=100)
        assert len(df) == 100


class TestSFMTAStops:
    def test_returns_dataframe(self, loader):
        df = loader.load_sfmta_stops()
        assert isinstance(df, pd.DataFrame)

    def test_has_lat_lon(self, loader):
        df = loader.load_sfmta_stops()
        assert "latitude"  in df.columns
        assert "longitude" in df.columns

    def test_coords_in_sf_bbox(self, loader):
        df = loader.load_sfmta_stops()
        assert df["latitude"].between(37.70, 37.84).all()
        assert df["longitude"].between(-122.53, -122.34).all()


class TestNetworkNodes:
    def test_returns_33_nodes(self, loader):
        df = loader.load_network_nodes()
        assert len(df) == 33

    def test_has_geo_columns(self, loader):
        df = loader.load_network_nodes()
        assert "latitude"  in df.columns
        assert "longitude" in df.columns
        assert "node_type" in df.columns

    def test_node_types_valid(self, loader):
        df = loader.load_network_nodes()
        valid = {"substation", "load", "generator", "junction"}
        assert set(df["node_type"].unique()).issubset(valid)


class TestProcessedFeatures:
    def test_no_nans_in_features(self, loader):
        df = loader.get_processed_features()
        assert df.isna().sum().sum() == 0

    def test_has_target_column(self, loader):
        df = loader.get_processed_features()
        assert "consumption_mw" in df.columns

    def test_positive_length(self, loader):
        df = loader.get_processed_features()
        assert len(df) > 0


class TestCaching:
    def test_cache_returns_same_object(self):
        from src.data.sf_data_loader import SFDataLoader
        import unittest.mock as mock
        with mock.patch("src.data.sf_data_loader.requests.get", side_effect=ConnectionError):
            loader_cached = SFDataLoader(use_cache=True)
            df1 = loader_cached.load_energy_data(n_hours=100)
            df2 = loader_cached.load_energy_data(n_hours=100)
            assert df1 is df2
