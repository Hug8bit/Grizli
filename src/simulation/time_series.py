"""
Time series simulation for 24-hour scenarios.
Generates realistic solar, wind, and load profiles.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pandapower as pp


class Season(Enum):
    """Season affecting renewable generation profiles."""
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"


@dataclass
class TimeSeriesConfig:
    """Configuration for time series simulation."""
    duration_hours: int = 24
    resolution_minutes: int = 15
    season: Season = Season.SUMMER
    cloud_variability: float = 0.2
    wind_variability: float = 0.3
    load_variability: float = 0.1

    @property
    def num_steps(self) -> int:
        """Number of time steps."""
        return self.duration_hours * 60 // self.resolution_minutes

    @property
    def time_index(self) -> pd.DatetimeIndex:
        """Generate time index."""
        return pd.date_range(
            start='2024-01-01',
            periods=self.num_steps,
            freq=f'{self.resolution_minutes}min'
        )


class ScenarioGenerator:
    """
    Generates realistic time-varying profiles for grid simulation.

    Supports solar, wind, and various load types with seasonal
    and random variations.
    """

    def __init__(
        self,
        config: Optional[TimeSeriesConfig] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the scenario generator.

        Args:
            config: Time series configuration
            seed: Random seed for reproducibility
        """
        self.config = config or TimeSeriesConfig()
        self._rng = np.random.default_rng(seed)

    def generate_solar_profile(
        self,
        capacity_mw: float = 1.0,
        peak_hour: float = 12.0
    ) -> pd.Series:
        """
        Generate solar generation profile.

        Uses Gaussian curve centered at peak_hour with cloud variability.

        Args:
            capacity_mw: Peak capacity in MW
            peak_hour: Hour of peak generation (default noon)

        Returns:
            Series with solar output over time
        """
        hours = np.arange(self.config.num_steps) * self.config.resolution_minutes / 60

        # Seasonal adjustment
        seasonal_factor = {
            Season.SUMMER: 1.0,
            Season.SPRING: 0.85,
            Season.AUTUMN: 0.75,
            Season.WINTER: 0.5,
        }[self.config.season]

        # Day length adjustment
        day_length = {
            Season.SUMMER: 14,
            Season.SPRING: 12,
            Season.AUTUMN: 11,
            Season.WINTER: 9,
        }[self.config.season]

        sunrise = peak_hour - day_length / 2
        sunset = peak_hour + day_length / 2

        # Gaussian profile
        sigma = day_length / 4
        base_profile = np.exp(-((hours - peak_hour) ** 2) / (2 * sigma ** 2))

        # Zero outside daylight hours
        base_profile = np.where(
            (hours >= sunrise) & (hours <= sunset),
            base_profile,
            0
        )

        # Add cloud variability
        noise = self._rng.normal(0, self.config.cloud_variability, len(hours))
        noise = np.clip(noise, -0.5, 0.5)

        profile = base_profile * (1 + noise) * seasonal_factor * capacity_mw
        profile = np.maximum(profile, 0)

        return pd.Series(
            profile,
            index=self.config.time_index,
            name='solar_mw'
        )

    def generate_wind_profile(
        self,
        capacity_mw: float = 1.0,
        base_capacity_factor: float = 0.35
    ) -> pd.Series:
        """
        Generate wind generation profile.

        Wind tends to be higher at night and in winter.

        Args:
            capacity_mw: Rated capacity in MW
            base_capacity_factor: Average capacity factor

        Returns:
            Series with wind output over time
        """
        hours = np.arange(self.config.num_steps) * self.config.resolution_minutes / 60

        # Seasonal adjustment (more wind in winter)
        seasonal_factor = {
            Season.WINTER: 1.3,
            Season.SPRING: 1.1,
            Season.AUTUMN: 1.0,
            Season.SUMMER: 0.8,
        }[self.config.season]

        # Diurnal pattern (stronger at night)
        diurnal = 1 + 0.2 * np.cos(2 * np.pi * (hours - 4) / 24)

        # Correlated noise (wind doesn't change instantly)
        noise = np.zeros(self.config.num_steps)
        noise[0] = self._rng.normal(0, self.config.wind_variability)
        for i in range(1, len(noise)):
            noise[i] = 0.8 * noise[i-1] + 0.2 * self._rng.normal(
                0, self.config.wind_variability
            )

        profile = (
            base_capacity_factor *
            seasonal_factor *
            diurnal *
            (1 + noise) *
            capacity_mw
        )
        profile = np.clip(profile, 0, capacity_mw)

        return pd.Series(
            profile,
            index=self.config.time_index,
            name='wind_mw'
        )

    def generate_residential_load(
        self,
        peak_mw: float = 1.0
    ) -> pd.Series:
        """
        Generate residential load profile.

        Two peaks: morning (~8h) and evening (~19h).

        Args:
            peak_mw: Peak load in MW

        Returns:
            Series with load over time
        """
        hours = np.arange(self.config.num_steps) * self.config.resolution_minutes / 60

        # Double Gaussian for morning and evening peaks
        morning_peak = np.exp(-((hours - 8) ** 2) / 8)
        evening_peak = np.exp(-((hours - 19) ** 2) / 6)

        # Base load (never zero)
        base = 0.3

        profile = base + 0.3 * morning_peak + 0.7 * evening_peak

        # Seasonal adjustment (more in winter for heating)
        seasonal_factor = {
            Season.WINTER: 1.2,
            Season.SUMMER: 1.1,  # AC
            Season.SPRING: 0.9,
            Season.AUTUMN: 0.95,
        }[self.config.season]

        # Add variability
        noise = self._rng.normal(0, self.config.load_variability, len(hours))
        profile = profile * (1 + noise) * seasonal_factor * peak_mw
        profile = np.maximum(profile, 0.1 * peak_mw)

        return pd.Series(
            profile,
            index=self.config.time_index,
            name='residential_mw'
        )

    def generate_industrial_load(
        self,
        peak_mw: float = 1.0
    ) -> pd.Series:
        """
        Generate industrial load profile.

        Flat during working hours (6h-22h), low at night.

        Args:
            peak_mw: Peak load in MW

        Returns:
            Series with load over time
        """
        hours = np.arange(self.config.num_steps) * self.config.resolution_minutes / 60

        # Flat during day, low at night
        profile = np.where(
            (hours >= 6) & (hours <= 22),
            0.9,
            0.3
        )

        # Smooth transitions
        for i in range(1, len(profile)):
            if abs(profile[i] - profile[i-1]) > 0.1:
                profile[i] = 0.7 * profile[i] + 0.3 * profile[i-1]

        # Add variability
        noise = self._rng.normal(0, self.config.load_variability * 0.5, len(hours))
        profile = profile * (1 + noise) * peak_mw
        profile = np.maximum(profile, 0.2 * peak_mw)

        return pd.Series(
            profile,
            index=self.config.time_index,
            name='industrial_mw'
        )

    def generate_commercial_load(
        self,
        peak_mw: float = 1.0
    ) -> pd.Series:
        """
        Generate commercial load profile.

        Peak during business hours (8h-20h).

        Args:
            peak_mw: Peak load in MW

        Returns:
            Series with load over time
        """
        hours = np.arange(self.config.num_steps) * self.config.resolution_minutes / 60

        # Trapezoidal profile
        profile = np.zeros_like(hours)
        for i, h in enumerate(hours):
            if h < 7:
                profile[i] = 0.2
            elif h < 8:
                profile[i] = 0.2 + 0.7 * (h - 7)
            elif h < 18:
                profile[i] = 0.9
            elif h < 20:
                profile[i] = 0.9 - 0.35 * (h - 18)
            else:
                profile[i] = 0.2

        # Add variability
        noise = self._rng.normal(0, self.config.load_variability, len(hours))
        profile = profile * (1 + noise) * peak_mw
        profile = np.maximum(profile, 0.1 * peak_mw)

        return pd.Series(
            profile,
            index=self.config.time_index,
            name='commercial_mw'
        )

    def generate_network_scenario(
        self,
        net: pp.pandapowerNet
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate complete scenario for a network.

        Creates profiles for all generators and loads in the network.

        Args:
            net: PandaPower network

        Returns:
            Dictionary with 'sgen' and 'load' DataFrames
        """
        # Generator profiles
        sgen_profiles = pd.DataFrame(index=self.config.time_index)
        if len(net.sgen) > 0:
            for idx in net.sgen.index:
                gen_type = net.sgen.at[idx, 'type'] if 'type' in net.sgen.columns else 'PV'
                capacity = net.sgen.at[idx, 'p_mw']

                if gen_type == 'PV':
                    profile = self.generate_solar_profile(capacity)
                else:  # Wind
                    profile = self.generate_wind_profile(capacity)

                sgen_profiles[f'sgen_{idx}'] = profile.values

        # Load profiles
        load_profiles = pd.DataFrame(index=self.config.time_index)
        if len(net.load) > 0:
            for idx in net.load.index:
                name = net.load.at[idx, 'name'] if 'name' in net.load.columns else ''
                peak = net.load.at[idx, 'p_mw']

                if 'industrial' in name.lower():
                    profile = self.generate_industrial_load(peak)
                elif 'commercial' in name.lower():
                    profile = self.generate_commercial_load(peak)
                else:  # Default residential
                    profile = self.generate_residential_load(peak)

                load_profiles[f'load_{idx}'] = profile.values

        return {
            'sgen': sgen_profiles,
            'load': load_profiles
        }


class TimeSeriesSimulator:
    """
    Runs time series power flow simulation.

    Applies time-varying profiles and collects results.
    """

    def __init__(self, net: pp.pandapowerNet):
        """
        Initialize the time series simulator.

        Args:
            net: PandaPower network
        """
        self.net = net
        self._results: Optional[pd.DataFrame] = None

    def run_simulation(
        self,
        sgen_profiles: pd.DataFrame,
        load_profiles: pd.DataFrame,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Run time series simulation.

        Args:
            sgen_profiles: Generator power profiles (columns: sgen_0, sgen_1, ...)
            load_profiles: Load power profiles (columns: load_0, load_1, ...)
            verbose: Print progress

        Returns:
            DataFrame with results for each timestep
        """
        results = []
        time_index = sgen_profiles.index if len(sgen_profiles) > 0 else load_profiles.index

        # Store original values
        original_sgen = self.net.sgen.p_mw.copy() if len(self.net.sgen) > 0 else None
        original_load = self.net.load.p_mw.copy() if len(self.net.load) > 0 else None

        try:
            for i, t in enumerate(time_index):
                # Apply profiles
                for col in sgen_profiles.columns:
                    idx = int(col.split('_')[1])
                    if idx < len(self.net.sgen):
                        self.net.sgen.at[idx, 'p_mw'] = sgen_profiles.at[t, col]

                for col in load_profiles.columns:
                    idx = int(col.split('_')[1])
                    if idx < len(self.net.load):
                        self.net.load.at[idx, 'p_mw'] = load_profiles.at[t, col]

                # Run power flow
                try:
                    pp.runpp(self.net, numba=True)
                    converged = True
                except Exception:
                    converged = False

                # Collect results
                if converged:
                    max_loading = self.net.res_line.loading_percent.max()
                    total_losses = self.net.res_line.pl_mw.sum()
                    min_voltage = self.net.res_bus.vm_pu.min()
                    max_voltage = self.net.res_bus.vm_pu.max()
                    num_overloaded = (self.net.res_line.loading_percent > 100).sum()
                else:
                    max_loading = 0
                    total_losses = 0
                    min_voltage = 1
                    max_voltage = 1
                    num_overloaded = 0

                total_gen = self.net.sgen.p_mw.sum() if len(self.net.sgen) > 0 else 0
                total_load = self.net.load.p_mw.sum() if len(self.net.load) > 0 else 0

                results.append({
                    'time': t,
                    'converged': converged,
                    'max_loading_percent': max_loading,
                    'total_losses_mw': total_losses,
                    'min_voltage_pu': min_voltage,
                    'max_voltage_pu': max_voltage,
                    'num_overloaded': num_overloaded,
                    'total_generation_mw': total_gen,
                    'total_load_mw': total_load,
                })

                if verbose and (i + 1) % 10 == 0:
                    print(f"Step {i+1}/{len(time_index)}: "
                          f"Max loading={max_loading:.1f}%, "
                          f"Overloaded={num_overloaded}")

        finally:
            # Restore original values
            if original_sgen is not None:
                self.net.sgen.p_mw = original_sgen
            if original_load is not None:
                self.net.load.p_mw = original_load

        self._results = pd.DataFrame(results)
        return self._results

    def get_congestion_statistics(self) -> Dict:
        """
        Get congestion statistics from simulation.

        Returns:
            Dictionary with congestion KPIs
        """
        if self._results is None:
            raise ValueError("No simulation results. Run simulation first.")

        df = self._results

        return {
            "total_steps": len(df),
            "converged_steps": int(df.converged.sum()),
            "convergence_rate": float(df.converged.mean()),
            "max_loading_peak": float(df.max_loading_percent.max()),
            "max_loading_mean": float(df.max_loading_percent.mean()),
            "steps_with_overload": int((df.num_overloaded > 0).sum()),
            "overload_rate": float((df.num_overloaded > 0).mean()),
            "total_overloaded_events": int(df.num_overloaded.sum()),
            "total_losses_mwh": float(df.total_losses_mw.sum() * 0.25),  # 15-min resolution
            "mean_losses_mw": float(df.total_losses_mw.mean()),
            "min_voltage_overall": float(df.min_voltage_pu.min()),
            "max_voltage_overall": float(df.max_voltage_pu.max()),
        }

    def get_hourly_summary(self) -> pd.DataFrame:
        """
        Get hourly aggregated results.

        Returns:
            DataFrame with hourly statistics
        """
        if self._results is None:
            raise ValueError("No simulation results. Run simulation first.")

        df = self._results.copy()
        df['hour'] = pd.to_datetime(df['time']).dt.hour

        hourly = df.groupby('hour').agg({
            'max_loading_percent': ['mean', 'max'],
            'total_losses_mw': 'mean',
            'num_overloaded': 'sum',
            'total_generation_mw': 'mean',
            'total_load_mw': 'mean',
        })

        hourly.columns = ['_'.join(col).strip() for col in hourly.columns.values]
        return hourly
