import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


class DataProcessor:
    """
    Data Transformation and Calculations.
    Author: Student B (Data Analyst)
    """

    @staticmethod
    def process_raw_api_data(raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Transforms raw API responses into a clean Pandas DataFrame."""
        w_hourly = raw_data.get("weather", {}).get("hourly", {})
        a_hourly = raw_data.get("air_quality", {}).get("hourly", {})

        df_weather = pd.DataFrame(w_hourly)
        df_air = pd.DataFrame(a_hourly)

        if df_weather.empty or df_air.empty:
            return pd.DataFrame()

        df = pd.merge(df_weather, df_air, on="time", how="inner")
        df["datetime"] = pd.to_datetime(df["time"])
        df.sort_values("datetime", inplace=True)
        return df

    @staticmethod
    def process_daily_data(raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Daily aggregates (min/max temperature, precipitation total) indexed by calendar date."""
        daily = pd.DataFrame(raw_data.get("weather", {}).get("daily", {}))
        if daily.empty:
            return daily
        daily["date"] = pd.to_datetime(daily["time"]).dt.date
        return daily.set_index("date")

    @staticmethod
    def calculate_custom_health_index(df: pd.DataFrame) -> pd.Series:
        """
        Calculates a custom composite Comfort/Health Score (0-100 scale).
        Higher is better. Uses the apparent ("feels like") temperature when available.
        """
        temp = df["apparent_temperature"] if "apparent_temperature" in df else df["temperature_2m"]
        humidity = df["relativehumidity_2m"]
        pm2_5 = df["pm2_5"]

        # Penalty factors based on optimal comfort conditions
        temp_penalty = np.abs(temp - 21.0) * 1.5
        humidity_penalty = np.abs(humidity - 50.0) * 0.3
        pollution_penalty = pm2_5 * 1.2

        score = 100 - (temp_penalty + humidity_penalty + pollution_penalty)
        return score.clip(0, 100).round(1)

    @staticmethod
    def get_reference_time(raw_data: Dict[str, Any]) -> datetime.datetime:
        """
        Current time expressed in the timezone of the API data.
        Open-Meteo returns naive local times along with their UTC offset.
        """
        offset = raw_data.get("weather", {}).get("utc_offset_seconds")
        if offset is None:
            # Mock data is generated with the machine's local time
            return datetime.datetime.now()
        now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        return now_utc + datetime.timedelta(seconds=offset)

    @staticmethod
    def get_current_index(df: pd.DataFrame, reference_time: Optional[datetime.datetime] = None) -> int:
        """
        Position of the most recent row that is not in the future.
        The API also returns several days of forecasts, so the last row is not "now".
        """
        if reference_time is None:
            return len(df) - 1
        not_future = (df["datetime"] <= reference_time).to_numpy()
        if not not_future.any():
            return 0
        return int(np.flatnonzero(not_future)[-1])