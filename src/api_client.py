import os
import sqlite3
import datetime
import json
import logging
from contextlib import closing
from typing import Dict, Any, Optional

import requests
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DB = os.path.join(BASE_DIR, "data", "data_cache.db")
CACHE_TTL_MINUTES = 30
DEFAULT_CITIES = {
    "Paris": {"lat": 48.8566, "lon": 2.3522},
    "Lyon": {"lat": 45.7640, "lon": 4.8357},
    "Marseille": {"lat": 43.2965, "lon": 5.3698},
    "London": {"lat": 51.5074, "lon": -0.1278},
    "Madrid": {"lat": 40.4168, "lon": -3.7038},
    "Tokyo": {"lat": 35.6895, "lon": 139.6917},
    "Sydney": {"lat": -33.8688, "lon": 151.2093},
    "New York": {"lat": 40.7128, "lon": -74.0060},
    "Pyongyang": {"lat": 39.0392, "lon": 125.7625},
    "Abu Dhabi": {"lat": 24.4539, "lon": 54.3773},
    "Berlin": {"lat": 52.5200, "lon": 13.4050},
    "Moscow": {"lat": 55.7558, "lon": 37.6173},
    "Rome": {"lat": 41.9028, "lon": 12.4964}
}


class WeatherAPIClient:
    """
    Handles API calls to Open-Meteo REST APIs and manages SQLite local caching.
    Author: Student A (Data / API Engineer)
    """

    def __init__(self, db_path: str = CACHE_DB):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes SQLite cache database table."""
        try:
            with closing(sqlite3.connect(self.db_path)) as conn, conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_cache (
                        key TEXT PRIMARY KEY,
                        response TEXT,
                        timestamp DATETIME
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")

    def _get_cached_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached json response if within TTL limit."""
        try:
            with closing(sqlite3.connect(self.db_path)) as conn, conn:
                cursor = conn.cursor()
                cursor.execute("SELECT response, timestamp FROM api_cache WHERE key = ?", (key,))
                row = cursor.fetchone()

                if row:
                    cached_response, timestamp_str = row
                    timestamp = datetime.datetime.fromisoformat(timestamp_str)
                    if datetime.datetime.now() - timestamp < datetime.timedelta(minutes=CACHE_TTL_MINUTES):
                        logging.info(f"Cache hit for key: {key}")
                        return json.loads(cached_response)
        except Exception as e:
            logging.warning(f"Failed to read cache for key {key}: {e}")
        return None

    def _set_cached_data(self, key: str, data: Dict[str, Any]) -> None:
        """Saves API response to local SQLite cache."""
        try:
            with closing(sqlite3.connect(self.db_path)) as conn, conn:
                cursor = conn.cursor()
                now_str = datetime.datetime.now().isoformat()
                cursor.execute("""
                    INSERT OR REPLACE INTO api_cache (key, response, timestamp)
                    VALUES (?, ?, ?)
                """, (key, json.dumps(data), now_str))
                conn.commit()
        except Exception as e:
            logging.error(f"Failed to write cache for key {key}: {e}")

    def fetch_weather_and_air_quality(self, lat: float, lon: float, mock_mode: bool = False) -> Dict[str, Any]:
        """
        Fetches combined weather and air quality data from Open-Meteo or local mock generator.
        """
        cache_key = f"data_{lat}_{lon}"

        if mock_mode:
            return self._generate_mock_data(lat, lon)

        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current_weather=true&hourly=temperature_2m,relativehumidity_2m,surface_pressure,windspeed_10m"
            f"&timezone=auto"
        )
        air_quality_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
            f"&hourly=pm10,pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,us_aqi"
            f"&timezone=auto"
        )

        try:
            logging.info(f"Requesting remote APIs for coordinates ({lat}, {lon})...")
            weather_res = requests.get(weather_url, timeout=5)
            air_res = requests.get(air_quality_url, timeout=5)

            weather_res.raise_for_status()
            air_res.raise_for_status()

            combined = {
                "weather": weather_res.json(),
                "air_quality": air_res.json()
            }
            self._set_cached_data(cache_key, combined)
            return combined

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}. Falling back to mock data.")
            return self._generate_mock_data(lat, lon)

    def _generate_mock_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fallback mock data generator for disconnected mode or testing."""
        hours = 24 * 7
        now = datetime.datetime.now()
        times = [(now - datetime.timedelta(hours=i)).isoformat() for i in range(hours)][::-1]

        return {
            "weather": {
                "current_weather": {
                    "temperature": round(np.random.uniform(15, 30), 1),
                    "windspeed": round(np.random.uniform(5, 25), 1),
                    "winddirection": int(np.random.uniform(0, 360))
                },
                "hourly": {
                    "time": times,
                    "temperature_2m": np.random.normal(20, 5, hours).tolist(),
                    "relativehumidity_2m": np.random.uniform(40, 90, hours).tolist(),
                    "surface_pressure": np.random.uniform(1000, 1025, hours).tolist(),
                    "windspeed_10m": np.random.uniform(2, 30, hours).tolist()
                }
            },
            "air_quality": {
                "hourly": {
                    "time": times,
                    "pm2_5": np.random.uniform(5, 35, hours).tolist(),
                    "pm10": np.random.uniform(10, 60, hours).tolist(),
                    "nitrogen_dioxide": np.random.uniform(10, 55, hours).tolist(),
                    "ozone": np.random.uniform(20, 80, hours).tolist(),
                    "sulphur_dioxide": np.random.uniform(1, 15, hours).tolist(),
                    "carbon_monoxide": np.random.uniform(100, 500, hours).tolist(),
                    "us_aqi": np.random.randint(15, 120, hours).tolist()
                }
            }
        }

    def generate_mordor_data(self) -> Dict[str, Any]:
        """Returns deterministic mock weather and air-quality data for Mordor."""
        data = self._generate_mock_data(0, 0)
        n = len(data["weather"]["hourly"]["time"])

        data["weather"]["current_weather"]["temperature"] = 60.0
        data["weather"]["hourly"]["temperature_2m"] = [60.0] * n
        data["weather"]["hourly"]["relativehumidity_2m"] = [5.0] * n
        data["air_quality"]["hourly"]["pm2_5"] = [900.0] * n
        data["air_quality"]["hourly"]["pm10"] = [1200.0] * n
        data["air_quality"]["hourly"]["nitrogen_dioxide"] = [600.0] * n
        data["air_quality"]["hourly"]["us_aqi"] = [500] * n
        return data