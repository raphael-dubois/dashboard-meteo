import os
import sqlite3
import datetime
import json
import logging
from typing import Dict, Any, Optional

import requests
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DB = os.path.join(BASE_DIR, "data", "data_cache.db")
CACHE_TTL_MINUTES = 30
# Days of history requested from the APIs (needed for 7-day trends)
PAST_DAYS = 7
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

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

    def __init__(self, db_path: str = CACHE_DB):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def search_cities(self, query: str, count: int = 8, language: str = "fr") -> list[Dict[str, Any]]:
        """
        Searches for cities using the Open-Meteo Geocoding API.
        Returns a list of dicts with keys: name, label, latitude, longitude, country, admin1, population.
        Results are cached locally to avoid redundant network calls.
        """
        query = query.strip()
        if len(query) < 2:
            return []

        cache_key = f"geocoding_{query.lower()}_{language}_{count}"
        cached = self._get_cached_data(cache_key)
        if cached is not None:
            return cached

        try:
            response = requests.get(
                self.GEOCODING_URL,
                params={"name": query, "count": count, "language": language, "format": "json"},
                timeout=4,
            )
            response.raise_for_status()
            results = response.json().get("results", [])

            cities = []
            for r in results:
                parts = [r.get("name", "")]
                if r.get("admin1"):
                    parts.append(r["admin1"])
                if r.get("country"):
                    parts.append(r["country"])
                cities.append({
                    "name": r.get("name", ""),
                    "label": ", ".join(parts),
                    "lat": r.get("latitude", 0.0),
                    "lon": r.get("longitude", 0.0),
                    "country": r.get("country", ""),
                    "admin1": r.get("admin1", ""),
                    "population": r.get("population", 0),
                })

            self._set_cached_data(cache_key, cities)
            return cities

        except requests.exceptions.RequestException as e:
            logging.warning(f"Geocoding request failed: {e}")
            return []

    def _init_db(self) -> None:
        """Initializes SQLite cache database table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
            with sqlite3.connect(self.db_path) as conn:
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
            with sqlite3.connect(self.db_path) as conn:
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
        # "v2": responses now include 7 days of history and extra variables
        cache_key = f"data_v2_{lat}_{lon}"

        if mock_mode:
            return self._generate_mock_data(lat, lon)

        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&hourly=temperature_2m,relativehumidity_2m,apparent_temperature,precipitation,cloud_cover,"
            f"surface_pressure,windspeed_10m,wind_direction_10m"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&past_days={PAST_DAYS}&timezone=auto"
        )
        air_quality_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
            f"&hourly=pm10,pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,us_aqi"
            f"&past_days={PAST_DAYS}&timezone=auto"
        )

        try:
            logging.info(f"Requesting remote APIs for coordinates ({lat}, {lon})...")
            weather_res = requests.get(weather_url, timeout=5)
            air_res = requests.get(air_quality_url, timeout=5)

            weather_res.raise_for_status()
            air_res.raise_for_status()

            combined = {
                "source": "api",
                "weather": weather_res.json(),
                "air_quality": air_res.json()
            }
            self._set_cached_data(cache_key, combined)
            return combined

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}. Falling back to mock data.")
            return self._generate_mock_data(lat, lon)

    def _generate_mock_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fallback mock data generator for disconnected mode or testing.
        The "source" key lets the UI warn that the data is simulated.
        """
        hours = 24 * 7
        now = datetime.datetime.now()
        times = [(now - datetime.timedelta(hours=i)).isoformat() for i in range(hours)][::-1]
        temperatures = np.random.normal(20, 5, hours)
        precipitation = np.where(np.random.rand(hours) < 0.15, np.random.uniform(0.1, 3.0, hours), 0.0)

        # Daily aggregates derived from the hourly series, like the API's "daily" block
        days = sorted({t[:10] for t in times})
        day_masks = [np.array([t.startswith(day) for t in times]) for day in days]

        return {
            "source": "mock",
            "weather": {
                "current_weather": {
                    "temperature": round(np.random.uniform(15, 30), 1),
                    "windspeed": round(np.random.uniform(5, 25), 1),
                    "winddirection": int(np.random.uniform(0, 360))
                },
                "hourly": {
                    "time": times,
                    "temperature_2m": temperatures.tolist(),
                    "relativehumidity_2m": np.random.uniform(40, 90, hours).tolist(),
                    "apparent_temperature": (temperatures + np.random.uniform(-3, 2, hours)).tolist(),
                    "precipitation": precipitation.tolist(),
                    "cloud_cover": np.random.uniform(0, 100, hours).tolist(),
                    "surface_pressure": np.random.uniform(1000, 1025, hours).tolist(),
                    "windspeed_10m": np.random.uniform(2, 30, hours).tolist(),
                    "wind_direction_10m": np.random.uniform(0, 360, hours).tolist()
                },
                "daily": {
                    "time": days,
                    "temperature_2m_max": [float(temperatures[mask].max()) for mask in day_masks],
                    "temperature_2m_min": [float(temperatures[mask].min()) for mask in day_masks],
                    "precipitation_sum": [float(precipitation[mask].sum()) for mask in day_masks]
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