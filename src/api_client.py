import os
import copy
import math
import sqlite3
import datetime
import functools
import json
import logging
from typing import Dict, Any, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DB = os.path.join(BASE_DIR, "data", "data_cache.db")
# Offline dataset: real API responses recorded for every reference city (python -m src.api_client)
MOCK_DATA_PATH = os.path.join(BASE_DIR, "data", "mock_responses.json")
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


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points (haversine formula)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi, d_lambda = phi2 - phi1, math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def _city_now(utc_offset_seconds: int) -> datetime.datetime:
    """Current local time of a city from its UTC offset (naive, like the API times)."""
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    return now_utc + datetime.timedelta(seconds=utc_offset_seconds)


def _shift_dates(values: List[str], days: int) -> List[str]:
    """Moves ISO dates or date-times ("2026-10-20" / "2026-10-20T14:00") by whole days."""
    delta = datetime.timedelta(days=days)
    shifted = []
    for value in values:
        if "T" in value:
            shifted.append((datetime.datetime.fromisoformat(value) + delta).strftime("%Y-%m-%dT%H:%M"))
        else:
            shifted.append((datetime.date.fromisoformat(value) + delta).isoformat())
    return shifted


@functools.lru_cache(maxsize=4)
def _read_snapshot(path: str) -> Dict[str, Any]:
    """Reads the offline JSON dataset once per process."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class WeatherAPIClient:
    """
    Handles API calls to Open-Meteo REST APIs and manages SQLite local caching.
    Author: Student A (Data / API Engineer)
    """

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def __init__(self, db_path: str = CACHE_DB, mock_path: str = MOCK_DATA_PATH):
        self.db_path = db_path
        self.mock_path = mock_path
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
            return self._load_mock_data(lat, lon)

        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        try:
            logging.info(f"Requesting remote APIs for coordinates ({lat}, {lon})...")
            combined = self._request_api(lat, lon)
            self._set_cached_data(cache_key, combined)
            return combined

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}. Falling back to offline data.")
            return self._load_mock_data(lat, lon)

    def _request_api(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Raw weather and air quality responses, without cache.
        Raises requests.exceptions.RequestException (network error, HTTP error, invalid JSON).
        """
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
        weather_res = requests.get(weather_url, timeout=5)
        air_res = requests.get(air_quality_url, timeout=5)

        weather_res.raise_for_status()
        air_res.raise_for_status()

        return {
            "source": "api",
            "weather": weather_res.json(),
            "air_quality": air_res.json()
        }

    def fetch_current_aqi(
        self, cities: Dict[str, Dict[str, float]], mock_mode: bool = False
    ) -> Dict[str, Optional[float]]:
        """
        Current US AQI of several cities in a single request: Open-Meteo accepts comma-separated
        coordinates and answers with one result per location, in the same order.
        Returns {city name: AQI}, with None when the value is unavailable.
        Offline (mode or network failure), values come from the JSON dataset.
        """
        names = list(cities)
        if mock_mode:
            return self._mock_current_aqi(cities)

        cache_key = "aqi_" + "_".join(f"{c['lat']},{c['lon']}" for c in cities.values())
        cached = self._get_cached_data(cache_key)
        if cached is not None:
            return cached

        try:
            response = requests.get(
                self.AIR_QUALITY_URL,
                params={
                    "latitude": ",".join(str(c["lat"]) for c in cities.values()),
                    "longitude": ",".join(str(c["lon"]) for c in cities.values()),
                    "current": "us_aqi",
                    "timezone": "auto",
                },
                timeout=5,
            )
            response.raise_for_status()
            payload = response.json()
            # A single location comes back as an object, several as a list
            locations = payload if isinstance(payload, list) else [payload]
            result = {name: loc.get("current", {}).get("us_aqi") for name, loc in zip(names, locations)}
            self._set_cached_data(cache_key, result)
            return result
        except requests.exceptions.RequestException as e:
            logging.warning(f"Batch AQI request failed: {e}. Falling back to offline data.")
            return self._mock_current_aqi(cities)

    def _mock_current_aqi(self, cities: Dict[str, Dict[str, float]]) -> Dict[str, Optional[float]]:
        """Current AQI of each city read from the offline dataset (nearest reference city)."""
        result = {}
        for name, coords in cities.items():
            data = self._load_mock_data(coords["lat"], coords["lon"])
            hourly = data.get("air_quality", {}).get("hourly", {})
            offset = data.get("weather", {}).get("utc_offset_seconds", 0)
            current_hour = _city_now(offset).strftime("%Y-%m-%dT%H:00")
            times = hourly.get("time", [])
            result[name] = hourly["us_aqi"][times.index(current_hour)] if current_hour in times else None
        return result

    def _load_mock_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Offline dataset for disconnected mode or network failures: real API responses recorded
        in data/mock_responses.json. The nearest reference city is used, and its dates are moved
        by whole days so that the recording day becomes today (the day/night cycle stays aligned
        with the clock). The "source" and "mock_*" keys let the UI explain where the data comes from.
        """
        try:
            snapshot = _read_snapshot(self.mock_path)
        except (OSError, ValueError) as e:
            logging.error(f"Offline dataset unavailable ({self.mock_path}): {e}")
            return {"source": "mock", "weather": {}, "air_quality": {}}

        city, record = min(
            snapshot["cities"].items(),
            key=lambda item: _distance_km(lat, lon, item[1]["lat"], item[1]["lon"]),
        )
        weather = copy.deepcopy(record["weather"])
        air_quality = copy.deepcopy(record["air_quality"])

        offset = weather.get("utc_offset_seconds", 0)
        captured_utc = datetime.datetime.fromisoformat(snapshot["captured_at_utc"])
        captured_day = (captured_utc + datetime.timedelta(seconds=offset)).date()
        shift_days = (_city_now(offset).date() - captured_day).days

        for block in (weather.get("hourly"), weather.get("daily"), air_quality.get("hourly")):
            if block:
                block["time"] = _shift_dates(block["time"], shift_days)
        if "current_weather" in weather:
            weather["current_weather"]["time"] = _shift_dates([weather["current_weather"]["time"]], shift_days)[0]

        return {
            "source": "mock",
            "mock_city": city,
            "mock_captured_at": snapshot["captured_at_utc"],
            "weather": weather,
            "air_quality": air_quality,
        }

    def capture_mock_snapshot(self, path: Optional[str] = None) -> str:
        """
        Records the current API responses of every reference city into the offline dataset.
        Run with: python -m src.api_client
        """
        path = path or self.mock_path
        snapshot = {
            "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
            "cities": {},
        }
        for name, coords in DEFAULT_CITIES.items():
            logging.info(f"Recording offline data for {name}...")
            response = self._request_api(coords["lat"], coords["lon"])
            snapshot["cities"][name] = {
                "lat": coords["lat"],
                "lon": coords["lon"],
                "weather": response["weather"],
                "air_quality": response["air_quality"],
            }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, separators=(",", ":"))
        _read_snapshot.cache_clear()
        return path


if __name__ == "__main__":
    # Refreshes the offline dataset from the live APIs: python -m src.api_client
    saved_path = WeatherAPIClient().capture_mock_snapshot()
    print(f"Instantané enregistré : {saved_path}")
