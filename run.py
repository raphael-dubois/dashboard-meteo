import os
import sqlite3
import datetime
import json
import logging
import unittest
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CACHE_DB = "data_cache.db"
CACHE_TTL_MINUTES = 30
DEFAULT_CITIES = {
    "Paris": {"lat": 48.8566, "lon": 2.3522},
    "Lyon": {"lat": 45.7640, "lon": 4.8357},
    "Marseille": {"lat": 43.2965, "lon": 5.3698},
    "London": {"lat": 51.5074, "lon": -0.1278},
    "Madrid": {"lat": 40.4168, "lon": -3.7038},
}

# Thresholds for alerts (WHO Guidelines)
THRESHOLDS = {
    "pm2_5": {"warning": 15.0, "danger": 25.0, "unit": "µg/m³"},
    "pm10": {"warning": 45.0, "danger": 50.0, "unit": "µg/m³"},
    "no2": {"warning": 25.0, "danger": 50.0, "unit": "µg/m³"},
    "temp_max": {"warning": 33.0, "danger": 38.0, "unit": "°C"}
}


class WeatherAPIClient:
    """
    Handles API calls to Open-Meteo REST APIs and manages SQLite local caching.
    Author: Student A (Data / API Engineer)
    """

    def __init__(self, db_path: str = CACHE_DB):
        self.db_path = db_path
        self._init_db()

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
        cache_key = f"data_{lat}_{lon}"
        
        if mock_mode:
            return self._generate_mock_data(lat, lon)

        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current_weather=true&hourly=temperature_2m,relativehumidity_2m,surface_pressure,windspeed_10m"
        )
        air_quality_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
            f"&hourly=pm10,pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,us_aqi"
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


class DataProcessor:
    """
    Data Transformation, Calculations, and Alert Engine.
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
    def calculate_custom_health_index(df: pd.DataFrame) -> pd.Series:
        """
        Calculates a custom composite Comfort/Health Score (0-100 scale).
        Higher is better.
        """
        temp = df["temperature_2m"]
        humidity = df["relativehumidity_2m"]
        pm2_5 = df["pm2_5"]

        # Penalty factors based on optimal comfort conditions
        temp_penalty = np.abs(temp - 21.0) * 1.5
        humidity_penalty = np.abs(humidity - 50.0) * 0.3
        pollution_penalty = pm2_5 * 1.2

        score = 100 - (temp_penalty + humidity_penalty + pollution_penalty)
        return score.clip(0, 100).round(1)

    @staticmethod
    def detect_alerts(df: pd.DataFrame) -> List[Dict[str, str]]:
        """Scans current/recent data points for threshold violations."""
        alerts = []
        if df.empty:
            return alerts

        latest = df.iloc[-1]

        # PM2.5 check
        if latest["pm2_5"] >= THRESHOLDS["pm2_5"]["danger"]:
            alerts.append({
                "level": "danger",
                "message": f"Alerte Rouge Pollution (PM2.5) : {latest['pm2_5']:.1f} {THRESHOLDS['pm2_5']['unit']} (Seuil: {THRESHOLDS['pm2_5']['danger']})"
            })
        elif latest["pm2_5"] >= THRESHOLDS["pm2_5"]["warning"]:
            alerts.append({
                "level": "warning",
                "message": f"Vigilance Pollution (PM2.5) : {latest['pm2_5']:.1f} {THRESHOLDS['pm2_5']['unit']}"
            })

        # NO2 check
        if latest["nitrogen_dioxide"] >= THRESHOLDS["no2"]["danger"]:
            alerts.append({
                "level": "danger",
                "message": f"Alerte Dépassement NO2 : {latest['nitrogen_dioxide']:.1f} µg/m³"
            })

        # Temperature check
        if latest["temperature_2m"] >= THRESHOLDS["temp_max"]["warning"]:
            alerts.append({
                "level": "warning",
                "message": f"Vigilance Forte Chaleur / Canicule : {latest['temperature_2m']:.1f} °C"
            })

        return alerts


def render_dashboard():
    """
    Renders Streamlit Web Application Interface.
    Author: Student C (Lead UI/UX)
    """
    st.set_page_config(page_title="Dashboard Météo & AQI", page_icon="🌤️", layout="wide")

    st.title("🌤️ Dashboard Météo & Surveillance Qualité de l'Air")
    st.markdown("Projet 3ᵉ Année Ingénieur — Analyse de données environnementales en temps réel")

    client = WeatherAPIClient()

    # Sidebar Options
    st.sidebar.header("⚙️ Configuration")
    selected_city = st.sidebar.selectbox("Sélectionner une ville", list(DEFAULT_CITIES.keys()))
    mock_mode = st.sidebar.checkbox("Mode Déconnecté / Data Mocked", value=False)

    city_info = DEFAULT_CITIES[selected_city]
    
    # Fetch Data
    with st.spinner(f"Chargement des données pour {selected_city}..."):
        raw_data = client.fetch_weather_and_air_quality(city_info["lat"], city_info["lon"], mock_mode=mock_mode)
        df = DataProcessor.process_raw_api_data(raw_data)

    if df.empty:
        st.error("Impossible de récupérer ou d'afficher les données.")
        return

    # Add calculated metrics
    df["health_score"] = DataProcessor.calculate_custom_health_index(df)
    latest = df.iloc[-1]

    # Alert Box Section
    alerts = DataProcessor.detect_alerts(df)
    if alerts:
        for alert in alerts:
            if alert["level"] == "danger":
                st.error(f"🚨 **DANGER** : {alert['message']}")
            else:
                st.warning(f"⚠️ **ATTENTION** : {alert['message']}")
    else:
        st.success("✅ Qualité de l'air et conditions météorologiques stables. Aucune alerte active.")

    # KPI Summary Cards
    aqi = latest["us_aqi"]
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Température", f"{latest['temperature_2m']:.1f} °C")
    col2.metric("Humidité", f"{latest['relativehumidity_2m']:.0f} %")
    col3.metric("Indice AQI (US)", "N/D" if pd.isna(aqi) else str(int(aqi)))
    col4.metric("PM 2.5", f"{latest['pm2_5']:.1f} µg/m³")
    col5.metric("Score Confort (0-100)", f"{latest['health_score']:.1f}")

    # Visualizations
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Évolution Météo", "🏭 Qualité de l'Air", "🗺️ Carte Interactive"])

    with tab1:
        st.subheader("Température et Humidité (7 derniers jours / Prévisions)")
        fig_temp = px.line(
            df, x="datetime", y=["temperature_2m", "relativehumidity_2m"],
            labels={"value": "Valeur", "datetime": "Date/Heure", "variable": "Métrique"},
            title="Série temporelle Météo"
        )
        st.plotly_chart(fig_temp, width="stretch")

    with tab2:
        st.subheader("Concentration des Polluants Clés (µg/m³)")
        fig_air = px.bar(
            df.tail(24), x="datetime", y=["pm2_5", "pm10", "nitrogen_dioxide"],
            barmode="group",
            title="Exposition aux polluants sur les 24 dernières heures"
        )
        st.plotly_chart(fig_air, width="stretch")

    with tab3:
        st.subheader("Localisation de la station de suivi")
        map_df = pd.DataFrame([{
            "lat": city_info["lat"],
            "lon": city_info["lon"],
            "city": selected_city,
            "aqi": latest["us_aqi"]
        }])
        fig_map = px.scatter_map(
            map_df, lat="lat", lon="lon", hover_name="city", hover_data=["aqi"],
            color_discrete_sequence=["fuchsia"], zoom=12, height=400,
            center={"lat": city_info["lat"], "lon": city_info["lon"]},
            map_style="open-street-map"
        )
        st.plotly_chart(fig_map, width="stretch")


class TestDashboardModules(unittest.TestCase):
    """
    Unit testing module for verifying data processing & scoring logic.
    """

    def setUp(self):
        self.mock_df = pd.DataFrame({
            "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
            "temperature_2m": [21.0, 40.0],
            "relativehumidity_2m": [50.0, 20.0],
            "pm2_5": [10.0, 30.0],
            "pm10": [20.0, 55.0],
            "nitrogen_dioxide": [15.0, 60.0]
        })

    def test_health_index_calculation(self):
        scores = DataProcessor.calculate_custom_health_index(self.mock_df)
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores.iloc[0], scores.iloc[1])

    def test_alerts_detection(self):
        alerts = DataProcessor.detect_alerts(self.mock_df)
        self.assertTrue(len(alerts) > 0)


if __name__ == "__main__":
    # Standard Execution runs Streamlit application
    render_dashboard()
