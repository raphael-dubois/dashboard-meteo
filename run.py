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
from plotly.subplots import make_subplots
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
    "Tokyo": {"lat": 35.6895, "lon": 139.6917},
    "Sydney": {"lat": -33.8688, "lon": 151.2093},
    "New York": {"lat": 40.7128, "lon": -74.0060},
    "Pyongyang": {"lat": 39.0392, "lon": 125.7625},
    "Abu Dhabi": {"lat": 24.4539, "lon": 54.3773},
    "Berlin": {"lat": 52.5200, "lon": 13.4050},
    "Moscow": {"lat": 55.7558, "lon": 37.6173},
    "Rome": {"lat": 41.9028, "lon": 12.4964}
}

# Thresholds for alerts (WHO Guidelines)
THRESHOLDS = {
    "pm2_5": {"warning": 15.0, "danger": 25.0, "unit": "µg/m³"},
    "pm10": {"warning": 45.0, "danger": 50.0, "unit": "µg/m³"},
    "no2": {"warning": 25.0, "danger": 50.0, "unit": "µg/m³"},
    "temp_max": {"warning": 33.0, "danger": 38.0, "unit": "°C"}
}

# Editorial palette (kept in sync with .streamlit/config.toml)
PALETTE = {
    "text": "#2B2620",
    "muted": "#857B6E",
    "line": "#DDD5C7",
    "grid": "#E6DFD3",
    "accent": "#A84B24",
    "slate": "#5B7A99",
    "ochre": "#C08A3E",
    "warning": "#A8661A",
    "danger": "#B3261E",
}

# US AQI categories: (upper bound, label, color)
AQI_CATEGORIES = [
    (50, "Air de bonne qualité", "#5B8C3A"),
    (100, "Qualité de l'air modérée", "#C9A227"),
    (150, "Mauvais pour les personnes sensibles", "#D9822B"),
    (200, "Air de mauvaise qualité", "#B3261E"),
    (300, "Air de très mauvaise qualité", "#7B3F8C"),
    (float("inf"), "Air dangereux", "#6E1E2B"),
]

FRENCH_DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
FRENCH_MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre"
]

DASHBOARD_CSS = f"""
<style>
.masthead {{ border-bottom: 1px solid {PALETTE["line"]}; padding-bottom: 0.75rem; margin-bottom: 0.5rem; }}
.masthead-title {{ font-family: "Source Serif 4", "Source Serif Pro", Georgia, serif; font-size: 1.35rem; color: {PALETTE["text"]}; }}
.masthead-sub {{ font-size: 0.85rem; color: {PALETTE["muted"]}; }}
.hero {{ display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 2rem; align-items: center; padding: 1.25rem 0 1rem; }}
.hero-date {{ font-size: 0.85rem; color: {PALETTE["muted"]}; }}
.hero-city {{ font-family: "Source Serif 4", "Source Serif Pro", Georgia, serif; font-size: 2.75rem; line-height: 1.15; color: {PALETTE["text"]}; }}
.hero-temp-row {{ display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }}
.hero-temp {{ font-size: 4.25rem; font-weight: 600; line-height: 1.05; color: {PALETTE["text"]}; }}
.hero-sub {{ font-size: 0.95rem; color: {PALETTE["muted"]}; }}
.hero-badge {{ display: inline-block; margin-left: 0.5rem; padding: 0.1rem 0.5rem; border: 1px solid {PALETTE["line"]}; border-radius: 4px; font-size: 0.75rem; color: {PALETTE["muted"]}; }}
.hero-aqi {{ display: flex; align-items: center; gap: 1rem; }}
.aqi-ring {{ flex: none; width: 96px; height: 96px; border-radius: 50%; display: grid; place-items: center; }}
.aqi-ring-inner {{ width: 78px; height: 78px; border-radius: 50%; background: #F7F4EE; display: grid; place-items: center; font-size: 1.4rem; font-weight: 600; color: {PALETTE["text"]}; }}
.hero-aqi-label {{ font-size: 1.05rem; font-weight: 600; color: {PALETTE["text"]}; }}
.hero-aqi-caption {{ font-size: 0.85rem; color: {PALETTE["muted"]}; }}
.stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border-top: 1px solid {PALETTE["line"]}; padding-top: 0.9rem; margin-bottom: 1.25rem; }}
.stat {{ padding: 0 1.25rem; border-left: 1px solid {PALETTE["line"]}; }}
.stat:first-child {{ padding-left: 0; border-left: none; }}
.stat-label {{ font-size: 0.85rem; color: {PALETTE["muted"]}; }}
.stat-value {{ font-size: 1.6rem; font-weight: 600; color: {PALETTE["text"]}; }}
.stat-unit {{ font-size: 0.9rem; font-weight: 400; color: {PALETTE["muted"]}; margin-left: 0.25rem; }}
.meteo-alert {{ border-left: 3px solid; border-radius: 0; padding: 0.65rem 1rem; margin-bottom: 0.6rem; }}
.meteo-alert p {{ margin: 0; }}
.meteo-alert--danger {{ border-color: {PALETTE["danger"]}; background: #F7E4E0; color: #7A1A14; }}
.meteo-alert--warning {{ border-color: {PALETTE["warning"]}; background: #F8EDD9; color: #6B4510; }}
.meteo-alert--ok {{ border-color: #5B8C3A; background: #E9EFDF; color: #34501F; }}
@media (max-width: 640px) {{
  .hero {{ grid-template-columns: 1fr; gap: 1.25rem; }}
  .stats {{ grid-template-columns: 1fr; gap: 0.75rem; }}
  .stat {{ padding-left: 0; border-left: none; }}
}}
</style>
"""


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

    @staticmethod
    def detect_alerts(df: pd.DataFrame, reference_time: Optional[datetime.datetime] = None) -> List[Dict[str, str]]:
        """Scans the current data point for threshold violations."""
        alerts = []
        if df.empty:
            return alerts

        latest = df.iloc[DataProcessor.get_current_index(df, reference_time)]

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


def _format_french_datetime(ts: pd.Timestamp) -> str:
    """Formats a timestamp as 'Lundi 20 octobre · 14:30' without relying on system locale."""
    day = FRENCH_DAYS[ts.weekday()].capitalize()
    return f"{day} {ts.day} {FRENCH_MONTHS[ts.month - 1]} · {ts:%H:%M}"


def _format_coordinates(lat: float, lon: float) -> str:
    """Formats coordinates as '48.86° N · 2.35° E'."""
    lat_ref = "N" if lat >= 0 else "S"
    lon_ref = "E" if lon >= 0 else "O"
    return f"{abs(lat):.2f}° {lat_ref} · {abs(lon):.2f}° {lon_ref}"


def _aqi_category(aqi: float) -> Tuple[str, str]:
    """Returns the (label, color) pair of the US AQI category."""
    for upper_bound, label, color in AQI_CATEGORIES:
        if aqi <= upper_bound:
            return label, color
    return AQI_CATEGORIES[-1][1], AQI_CATEGORIES[-1][2]


def _threshold_color(value: float, key: str) -> str:
    """Colors a pollutant value according to its warning/danger thresholds."""
    if value >= THRESHOLDS[key]["danger"]:
        return PALETTE["danger"]
    if value >= THRESHOLDS[key]["warning"]:
        return PALETTE["warning"]
    return PALETTE["text"]


def _aqi_gauge_html(aqi: float, color: str) -> str:
    """Ring gauge for the US AQI (0-300 scale, capped). Pure CSS: st.html strips SVG."""
    percent = min(max(aqi, 0) / 300, 1) * 100
    return (
        f'<div class="aqi-ring" role="img" aria-label="AQI {int(aqi)}" '
        f'style="background: conic-gradient({color} 0 {percent:.1f}%, {PALETTE["grid"]} {percent:.1f}% 100%)">'
        f'<div class="aqi-ring-inner">{int(aqi)}</div></div>'
    )


def _render_hero(city: str, latest: pd.Series, mock_mode: bool) -> None:
    """Featured block: city, temperature, AQI gauge and secondary stats."""
    aqi = latest["us_aqi"]
    if pd.isna(aqi):
        aqi_block = (
            '<div class="hero-aqi"><div><div class="hero-aqi-label">Indice indisponible</div>'
            '<div class="hero-aqi-caption">Indice AQI (US)</div></div></div>'
        )
    else:
        aqi_label, aqi_color = _aqi_category(aqi)
        aqi_block = (
            f'<div class="hero-aqi">{_aqi_gauge_html(aqi, aqi_color)}'
            f'<div><div class="hero-aqi-label">{aqi_label}</div>'
            f'<div class="hero-aqi-caption">Indice AQI (US)</div></div></div>'
        )

    mock_badge = '<span class="hero-badge">Données simulées</span>' if mock_mode else ""
    pm25_color = _threshold_color(latest["pm2_5"], "pm2_5")

    st.html(f"""
    <div class="hero">
      <div>
        <div class="hero-date">{_format_french_datetime(latest["datetime"])}{mock_badge}</div>
        <div class="hero-city">{city}</div>
        <div class="hero-temp-row">
          <span class="hero-temp">{latest["temperature_2m"]:.0f}°</span>
          <span class="hero-sub">Vent {latest["windspeed_10m"]:.0f} km/h · {latest["surface_pressure"]:.0f} hPa</span>
        </div>
      </div>
      {aqi_block}
    </div>
    <div class="stats">
      <div class="stat">
        <div class="stat-label">Humidité</div>
        <div class="stat-value">{latest["relativehumidity_2m"]:.0f}<span class="stat-unit">%</span></div>
      </div>
      <div class="stat">
        <div class="stat-label">PM2.5</div>
        <div class="stat-value" style="color: {pm25_color}">{latest["pm2_5"]:.1f}<span class="stat-unit">µg/m³</span></div>
      </div>
      <div class="stat">
        <div class="stat-label">Score confort</div>
        <div class="stat-value">{latest["health_score"]:.0f}<span class="stat-unit">/100</span></div>
      </div>
    </div>
    """)


def _render_alerts(alerts: List[Dict[str, str]]) -> None:
    """Editorial alert banners (left accent border, Material icon)."""
    if not alerts:
        alerts = [{
            "level": "ok",
            "message": "Qualité de l'air et conditions météorologiques stables. Aucune alerte active."
        }]

    styles = {
        "danger": (":material/error:", "Danger"),
        "warning": (":material/warning:", "Attention"),
        "ok": (":material/check_circle:", "Situation normale"),
    }
    for alert in alerts:
        icon, title = styles.get(alert["level"], styles["warning"])
        # Blank lines let Streamlit parse the inner Markdown (Material icon + bold)
        st.markdown(
            f'<div class="meteo-alert meteo-alert--{alert["level"]}">\n\n'
            f'{icon} **{title}** · {alert["message"]}\n\n'
            f'</div>',
            unsafe_allow_html=True,
        )


def _style_figure(fig: go.Figure, height: int = 420) -> go.Figure:
    """Applies the editorial look to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], size=13),
        margin=dict(l=64, r=64, t=48, b=48),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFDF9", bordercolor=PALETTE["line"], font_color=PALETTE["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    )
    fig.update_xaxes(
        showgrid=False, linecolor=PALETTE["line"], ticks="outside", tickcolor=PALETTE["line"],
        tickformat="%d/%m\n%Hh", automargin=True,
    )
    fig.update_yaxes(gridcolor=PALETTE["grid"], zeroline=False, title_font_color=PALETTE["muted"], automargin=True)
    return fig


def _weather_figure(df: pd.DataFrame) -> go.Figure:
    """Temperature and humidity on two y-axes (different units)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["temperature_2m"], name="Température",
            line=dict(color=PALETTE["accent"], width=2.5),
            hovertemplate="%{y:.1f} °C<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["relativehumidity_2m"], name="Humidité",
            line=dict(color=PALETTE["slate"], width=1.5, dash="dot"),
            hovertemplate="%{y:.0f} %<extra></extra>",
        ),
        secondary_y=True,
    )
    _style_figure(fig)
    fig.update_yaxes(title_text="Température (°C)", secondary_y=False)
    fig.update_yaxes(
        title_text="Humidité (%)", range=[0, 100], tickmode="linear", tick0=0, dtick=20,
        showgrid=False, secondary_y=True,
    )
    return fig


def _pollutants_figure(df: pd.DataFrame) -> go.Figure:
    """Key pollutant concentrations over the whole period."""
    series = [
        ("pm2_5", "PM2.5", PALETTE["accent"]),
        ("pm10", "PM10", PALETTE["ochre"]),
        ("nitrogen_dioxide", "NO₂", PALETTE["slate"]),
    ]
    fig = go.Figure()
    for column, name, color in series:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df[column], name=name,
            line=dict(color=color, width=2),
            hovertemplate="%{y:.1f} µg/m³<extra></extra>",
        ))
    _style_figure(fig)
    fig.update_layout(margin_r=16)
    fig.update_yaxes(title_text="Concentration (µg/m³)", rangemode="tozero")
    return fig


def _render_sidebar() -> Tuple[str, bool]:
    """Sidebar: location and data source settings. Returns (city, mock_mode)."""
    with st.sidebar:
        st.subheader(":material/location_on: Lieu")
        selected_city = st.selectbox("Ville", list(DEFAULT_CITIES.keys()))
        city_info = DEFAULT_CITIES[selected_city]
        st.caption(_format_coordinates(city_info["lat"], city_info["lon"]))

        st.divider()
        st.subheader(":material/database: Données")
        mock_mode = st.toggle(
            "Mode déconnecté",
            value=False,
            help="Utilise des données simulées au lieu des API Open-Meteo.",
        )
        st.caption(f"Cache local des réponses API : {CACHE_TTL_MINUTES} min")

        st.divider()
        st.caption(
            "Sources : Open-Meteo (prévisions et qualité de l'air). "
            "Seuils d'alerte : recommandations de l'OMS."
        )
    return selected_city, mock_mode


def render_dashboard():
    """
    Renders Streamlit Web Application Interface.
    Author: Student C (Lead UI/UX)
    """
    st.set_page_config(page_title="Dashboard Météo & AQI", page_icon=":material/partly_cloudy_day:", layout="wide")
    st.html(DASHBOARD_CSS)

    st.html("""
    <div class="masthead">
      <div class="masthead-title">Dashboard Météo & Surveillance Qualité de l'Air</div>
      <div class="masthead-sub">Projet 3ᵉ Année Ingénieur — Analyse de données environnementales en temps réel</div>
    </div>
    """)

    client = WeatherAPIClient()
    selected_city, mock_mode = _render_sidebar()
    city_info = DEFAULT_CITIES[selected_city]

    # Fetch Data
    with st.spinner(f"Chargement des données pour {selected_city}…"):
        raw_data = client.fetch_weather_and_air_quality(city_info["lat"], city_info["lon"], mock_mode=mock_mode)
        df = DataProcessor.process_raw_api_data(raw_data)

    if df.empty:
        st.error("Impossible de récupérer ou d'afficher les données.", icon=":material/error:")
        return

    # Add calculated metrics
    df["health_score"] = DataProcessor.calculate_custom_health_index(df)
    reference_time = DataProcessor.get_reference_time(raw_data)
    latest = df.iloc[DataProcessor.get_current_index(df, reference_time)]

    _render_hero(selected_city, latest, mock_mode)
    _render_alerts(DataProcessor.detect_alerts(df, reference_time))

    # Visualizations
    tab1, tab2, tab3 = st.tabs([
        ":material/thermostat: Évolution Météo",
        ":material/air: Qualité de l'Air",
        ":material/map: Carte Interactive",
    ])

    with tab1:
        st.subheader("Température et humidité")
        st.caption("Série horaire sur la période couverte par les API (historique récent et prévisions).")
        st.plotly_chart(_weather_figure(df), width="stretch", theme=None)

    with tab2:
        st.subheader("Concentration des polluants clés")
        st.caption("PM2.5, PM10 et dioxyde d'azote, en µg/m³.")
        st.plotly_chart(_pollutants_figure(df), width="stretch", theme=None)

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
            color_discrete_sequence=[PALETTE["accent"]], zoom=12, height=600,
            center={"lat": city_info["lat"], "lon": city_info["lon"]},
            map_style="open-street-map"
        )
        fig_map.update_traces(marker=dict(size=18))
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

    def test_current_index_skips_forecasts(self):
        df = self.mock_df.assign(datetime=pd.to_datetime(self.mock_df["time"]))
        reference_time = datetime.datetime(2026, 1, 1, 0, 30)
        self.assertEqual(DataProcessor.get_current_index(df, reference_time), 0)
        # Only the forecast row (40 °C, high pollution) would trigger alerts
        self.assertEqual(DataProcessor.detect_alerts(df, reference_time), [])

    def test_reference_time_uses_api_offset(self):
        reference_time = DataProcessor.get_reference_time({"weather": {"utc_offset_seconds": 3600}})
        now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        delta = reference_time - now_utc
        self.assertAlmostEqual(delta.total_seconds(), 3600, delta=5)


if __name__ == "__main__":
    # Standard Execution runs Streamlit application
    render_dashboard()
