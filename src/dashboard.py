from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from .alerts import THRESHOLDS, detect_alerts
from .api_client import WeatherAPIClient, DEFAULT_CITIES, CACHE_TTL_MINUTES
from .data_processing import DataProcessor

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
    _render_alerts(detect_alerts(df, reference_time))

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
