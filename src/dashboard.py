from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from .alerts import THRESHOLDS, detect_alerts
from .api_client import WeatherAPIClient, DEFAULT_CITIES, CACHE_TTL_MINUTES
from .data_processing import DataProcessor

# Palettes for the custom blocks and charts.
# "editorial" adapts to Streamlit's light/dark mode on its own: neutral tones derive from the
# inherited text color (currentColor), accents are mid tones readable on both backgrounds.
# Switching mode in the Settings menu does not rerun the script, so no Python-side detection.
# "retro" is a fixed palette applied on top via CSS.
THEMES: Dict[str, Dict[str, Any]] = {
    "editorial": {
        "adaptive": True,
        "bg": "transparent", "surface": "transparent",
        "text": "inherit",
        "muted": "color-mix(in srgb, currentColor 62%, transparent)",
        "line": "color-mix(in srgb, currentColor 16%, transparent)",
        "grid": "color-mix(in srgb, currentColor 12%, transparent)",
        "accent": "#C2603A", "slate": "#6B8CB0", "ochre": "#C9973F",
        "warning": "#C9821F", "danger": "#D9443A", "ok": "#6BA34A",
        "danger_bg": "color-mix(in srgb, #D9443A 14%, transparent)",
        "danger_fg": "color-mix(in srgb, #D9443A 60%, currentColor)",
        "warning_bg": "color-mix(in srgb, #C9821F 16%, transparent)",
        "warning_fg": "color-mix(in srgb, #C9821F 60%, currentColor)",
        "ok_bg": "color-mix(in srgb, #6BA34A 15%, transparent)",
        "ok_fg": "color-mix(in srgb, #6BA34A 60%, currentColor)",
        "aqi": ["#6BA34A", "#D4AE2C", "#E08A35", "#D9443A", "#9A5BB5", "#A3394F"],
        "map_style": "open-street-map",
    },
    "retro": {
        "adaptive": False,
        "bg": "#07060F", "surface": "#110D22",
        "text": "#E9F1FF", "muted": "#9A95C9", "line": "#FF2BD6", "grid": "rgba(0, 240, 255, 0.14)",
        "accent": "#FF2BD6", "slate": "#00F0FF", "ochre": "#FFE600",
        "warning": "#FFB000", "danger": "#FF3860", "ok": "#39FF88",
        "danger_bg": "rgba(255, 56, 96, 0.10)", "danger_fg": "#FF8FA6",
        "warning_bg": "rgba(255, 176, 0, 0.10)", "warning_fg": "#FFD37A",
        "ok_bg": "rgba(57, 255, 136, 0.08)", "ok_fg": "#9DFFC4",
        "aqi": ["#39FF88", "#FFE600", "#FFB000", "#FF3860", "#C04BFF", "#FF2BD6"],
        "map_style": "carto-darkmatter",
    },
}

# US AQI categories: (upper bound, label); colors come from the active palette
AQI_CATEGORIES = [
    (50, "Air de bonne qualité"),
    (100, "Qualité de l'air modérée"),
    (150, "Mauvais pour les personnes sensibles"),
    (200, "Air de mauvaise qualité"),
    (300, "Air de très mauvaise qualité"),
    (float("inf"), "Air dangereux"),
]

# Pollutants shown under the weather tiles: (DataFrame column, label, THRESHOLDS key)
POLLUTANTS = [
    ("pm2_5", "PM2.5", "pm2_5"),
    ("pm10", "PM10", "pm10"),
    ("nitrogen_dioxide", "NO₂", "no2"),
    ("ozone", "O₃", "o3"),
    ("sulphur_dioxide", "SO₂", "so2"),
    ("carbon_monoxide", "CO", "co"),
]

FRENCH_DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
FRENCH_MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre"
]

SERIF = '"Source Serif 4", "Source Serif Pro", Georgia, serif'
PIXEL = '"Press Start 2P", monospace'
TERMINAL = '"VT323", monospace'


def _build_css(p: Dict[str, Any]) -> str:
    """Styles of the custom blocks (masthead, hero, stats, alerts) for a palette."""
    return f"""
.masthead {{ border-bottom: 1px solid {p["line"]}; padding-bottom: 0.75rem; margin-bottom: 0.5rem; }}
.masthead-title {{ font-family: {SERIF}; font-size: 1.35rem; color: {p["text"]}; }}
.masthead-sub {{ font-size: 0.85rem; color: {p["muted"]}; }}
.hero {{ display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 2rem; align-items: center; padding: 1.25rem 0 1rem; }}
.hero-date {{ font-size: 0.85rem; color: {p["muted"]}; }}
.hero-city {{ font-family: {SERIF}; font-size: 2.75rem; line-height: 1.15; color: {p["text"]}; }}
.hero-temp-row {{ display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }}
.hero-temp {{ font-size: 4.25rem; font-weight: 600; line-height: 1.05; color: {p["text"]}; }}
.hero-sub {{ font-size: 0.95rem; color: {p["muted"]}; }}
.hero-badge {{ display: inline-block; margin-left: 0.5rem; padding: 0.1rem 0.5rem; border: 1px solid {p["line"]}; border-radius: 4px; font-size: 0.75rem; color: {p["muted"]}; }}
.hero-badge--warning {{ border-color: {p["warning"]}; color: {p["warning_fg"]}; background: {p["warning_bg"]}; }}
.hero-aqi {{ display: flex; align-items: center; gap: 1rem; }}
.aqi-ring {{ flex: none; position: relative; width: 96px; height: 96px; border-radius: 50%; display: grid; place-items: center; }}
.aqi-ring::before {{ content: ""; position: absolute; inset: 0; border-radius: 50%; background: var(--aqi-fill); -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 9px), #000 calc(100% - 9px)); mask: radial-gradient(farthest-side, transparent calc(100% - 9px), #000 calc(100% - 9px)); }}
.aqi-ring-inner {{ position: relative; font-size: 1.4rem; font-weight: 600; color: {p["text"]}; }}
.hero-aqi-label {{ font-size: 1.05rem; font-weight: 600; color: {p["text"]}; }}
.hero-aqi-caption {{ font-size: 0.85rem; color: {p["muted"]}; }}
.stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); row-gap: 0.9rem; border-top: 1px solid {p["line"]}; padding-top: 0.9rem; margin-bottom: 0.9rem; }}
.stat {{ padding: 0 1.25rem; border-left: 1px solid {p["line"]}; }}
.stat:nth-child(3n+1) {{ padding-left: 0; border-left: none; }}
.stat-label {{ font-size: 0.85rem; color: {p["muted"]}; }}
.stat-value {{ font-size: 1.6rem; font-weight: 600; color: {p["text"]}; }}
.stat-unit {{ font-size: 0.9rem; font-weight: 400; color: {p["muted"]}; margin-left: 0.25rem; }}
.wind-dir {{ font-size: 0.9rem; font-weight: 400; color: {p["muted"]}; margin-left: 0.5rem; }}
.wind-arrow {{ display: inline-block; }}
.pollutants {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 1.5rem; border-top: 1px solid {p["line"]}; padding: 0.75rem 0; margin-bottom: 1rem; }}
.pollutant {{ display: flex; align-items: center; gap: 0.4rem; font-size: 0.95rem; color: {p["text"]}; }}
.pollutant-dot {{ width: 0.6rem; height: 0.6rem; border-radius: 50%; background: var(--dot); }}
.pollutant-label {{ color: {p["muted"]}; }}
.pollutants-caption {{ font-size: 0.8rem; color: {p["muted"]}; }}
.meteo-alert {{ border-left: 3px solid; border-radius: 0; padding: 0.65rem 1rem; margin-bottom: 0.6rem; }}
.meteo-alert p {{ margin: 0; }}
.meteo-alert--danger {{ border-color: {p["danger"]}; background: {p["danger_bg"]}; color: {p["danger_fg"]}; }}
.meteo-alert--warning {{ border-color: {p["warning"]}; background: {p["warning_bg"]}; color: {p["warning_fg"]}; }}
.meteo-alert--ok {{ border-color: {p["ok"]}; background: {p["ok_bg"]}; color: {p["ok_fg"]}; }}
@media (max-width: 640px) {{
  .hero {{ grid-template-columns: 1fr; gap: 1.25rem; }}
  .stats {{ grid-template-columns: 1fr; gap: 0.75rem; }}
  .stat {{ padding-left: 0; border-left: none; }}
}}
"""


def _build_retro_css(p: Dict[str, Any]) -> str:
    """Arcade neon overrides: pixel fonts, glows, scanlines and native Streamlit widgets."""
    cyan, magenta, yellow = p["slate"], p["accent"], p["ochre"]
    return f"""
.stApp {{
  background-color: {p["bg"]};
  background-image: repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.025) 0 1px, transparent 1px 3px);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ border-right: 2px solid {magenta}; box-shadow: 0 0 18px rgba(255, 43, 214, 0.35); }}
[data-testid="stSidebar"] > div:first-child {{ background: #0D0A1C; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255, 43, 214, 0.4); }}
.stApp p, .stApp label, .stApp li, .stApp input {{
  font-family: {TERMINAL} !important; color: {p["text"]};
}}
.stApp p, .stApp label, .stApp li {{ font-size: 1.2rem; }}
.stApp h1, .stApp h2, .stApp h3 {{
  font-family: {PIXEL} !important; font-size: 1rem !important; line-height: 1.7;
  color: {cyan}; text-shadow: 0 0 8px rgba(0, 240, 255, 0.6);
}}
[data-testid="stCaptionContainer"] p {{ color: {p["muted"]}; }}
[data-testid="stTabs"] [role="tablist"] {{ border-bottom-color: rgba(255, 43, 214, 0.3) !important; }}
[data-testid="stTab"] p {{ font-family: {PIXEL} !important; font-size: 0.65rem; }}
[data-testid="stTab"][aria-selected="true"] {{ border-bottom-color: {magenta} !important; box-shadow: 0 6px 8px -6px {magenta}; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: {magenta}; text-shadow: 0 0 6px {magenta}; }}
[data-testid="stSelectbox"] [role="group"] {{ background: {p["surface"]} !important; border: 2px solid {cyan} !important; border-radius: 0 !important; color: {p["text"]}; }}
[data-testid="stSelectbox"] [role="group"] * {{ color: inherit !important; font-family: {TERMINAL} !important; font-size: 1.15rem; }}
[data-testid="stCheckbox"] label > div:first-of-type {{ background: rgba(255, 255, 255, 0.15) !important; border-radius: 0 !important; }}
[data-testid="stCheckbox"] label > div:first-of-type > div {{ background: {p["text"]} !important; border-radius: 0 !important; }}
[data-testid="stCheckbox"] label[data-selected] > div:first-of-type {{ background: {magenta} !important; box-shadow: 0 0 8px {magenta}; }}
.stApp hr {{ border-color: rgba(255, 43, 214, 0.4); }}
.stApp [data-testid="stDownloadButton"] button, .stApp [data-testid="stButton"] button {{
  background: transparent !important; border: 2px solid {cyan} !important; border-radius: 0 !important;
  box-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
}}
.stApp [data-testid="stDownloadButton"] button p, .stApp [data-testid="stButton"] button p {{
  font-family: {PIXEL} !important; font-size: 0.65rem; color: {cyan};
}}
.stApp [data-testid="stDownloadButton"] button:hover, .stApp [data-testid="stButton"] button:hover {{
  border-color: {magenta} !important; box-shadow: 0 0 14px {magenta};
}}
.stApp [data-testid="stDownloadButton"] button:hover p, .stApp [data-testid="stButton"] button:hover p {{ color: {magenta}; }}
.masthead {{ border-bottom: 2px solid {magenta}; box-shadow: 0 6px 12px -8px {magenta}; }}
.masthead-title {{ font-family: {PIXEL}; font-size: 0.95rem; line-height: 1.8; color: {cyan}; text-shadow: 0 0 8px rgba(0, 240, 255, 0.7); }}
.masthead-sub, .hero-date, .hero-sub, .hero-aqi-caption, .stat-label, .stat-unit, .wind-dir, .pollutant, .pollutants-caption {{ font-family: {TERMINAL}; font-size: 1.15rem; }}
.hero-city {{ font-family: {PIXEL}; font-size: 2rem; line-height: 1.5; color: {magenta}; text-shadow: 0 0 10px {magenta}, 0 0 24px rgba(255, 43, 214, 0.5); }}
.hero-temp {{ font-family: {PIXEL}; font-size: 3rem; font-weight: 400; line-height: 1.4; color: {yellow}; text-shadow: 0 0 10px rgba(255, 230, 0, 0.7); }}
.hero-badge {{ font-family: {PIXEL}; font-size: 0.55rem; border: 2px solid {yellow}; border-radius: 0; color: {yellow}; }}
.hero-aqi-label {{ font-family: {PIXEL}; font-size: 0.75rem; font-weight: 400; line-height: 1.7; }}
.aqi-ring {{ box-shadow: 0 0 16px rgba(255, 43, 214, 0.45); }}
.aqi-ring-inner {{ font-family: {PIXEL}; font-size: 1rem; font-weight: 400; }}
.stats {{ gap: 0.75rem; border-top: none; }}
.stat, .stat:nth-child(3n+1) {{ padding: 0.6rem 0.9rem; border: 2px solid rgba(0, 240, 255, 0.6); box-shadow: 0 0 10px rgba(0, 240, 255, 0.25); }}
.stat-value {{ font-family: {PIXEL}; font-size: 1.1rem; font-weight: 400; line-height: 1.9; }}
.stats {{ column-gap: 0.75rem; }}
.pollutants {{ border-top: none; }}
.pollutant-dot {{ border-radius: 0; box-shadow: 0 0 6px var(--dot); }}
.meteo-alert {{ border: 2px solid; box-shadow: 0 0 12px currentColor; }}
.meteo-alert p {{ font-size: 1.25rem; color: inherit !important; }}
.meteo-alert strong {{ font-family: {PIXEL}; font-size: 0.65rem; font-weight: 400; }}
"""


def _get_theme_name(retro_mode: bool) -> str:
    """Active palette: retro if toggled, otherwise the editorial palette (light and dark)."""
    return "retro" if retro_mode else "editorial"


def _inject_styles(theme_name: str) -> None:
    """Injects the CSS for the active palette."""
    p = THEMES[theme_name]
    css = _build_css(p)
    if theme_name == "retro":
        fonts = "@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');"
        css = fonts + css + _build_retro_css(p)
    st.html(f"<style>{css}</style>")


def _format_french_datetime(ts: pd.Timestamp) -> str:
    """Formats a timestamp as 'Lundi 20 octobre · 14:30' without relying on system locale."""
    day = FRENCH_DAYS[ts.weekday()].capitalize()
    return f"{day} {ts.day} {FRENCH_MONTHS[ts.month - 1]} · {ts:%H:%M}"


def _format_coordinates(lat: float, lon: float) -> str:
    """Formats coordinates as '48.86° N · 2.35° E'."""
    lat_ref = "N" if lat >= 0 else "S"
    lon_ref = "E" if lon >= 0 else "O"
    return f"{abs(lat):.2f}° {lat_ref} · {abs(lon):.2f}° {lon_ref}"


def _aqi_category(aqi: float, p: Dict[str, Any]) -> Tuple[str, str]:
    """Returns the (label, color) pair of the US AQI category."""
    for (upper_bound, label), color in zip(AQI_CATEGORIES, p["aqi"]):
        if aqi <= upper_bound:
            return label, color
    return AQI_CATEGORIES[-1][1], p["aqi"][-1]


def _threshold_level(value: float, key: str) -> str:
    """Level of a value against its WHO thresholds: "ok", "warning" or "danger"."""
    if value >= THRESHOLDS[key]["danger"]:
        return "danger"
    if value >= THRESHOLDS[key]["warning"]:
        return "warning"
    return "ok"


def _level_color(level: str, p: Dict[str, Any]) -> str:
    """Color of a threshold level, readable in light and dark mode."""
    color = p[level]
    # Pull the hue toward the text color so it stays readable in light and dark mode
    return f"color-mix(in srgb, {color} 75%, currentColor)" if p["adaptive"] else color


def _wind_cardinal(degrees: float) -> str:
    """8-point French compass label of the direction the wind comes from."""
    labels = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    return labels[int((degrees % 360) / 45 + 0.5) % 8]


def _aqi_gauge_html(aqi: float, color: str, p: Dict[str, Any]) -> str:
    """Ring gauge for the US AQI (0-300 scale, capped). Pure CSS (masked conic gradient): st.html strips SVG."""
    percent = min(max(aqi, 0) / 300, 1) * 100
    return (
        f'<div class="aqi-ring" role="img" aria-label="AQI {int(aqi)}" '
        f'style="--aqi-fill: conic-gradient({color} 0 {percent:.1f}%, {p["grid"]} {percent:.1f}% 100%)">'
        f'<div class="aqi-ring-inner">{int(aqi)}</div></div>'
    )


def _pollutants_html(latest: pd.Series, p: Dict[str, Any]) -> str:
    """Strip with the six monitored pollutants, colored against WHO thresholds when defined."""
    items = []
    for column, label, threshold_key in POLLUTANTS:
        value = latest.get(column)
        if value is None or pd.isna(value):
            continue
        level = _threshold_level(value, threshold_key) if threshold_key in THRESHOLDS else "ok"
        items.append(
            f'<span class="pollutant"><span class="pollutant-dot" style="--dot: {_level_color(level, p)}"></span>'
            f'<span class="pollutant-label">{label}</span> {value:.1f}</span>'
        )
    return f'<div class="pollutants">{"".join(items)}<span class="pollutants-caption">µg/m³</span></div>'


def _render_hero(
    city: str, latest: pd.Series, today: Optional[pd.Series], badge: str, p: Dict[str, Any]
) -> None:
    """Featured block: city, temperatures, AQI gauge, weather tiles and pollutant strip."""
    aqi = latest["us_aqi"]
    if pd.isna(aqi):
        aqi_block = (
            '<div class="hero-aqi"><div><div class="hero-aqi-label">Indice indisponible</div>'
            '<div class="hero-aqi-caption">Indice AQI (US)</div></div></div>'
        )
    else:
        aqi_label, aqi_color = _aqi_category(aqi, p)
        aqi_block = (
            f'<div class="hero-aqi">{_aqi_gauge_html(aqi, aqi_color, p)}'
            f'<div><div class="hero-aqi-label">{aqi_label}</div>'
            f'<div class="hero-aqi-caption">Indice AQI (US)</div></div></div>'
        )

    temp_details = [f"Ressentie {latest['apparent_temperature']:.0f}°"]
    if today is not None:
        temp_details.append(f"min {today['temperature_2m_min']:.0f}° / max {today['temperature_2m_max']:.0f}°")
    precipitation = today["precipitation_sum"] if today is not None else latest["precipitation"]
    wind_direction = latest["wind_direction_10m"]
    # Meteorological convention: the direction the wind comes from; the arrow shows where it blows
    wind_arrow_angle = (wind_direction + 180) % 360

    st.html(f"""
    <div class="hero">
      <div>
        <div class="hero-date">{_format_french_datetime(latest["datetime"])}{badge}</div>
        <div class="hero-city">{city}</div>
        <div class="hero-temp-row">
          <span class="hero-temp">{latest["temperature_2m"]:.0f}°</span>
          <span class="hero-sub">{" · ".join(temp_details)}</span>
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
        <div class="stat-label">Vent</div>
        <div class="stat-value">{latest["windspeed_10m"]:.0f}<span class="stat-unit">km/h</span><span class="wind-dir" title="Vent venant de {wind_direction:.0f}°"><span class="wind-arrow" style="transform: rotate({wind_arrow_angle:.0f}deg)">↑</span> {_wind_cardinal(wind_direction)}</span></div>
      </div>
      <div class="stat">
        <div class="stat-label">Précipitations du jour</div>
        <div class="stat-value">{precipitation:.1f}<span class="stat-unit">mm</span></div>
      </div>
      <div class="stat">
        <div class="stat-label">Couverture nuageuse</div>
        <div class="stat-value">{latest["cloud_cover"]:.0f}<span class="stat-unit">%</span></div>
      </div>
      <div class="stat">
        <div class="stat-label">Pression</div>
        <div class="stat-value">{latest["surface_pressure"]:.0f}<span class="stat-unit">hPa</span></div>
      </div>
      <div class="stat">
        <div class="stat-label">Score confort</div>
        <div class="stat-value">{latest["health_score"]:.0f}<span class="stat-unit">/100</span></div>
      </div>
    </div>
    {_pollutants_html(latest, p)}
    """)


def _render_alerts(alerts: List[Dict[str, str]]) -> None:
    """Alert banners (accent border, Material icon)."""
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


def _style_figure(fig: go.Figure, p: Dict[str, Any], retro: bool, height: int = 420) -> go.Figure:
    """
    Layout shared by the charts. In editorial mode, text and grid colors are left to
    Streamlit's chart theme so they follow light/dark switches; retro sets everything.
    """
    fig.update_layout(
        height=height,
        margin=dict(l=64, r=64, t=48, b=48),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    )
    fig.update_xaxes(showgrid=False, ticks="outside", tickformat="%d/%m\n%Hh", automargin=True)
    fig.update_yaxes(zeroline=False, automargin=True)
    if retro:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=p["text"], family=TERMINAL.replace('"', ""), size=17),
            hoverlabel=dict(bgcolor=p["surface"], bordercolor=p["line"], font_color=p["text"]),
        )
        fig.update_xaxes(linecolor=p["line"], tickcolor=p["line"])
        fig.update_yaxes(gridcolor=p["grid"], title_font_color=p["muted"])
    return fig


def _add_line(
    fig: go.Figure, x: pd.Series, y: pd.Series, name: str, color: str, width: float,
    hovertemplate: str, retro: bool, dash: Optional[str] = None, secondary_y: Optional[bool] = None,
) -> None:
    """Adds a line trace; in retro mode a wide translucent copy underneath fakes a neon glow."""
    extra = {} if secondary_y is None else {"secondary_y": secondary_y}
    if retro:
        fig.add_trace(go.Scatter(
            x=x, y=y, line=dict(color=color, width=width * 4), opacity=0.22,
            showlegend=False, hoverinfo="skip",
        ), **extra)
    fig.add_trace(go.Scatter(
        x=x, y=y, name=name, line=dict(color=color, width=width, dash=dash),
        hovertemplate=hovertemplate,
    ), **extra)


def _weather_figure(df: pd.DataFrame, p: Dict[str, Any], retro: bool) -> go.Figure:
    """Temperature and humidity on two y-axes (different units)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    _add_line(
        fig, df["datetime"], df["temperature_2m"], "Température", p["accent"], 2.5,
        "%{y:.1f} °C<extra></extra>", retro, secondary_y=False,
    )
    _add_line(
        fig, df["datetime"], df["relativehumidity_2m"], "Humidité", p["slate"], 1.5,
        "%{y:.0f} %<extra></extra>", retro, dash="dot", secondary_y=True,
    )
    _style_figure(fig, p, retro)
    fig.update_yaxes(title_text="Température (°C)", secondary_y=False)
    fig.update_yaxes(
        title_text="Humidité (%)", range=[0, 100], tickmode="linear", tick0=0, dtick=20,
        showgrid=False, secondary_y=True,
    )
    return fig


def _pollutants_figure(df: pd.DataFrame, p: Dict[str, Any], retro: bool) -> go.Figure:
    """Key pollutant concentrations over the whole period."""
    series = [
        ("pm2_5", "PM2.5", p["accent"]),
        ("pm10", "PM10", p["ochre"]),
        ("nitrogen_dioxide", "NO₂", p["slate"]),
    ]
    fig = go.Figure()
    for column, name, color in series:
        _add_line(fig, df["datetime"], df[column], name, color, 2, "%{y:.1f} µg/m³<extra></extra>", retro)
    _style_figure(fig, p, retro)
    fig.update_layout(margin_r=16)
    fig.update_yaxes(title_text="Concentration (µg/m³)", rangemode="tozero")
    return fig


def _render_sidebar() -> Tuple[Dict[str, Any], bool, bool]:
    """Sidebar: location search, data source and appearance. Returns (city_info, mock_mode, retro_mode).

    city_info is a dict with keys: name, lat, lon.
    """
    # Initialise session state for the city search
    if "city_search_results" not in st.session_state:
        st.session_state.city_search_results = []
    if "selected_city_info" not in st.session_state:
        # Default to the first city in DEFAULT_CITIES
        first_city = next(iter(DEFAULT_CITIES))
        st.session_state.selected_city_info = {
            "name": first_city,
            "lat": DEFAULT_CITIES[first_city]["lat"],
            "lon": DEFAULT_CITIES[first_city]["lon"],
        }

    client = WeatherAPIClient()

    with st.sidebar:
        st.subheader(":material/search: Rechercher une ville")
        search_query = st.text_input(
            "Recherche",
            placeholder="Tapez un nom de ville…",
            label_visibility="collapsed",
        )

        if search_query and len(search_query.strip()) >= 2:
            results = client.search_cities(search_query)
            st.session_state.city_search_results = results
        elif not search_query:
            # When the field is empty, show default cities
            st.session_state.city_search_results = [
                {"name": name, "label": name, "lat": info["lat"], "lon": info["lon"],
                 "country": "", "admin1": "", "population": 0}
                for name, info in DEFAULT_CITIES.items()
            ]

        results = st.session_state.city_search_results
        if results:
            labels = [r["label"] for r in results]
            # Try to keep the current selection visible in the list
            current_name = st.session_state.selected_city_info["name"]
            default_idx = 0
            for i, r in enumerate(results):
                if r["name"] == current_name:
                    default_idx = i
                    break

            chosen_label = st.selectbox(
                "Résultats",
                labels,
                index=default_idx,
                label_visibility="collapsed",
            )
            chosen = results[labels.index(chosen_label)]
            st.session_state.selected_city_info = {
                "name": chosen["name"],
                "lat": chosen["lat"],
                "lon": chosen["lon"],
            }
        else:
            if search_query:
                st.caption("Aucun résultat trouvé.")

        city_info = st.session_state.selected_city_info
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
        st.subheader(":material/palette: Apparence")
        retro_mode = st.toggle(
            "Mode rétro gaming",
            value=False,
            help="Style arcade néon. Les modes clair et sombre se choisissent dans le menu Settings.",
        )

        st.divider()
        st.caption(
            "Sources : Open-Meteo (prévisions et qualité de l'air). "
            "Seuils d'alerte : recommandations de l'OMS."
        )
    return city_info, mock_mode, retro_mode


def _all_cities_csv(client: WeatherAPIClient, mock_mode: bool) -> bytes:
    """Builds a CSV containing 2 days of data sampled every 2 hours for every configured city."""
    city_frames = []
    unavailable_cities = []

    for city, coordinates in DEFAULT_CITIES.items():
        raw_data = client.fetch_weather_and_air_quality(
            coordinates["lat"], coordinates["lon"], mock_mode=mock_mode
        )
        city_df = DataProcessor.process_raw_api_data(raw_data)
        if city_df.empty:
            unavailable_cities.append(city)
            continue

        reference_time = DataProcessor.get_reference_time(raw_data)
        current_idx = DataProcessor.get_current_index(city_df, reference_time)
        start_idx = max(0, min(current_idx, len(city_df) - 48))
        city_df = city_df.iloc[start_idx : start_idx + 48 : 2].copy()

        city_df["health_score"] = DataProcessor.calculate_custom_health_index(city_df)
        city_df.insert(0, "ville", city)
        city_df.insert(1, "latitude", coordinates["lat"])
        city_df.insert(2, "longitude", coordinates["lon"])
        city_frames.append(city_df)

    if unavailable_cities:
        raise ValueError(
            "Données indisponibles pour : " + ", ".join(unavailable_cities)
        )

    csv_text = pd.concat(city_frames, ignore_index=True).to_csv(
        index=False, sep=";", float_format="%.17g"
    )
    return csv_text.encode("utf-8-sig")


def render_dashboard():
    """
    Renders Streamlit Web Application Interface.
    Author: Student C (Lead UI/UX)
    """
    st.set_page_config(page_title="Dashboard Météo & AQI", page_icon=":material/partly_cloudy_day:", layout="wide")

    city_info, mock_mode, retro_mode = _render_sidebar()
    city_name = city_info["name"]
    theme_name = _get_theme_name(retro_mode)
    palette = THEMES[theme_name]
    _inject_styles(theme_name)

    st.html("""
    <div class="masthead">
      <div class="masthead-title">Dashboard Météo & Surveillance Qualité de l'Air</div>
      <div class="masthead-sub">Projet 3ᵉ Année Ingénieur — Analyse de données environnementales en temps réel</div>
    </div>
    """)

    client = WeatherAPIClient()

    # Fetch Data
    with st.spinner(f"Chargement des données pour {city_name}…"):
        raw_data = client.fetch_weather_and_air_quality(city_info["lat"], city_info["lon"], mock_mode=mock_mode)
        df = DataProcessor.process_raw_api_data(raw_data)

    if df.empty:
        st.error("Impossible de récupérer ou d'afficher les données.", icon=":material/error:")
        return

    # Add calculated metrics
    df["health_score"] = DataProcessor.calculate_custom_health_index(df)
    reference_time = DataProcessor.get_reference_time(raw_data)
    latest = df.iloc[DataProcessor.get_current_index(df, reference_time)]

    daily = DataProcessor.process_daily_data(raw_data)
    today_date = latest["datetime"].date()
    today = daily.loc[today_date] if today_date in daily.index else None
    if mock_mode:
        badge = '<span class="hero-badge">Données simulées</span>'
    elif raw_data.get("source") == "mock":
        badge = '<span class="hero-badge hero-badge--warning">Données simulées : API indisponible</span>'
    else:
        badge = ""
    _render_hero(city_name, latest, today, badge, palette)
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
        st.plotly_chart(_weather_figure(df, palette, retro_mode), width="stretch", theme=None if retro_mode else "streamlit")

    with tab2:
        st.subheader("Concentration des polluants clés")
        st.caption("PM2.5, PM10 et dioxyde d'azote, en µg/m³.")
        st.plotly_chart(_pollutants_figure(df, palette, retro_mode), width="stretch", theme=None if retro_mode else "streamlit")

    with tab3:
        st.subheader("Localisation de la station de suivi")
        map_df = pd.DataFrame([{
            "lat": city_info["lat"],
            "lon": city_info["lon"],
            "city": city_name,
            "aqi": latest["us_aqi"]
        }])
        fig_map = px.scatter_map(
            map_df, lat="lat", lon="lon", hover_name="city", hover_data=["aqi"],
            color_discrete_sequence=[palette["accent"]], zoom=12, height=600,
            center={"lat": city_info["lat"], "lon": city_info["lon"]},
            map_style=palette["map_style"]
        )
        fig_map.update_traces(marker=dict(size=18))
        st.plotly_chart(fig_map, width="stretch")

    st.divider()
    st.download_button(
        "Télécharger les données de toutes les villes (CSV)",
        data=lambda: _all_cities_csv(client, mock_mode),
        file_name="donnees_toutes_les_villes.csv",
        mime="text/csv",
        on_click="ignore",
        width="stretch",
    )
