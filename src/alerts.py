import datetime
from typing import Dict, List, Optional

import pandas as pd

from .data_processing import DataProcessor

# Thresholds for alerts (WHO 2021 Air Quality Guidelines).
# "warning" is the guideline level, "danger" the last interim target. Each value is compared
# to its WHO averaging period (24 h, 8 h for ozone); temperature uses the current hour.
THRESHOLDS = {
    "pm2_5": {"column": "pm2_5", "label": "PM2.5", "window_hours": 24, "warning": 15.0, "danger": 25.0, "unit": "µg/m³"},
    "pm10": {"column": "pm10", "label": "PM10", "window_hours": 24, "warning": 45.0, "danger": 50.0, "unit": "µg/m³"},
    "no2": {"column": "nitrogen_dioxide", "label": "NO₂", "window_hours": 24, "warning": 25.0, "danger": 50.0, "unit": "µg/m³"},
    "o3": {"column": "ozone", "label": "O₃", "window_hours": 8, "warning": 100.0, "danger": 120.0, "unit": "µg/m³"},
    "so2": {"column": "sulphur_dioxide", "label": "SO₂", "window_hours": 24, "warning": 40.0, "danger": 50.0, "unit": "µg/m³"},
    "co": {"column": "carbon_monoxide", "label": "CO", "window_hours": 24, "warning": 4000.0, "danger": 7000.0, "unit": "µg/m³"},
    "temp_max": {"column": "temperature_2m", "label": "Température", "window_hours": 1, "warning": 33.0, "danger": 38.0, "unit": "°C"},
}

POLLUTANT_KEYS = ["pm2_5", "pm10", "no2", "o3", "so2", "co"]


def current_means(df: pd.DataFrame, reference_time: Optional[datetime.datetime] = None) -> Dict[str, float]:
    """Current value of every monitored variable over its WHO averaging period."""
    if df.empty:
        return {}
    if "datetime" not in df:
        df = df.assign(datetime=pd.to_datetime(df["time"]))

    means = {}
    for key, rule in THRESHOLDS.items():
        if rule["column"] in df:
            means[key] = DataProcessor.rolling_mean(df, rule["column"], rule["window_hours"], reference_time)
    return means


def _build_alert(key: str, value: float) -> Optional[Dict[str, str]]:
    """Alert message for a value above its warning or danger threshold."""
    rule = THRESHOLDS[key]
    level = "danger" if value >= rule["danger"] else "warning" if value >= rule["warning"] else None
    if level is None:
        return None

    if key == "temp_max":
        title = "Alerte canicule" if level == "danger" else "Vigilance forte chaleur"
        return {"level": level, "message": f"{title} : {value:.1f} °C (seuil {rule[level]:g} °C)"}

    period = f"en moyenne sur {rule['window_hours']} h"
    if level == "danger":
        message = (f"Alerte pic de pollution {rule['label']} : {value:.1f} {rule['unit']} {period} "
                   f"(seuil d'alerte {rule['danger']:g})")
    else:
        message = (f"Vigilance {rule['label']} : {value:.1f} {rule['unit']} {period} "
                   f"(recommandation OMS {rule['warning']:g})")
    return {"level": level, "message": message}


def detect_alerts(df: pd.DataFrame, reference_time: Optional[datetime.datetime] = None) -> List[Dict[str, str]]:
    """Compares the current WHO-period averages to the thresholds. Danger alerts come first."""
    alerts = []
    for key, value in current_means(df, reference_time).items():
        if pd.isna(value):
            continue
        alert = _build_alert(key, value)
        if alert:
            alerts.append(alert)
    return sorted(alerts, key=lambda alert: alert["level"] != "danger")
