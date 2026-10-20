import datetime
from typing import Dict, List, Optional

import pandas as pd

from .data_processing import DataProcessor

# Thresholds for alerts (WHO Guidelines)
THRESHOLDS = {
    "pm2_5": {"warning": 15.0, "danger": 25.0, "unit": "µg/m³"},
    "pm10": {"warning": 45.0, "danger": 50.0, "unit": "µg/m³"},
    "no2": {"warning": 25.0, "danger": 50.0, "unit": "µg/m³"},
    "temp_max": {"warning": 33.0, "danger": 38.0, "unit": "°C"}
}


def detect_alerts(
    df: pd.DataFrame,
    reference_time: Optional[datetime.datetime] = None,
    mordor_mode: bool = False,
) -> List[Dict[str, str]]:
    """Scans the current data point for threshold violations."""
    alerts = []
    if df.empty:
        return alerts

    latest = df.iloc[DataProcessor.get_current_index(df, reference_time)]

    # PM2.5 check
    if latest["pm2_5"] >= THRESHOLDS["pm2_5"]["danger"]:
        alerts.append({
            "level": "danger",
            "message": (
                "Alerte Rouge Pollution : Sauron a rallumé les hauts fourneaux"
                if mordor_mode
                else f"Alerte Rouge Pollution (PM2.5) : {latest['pm2_5']:.1f} "
                f"{THRESHOLDS['pm2_5']['unit']} (Seuil: {THRESHOLDS['pm2_5']['danger']})"
            )
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