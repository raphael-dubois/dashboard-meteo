"""
PDF report of the dashboard (2 A4 pages): current conditions, pollutants and alerts on page 1,
charts and 24 h statistics on page 2. fpdf2 lays out the document, matplotlib draws the charts.
"""
import datetime
import io
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Charts are rendered to PNG in memory, no display needed

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

from .alerts import AQI_SHORT_LABELS, POLLUTANT_KEYS, THRESHOLDS, aqi_index, current_means, detect_alerts
from .data_processing import DataProcessor

# fpdf2 subsets fonts with fontTools, which logs every step at INFO level
logging.getLogger("fontTools").setLevel(logging.WARNING)

# DejaVu fonts ship with matplotlib: the report renders accents, "₂", "µ" and "³" on any machine
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

# Print palette (editorial light theme), RGB
COLORS = {
    "text": (43, 38, 32), "muted": (133, 123, 110), "line": (221, 213, 199),
    "ok": (79, 122, 53), "warning": (168, 102, 26), "danger": (179, 38, 30),
    "warning_bg": (248, 237, 217), "danger_bg": (247, 228, 224), "ok_bg": (233, 239, 223),
}
AQI_COLORS = [(91, 140, 58), (176, 138, 30), (208, 122, 38), (179, 38, 30), (123, 63, 140), (110, 30, 43)]
CHART_COLORS = ["#C2603A", "#6B8CB0"]
STATUS_LABELS = {"ok": "Conforme", "warning": "Vigilance", "danger": "Alerte"}

# Page 2 charts: (columns, labels, title)
CHARTS = [
    (["temperature_2m", "apparent_temperature"], ["Température", "Ressentie"], "Température et ressentie (°C)"),
    (["pm2_5", "nitrogen_dioxide"], ["PM2.5", "NO₂"], "PM2.5 et NO₂ (µg/m³)"),
]
# Statistics table rows: (column, label, decimals)
STATS_ROWS = [
    ("temperature_2m", "Température (°C)", 1),
    ("apparent_temperature", "Ressentie (°C)", 1),
    ("relativehumidity_2m", "Humidité (%)", 0),
    ("windspeed_10m", "Vent (km/h)", 0),
    ("pm2_5", "PM2.5 (µg/m³)", 1),
    ("nitrogen_dioxide", "NO₂ (µg/m³)", 1),
    ("ozone", "O₃ (µg/m³)", 1),
]

PAGE_WIDTH = 178  # A4 width (210 mm) minus 16 mm margins


def _fmt(value: Any, decimals: int = 1, signed: bool = False) -> str:
    """French number format ("12,6"), "–" when the value is missing."""
    if value is None or pd.isna(value):
        return "–"
    text = f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"
    return text.replace(".", ",")


def report_file_name(city: str, when: datetime.datetime) -> str:
    """File name such as 'rapport_meteo_paris_2026-10-20.pdf' (ASCII only)."""
    ascii_city = unicodedata.normalize("NFKD", city).encode("ascii", "ignore").decode()
    slug = "".join(c if c.isalnum() else "_" for c in ascii_city.lower()).strip("_")
    return f"rapport_meteo_{slug}_{when:%Y-%m-%d}.pdf"


class _ReportPDF(FPDF):
    """A4 document with the report fonts and a footer on every page."""

    def __init__(self, footer_note: str):
        super().__init__(format="A4")
        self.footer_note = footer_note
        self.add_font("Sans", "", FONT_DIR / "DejaVuSans.ttf")
        self.add_font("Sans", "B", FONT_DIR / "DejaVuSans-Bold.ttf")
        self.add_font("Serif", "", FONT_DIR / "DejaVuSerif.ttf")
        self.set_margins(16, 16, 16)
        self.set_auto_page_break(True, margin=18)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_draw_color(*COLORS["line"])
        self.line(16, self.get_y(), 16 + PAGE_WIDTH, self.get_y())
        self.set_font("Sans", "", 8)
        self.set_text_color(*COLORS["muted"])
        self.cell(PAGE_WIDTH / 2, 6, self.footer_note)
        self.cell(PAGE_WIDTH / 2, 6, f"{self.page_no()} / {{nb}}", align="R")

    def heading(self, text: str) -> None:
        """Section title with a thin rule underneath."""
        self.ln(4)
        self.set_font("Serif", "", 12)
        self.set_text_color(*COLORS["text"])
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*COLORS["line"])
        self.line(16, self.get_y(), 16 + PAGE_WIDTH, self.get_y())
        self.ln(2)

    def table_row(self, cells: List[str], widths: List[float], header: bool = False,
                  colors: Optional[List[Optional[Tuple[int, int, int]]]] = None,
                  aligns: Optional[str] = None) -> None:
        """
        One table row with a thin bottom rule. aligns: one letter per column ("L"/"R"),
        by default the first column left-aligned and the others right-aligned.
        """
        aligns = aligns or "L" + "R" * (len(cells) - 1)
        self.set_font("Sans", "", 8 if header else 9)
        for index, (text, width) in enumerate(zip(cells, widths)):
            color = (colors[index] if colors else None) or (COLORS["muted"] if header else COLORS["text"])
            self.set_text_color(*color)
            self.cell(width, 6.5, text, align=aligns[index])
        self.ln()
        self.set_draw_color(*COLORS["line"])
        self.line(16, self.get_y(), 16 + sum(widths), self.get_y())


def _chart_png(df: pd.DataFrame, columns: List[str], labels: List[str], title: str, now: pd.Timestamp) -> bytes:
    """Line chart over the selected period, forecasts shaded after "now"."""
    fig, ax = plt.subplots(figsize=(7.0, 2.35), dpi=200)
    for column, label, color in zip(columns, labels, CHART_COLORS):
        ax.plot(df["datetime"], df[column], label=label, color=color, linewidth=1.4)

    start, end = df["datetime"].min(), df["datetime"].max()
    if start <= now < end:
        ax.axvspan(now, end, color="#F1EDE6", zorder=0)
        ax.axvline(now, color="#A39786", linewidth=0.8, linestyle=":")
        ax.text(now, 1.0, " Prévisions", transform=ax.get_xaxis_transform(), va="top", fontsize=7, color="#857B6E")

    ax.set_title(title, loc="left", fontsize=8, color="#2B2620", pad=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.tick_params(labelsize=7, colors="#857B6E", length=0)
    ax.grid(axis="y", color="#E6DFD3", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2,
              borderaxespad=0.2, handlelength=1.5)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def _page_conditions(pdf: _ReportPDF, city: Dict[str, Any], latest: pd.Series, today: Optional[pd.Series],
                     temp_change: float, data_note: str) -> None:
    """Page 1, top: title, city, current weather and AQI."""
    pdf.set_font("Serif", "", 17)
    pdf.set_text_color(*COLORS["text"])
    pdf.cell(PAGE_WIDTH * 0.75, 9, "Rapport météo et qualité de l'air")
    pdf.set_font("Sans", "", 9)
    pdf.set_text_color(*COLORS["muted"])
    pdf.cell(PAGE_WIDTH * 0.25, 9, f"{latest['datetime']:%d/%m/%Y · %H:%M}", align="R", new_x="LMARGIN", new_y="NEXT")
    lat_ref, lon_ref = ("N" if city["lat"] >= 0 else "S"), ("E" if city["lon"] >= 0 else "O")
    pdf.cell(0, 5, f"{city['name']} · {abs(city['lat']):.2f}° {lat_ref}, {abs(city['lon']):.2f}° {lon_ref}",
             new_x="LMARGIN", new_y="NEXT")
    if data_note:
        pdf.set_text_color(*COLORS["warning"])
        pdf.cell(0, 5, data_note, new_x="LMARGIN", new_y="NEXT")

    pdf.heading("Conditions actuelles")
    top = pdf.get_y()
    pdf.set_font("Sans", "B", 30)
    pdf.set_text_color(*COLORS["text"])
    pdf.cell(28, 13, f"{latest['temperature_2m']:.0f}°")
    pdf.set_font("Sans", "", 9)
    pdf.set_text_color(*COLORS["muted"])
    details = f"Ressentie {latest['apparent_temperature']:.0f}°"
    if today is not None:
        details += f" · min {today['temperature_2m_min']:.0f}° / max {today['temperature_2m_max']:.0f}°"
    pdf.set_xy(16 + 30, top + 1.5)
    pdf.cell(100, 5, details)
    pdf.set_xy(16 + 30, top + 6.5)
    if not pd.isna(temp_change):
        pdf.cell(100, 5, f"{_fmt(temp_change, 1, signed=True)}° par rapport à hier à la même heure")

    aqi = latest["us_aqi"]
    pdf.set_xy(16 + PAGE_WIDTH - 50, top)
    pdf.set_font("Sans", "B", 16)
    if pd.isna(aqi):
        pdf.set_text_color(*COLORS["muted"])
        pdf.cell(50, 8, "AQI indisponible", align="R")
    else:
        pdf.set_text_color(*AQI_COLORS[aqi_index(aqi)])
        pdf.cell(50, 8, f"AQI {aqi:.0f}", align="R")
        pdf.set_xy(16 + PAGE_WIDTH - 50, top + 8)
        pdf.set_font("Sans", "", 9)
        pdf.set_text_color(*COLORS["muted"])
        pdf.cell(50, 5, f"{AQI_SHORT_LABELS[aqi_index(aqi)]} · indice US", align="R")
    pdf.set_xy(16, top + 16)

    precipitation = today["precipitation_sum"] if today is not None else latest["precipitation"]
    rows = [
        ("Humidité", f"{latest['relativehumidity_2m']:.0f} %", "Vent", f"{latest['windspeed_10m']:.0f} km/h"),
        ("Précipitations du jour", f"{_fmt(precipitation)} mm", "Couverture nuageuse", f"{latest['cloud_cover']:.0f} %"),
        ("Pression", f"{latest['surface_pressure']:.0f} hPa", "Score confort", f"{latest['health_score']:.0f}/100"),
    ]
    widths = [50, 34, 10, 50, 34]
    for label, value, label_2, value_2 in rows:
        pdf.table_row([label, value, "", label_2, value_2], widths, aligns="LRLLR",
                      colors=[COLORS["muted"], None, None, COLORS["muted"], None])


def _page_air_quality(pdf: _ReportPDF, means: Dict[str, float], alerts: List[Dict[str, str]]) -> None:
    """Page 1, bottom: pollutants against WHO thresholds, then active alerts."""
    pdf.heading("Polluants (moyennes sur les périodes OMS)")
    widths = [40, 38, 30, 34, 36]
    pdf.table_row(["Polluant", "Moyenne (µg/m³)", "Période", "Seuil OMS", "Statut"], widths, header=True)
    for key in POLLUTANT_KEYS:
        rule, value = THRESHOLDS[key], means.get(key)
        if value is None or pd.isna(value):
            level = None
        else:
            level = "danger" if value >= rule["danger"] else "warning" if value >= rule["warning"] else "ok"
        pdf.table_row(
            [rule["label"], _fmt(value, 0 if key == "co" else 1), f"{rule['window_hours']} h",
             _fmt(rule["warning"], 0), STATUS_LABELS.get(level, "–")],
            widths, colors=[None, None, COLORS["muted"], COLORS["muted"], COLORS[level] if level else None],
        )

    pdf.heading("Alertes")
    if not alerts:
        alerts = [{"level": "ok", "message": "Aucune alerte active : qualité de l'air et conditions météorologiques stables."}]
    pdf.set_font("Sans", "", 9)
    for alert in alerts:
        level = alert["level"] if alert["level"] in ("ok", "warning", "danger") else "warning"
        y = pdf.get_y()
        pdf.set_fill_color(*COLORS[f"{level}_bg"])
        pdf.rect(16, y, PAGE_WIDTH, 8, style="F")
        pdf.set_fill_color(*COLORS[level])
        pdf.rect(16, y, 1.2, 8, style="F")
        pdf.set_xy(20, y + 1.5)
        pdf.set_text_color(*COLORS["text"])
        pdf.cell(PAGE_WIDTH - 6, 5, alert["message"])
        pdf.set_xy(16, y + 10)


def _page_trends(pdf: _ReportPDF, period_df: pd.DataFrame, stats: pd.DataFrame, now: pd.Timestamp) -> None:
    """Page 2: charts over the selected period and 24 h statistics."""
    start, end = period_df["datetime"].min(), period_df["datetime"].max()
    pdf.heading(f"Évolution du {start:%d/%m} au {end:%d/%m}")
    for columns, labels, title in CHARTS:
        pdf.image(io.BytesIO(_chart_png(period_df, columns, labels, title, now)), w=PAGE_WIDTH)
        pdf.ln(2)
    pdf.set_font("Sans", "", 8)
    pdf.set_text_color(*COLORS["muted"])
    pdf.cell(0, 5, "Pas horaire. La zone grisée correspond aux prévisions.", new_x="LMARGIN", new_y="NEXT")

    pdf.heading("Statistiques sur les dernières 24 h")
    widths = [44, 22, 22, 18, 18, 26, 28]
    pdf.table_row(["Métrique", "Moyenne", "Médiane", "Min", "Max", "Sur 24 h", "Sur 7 j (/jour)"], widths, header=True)
    for column, label, decimals in STATS_ROWS:
        if column not in stats.index:
            continue
        row = stats.loc[column]
        pdf.table_row(
            [label, _fmt(row["mean"], decimals), _fmt(row["median"], decimals), _fmt(row["min"], decimals),
             _fmt(row["max"], decimals), _fmt(row["change_24h"], decimals, signed=True),
             _fmt(row["slope_7d"], decimals + 1, signed=True)],
            widths, colors=[None, None, None, None, None, COLORS["muted"], COLORS["muted"]],
        )
    pdf.ln(2)
    pdf.set_font("Sans", "", 8)
    pdf.set_text_color(*COLORS["muted"])
    pdf.multi_cell(PAGE_WIDTH, 4.5, (
        "Sur 24 h : valeur actuelle moins celle d'il y a 24 h. Sur 7 j : pente d'une régression linéaire "
        "sur les 7 derniers jours. Calculs sur les observations uniquement, prévisions exclues."
    ))


def build_pdf_report(
    city: Dict[str, Any], df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date,
    today: Optional[pd.Series], reference_time: Optional[datetime.datetime], data_note: str = "",
) -> bytes:
    """
    Builds the 2-page PDF report of a city.
    city: {"name", "lat", "lon"}; df: processed hourly data (with health_score);
    start_date/end_date: period of the charts; today: daily aggregates of the current day.
    """
    if "health_score" not in df:
        df = df.assign(health_score=DataProcessor.calculate_custom_health_index(df))
    latest = df.iloc[DataProcessor.get_current_index(df, reference_time)]
    period_df = df[(df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)]
    stats = DataProcessor.compute_statistics(df, [column for column, _, _ in STATS_ROWS], 24, reference_time)

    generated = datetime.datetime.now()
    pdf = _ReportPDF(f"Source : Open-Meteo · seuils OMS 2021 · généré le {generated:%d/%m/%Y à %H:%M}")
    pdf.set_title(f"Rapport météo et qualité de l'air - {city['name']}")

    pdf.add_page()
    _page_conditions(pdf, city, latest, today, stats.loc["temperature_2m", "change_24h"], data_note)
    _page_air_quality(pdf, current_means(df, reference_time), detect_alerts(df, reference_time))

    pdf.add_page()
    if period_df.empty:
        pdf.heading("Évolution")
        pdf.cell(0, 6, "Aucune donnée sur la période sélectionnée.")
    else:
        _page_trends(pdf, period_df, stats, latest["datetime"])

    return bytes(pdf.output())
