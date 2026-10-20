import pandas as pd
import plotly.express as px
import streamlit as st

from .alerts import detect_alerts
from .api_client import WeatherAPIClient, DEFAULT_CITIES
from .data_processing import DataProcessor


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
    alerts = detect_alerts(df)
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
            color_discrete_sequence=["fuchsia"], zoom=9, height=400,
            map_style="open-street-map"
        )
        st.plotly_chart(fig_map, width="stretch")