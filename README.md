# weather-dashboard

Dashboard Streamlit de suivi météo et de qualité de l'air, basé sur l'API [Open-Meteo](https://open-meteo.com/) (gratuite, sans clé).

## Fonctionnalités

- Conditions actuelles : température, ressentie, min/max, humidité, vent, précipitations, nuages, pression, score confort
- Qualité de l'air : indice AQI (US) et 6 polluants (PM2.5, PM10, NO₂, O₃, SO₂, CO)
- Alertes sur les seuils OMS 2021, calculées sur des moyennes glissantes (24 h, 8 h pour l'ozone)
- Graphiques sur 7 jours d'historique et 4 jours de prévisions, période et courbes au choix
- Statistiques (moyenne, médiane, min, max) et tendances sur 24 h et 7 jours
- Carte des villes colorée selon l'AQI
- Export PDF (rapport de la ville) et CSV (toutes les villes)
- Mode déconnecté sur un jeu de données JSON, thèmes clair, sombre et rétro

## Lancer en local

Python 3.15 :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run run.py
```

Le dashboard est disponible sur http://localhost:8501.

## Lancer avec Docker

```bash
docker compose up -d --build
```

| Variable | Rôle | Défaut |
|---|---|---|
| `PORT` | Port d'écoute | `8501` |
| `LOG_LEVEL` | Niveau de log | `WARNING` (Docker), `INFO` (local) |
| `CACHE_DB_PATH` | Emplacement du cache SQLite | `/app/cache/data_cache.db` (Docker), `data/data_cache.db` (local) |
| `STREAMLIT_SERVER_BASE_URL_PATH` | Sous-chemin derrière un reverse proxy | aucun |

## Tests

```bash
pytest
```

## Mode déconnecté

Sans connexion, ou avec l'option « Mode déconnecté », le dashboard utilise `data/mock_responses.json` : de vraies réponses API enregistrées pour les villes de référence, recalées sur la date du jour. Pour régénérer ce fichier :

```bash
python -m src.api_client
```

## Structure

```
run.py                     Point d'entrée
src/api_client.py          Appels API, cache SQLite (TTL 30 min), données hors ligne
src/data_processing.py     Mise en forme, score confort, statistiques et tendances
src/alerts.py              Seuils OMS, alertes, catégories AQI
src/dashboard.py           Interface Streamlit
src/export.py              Rapport PDF
data/mock_responses.json   Jeu de données du mode déconnecté
tests/                     Tests pytest
```
