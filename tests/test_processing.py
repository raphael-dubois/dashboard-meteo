import datetime
import io
import os
import tempfile
import unittest

import pandas as pd

from src.alerts import detect_alerts
from src.api_client import DEFAULT_CITIES, WeatherAPIClient
from src.dashboard import _all_cities_csv
from src.data_processing import DataProcessor


class TestDashboardModules(unittest.TestCase):
    """Unit testing module for verifying data processing & scoring logic."""

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
        alerts = detect_alerts(self.mock_df)
        self.assertTrue(len(alerts) > 0)

    def test_mordor_alert_and_zero_health_score(self):
        mordor_df = pd.DataFrame({
            "temperature_2m": [60.0],
            "relativehumidity_2m": [5.0],
            "pm2_5": [900.0],
            "nitrogen_dioxide": [600.0],
        })
        self.assertEqual(
            DataProcessor.calculate_custom_health_index(mordor_df).iloc[0],
            0.0,
        )
        alerts = detect_alerts(mordor_df, mordor_mode=True)
        self.assertEqual(
            alerts[0]["message"],
            "Alerte Rouge Pollution : Sauron a rallumé les hauts fourneaux",
        )

    def test_no_alerts_on_empty_df(self):
        self.assertEqual(detect_alerts(pd.DataFrame()), [])

    def test_current_index_skips_forecasts(self):
        df = self.mock_df.assign(datetime=pd.to_datetime(self.mock_df["time"]))
        reference_time = datetime.datetime(2026, 1, 1, 0, 30)
        self.assertEqual(DataProcessor.get_current_index(df, reference_time), 0)
        # Only the forecast row (40 °C, high pollution) would trigger alerts
        self.assertEqual(detect_alerts(df, reference_time), [])

    def test_reference_time_uses_api_offset(self):
        reference_time = DataProcessor.get_reference_time({"weather": {"utc_offset_seconds": 3600}})
        now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        delta = reference_time - now_utc
        self.assertAlmostEqual(delta.total_seconds(), 3600, delta=5)

    def test_all_cities_csv_2_days_every_2_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = WeatherAPIClient(db_path=os.path.join(tmp, "test_cache.db"))
            csv_bytes = _all_cities_csv(client, mock_mode=True)
            exported_df = pd.read_csv(io.BytesIO(csv_bytes), sep=";")
            # 2 days (48h) sampled every 2h = 24 rows per city
            self.assertEqual(len(exported_df), len(DEFAULT_CITIES) * 24)
            paris_times = pd.to_datetime(exported_df.loc[exported_df["ville"] == "Paris", "datetime"])
            diffs = paris_times.diff().dropna()
            self.assertTrue((diffs == pd.Timedelta(hours=2)).all())


if __name__ == "__main__":
    unittest.main()