import unittest

import pandas as pd

from src.alerts import detect_alerts
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

    def test_no_alerts_on_empty_df(self):
        self.assertEqual(detect_alerts(pd.DataFrame()), [])


if __name__ == "__main__":
    unittest.main()