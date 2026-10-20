import os
import tempfile
import unittest

from src.api_client import WeatherAPIClient


class TestWeatherAPIClient(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.client = WeatherAPIClient(db_path=os.path.join(self.tmp.name, "test_cache.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_mock_data_structure(self):
        data = self.client.fetch_weather_and_air_quality(48.85, 2.35, mock_mode=True)
        self.assertIn("weather", data)
        self.assertIn("air_quality", data)
        self.assertEqual(len(data["weather"]["hourly"]["time"]), 24 * 7)
        self.assertIn("pm2_5", data["air_quality"]["hourly"])

    def test_mock_data_flags_source_and_has_full_weather(self):
        data = self.client.fetch_weather_and_air_quality(48.85, 2.35, mock_mode=True)
        self.assertEqual(data["source"], "mock")
        hourly = data["weather"]["hourly"]
        for column in ("apparent_temperature", "precipitation", "cloud_cover", "wind_direction_10m"):
            self.assertEqual(len(hourly[column]), len(hourly["time"]))
        daily = data["weather"]["daily"]
        self.assertEqual(len(daily["time"]), len(daily["temperature_2m_max"]))
        self.assertTrue(all(lo <= hi for lo, hi in zip(daily["temperature_2m_min"], daily["temperature_2m_max"])))

    def test_cache_roundtrip(self):
        payload = {"hello": "world"}
        self.client._set_cached_data("test_key", payload)
        self.assertEqual(self.client._get_cached_data("test_key"), payload)

    def test_cache_miss(self):
        self.assertIsNone(self.client._get_cached_data("unknown_key"))


if __name__ == "__main__":
    unittest.main()