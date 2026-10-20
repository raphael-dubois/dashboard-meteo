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

    def test_mordor_mock_data_values(self):
        data = self.client.generate_mordor_data()
        self.assertEqual(data["weather"]["current_weather"]["temperature"], 60.0)
        self.assertEqual(set(data["weather"]["hourly"]["temperature_2m"]), {60.0})
        self.assertEqual(set(data["air_quality"]["hourly"]["pm2_5"]), {900.0})

    def test_cache_roundtrip(self):
        payload = {"hello": "world"}
        self.client._set_cached_data("test_key", payload)
        self.assertEqual(self.client._get_cached_data("test_key"), payload)

    def test_cache_miss(self):
        self.assertIsNone(self.client._get_cached_data("unknown_key"))


if __name__ == "__main__":
    unittest.main()