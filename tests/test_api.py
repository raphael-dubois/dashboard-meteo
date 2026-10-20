import datetime
import os
import tempfile
import unittest
from unittest import mock

import requests

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
        # 7 days of history + forecasts, like the live API
        self.assertGreater(len(data["weather"]["hourly"]["time"]), 24 * 7)
        self.assertIn("pm2_5", data["air_quality"]["hourly"])

    def test_mock_data_is_stable_and_realigned_on_today(self):
        first = self.client.fetch_weather_and_air_quality(48.85, 2.35, mock_mode=True)
        second = self.client.fetch_weather_and_air_quality(48.85, 2.35, mock_mode=True)
        self.assertEqual(first, second)
        offset = first["weather"]["utc_offset_seconds"]
        today = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=offset)).date()
        self.assertIn(f"{today:%Y-%m-%d}T00:00", first["weather"]["hourly"]["time"])
        self.assertIn(f"{today:%Y-%m-%d}T00:00", first["air_quality"]["hourly"]["time"])
        self.assertIn(today.isoformat(), first["weather"]["daily"]["time"])

    def test_mock_data_uses_nearest_reference_city(self):
        versailles = self.client.fetch_weather_and_air_quality(48.80, 2.13, mock_mode=True)
        kyoto = self.client.fetch_weather_and_air_quality(35.01, 135.77, mock_mode=True)
        self.assertEqual(versailles["mock_city"], "Paris")
        self.assertEqual(kyoto["mock_city"], "Tokyo")

    def test_missing_offline_dataset_returns_empty_data(self):
        client = WeatherAPIClient(
            db_path=os.path.join(self.tmp.name, "other.db"),
            mock_path=os.path.join(self.tmp.name, "missing.json"),
        )
        data = client.fetch_weather_and_air_quality(48.85, 2.35, mock_mode=True)
        self.assertEqual((data["source"], data["weather"], data["air_quality"]), ("mock", {}, {}))

    def test_network_failure_falls_back_to_offline_dataset(self):
        with mock.patch("src.api_client.requests.get", side_effect=requests.exceptions.ConnectionError):
            data = self.client.fetch_weather_and_air_quality(48.85, 2.35)
        self.assertEqual(data["source"], "mock")
        self.assertEqual(data["mock_city"], "Paris")

    def test_mock_data_flags_source_and_has_full_weather(self):
        data = self.client.fetch_weather_and_air_quality(48.85, 2.35, mock_mode=True)
        self.assertEqual(data["source"], "mock")
        hourly = data["weather"]["hourly"]
        for column in ("apparent_temperature", "precipitation", "cloud_cover", "wind_direction_10m"):
            self.assertEqual(len(hourly[column]), len(hourly["time"]))
        daily = data["weather"]["daily"]
        self.assertEqual(len(daily["time"]), len(daily["temperature_2m_max"]))
        self.assertTrue(all(lo <= hi for lo, hi in zip(daily["temperature_2m_min"], daily["temperature_2m_max"])))

    def test_batch_aqi_maps_results_to_cities_in_order(self):
        cities = {"Paris": {"lat": 48.85, "lon": 2.35}, "Lyon": {"lat": 45.76, "lon": 4.83}}
        response = mock.Mock()
        response.json.return_value = [{"current": {"us_aqi": 42}}, {"current": {"us_aqi": 87}}]
        with mock.patch("src.api_client.requests.get", return_value=response) as get:
            result = self.client.fetch_current_aqi(cities)
        self.assertEqual(result, {"Paris": 42, "Lyon": 87})
        self.assertEqual(get.call_args.kwargs["params"]["latitude"], "48.85,45.76")

    def test_batch_aqi_falls_back_to_offline_dataset_when_network_fails(self):
        cities = {"Paris": {"lat": 48.85, "lon": 2.35}, "Tokyo": {"lat": 35.69, "lon": 139.69}}
        with mock.patch("src.api_client.requests.get", side_effect=requests.exceptions.ConnectionError):
            result = self.client.fetch_current_aqi(cities)
        self.assertEqual(set(result), {"Paris", "Tokyo"})
        self.assertTrue(all(isinstance(aqi, (int, float)) for aqi in result.values()))

    def test_cache_roundtrip(self):
        payload = {"hello": "world"}
        self.client._set_cached_data("test_key", payload)
        self.assertEqual(self.client._get_cached_data("test_key"), payload)

    def test_cache_miss(self):
        self.assertIsNone(self.client._get_cached_data("unknown_key"))


if __name__ == "__main__":
    unittest.main()