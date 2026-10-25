# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for weather tool functions and data contracts."""

from unittest.mock import MagicMock, patch

from app.agent import _f_to_c, get_weather


def test_f_to_c() -> None:
    """Verify Fahrenheit to Celsius conversion."""
    assert _f_to_c(32.0) == 0.0
    assert _f_to_c(212.0) == 100.0
    assert _f_to_c(68.0) == 20.0


@patch("app.agent.requests.get")
def test_get_weather_city_contract(mock_get: MagicMock) -> None:
    """Verify get_weather returns expected dictionary structure for a city."""
    geo_resp = MagicMock()
    geo_resp.status_code = 200
    geo_resp.json.return_value = {
        "results": [
            {
                "name": "Seattle",
                "admin1": "Washington",
                "country": "United States",
                "country_code": "US",
                "latitude": 47.6062,
                "longitude": -122.3321,
            }
        ]
    }

    forecast_resp = MagicMock()
    forecast_resp.status_code = 200
    forecast_resp.json.return_value = {
        "timezone": "America/Los_Angeles",
        "current": {
            "temperature_2m": 59.0,
            "relative_humidity_2m": 72,
            "apparent_temperature": 57.5,
            "weather_code": 2,
            "wind_speed_10m": 6.5,
        },
        "daily": {
            "time": ["2026-10-24", "2026-10-25", "2026-10-26"],
            "weather_code": [2, 61, 0],
            "temperature_2m_max": [62.0, 58.0, 64.0],
            "temperature_2m_min": [48.0, 46.0, 47.0],
            "precipitation_probability_max": [20, 80, 5],
        },
    }

    mock_get.side_effect = [geo_resp, forecast_resp]

    result = get_weather("Seattle, WA")
    assert result["status"] == "success"
    assert "Seattle" in result["location"]
    assert result["current"]["temperature_f"] == 59.0
    assert result["current"]["temperature_c"] == 15.0
    assert result["current"]["condition"] == "Partly cloudy"
    assert len(result["forecast"]) == 3


@patch("app.agent.requests.get")
def test_get_weather_unknown_location(mock_get: MagicMock) -> None:
    """Verify get_weather returns an error status when location is not found."""
    empty_resp = MagicMock()
    empty_resp.status_code = 200
    empty_resp.json.return_value = {"results": []}
    mock_get.return_value = empty_resp

    result = get_weather("ZzzNotARealPlace99999")
    assert result["status"] == "error"
    assert "error_message" in result
