# ruff: noqa
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

import re
from typing import Any

import requests
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

load_dotenv()

MODEL = "gemini-2.5-flash"

WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

US_STATE_ABBR: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}


def _f_to_c(temp_f: float) -> float:
    """Convert Fahrenheit to Celsius rounded to 1 decimal place."""
    return round((temp_f - 32.0) * 5.0 / 9.0, 1)


def _resolve_location(location: str) -> dict[str, Any] | None:
    """Resolve a city name or zip/postal code to coordinates and display name."""
    cleaned = location.strip()
    if not cleaned:
        return None

    # 1. Check if input is (or contains) a 5-digit US zip code
    zip_match = re.fullmatch(r"(?:zip(?:\s*code)?\s*)?(\d{5})(?:-\d{4})?", cleaned, re.IGNORECASE)
    if zip_match:
        zip_code = zip_match.group(1)
        try:
            resp = requests.get(f"https://api.zippopotam.us/us/{zip_code}", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                places = data.get("places", [])
                if places:
                    place = places[0]
                    city = place.get("place name", zip_code)
                    state = place.get("state", "")
                    state_abbr = place.get("state abbreviation", "")
                    return {
                        "name": f"{city}, {state_abbr or state} ({zip_code}), United States",
                        "latitude": float(place["latitude"]),
                        "longitude": float(place["longitude"]),
                    }
        except requests.RequestException:
            pass

    # 2. Try Open-Meteo Geocoding API
    # Handle "City, State" or "City, Country" inputs gracefully
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    search_candidates = [cleaned]
    region_hint = None
    if len(parts) >= 2:
        search_candidates.append(parts[0])
        region_hint = parts[1].upper()
        if region_hint in US_STATE_ABBR:
            region_hint = US_STATE_ABBR[region_hint].lower()
        else:
            region_hint = parts[1].lower()

    for candidate in search_candidates:
        try:
            resp = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": candidate, "count": 10, "language": "en", "format": "json"},
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if not results:
                continue

            chosen = results[0]
            if region_hint:
                for item in results:
                    admin1 = (item.get("admin1") or "").lower()
                    country = (item.get("country") or "").lower()
                    country_code = (item.get("country_code") or "").lower()
                    if (
                        region_hint in admin1
                        or region_hint == country
                        or region_hint == country_code
                    ):
                        chosen = item
                        break

            label_parts = [chosen.get("name", candidate)]
            if chosen.get("admin1"):
                label_parts.append(chosen["admin1"])
            if chosen.get("country"):
                label_parts.append(chosen["country"])
            return {
                "name": ", ".join(label_parts),
                "latitude": float(chosen["latitude"]),
                "longitude": float(chosen["longitude"]),
            }
        except requests.RequestException:
            continue

    return None


def get_weather(location: str) -> dict[str, Any]:
    """Fetches current weather conditions and a 3-day forecast for a city or zip code.

    Args:
        location: The city name (e.g., 'Austin, TX', 'Tokyo') or postal/zip code (e.g., '10001').

    Returns:
        A dictionary containing 'status', resolved 'location', 'current' weather metrics,
        and a 3-day 'forecast' list, or an 'error_message' if the location cannot be resolved.
    """
    resolved = _resolve_location(location)
    if not resolved:
        return {
            "status": "error",
            "error_message": (
                f"Could not find location '{location}'. Please provide a valid city name "
                "(e.g., 'Seattle, WA') or 5-digit zip code (e.g., '10001')."
            ),
        }

    try:
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": resolved["latitude"],
                "longitude": resolved["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "auto",
                "forecast_days": 3,
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        data = weather_resp.json()
    except requests.RequestException as exc:
        return {
            "status": "error",
            "error_message": f"Failed to retrieve weather data for '{resolved['name']}': {exc}",
        }

    current = data.get("current", {})
    temp_f = float(current.get("temperature_2m", 0.0))
    feels_f = float(current.get("apparent_temperature", temp_f))
    wind_mph = float(current.get("wind_speed_10m", 0.0))
    wmo_code = int(current.get("weather_code", 0))

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precip_probs = daily.get("precipitation_probability_max", [])
    daily_codes = daily.get("weather_code", [])

    forecast_days = []
    for idx, day_date in enumerate(dates):
        hi_f = float(max_temps[idx]) if idx < len(max_temps) else 0.0
        lo_f = float(min_temps[idx]) if idx < len(min_temps) else 0.0
        d_code = int(daily_codes[idx]) if idx < len(daily_codes) else 0
        precip = int(precip_probs[idx]) if idx < len(precip_probs) and precip_probs[idx] is not None else 0
        forecast_days.append(
            {
                "date": day_date,
                "condition": WMO_WEATHER_CODES.get(d_code, f"Code {d_code}"),
                "high_f": round(hi_f, 1),
                "high_c": _f_to_c(hi_f),
                "low_f": round(lo_f, 1),
                "low_c": _f_to_c(lo_f),
                "precipitation_chance_percent": precip,
            }
        )

    return {
        "status": "success",
        "location": resolved["name"],
        "coordinates": {
            "latitude": resolved["latitude"],
            "longitude": resolved["longitude"],
        },
        "timezone": data.get("timezone", "UTC"),
        "current": {
            "condition": WMO_WEATHER_CODES.get(wmo_code, f"Code {wmo_code}"),
            "temperature_f": round(temp_f, 1),
            "temperature_c": _f_to_c(temp_f),
            "feels_like_f": round(feels_f, 1),
            "feels_like_c": _f_to_c(feels_f),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_speed_mph": round(wind_mph, 1),
            "wind_speed_kmh": round(wind_mph * 1.60934, 1),
        },
        "forecast": forecast_days,
    }


root_agent = Agent(
    # Keep in sync with agents-cli-manifest.yaml: agents-cli derives this name
    # from the project `name:` recorded there, and telemetry reports it as
    # gen_ai.agent.name. Renaming the agent only here makes the two disagree,
    # and anything selecting traces by name stops finding this agent's.
    name="weather_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=128),
    ),
    description="Provides current weather conditions and multi-day forecasts for cities and zip codes.",
    instruction=(
        "You are a helpful weather assistant that provides accurate, real-time weather conditions "
        "and 3-day forecasts for cities and postal/zip codes worldwide.\n\n"
        "Rules:\n"
        "1. Always call the `get_weather` tool whenever a user asks for the weather or forecast "
        "for a city or zip code.\n"
        "2. In your response, clearly state the resolved location name, current temperature "
        "(in both °F and °C), feels-like temperature, current sky condition, humidity, wind speed, "
        "and a summary of the 3-day forecast (high/low temperatures and precipitation chance).\n"
        "3. Never guess, invent, or hallucinate weather data. If `get_weather` returns an error "
        "status, politely inform the user that the location could not be found and ask for a valid "
        "city name or 5-digit zip code.\n"
        "4. Only answer questions related to weather and forecasts. If a user asks about unrelated "
        "topics (such as coding, medical advice, finance, or general trivia), politely decline and "
        "explain that you only assist with weather information."
    ),
    tools=[get_weather],
)

app = App(
    root_agent=root_agent,
    name="app",
)
