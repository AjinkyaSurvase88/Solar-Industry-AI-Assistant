"""
Weather Service
===============
Fetches real-time and historical weather/solar data.

Priority order:
  1. Open-Meteo API      — free, no key, real-time + historical
  2. OpenWeatherMap API  — free tier with API key (more detailed)
  3. NASA POWER monthly  — long-term averages as ultimate fallback
"""

import requests
import os
import math
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY", "")


def fetch_openmeteo(lat: float, lon: float) -> Optional[dict]:
    """
    Fetch current solar & weather data from Open-Meteo.
    Free, no API key, globally available.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "cloud_cover",
                "wind_speed_10m",
                "weather_code",
            ],
            "daily": [
                "shortwave_radiation_sum",     # GHI kWh/m²
                "temperature_2m_max",
                "temperature_2m_min",
                "sunshine_duration",           # seconds
                "precipitation_sum",
            ],
            "timezone": "auto",
            "forecast_days": 7,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        daily   = data.get("daily", {})

        # Use today's GHI + 7-day average for robustness
        ghi_list = [v for v in daily.get("shortwave_radiation_sum", []) if v is not None]
        ghi_today = ghi_list[0] if ghi_list else None

        sunshine_list = daily.get("sunshine_duration", [])
        sunshine_today = sunshine_list[0] if sunshine_list else None  # seconds

        return {
            "source":           "open-meteo",
            "temperature":      current.get("temperature_2m"),
            "humidity":         current.get("relative_humidity_2m"),
            "cloud_cover":      current.get("cloud_cover"),
            "wind_speed":       current.get("wind_speed_10m"),
            "ghi":              ghi_today,                            # kWh/m²/day
            "sunshine_hours":   sunshine_today / 3600 if sunshine_today else None,
            "weather_code":     current.get("weather_code"),
            "temp_max_7d":      max(v for v in daily.get("temperature_2m_max", [0]) if v),
            "temp_min_7d":      min(v for v in daily.get("temperature_2m_min", [0]) if v),
            "ghi_7d_avg":       sum(ghi_list) / len(ghi_list) if ghi_list else None,
        }
    except Exception as e:
        print(f"  Open-Meteo error: {e}")
        return None


def fetch_openweather(lat: float, lon: float) -> Optional[dict]:
    """Fetch current weather from OpenWeatherMap (requires API key)."""
    if not OPENWEATHER_KEY or OPENWEATHER_KEY == "your_openweather_api_key_here":
        return None

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHER_KEY,
            "units": "metric",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        d = resp.json()

        clouds    = d.get("clouds", {}).get("all", 0)
        uv_index  = _estimate_uv_from_clouds(clouds, lat)

        return {
            "source":       "openweathermap",
            "temperature":  d["main"]["temp"],
            "humidity":     d["main"]["humidity"],
            "cloud_cover":  clouds,
            "wind_speed":   d["wind"]["speed"],
            "description":  d["weather"][0]["description"],
            "ghi":          _estimate_ghi(uv_index, clouds),
            "uv_index":     uv_index,
            "city_name":    d.get("name", ""),
        }
    except Exception as e:
        print(f"  OpenWeatherMap error: {e}")
        return None


def fetch_nasa_power_current(lat: float, lon: float) -> Optional[dict]:
    """
    Fetch current month's 20-year average from NASA POWER.
    Used as ultimate fallback — always works, no key needed.
    """
    try:
        month = datetime.now().month
        year  = datetime.now().year
        # Get this month across 20 years for climatological average
        url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
        params = {
            "parameters": "ALLSKY_SFC_SW_DWN,T2M,RH2M,CLOUD_AMT,WS10M",
            "community":  "RE",
            "longitude":  lon,
            "latitude":   lat,
            "format":     "JSON",
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        props = data["properties"]["parameter"]

        month_key = f"{month:02d}"

        return {
            "source":      "nasa_power_climatology",
            "ghi":         props.get("ALLSKY_SFC_SW_DWN", {}).get(month_key),
            "temperature": props.get("T2M", {}).get(month_key),
            "humidity":    props.get("RH2M", {}).get(month_key),
            "cloud_cover": props.get("CLOUD_AMT", {}).get(month_key),
            "wind_speed":  props.get("WS10M", {}).get(month_key),
            "is_climatology": True,
        }
    except Exception as e:
        print(f"  NASA POWER error: {e}")
        return None


def _estimate_uv_from_clouds(cloud_pct: float, lat: float) -> float:
    """Estimate UV index from cloud cover and latitude."""
    max_uv = max(1.0, 10.0 - abs(lat) * 0.1)  # Reduce UV with latitude
    reduction = 1.0 - (cloud_pct / 100.0) * 0.7
    return round(max_uv * reduction, 1)


def _estimate_ghi(uv_index: float, cloud_pct: float) -> float:
    """
    Estimate GHI (kWh/m²/day) from UV index and cloud cover.
    Approximate conversion: GHI ≈ UV_index × 0.4 (empirical)
    """
    clear_ghi = uv_index * 0.45
    cloud_factor = 1 - (cloud_pct / 100.0) * 0.65
    return round(clear_ghi * cloud_factor, 2)


def get_weather_data(lat: float, lon: float) -> dict:
    """
    Main weather fetch function. Tries sources in priority order.
    Always returns a valid dict (uses climatology as final fallback).
    """
    # Priority 1: Open-Meteo (best free real-time source)
    data = fetch_openmeteo(lat, lon)
    if data and data.get("temperature") is not None:
        return _normalize(data, lat, lon)

    # Priority 2: OpenWeatherMap (if key available)
    data = fetch_openweather(lat, lon)
    if data and data.get("temperature") is not None:
        return _normalize(data, lat, lon)

    # Priority 3: NASA POWER Climatology (always works)
    data = fetch_nasa_power_current(lat, lon)
    if data and data.get("temperature") is not None:
        return _normalize(data, lat, lon)

    # Ultimate fallback: estimated from latitude
    return _latitude_based_estimate(lat, lon)


def _normalize(data: dict, lat: float, lon: float) -> dict:
    """Ensure all expected fields exist with sensible defaults."""
    # Estimate GHI if missing
    if not data.get("ghi"):
        clouds = data.get("cloud_cover", 30)
        uv = _estimate_uv_from_clouds(clouds, lat)
        data["ghi"] = _estimate_ghi(uv, clouds)

    # Ensure clearness index
    clear_sky_ghi = data.get("clearsky_ghi", data["ghi"] / max(0.1, 1 - (data.get("cloud_cover", 30) / 100) * 0.3))
    data["clearness_index"] = min(1.0, data["ghi"] / max(0.1, clear_sky_ghi))

    data.setdefault("lat",          lat)
    data.setdefault("lon",          lon)
    data.setdefault("temperature",  25.0)
    data.setdefault("humidity",     50.0)
    data.setdefault("cloud_cover",  30.0)
    data.setdefault("wind_speed",   3.0)
    data.setdefault("temp_max",     data["temperature"] + 5)
    data.setdefault("temp_min",     data["temperature"] - 5)

    return data


def _latitude_based_estimate(lat: float, lon: float) -> dict:
    """Estimate solar data purely from geography when all APIs fail."""
    month = datetime.now().month
    abs_lat = abs(lat)

    # GHI estimation based on latitude and season
    base_ghi = 6.0 - abs_lat * 0.05  # Higher GHI near equator
    # Seasonal adjustment for northern hemisphere summer
    season_adj = math.cos(math.radians((month - 6) * 30)) * (abs_lat / 90) * 2
    ghi = max(1.0, base_ghi + season_adj)

    # Temperature from latitude
    base_temp = 30 - abs_lat * 0.5
    temp = max(-5, base_temp)

    return {
        "source":          "latitude_estimate",
        "ghi":             round(ghi, 2),
        "temperature":     round(temp, 1),
        "humidity":        55.0,
        "cloud_cover":     25.0,
        "wind_speed":      3.5,
        "clearness_index": 0.65,
        "lat":             lat,
        "lon":             lon,
        "is_estimate":     True,
    }


def get_weather_description(weather_code: int) -> str:
    """Convert WMO weather code to description."""
    codes = {
        0: "☀️ Clear sky", 1: "🌤 Mainly clear", 2: "⛅ Partly cloudy",
        3: "☁️ Overcast", 45: "🌫 Foggy", 48: "🌫 Icy fog",
        51: "🌦 Light drizzle", 61: "🌧 Light rain", 63: "🌧 Moderate rain",
        71: "🌨 Light snow", 80: "🌦 Rain showers", 95: "⛈ Thunderstorm",
    }
    for k, v in codes.items():
        if weather_code == k:
            return v
    return "🌤 Partly cloudy"
