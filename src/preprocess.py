"""
Preprocessing & Feature Engineering
=====================================
Transforms raw user inputs + weather data into ML-ready feature vectors.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional


PROPERTY_TYPE_MAP = {"Residential": 0, "Commercial": 1}


def days_in_month(month: int) -> int:
    """Return days in month (1-12). Returns 30 for invalid input."""
    if not isinstance(month, int) or month < 1 or month > 12:
        return 30
    days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return days[month]


def get_season(month: int, lat: float) -> int:
    """Return season (0=Winter, 1=Spring, 2=Summer, 3=Autumn) accounting for hemisphere."""
    season_north = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                    6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    s = season_north[month]
    if lat < 0:  # Southern hemisphere - invert summer/winter
        s = (s + 2) % 4
    return s


def get_climate_zone(lat: float) -> int:
    """Classify climate zone by latitude."""
    abs_lat = abs(lat)
    if abs_lat <= 15:  return 0   # Tropical
    elif abs_lat <= 30: return 1  # Subtropical
    elif abs_lat <= 45: return 2  # Temperate
    elif abs_lat <= 60: return 3  # Subarctic
    else:               return 4  # Arctic


def build_feature_vector(
    lat: float,
    lon: float,
    ghi: float,
    temperature: float,
    cloud_cover: float,
    humidity: float,
    wind_speed: float,
    clearness_index: float,
    roof_area: float,
    monthly_consumption_kwh: float,
    property_type: str = "Residential",
    month: Optional[int] = None,
    panel_efficiency: float = 0.20,
) -> dict:
    """
    Build the full feature vector for ML prediction.
    Matches exactly the features used during training.
    """
    if month is None:
        month = datetime.now().month

    # ── Computed fields ──────────────────────────────────────────────────────
    season       = get_season(month, lat)
    climate_zone = get_climate_zone(lat)
    is_southern  = 1 if lat < 0 else 0
    month_days   = days_in_month(month)

    # Clear-sky GHI estimate
    clearsky_ghi = ghi / max(0.01, clearness_index) if clearness_index > 0 else ghi * 1.2

    # Monthly solar energy per m²
    monthly_energy_per_sqm = ghi * month_days

    # Module temperature (ambient + NOCT heating)
    module_temp = temperature + 25

    # Temperature correction factor (efficiency drops 0.4%/°C above 25°C)
    temp_correction = max(0.70, 1 - 0.004 * max(0, module_temp - 25))

    # Cloud factor
    cloud_factor = 1 - (cloud_cover / 100) * 0.3

    # Performance ratio
    performance_ratio = min(0.92, max(0.55, 0.80 * temp_correction * cloud_factor))

    # Panel coverage and system size
    panel_coverage = 0.75  # 75% of roof covered by panels
    system_size_kwp = roof_area * panel_coverage * panel_efficiency

    # Property type encoding
    property_type_enc = PROPERTY_TYPE_MAP.get(property_type, 0)

    return {
        # Location
        "lat":                    lat,
        "lon":                    lon,
        # Solar irradiance
        "ghi":                    ghi,
        "clearsky_ghi":           clearsky_ghi,
        "clearness_index":        clearness_index,
        # Meteorology
        "temperature":            temperature,
        "temp_max":               temperature + 5,
        "temp_min":               temperature - 5,
        "humidity":               humidity,
        "cloud_cover":            cloud_cover,
        "wind_speed":             wind_speed,
        # Time features
        "month":                  month,
        "season":                 season,
        "climate_zone":           climate_zone,
        "is_southern":            is_southern,
        # Derived solar features
        "peak_sun_hours":         ghi,
        "monthly_energy_per_sqm": monthly_energy_per_sqm,
        "temp_correction":        temp_correction,
        "performance_ratio":      performance_ratio,
        # Installation
        "roof_area":              roof_area,
        "panel_efficiency":       panel_efficiency,
        "system_size_kwp":        system_size_kwp,
        "monthly_consumption_kwh": monthly_consumption_kwh,
        # Categorical (encoded)
        "property_type_enc":      property_type_enc,
    }


def feature_vector_to_df(feature_dict: dict, feature_names: list) -> pd.DataFrame:
    """Convert feature dict to properly ordered DataFrame for model input."""
    row = {k: feature_dict.get(k, np.nan) for k in feature_names}
    return pd.DataFrame([row])


def estimate_monthly_generation_simple(
    ghi: float,
    roof_area: float,
    performance_ratio: float = 0.80,
    panel_efficiency: float = 0.20,
    month: int = None,
) -> float:
    """
    Physics-based estimate (when model not available).
    Generation (kWh) = GHI × days × system_size × performance_ratio
    """
    if month is None:
        month = datetime.now().month
    month_days    = days_in_month(month)
    panel_coverage = 0.75
    system_size_kwp = roof_area * panel_coverage * panel_efficiency
    return round(ghi * month_days * system_size_kwp * performance_ratio, 1)


def get_system_size(roof_area: float, panel_efficiency: float = 0.20) -> float:
    """Estimate installed system size in kWp."""
    panel_coverage = 0.75
    return round(roof_area * panel_coverage * panel_efficiency, 2)


def calculate_solar_suitability_score(
    ghi: float,
    cloud_cover: float,
    roof_area: float,
    budget: float,
    monthly_consumption_kwh: float,
    lat: float,
) -> tuple[float, dict]:
    """
    Calculate 0-100 solar suitability score.

    Weights:
      - Solar irradiance (GHI):  35%
      - Cloud cover (inverted):  20%
      - Roof size:               20%
      - Budget vs cost:          15%
      - Location efficiency:     10%
    """
    # 1. GHI Score (0-1): normalize relative to world best (~8 kWh/m²/day)
    ghi_score = min(1.0, ghi / 8.0)

    # 2. Cloud Score (inverted): less cloud = better
    cloud_score = max(0.0, 1.0 - cloud_cover / 100.0)

    # 3. Roof Score: minimum viable = 10 m², excellent = 200+ m²
    roof_score = min(1.0, max(0.0, (roof_area - 10) / 190.0))

    # 4. Budget Score: vs estimated installation cost
    system_size = get_system_size(roof_area)
    est_cost_inr = system_size * 60_000  # ~₹60,000 per kWp (India avg 2024)
    budget_score = min(1.0, budget / max(1, est_cost_inr))

    # 5. Location Efficiency: tropical belt (lat 10-30) is optimal
    abs_lat = abs(lat)
    if abs_lat <= 10:
        loc_score = 0.80  # Very good but sometimes too cloudy
    elif abs_lat <= 30:
        loc_score = 1.00  # Optimal solar belt
    elif abs_lat <= 45:
        loc_score = 0.70
    elif abs_lat <= 60:
        loc_score = 0.45
    else:
        loc_score = 0.25

    # Weighted total
    total = (
        ghi_score    * 0.35 +
        cloud_score  * 0.20 +
        roof_score   * 0.20 +
        budget_score * 0.15 +
        loc_score    * 0.10
    ) * 100

    score = round(min(100.0, max(0.0, total)), 1)

    breakdown = {
        "ghi_score":    round(ghi_score * 100, 1),
        "cloud_score":  round(cloud_score * 100, 1),
        "roof_score":   round(roof_score * 100, 1),
        "budget_score": round(budget_score * 100, 1),
        "loc_score":    round(loc_score * 100, 1),
        "total":        score,
    }

    return score, breakdown


def classify_suitability(score: float) -> tuple[str, str]:
    """Return suitability label and color."""
    if score >= 71:
        return "Excellent", "#00C853"
    elif score >= 41:
        return "Average", "#FFB300"
    else:
        return "Poor", "#FF3D00"
