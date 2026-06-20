"""
ML Predictor
=============
Load trained models and run inference for:
  1. Solar monthly generation (kWh)
  2. Normalized yield (kWh/kWp/month)
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

ROOT_DIR   = Path(__file__).parent.parent
MODEL_DIR  = ROOT_DIR / "models"

# Cached models (loaded once)
_gen_model  = None
_norm_model = None
_feature_names = None
_metrics       = None


def load_models():
    """Load models from disk (cached in module globals)."""
    global _gen_model, _norm_model, _feature_names, _metrics

    if _gen_model is not None:
        return True  # Already loaded

    gen_path  = MODEL_DIR / "solar_prediction.pkl"
    norm_path = MODEL_DIR / "roi_prediction.pkl"
    feat_path = MODEL_DIR / "feature_names.json"
    metr_path = MODEL_DIR / "model_metrics.json"

    if not gen_path.exists():
        return False

    try:
        _gen_model  = joblib.load(gen_path)
        _norm_model = joblib.load(norm_path) if norm_path.exists() else _gen_model
        if feat_path.exists():
            with open(feat_path) as f:
                _feature_names = json.load(f)
        if metr_path.exists():
            with open(metr_path) as f:
                _metrics = json.load(f)
        return True
    except Exception as e:
        print(f"Model load error: {e}")
        return False


def get_feature_names() -> list:
    if _feature_names is None:
        load_models()
    return _feature_names or []


def get_model_metrics() -> dict:
    if _metrics is None:
        load_models()
    return _metrics or {}


def predict_generation(feature_dict: dict) -> float:
    """
    Predict monthly solar generation (kWh).
    Returns physics-based estimate if model not loaded.
    """
    if not load_models():
        # Physics fallback
        from src.preprocess import estimate_monthly_generation_simple
        return estimate_monthly_generation_simple(
            ghi        = feature_dict.get("ghi", 5.0),
            roof_area  = feature_dict.get("roof_area", 50),
            month      = feature_dict.get("month"),
        )

    feature_names = get_feature_names()
    row = {k: feature_dict.get(k, np.nan) for k in feature_names}
    X   = pd.DataFrame([row])

    pred = _gen_model.predict(X)[0]
    return max(0.0, round(float(pred), 1))


def predict_normalized_yield(feature_dict: dict) -> float:
    """
    Predict monthly kWh per kWp installed.
    Used to scale generation to any system size.
    """
    if not load_models() or _norm_model is None:
        # Physics fallback: GHI × days × performance_ratio
        from src.preprocess import days_in_month
        month = feature_dict.get("month", 6)
        ghi   = feature_dict.get("ghi", 5.0)
        pr    = feature_dict.get("performance_ratio", 0.80)
        return round(ghi * days_in_month(month) * pr, 1)

    feature_names = get_feature_names()
    row = {k: feature_dict.get(k, np.nan) for k in feature_names}
    X   = pd.DataFrame([row])

    pred = _norm_model.predict(X)[0]
    return max(0.0, round(float(pred), 1))


def predict_all_months(feature_dict: dict) -> list[dict]:
    """
    Predict solar generation for all 12 months.
    Returns list of {month, generation_kwh, kwh_per_kwp}.
    """
    from src.preprocess import get_season, days_in_month

    results = []
    lat = feature_dict.get("lat", 20)

    for month in range(1, 13):
        # Adjust GHI seasonally (approximate sinusoidal model)
        base_ghi = feature_dict.get("ghi", 5.0)
        # Peak in April-May for India, trough in December-January
        phase = 4.5  # Month of peak sun for India (April-May)
        if lat < 0:
            phase = 10.5  # Southern hemisphere peaks in Oct-Nov

        season_factor = 0.7 + 0.3 * np.cos(
            np.radians((month - phase) * 30)
        )
        month_ghi = max(1.0, base_ghi * season_factor)

        # Adjust temperature seasonally
        base_temp = feature_dict.get("temperature", 25)
        temp_season = base_temp + 5 * np.cos(np.radians((month - phase) * 30))

        # Cloud cover seasonal
        base_clouds = feature_dict.get("cloud_cover", 30)
        # Monsoon months (June-Sept for India)
        if 6 <= month <= 9 and 0 < lat < 35:
            cloud_season = min(85, base_clouds * 2.0)
        else:
            cloud_season = base_clouds * max(0.5, 1 - 0.2 * season_factor)

        month_features = {
            **feature_dict,
            "month":       month,
            "season":      get_season(month, lat),
            "ghi":         month_ghi,
            "temperature": temp_season,
            "cloud_cover": cloud_season,
        }

        gen  = predict_generation(month_features)
        norm = predict_normalized_yield(month_features)

        results.append({
            "month":          month,
            "month_name":     ["", "Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"][month],
            "generation_kwh": gen,
            "kwh_per_kwp":    norm,
            "ghi":            round(month_ghi, 2),
            "cloud_cover":    round(cloud_season, 1),
        })

    return results


def is_model_loaded() -> bool:
    """Check if models are available."""
    return load_models()
