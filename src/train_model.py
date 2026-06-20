"""
Solar ML Model Training Script
================================
Trains two models on real NASA POWER data:
  1. Solar Generation Predictor  → monthly_generation_kwh
  2. Normalized Yield Predictor  → monthly_kwh_per_kwp

Run:  python src/train_model.py
Outputs: models/solar_prediction.pkl
         models/roi_prediction.pkl
         models/feature_names.json
         models/model_metrics.json
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not installed. Using GradientBoosting as fallback.")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_PATH  = ROOT_DIR / "data" / "solar_dataset.csv"
MODEL_DIR  = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ─── Feature Configuration ────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "lat", "lon",
    "ghi", "clearsky_ghi", "clearness_index",
    "temperature", "temp_max", "temp_min",
    "humidity", "cloud_cover", "wind_speed",
    "month", "season", "climate_zone", "is_southern",
    "peak_sun_hours", "monthly_energy_per_sqm",
    "temp_correction", "performance_ratio",
    "roof_area", "panel_efficiency", "system_size_kwp",
    "monthly_consumption_kwh",
]
CATEGORICAL_FEATURES = ["property_type"]
TARGET_GEN  = "monthly_generation_kwh"
TARGET_NORM = "monthly_kwh_per_kwp"


def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset and return train/test splits."""
    print(f"\n[*]  Loading dataset: {path}")
    if not path.exists():
        print(f"[ERR] Dataset not found at {path}")
        print(f"   Run: python data/build_dataset.py  first")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"    Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Encode categorical
    le = LabelEncoder()
    df["property_type_enc"] = le.fit_transform(df["property_type"].fillna("Residential"))

    # Build feature matrix
    available_num = [f for f in NUMERIC_FEATURES if f in df.columns]
    all_features  = available_num + ["property_type_enc"]

    # Drop rows missing target
    df = df.dropna(subset=[TARGET_GEN, TARGET_NORM])
    df = df[df[TARGET_GEN] > 0]

    X = df[all_features].copy()
    y_gen  = df[TARGET_GEN]
    y_norm = df[TARGET_NORM]

    print(f"    Features: {len(all_features)} | Valid rows: {len(df):,}")
    print(f"    Target range (generation): {y_gen.min():.1f} to {y_gen.max():.1f} kWh")

    # Save feature names
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(all_features, f, indent=2)

    # Save label encoder classes
    label_map = {c: i for i, c in enumerate(le.classes_)}
    with open(MODEL_DIR / "label_encoders.json", "w") as f:
        json.dump({"property_type": label_map}, f, indent=2)

    return X, y_gen, y_norm, df


def build_model_pipeline(model_type: str = "xgboost"):
    """Build sklearn pipeline with imputer + model."""
    if model_type == "xgboost" and HAS_XGB:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   model),
    ])
    return pipeline


def evaluate_model(pipeline, X_test, y_test, name: str) -> dict:
    """Compute and print evaluation metrics."""
    y_pred = pipeline.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    mape   = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

    print(f"\n  [{name}] Test Performance:")
    print(f"       R2    = {r2:.4f}")
    print(f"       RMSE  = {rmse:.2f} kWh")
    print(f"       MAE   = {mae:.2f} kWh")
    print(f"       MAPE  = {mape:.1f}%")

    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}


def get_feature_importance(pipeline, feature_names: list) -> dict:
    """Extract feature importances from the trained model."""
    try:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi = dict(sorted(
                zip(feature_names, importances),
                key=lambda x: x[1], reverse=True
            ))
            return fi
    except Exception:
        pass
    return {}


def train_models():
    """Main training routine."""
    print("=" * 65)
    print("  Solar AI Assistant - Model Training")
    print("=" * 65)

    X, y_gen, y_norm, df = load_and_prepare_data(DATA_PATH)

    # Train/test split (stratified by climate zone if available)
    X_train, X_test, yg_train, yg_test, yn_train, yn_test = train_test_split(
        X, y_gen, y_norm,
        test_size=0.20,
        random_state=42,
    )

    feature_names = list(X.columns)
    metrics = {}

    # ── Model 1: Solar Generation Predictor ─────────────────────────────────
    print("\n" + "─" * 45)
    print("  Model 1: Solar Generation Predictor")
    print("  Target: monthly_generation_kwh")
    print("─" * 45)

    # Try XGBoost first, fallback to Random Forest
    model_type = "xgboost" if HAS_XGB else "random_forest"
    print(f"  Algorithm: {model_type.upper()}")

    gen_pipeline = build_model_pipeline(model_type)

    # Cross-validation
    print(f"\n  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(gen_pipeline, X_train, yg_train,
                                cv=5, scoring="r2", n_jobs=-1)
    print(f"  CV R² scores: {cv_scores.round(4)}")
    print(f"  CV R² mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit
    gen_pipeline.fit(X_train, yg_train)
    m1 = evaluate_model(gen_pipeline, X_test, yg_test, "Generation Model")
    metrics["generation_model"] = m1

    # Feature importance
    fi = get_feature_importance(gen_pipeline, feature_names)
    if fi:
        print(f"\n  Top 10 Feature Importances:")
        for feat, imp in list(fi.items())[:10]:
            bar = "#" * int(imp * 50)
            print(f"    {feat:<30} {bar} {imp:.4f}")

    # Save Model 1
    gen_path = MODEL_DIR / "solar_prediction.pkl"
    joblib.dump(gen_pipeline, gen_path)
    print(f"\n  [OK] Saved -> {gen_path}")

    # ── Model 2: Normalized Yield Model (for ROI) ────────────────────────────
    print("\n" + "─" * 45)
    print("  Model 2: Normalized Yield Predictor")
    print("  Target: monthly_kwh_per_kwp")
    print("─" * 45)

    norm_pipeline = build_model_pipeline(model_type)
    norm_pipeline.fit(X_train, yn_train)
    m2 = evaluate_model(norm_pipeline, X_test, yn_test, "Normalized Yield Model")
    metrics["normalized_yield_model"] = m2

    norm_path = MODEL_DIR / "roi_prediction.pkl"
    joblib.dump(norm_pipeline, norm_path)
    print(f"\n  [OK] Saved -> {norm_path}")

    # ── Baseline: Ridge Regression ────────────────────────────────────────────
    print("\n" + "─" * 45)
    print("  Baseline: Ridge Regression (Generation)")
    print("─" * 45)

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    ridge = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0)
    )
    ridge.fit(X_train, yg_train)
    m3 = evaluate_model(ridge, X_test, yg_test, "Ridge Baseline")
    metrics["ridge_baseline"] = m3

    # ── Save metadata ──────────────────────────────────────────────────────────
    meta = {
        "trained_at":      datetime.now().isoformat(),
        "rows_trained":    len(X_train),
        "rows_tested":     len(X_test),
        "feature_count":   len(feature_names),
        "algorithm":       model_type,
        "has_xgboost":     HAS_XGB,
        "metrics":         metrics,
        "data_source":     "NASA POWER API (https://power.larc.nasa.gov/)",
        "data_years":      "2000-2023",
        "cities_covered":  int(df["city"].nunique()),
        "countries_covered": int(df["country"].nunique()),
    }
    with open(MODEL_DIR / "model_metrics.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  [DONE] Training Complete!")
    print(f"      Generation Model R2: {metrics['generation_model']['r2']:.4f}")
    print(f"      Normalized Model R2: {metrics['normalized_yield_model']['r2']:.4f}")
    print(f"      Ridge Baseline R2:   {metrics['ridge_baseline']['r2']:.4f}")
    print(f"\n  Saved models:")
    print(f"      {MODEL_DIR}/solar_prediction.pkl")
    print(f"      {MODEL_DIR}/roi_prediction.pkl")
    print(f"      {MODEL_DIR}/model_metrics.json")
    print("=" * 65)

    return gen_pipeline, norm_pipeline, metrics


if __name__ == "__main__":
    # Force UTF-8 output on Windows to avoid cp1252 encoding errors
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    train_models()
