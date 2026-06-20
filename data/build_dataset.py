"""
Build the Solar AI Assistant training dataset using NASA POWER API.

Data Source: NASA POWER (Prediction Of Worldwide Energy Resources)
URL: https://power.larc.nasa.gov/
License: Public Domain - No API key required
Coverage: Global, 1984-present
Resolution: Monthly averages at point locations

Run this script once to build the training dataset:
    python data/build_dataset.py

Output:
    data/solar_dataset.csv     - Final merged training dataset (~25,000 rows)
    data/nasa_power_raw.csv    - Raw NASA POWER data
    data/city_coordinates.csv  - 80-city reference table
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

# ─── Directory Setup ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
DATA_DIR.mkdir(exist_ok=True)

# ─── City Database: 50 Indian + 30 Global ────────────────────────────────────
CITIES = [
    # ── India (50 cities) ──────────────────────────────────────────────────
    # High solar (Rajasthan / Gujarat belt)
    {"city": "Jaisalmer",     "state": "Rajasthan",        "country": "India", "lat": 26.91, "lon": 70.91, "region": "india"},
    {"city": "Jodhpur",       "state": "Rajasthan",        "country": "India", "lat": 26.29, "lon": 73.02, "region": "india"},
    {"city": "Jaipur",        "state": "Rajasthan",        "country": "India", "lat": 26.91, "lon": 75.79, "region": "india"},
    {"city": "Bikaner",       "state": "Rajasthan",        "country": "India", "lat": 28.02, "lon": 73.31, "region": "india"},
    {"city": "Udaipur",       "state": "Rajasthan",        "country": "India", "lat": 24.58, "lon": 73.68, "region": "india"},
    {"city": "Ahmedabad",     "state": "Gujarat",          "country": "India", "lat": 23.02, "lon": 72.57, "region": "india"},
    {"city": "Gandhinagar",   "state": "Gujarat",          "country": "India", "lat": 23.22, "lon": 72.65, "region": "india"},
    {"city": "Surat",         "state": "Gujarat",          "country": "India", "lat": 21.17, "lon": 72.83, "region": "india"},
    {"city": "Rajkot",        "state": "Gujarat",          "country": "India", "lat": 22.30, "lon": 70.80, "region": "india"},
    {"city": "Bhuj",          "state": "Gujarat",          "country": "India", "lat": 23.25, "lon": 69.67, "region": "india"},
    # Metro cities
    {"city": "New Delhi",     "state": "Delhi",            "country": "India", "lat": 28.61, "lon": 77.21, "region": "india"},
    {"city": "Mumbai",        "state": "Maharashtra",      "country": "India", "lat": 19.08, "lon": 72.88, "region": "india"},
    {"city": "Bangalore",     "state": "Karnataka",        "country": "India", "lat": 12.97, "lon": 77.59, "region": "india"},
    {"city": "Hyderabad",     "state": "Telangana",        "country": "India", "lat": 17.39, "lon": 78.49, "region": "india"},
    {"city": "Chennai",       "state": "Tamil Nadu",       "country": "India", "lat": 13.08, "lon": 80.27, "region": "india"},
    {"city": "Kolkata",       "state": "West Bengal",      "country": "India", "lat": 22.57, "lon": 88.36, "region": "india"},
    {"city": "Pune",          "state": "Maharashtra",      "country": "India", "lat": 18.52, "lon": 73.86, "region": "india"},
    {"city": "Nagpur",        "state": "Maharashtra",      "country": "India", "lat": 21.15, "lon": 79.09, "region": "india"},
    # Deccan / Central India
    {"city": "Bhopal",        "state": "Madhya Pradesh",   "country": "India", "lat": 23.26, "lon": 77.41, "region": "india"},
    {"city": "Indore",        "state": "Madhya Pradesh",   "country": "India", "lat": 22.72, "lon": 75.86, "region": "india"},
    {"city": "Raipur",        "state": "Chhattisgarh",     "country": "India", "lat": 21.25, "lon": 81.63, "region": "india"},
    {"city": "Aurangabad",    "state": "Maharashtra",      "country": "India", "lat": 19.88, "lon": 75.32, "region": "india"},
    {"city": "Nashik",        "state": "Maharashtra",      "country": "India", "lat": 19.99, "lon": 73.78, "region": "india"},
    # South India
    {"city": "Coimbatore",    "state": "Tamil Nadu",       "country": "India", "lat": 11.02, "lon": 76.96, "region": "india"},
    {"city": "Madurai",       "state": "Tamil Nadu",       "country": "India", "lat": 9.92,  "lon": 78.12, "region": "india"},
    {"city": "Tirunelveli",   "state": "Tamil Nadu",       "country": "India", "lat": 8.73,  "lon": 77.70, "region": "india"},
    {"city": "Kochi",         "state": "Kerala",           "country": "India", "lat": 9.93,  "lon": 76.27, "region": "india"},
    {"city": "Thiruvananthapuram", "state": "Kerala",      "country": "India", "lat": 8.52,  "lon": 76.94, "region": "india"},
    {"city": "Vizag",         "state": "Andhra Pradesh",   "country": "India", "lat": 17.69, "lon": 83.22, "region": "india"},
    {"city": "Vijayawada",    "state": "Andhra Pradesh",   "country": "India", "lat": 16.51, "lon": 80.63, "region": "india"},
    # North India
    {"city": "Lucknow",       "state": "Uttar Pradesh",    "country": "India", "lat": 26.85, "lon": 80.95, "region": "india"},
    {"city": "Varanasi",      "state": "Uttar Pradesh",    "country": "India", "lat": 25.32, "lon": 83.00, "region": "india"},
    {"city": "Agra",          "state": "Uttar Pradesh",    "country": "India", "lat": 27.18, "lon": 78.01, "region": "india"},
    {"city": "Kanpur",        "state": "Uttar Pradesh",    "country": "India", "lat": 26.46, "lon": 80.33, "region": "india"},
    {"city": "Chandigarh",    "state": "Punjab/Haryana",   "country": "India", "lat": 30.73, "lon": 76.78, "region": "india"},
    {"city": "Amritsar",      "state": "Punjab",           "country": "India", "lat": 31.63, "lon": 74.87, "region": "india"},
    {"city": "Ludhiana",      "state": "Punjab",           "country": "India", "lat": 30.90, "lon": 75.85, "region": "india"},
    # East India
    {"city": "Patna",         "state": "Bihar",            "country": "India", "lat": 25.59, "lon": 85.14, "region": "india"},
    {"city": "Ranchi",        "state": "Jharkhand",        "country": "India", "lat": 23.34, "lon": 85.31, "region": "india"},
    {"city": "Bhubaneswar",   "state": "Odisha",           "country": "India", "lat": 20.30, "lon": 85.85, "region": "india"},
    {"city": "Guwahati",      "state": "Assam",            "country": "India", "lat": 26.14, "lon": 91.74, "region": "india"},
    # High altitude / low solar
    {"city": "Shimla",        "state": "Himachal Pradesh", "country": "India", "lat": 31.10, "lon": 77.17, "region": "india"},
    {"city": "Dehradun",      "state": "Uttarakhand",      "country": "India", "lat": 30.32, "lon": 78.04, "region": "india"},
    {"city": "Leh",           "state": "Ladakh",           "country": "India", "lat": 34.17, "lon": 77.58, "region": "india"},
    {"city": "Srinagar",      "state": "J&K",              "country": "India", "lat": 34.08, "lon": 74.80, "region": "india"},
    # Northeast
    {"city": "Shillong",      "state": "Meghalaya",        "country": "India", "lat": 25.57, "lon": 91.88, "region": "india"},
    {"city": "Imphal",        "state": "Manipur",          "country": "India", "lat": 24.80, "lon": 93.94, "region": "india"},
    {"city": "Agartala",      "state": "Tripura",          "country": "India", "lat": 23.84, "lon": 91.28, "region": "india"},
    # Islands
    {"city": "Port Blair",    "state": "Andaman & Nicobar","country": "India", "lat": 11.67, "lon": 92.75, "region": "india"},
    {"city": "Kavaratti",     "state": "Lakshadweep",      "country": "India", "lat": 10.56, "lon": 72.64, "region": "india"},
    {"city": "Panaji",        "state": "Goa",              "country": "India", "lat": 15.49, "lon": 73.83, "region": "india"},

    # ── Global (30 cities) ─────────────────────────────────────────────────
    # Middle East / North Africa (excellent solar)
    {"city": "Dubai",         "state": "Dubai",            "country": "UAE",           "lat": 25.20, "lon": 55.27, "region": "global"},
    {"city": "Riyadh",        "state": "Riyadh",           "country": "Saudi Arabia",  "lat": 24.69, "lon": 46.72, "region": "global"},
    {"city": "Cairo",         "state": "Cairo",            "country": "Egypt",         "lat": 30.06, "lon": 31.25, "region": "global"},
    {"city": "Muscat",        "state": "Muscat",           "country": "Oman",          "lat": 23.61, "lon": 58.59, "region": "global"},
    {"city": "Casablanca",    "state": "Casablanca",       "country": "Morocco",       "lat": 33.59, "lon": -7.62, "region": "global"},
    # Sub-Saharan Africa
    {"city": "Nairobi",       "state": "Nairobi",          "country": "Kenya",         "lat": -1.29, "lon": 36.82, "region": "global"},
    {"city": "Cape Town",     "state": "Western Cape",     "country": "South Africa",  "lat": -33.93,"lon": 18.42, "region": "global"},
    {"city": "Lagos",         "state": "Lagos",            "country": "Nigeria",       "lat": 6.52,  "lon": 3.38,  "region": "global"},
    # Americas - Sun Belt
    {"city": "Phoenix",       "state": "Arizona",          "country": "USA",           "lat": 33.45, "lon": -112.07,"region": "global"},
    {"city": "Los Angeles",   "state": "California",       "country": "USA",           "lat": 34.05, "lon": -118.24,"region": "global"},
    {"city": "Mexico City",   "state": "CDMX",             "country": "Mexico",        "lat": 19.43, "lon": -99.13, "region": "global"},
    {"city": "Buenos Aires",  "state": "Buenos Aires",     "country": "Argentina",     "lat": -34.60,"lon": -58.38, "region": "global"},
    {"city": "São Paulo",     "state": "São Paulo",        "country": "Brazil",        "lat": -23.55,"lon": -46.63, "region": "global"},
    # Americas - Northern
    {"city": "Toronto",       "state": "Ontario",          "country": "Canada",        "lat": 43.65, "lon": -79.38, "region": "global"},
    {"city": "New York",      "state": "New York",         "country": "USA",           "lat": 40.71, "lon": -74.01, "region": "global"},
    # Europe (varied)
    {"city": "Madrid",        "state": "Madrid",           "country": "Spain",         "lat": 40.42, "lon": -3.70,  "region": "global"},
    {"city": "Rome",          "state": "Lazio",            "country": "Italy",         "lat": 41.90, "lon": 12.50,  "region": "global"},
    {"city": "Berlin",        "state": "Berlin",           "country": "Germany",       "lat": 52.52, "lon": 13.40,  "region": "global"},
    {"city": "London",        "state": "England",          "country": "UK",            "lat": 51.51, "lon": -0.13,  "region": "global"},
    {"city": "Oslo",          "state": "Oslo",             "country": "Norway",        "lat": 59.91, "lon": 10.75,  "region": "global"},
    # Asia-Pacific
    {"city": "Singapore",     "state": "Singapore",        "country": "Singapore",     "lat": 1.35,  "lon": 103.82, "region": "global"},
    {"city": "Bangkok",       "state": "Bangkok",          "country": "Thailand",      "lat": 13.76, "lon": 100.50, "region": "global"},
    {"city": "Tokyo",         "state": "Tokyo",            "country": "Japan",         "lat": 35.68, "lon": 139.69, "region": "global"},
    {"city": "Shanghai",      "state": "Shanghai",         "country": "China",         "lat": 31.23, "lon": 121.47, "region": "global"},
    {"city": "Beijing",       "state": "Beijing",          "country": "China",         "lat": 39.90, "lon": 116.40, "region": "global"},
    {"city": "Sydney",        "state": "NSW",              "country": "Australia",     "lat": -33.87,"lon": 151.21, "region": "global"},
    {"city": "Perth",         "state": "WA",               "country": "Australia",     "lat": -31.95,"lon": 115.86, "region": "global"},
    # South Asia neighbors
    {"city": "Karachi",       "state": "Sindh",            "country": "Pakistan",      "lat": 24.86, "lon": 67.01,  "region": "global"},
    {"city": "Dhaka",         "state": "Dhaka",            "country": "Bangladesh",    "lat": 23.81, "lon": 90.41,  "region": "global"},
    {"city": "Colombo",       "state": "Western",          "country": "Sri Lanka",     "lat": 6.93,  "lon": 79.84,  "region": "global"},
]

# ─── NASA POWER API Configuration ────────────────────────────────────────────
NASA_API_BASE = "https://power.larc.nasa.gov/api/temporal/monthly/point"
NASA_PARAMS   = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,T2M_MAX,T2M_MIN,RH2M,CLOUD_AMT,WS10M,ALLSKY_KT"
START_YEAR    = 2000
END_YEAR      = 2023  # 24 years of data
COMMUNITY     = "RE"


def fetch_nasa_power(lat: float, lon: float, city: str) -> pd.DataFrame | None:
    """Fetch monthly NASA POWER data for a single city (2000-2023)."""
    params = {
        "parameters": NASA_PARAMS,
        "community": COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": START_YEAR,
        "end": END_YEAR,
        "format": "JSON",
    }
    try:
        resp = requests.get(NASA_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        raw = data["properties"]["parameter"]

        # Build long-format dataframe
        records = []
        for yyyymm, ghi in raw["ALLSKY_SFC_SW_DWN"].items():
            if ghi == -999:
                continue
            year  = int(yyyymm[:4])
            month = int(yyyymm[4:])
            if month > 12:   # Skip month=13 (NASA annual average row)
                continue

            def safe(param, key):
                v = raw.get(param, {}).get(key, np.nan)
                return np.nan if v == -999 else v

            records.append({
                "year":              year,
                "month":             month,
                "ghi":               ghi,                              # kWh/m²/day
                "clearsky_ghi":      safe("CLRSKY_SFC_SW_DWN", yyyymm),
                "temperature":       safe("T2M", yyyymm),             # °C
                "temp_max":          safe("T2M_MAX", yyyymm),
                "temp_min":          safe("T2M_MIN", yyyymm),
                "humidity":          safe("RH2M", yyyymm),            # %
                "cloud_cover":       safe("CLOUD_AMT", yyyymm),       # %
                "wind_speed":        safe("WS10M", yyyymm),           # m/s
                "clearness_index":   safe("ALLSKY_KT", yyyymm),       # 0-1
            })

        df = pd.DataFrame(records)
        df["city"]    = city
        df["lat"]     = lat
        df["lon"]     = lon
        return df

    except Exception as e:
        print(f"  [FAIL] {city}: {e}")
        return None


def days_in_month(month: int, year: int = 2020) -> int:
    """Return number of days in given month. Returns 0 for month 13 (NASA annual avg)."""
    if month < 1 or month > 12:
        return 0
    days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        return 29
    return days[month]


def compute_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain-specific solar energy features."""
    # Peak sun hours = GHI (kWh/m²/day) is already equivalent to peak sun hours
    df["peak_sun_hours"] = df["ghi"]

    # Days in month for each row
    df["month_days"] = df.apply(lambda r: days_in_month(int(r["month"]), int(r["year"])), axis=1)

    # Monthly solar energy per m² (kWh/m²/month)
    df["monthly_energy_per_sqm"] = df["ghi"] * df["month_days"]

    # Temperature correction factor (panels lose ~0.4% efficiency per °C above 25°C NOCT)
    # Standard test condition = 25°C; module temp ≈ ambient + 25 (NOCT correction)
    df["module_temp"] = df["temperature"] + 25  # estimated module temperature
    df["temp_correction"] = 1 - 0.004 * (df["module_temp"] - 25).clip(lower=0)

    # Cloud cover factor (clear sky ratio - clouds reduce output)
    df["cloud_factor"] = 1 - (df["cloud_cover"] / 100) * 0.3  # clouds have partial effect since GHI already accounts

    # Clearness index: ratio of actual to clear-sky irradiance (0=overcast, 1=perfect)
    # Already in data as clearness_index (ALLSKY_KT)
    df["clearness_index"] = df["clearness_index"].fillna(df["ghi"] / (df["clearsky_ghi"] + 1e-6))
    df["clearness_index"] = df["clearness_index"].clip(0, 1)

    # Season encoding (Northern Hemisphere)
    df["season"] = df["month"].map({
        12: 0, 1: 0, 2: 0,   # Winter
        3: 1,  4: 1, 5: 1,   # Spring
        6: 2,  7: 2, 8: 2,   # Summer
        9: 3,  10: 3, 11: 3  # Autumn
    })

    # Hemisphere flag (Southern hemisphere seasons are inverted)
    df["is_southern"] = (df["lat"] < 0).astype(int)
    # Adjust season for southern hemisphere
    df.loc[df["is_southern"] == 1, "season"] = (df.loc[df["is_southern"] == 1, "season"] + 2) % 4

    # Climate zone based on latitude
    df["climate_zone"] = pd.cut(
        df["lat"].abs(),
        bins=[0, 15, 30, 45, 60, 90],
        labels=[0, 1, 2, 3, 4],  # Tropical, Subtropical, Temperate, Subarctic, Arctic
        include_lowest=True
    ).astype(int)

    # System performance ratio (real-world efficiency including inverter, wiring, soiling)
    # Base: 0.80, adjusted for temperature and humidity
    df["performance_ratio"] = (0.80 * df["temp_correction"] * df["cloud_factor"]).clip(0.55, 0.92)

    # Target variable: kWh generated per kWp of installed capacity per month
    # This is the normalized metric used to scale to any system size
    df["monthly_kwh_per_kwp"] = df["peak_sun_hours"] * df["month_days"] * df["performance_ratio"]

    return df


def generate_installation_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each city-month row into multiple rows representing
    different roof sizes and consumption levels.
    This creates realistic diversity in the training set.
    """
    np.random.seed(42)
    scenarios = []

    roof_sizes     = [15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]  # m²
    property_types = ["Residential", "Commercial"]

    for _, row in df.iterrows():
        # Pick 3 random roof sizes per city-month to avoid explosion
        selected_roofs = np.random.choice(roof_sizes, size=3, replace=False)
        for roof_area in selected_roofs:
            prop_type = "Residential" if roof_area <= 100 else "Commercial"

            # Installed capacity: ~15% of roof area in kWp
            # (accounting for spacing, orientation, panel efficiency ~20%)
            panel_coverage = 0.75  # 75% of roof covered by panels
            panel_efficiency = np.random.uniform(0.18, 0.22)  # 18-22%
            system_size_kwp = roof_area * panel_coverage * panel_efficiency

            # Monthly electricity consumption (kWh) — realistic range
            if prop_type == "Residential":
                monthly_consumption = np.random.uniform(100, 600)
            else:
                monthly_consumption = np.random.uniform(500, 5000)

            # Electricity rate (INR/kWh) — varies by state/type
            elec_rate = np.random.uniform(5.0, 12.0)

            # Actual monthly solar generation (kWh)
            monthly_generation_kwh = row["monthly_kwh_per_kwp"] * system_size_kwp

            # Add small realistic noise (5% variation from soiling, shade, etc.)
            noise = np.random.normal(1.0, 0.05)
            monthly_generation_kwh *= max(0.85, noise)

            scenarios.append({
                **row.to_dict(),
                "roof_area":              roof_area,
                "panel_efficiency":       panel_efficiency,
                "system_size_kwp":        round(system_size_kwp, 2),
                "monthly_consumption_kwh": round(monthly_consumption, 1),
                "electricity_rate":       round(elec_rate, 2),
                "property_type":          prop_type,
                "monthly_generation_kwh": round(monthly_generation_kwh, 2),
            })

    return pd.DataFrame(scenarios)


def build_dataset():
    """Main function to build the complete training dataset."""
    print("=" * 65)
    print("  Solar AI Assistant - Dataset Builder")
    print("  Data Source: NASA POWER API (Public Domain, No Key Required)")
    print("=" * 65)

    # -- Save city coordinates
    city_df = pd.DataFrame(CITIES)
    coords_path = DATA_DIR / "city_coordinates.csv"
    city_df.to_csv(coords_path, index=False)
    print(f"\n[OK] Saved {len(CITIES)} city coordinates -> {coords_path}")

    # -- Fetch NASA POWER data for all cities
    raw_path = DATA_DIR / "nasa_power_raw.csv"

    # Resume from existing file if interrupted
    if raw_path.exists():
        print(f"\n[*]  Found existing raw data at {raw_path}, loading...")
        all_data = pd.read_csv(raw_path)
        fetched_cities = set(all_data["city"].unique())
        remaining = [c for c in CITIES if c["city"] not in fetched_cities]
        print(f"    Already fetched: {len(fetched_cities)} cities | Remaining: {len(remaining)}")
    else:
        all_data = pd.DataFrame()
        remaining = CITIES

    total = len(remaining)
    if total > 0:
        print(f"\n[*]  Fetching NASA POWER data for {total} cities...\n")

    for i, city_info in enumerate(remaining, 1):
        city    = city_info["city"]
        lat     = city_info["lat"]
        lon     = city_info["lon"]
        country = city_info["country"]

        print(f"  [{i:02d}/{total:02d}] {city}, {country} ({lat:.2f}, {lon:.2f}) ...", end=" ", flush=True)
        df_city = fetch_nasa_power(lat, lon, city)

        if df_city is not None and not df_city.empty:
            df_city["state"]   = city_info["state"]
            df_city["country"] = city_info["country"]
            df_city["region"]  = city_info["region"]
            all_data = pd.concat([all_data, df_city], ignore_index=True)
            print(f"OK  {len(df_city)} records")
        else:
            print("SKIP")

        # Save checkpoint every 5 cities
        if i % 5 == 0:
            all_data.to_csv(raw_path, index=False)
            print(f"    [SAVED] Checkpoint: {len(all_data)} records")

        # Rate limiting
        time.sleep(2.0)

    # Final save of raw data
    all_data.to_csv(raw_path, index=False)
    print(f"\n[OK] Raw data saved -> {raw_path}")
    print(f"    Total records: {len(all_data):,}")

    # -- Feature Engineering
    print("\n[*]  Engineering solar features...")
    all_data = all_data.dropna(subset=["ghi", "temperature"])
    all_data = all_data[all_data["ghi"] > 0]
    all_data = all_data[all_data["month"] <= 12]  # Drop annual-average rows (month=13)
    all_data = all_data.reset_index(drop=True)
    all_data = compute_solar_features(all_data)
    print(f"    After cleaning: {len(all_data):,} records")

    # -- Expand with installation scenarios
    print("\n[*]  Generating installation scenarios...")
    dataset = generate_installation_scenarios(all_data)
    print(f"    Total training rows: {len(dataset):,}")

    # -- Save final dataset
    cols_to_keep = [
        "city", "state", "country", "region", "lat", "lon",
        "year", "month", "season", "climate_zone", "is_southern",
        "ghi", "clearsky_ghi", "clearness_index",
        "temperature", "temp_max", "temp_min",
        "humidity", "cloud_cover", "wind_speed",
        "peak_sun_hours", "month_days", "monthly_energy_per_sqm",
        "temp_correction", "performance_ratio",
        "roof_area", "panel_efficiency", "system_size_kwp",
        "monthly_consumption_kwh", "electricity_rate", "property_type",
        "monthly_generation_kwh",
        "monthly_kwh_per_kwp",
    ]
    cols_to_keep = [c for c in cols_to_keep if c in dataset.columns]
    dataset = dataset[cols_to_keep]

    out_path = DATA_DIR / "solar_dataset.csv"
    dataset.to_csv(out_path, index=False)

    print(f"\n{'='*65}")
    print(f"[DONE] Dataset built successfully!")
    print(f"    Records:   {len(dataset):,}")
    print(f"    Cities:    {dataset['city'].nunique()}")
    print(f"    Countries: {dataset['country'].nunique()}")
    print(f"    Years:     {int(dataset['year'].min())}-{int(dataset['year'].max())}")
    print(f"    Output:    {out_path}")
    print(f"\n  GHI Range (kWh/m2/day):")
    print(f"    Min: {dataset['ghi'].min():.2f}  |  Max: {dataset['ghi'].max():.2f}  |  Mean: {dataset['ghi'].mean():.2f}")
    print(f"\n  Monthly Generation Range (kWh):")
    print(f"    Min: {dataset['monthly_generation_kwh'].min():.1f}  |  Max: {dataset['monthly_generation_kwh'].max():.1f}")
    print(f"{'='*65}")

    return dataset


if __name__ == "__main__":
    build_dataset()
