"""
Location Service
================
Convert city names → (latitude, longitude) using free geocoders.
No API key required for the primary Nominatim service.

Primary:   Nominatim (OpenStreetMap) — free, no key
Secondary: Hardcoded city database from our 80-city reference
"""

import requests
import time
import json
from pathlib import Path
from typing import Optional

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

# Embedded city database for instant offline lookup
CITY_DB = {
    # India
    "jaipur":           (26.91, 75.79, "Jaipur, Rajasthan, India"),
    "jodhpur":          (26.29, 73.02, "Jodhpur, Rajasthan, India"),
    "jaisalmer":        (26.91, 70.91, "Jaisalmer, Rajasthan, India"),
    "bikaner":          (28.02, 73.31, "Bikaner, Rajasthan, India"),
    "udaipur":          (24.58, 73.68, "Udaipur, Rajasthan, India"),
    "ahmedabad":        (23.02, 72.57, "Ahmedabad, Gujarat, India"),
    "surat":            (21.17, 72.83, "Surat, Gujarat, India"),
    "rajkot":           (22.30, 70.80, "Rajkot, Gujarat, India"),
    "new delhi":        (28.61, 77.21, "New Delhi, Delhi, India"),
    "delhi":            (28.61, 77.21, "Delhi, India"),
    "mumbai":           (19.08, 72.88, "Mumbai, Maharashtra, India"),
    "bangalore":        (12.97, 77.59, "Bangalore, Karnataka, India"),
    "bengaluru":        (12.97, 77.59, "Bengaluru, Karnataka, India"),
    "hyderabad":        (17.39, 78.49, "Hyderabad, Telangana, India"),
    "chennai":          (13.08, 80.27, "Chennai, Tamil Nadu, India"),
    "kolkata":          (22.57, 88.36, "Kolkata, West Bengal, India"),
    "pune":             (18.52, 73.86, "Pune, Maharashtra, India"),
    "nagpur":           (21.15, 79.09, "Nagpur, Maharashtra, India"),
    "bhopal":           (23.26, 77.41, "Bhopal, Madhya Pradesh, India"),
    "indore":           (22.72, 75.86, "Indore, Madhya Pradesh, India"),
    "raipur":           (21.25, 81.63, "Raipur, Chhattisgarh, India"),
    "lucknow":          (26.85, 80.95, "Lucknow, Uttar Pradesh, India"),
    "varanasi":         (25.32, 83.00, "Varanasi, Uttar Pradesh, India"),
    "agra":             (27.18, 78.01, "Agra, Uttar Pradesh, India"),
    "chandigarh":       (30.73, 76.78, "Chandigarh, India"),
    "amritsar":         (31.63, 74.87, "Amritsar, Punjab, India"),
    "coimbatore":       (11.02, 76.96, "Coimbatore, Tamil Nadu, India"),
    "madurai":          (9.92,  78.12, "Madurai, Tamil Nadu, India"),
    "kochi":            (9.93,  76.27, "Kochi, Kerala, India"),
    "thiruvananthapuram":(8.52, 76.94, "Thiruvananthapuram, Kerala, India"),
    "vizag":            (17.69, 83.22, "Visakhapatnam, Andhra Pradesh, India"),
    "visakhapatnam":    (17.69, 83.22, "Visakhapatnam, Andhra Pradesh, India"),
    "patna":            (25.59, 85.14, "Patna, Bihar, India"),
    "ranchi":           (23.34, 85.31, "Ranchi, Jharkhand, India"),
    "bhubaneswar":      (20.30, 85.85, "Bhubaneswar, Odisha, India"),
    "guwahati":         (26.14, 91.74, "Guwahati, Assam, India"),
    "shimla":           (31.10, 77.17, "Shimla, Himachal Pradesh, India"),
    "dehradun":         (30.32, 78.04, "Dehradun, Uttarakhand, India"),
    "leh":              (34.17, 77.58, "Leh, Ladakh, India"),
    "srinagar":         (34.08, 74.80, "Srinagar, J&K, India"),
    "shillong":         (25.57, 91.88, "Shillong, Meghalaya, India"),
    "panaji":           (15.49, 73.83, "Panaji, Goa, India"),
    "goa":              (15.49, 73.83, "Goa, India"),
    # Global
    "dubai":            (25.20, 55.27, "Dubai, UAE"),
    "riyadh":           (24.69, 46.72, "Riyadh, Saudi Arabia"),
    "cairo":            (30.06, 31.25, "Cairo, Egypt"),
    "casablanca":       (33.59, -7.62, "Casablanca, Morocco"),
    "nairobi":          (-1.29, 36.82, "Nairobi, Kenya"),
    "cape town":        (-33.93, 18.42, "Cape Town, South Africa"),
    "phoenix":          (33.45, -112.07, "Phoenix, AZ, USA"),
    "los angeles":      (34.05, -118.24, "Los Angeles, CA, USA"),
    "new york":         (40.71, -74.01, "New York, USA"),
    "toronto":          (43.65, -79.38, "Toronto, Canada"),
    "madrid":           (40.42, -3.70, "Madrid, Spain"),
    "rome":             (41.90, 12.50, "Rome, Italy"),
    "berlin":           (52.52, 13.40, "Berlin, Germany"),
    "london":           (51.51, -0.13, "London, UK"),
    "oslo":             (59.91, 10.75, "Oslo, Norway"),
    "singapore":        (1.35,  103.82, "Singapore"),
    "bangkok":          (13.76, 100.50, "Bangkok, Thailand"),
    "tokyo":            (35.68, 139.69, "Tokyo, Japan"),
    "sydney":           (-33.87, 151.21, "Sydney, Australia"),
    "perth":            (-31.95, 115.86, "Perth, Australia"),
    "karachi":          (24.86, 67.01, "Karachi, Pakistan"),
    "dhaka":            (23.81, 90.41, "Dhaka, Bangladesh"),
    "colombo":          (6.93,  79.84, "Colombo, Sri Lanka"),
    "sao paulo":        (-23.55, -46.63, "São Paulo, Brazil"),
    "buenos aires":     (-34.60, -58.38, "Buenos Aires, Argentina"),
    "mexico city":      (19.43, -99.13, "Mexico City, Mexico"),
}


def geocode_city(city_name: str) -> Optional[dict]:
    """
    Convert city name to coordinates.
    Returns dict with: lat, lon, display_name
    """
    # 1) Try offline database first (instant)
    key = city_name.lower().strip()
    if key in CITY_DB:
        lat, lon, display = CITY_DB[key]
        return {
            "lat": lat,
            "lon": lon,
            "display_name": display,
            "source": "offline_db"
        }

    # 2) Try Nominatim (OpenStreetMap) - free, no key
    if HAS_GEOPY:
        try:
            geolocator = Nominatim(user_agent="solar_ai_assistant_v1")
            location = geolocator.geocode(city_name, timeout=10)
            if location:
                return {
                    "lat": location.latitude,
                    "lon": location.longitude,
                    "display_name": location.address,
                    "source": "nominatim"
                }
        except (GeocoderTimedOut, GeocoderUnavailable):
            pass

    # 3) Fallback: Nominatim REST API directly
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": city_name,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
        }
        headers = {"User-Agent": "SolarAIAssistant/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        if data:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "display_name": data[0]["display_name"],
                "source": "nominatim_rest"
            }
    except Exception:
        pass

    return None


def get_location_info(city_or_lat: str, lon: float = None) -> dict:
    """
    Smart location resolver. Accepts:
    - City name string: "Jaipur"
    - Lat/lon pair: lat=26.91, lon=75.79
    
    Returns standardized location dict.
    """
    # If lat/lon passed directly
    if lon is not None:
        try:
            lat = float(city_or_lat)
            return {
                "lat": lat,
                "lon": float(lon),
                "display_name": f"({lat:.4f}, {lon:.4f})",
                "city": f"({lat:.2f}°, {lon:.2f}°)",
                "source": "coordinates"
            }
        except ValueError:
            pass

    # Geocode city name
    result = geocode_city(str(city_or_lat))
    if result:
        # Extract city name from display string
        parts = result["display_name"].split(",")
        result["city"] = parts[0].strip() if parts else city_or_lat
        return result

    return None


def get_india_cities() -> list[str]:
    """Return sorted list of Indian cities in our database."""
    cities = []
    for key, (lat, lon, display) in CITY_DB.items():
        if "India" in display:
            city_name = display.split(",")[0].strip()
            cities.append(city_name)
    return sorted(set(cities))


def get_global_cities() -> list[str]:
    """Return all cities in database."""
    cities = [display.split(",")[0].strip() for _, (_, _, display) in CITY_DB.items()]
    return sorted(set(cities))


if __name__ == "__main__":
    # Quick test
    test_cities = ["Jaipur", "Mumbai", "London", "Dubai", "Tokyo", "InvalidCity123"]
    print("Testing Location Service")
    print("-" * 40)
    for city in test_cities:
        result = get_location_info(city)
        if result:
            print(f"✔  {city}: ({result['lat']:.2f}, {result['lon']:.2f}) via {result['source']}")
        else:
            print(f"✗  {city}: Not found")
