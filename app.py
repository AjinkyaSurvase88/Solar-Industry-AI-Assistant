"""
Solar Industry AI Assistant
============================
Main Streamlit Application

Author: Solar AI Team
Data Source: NASA POWER API (https://power.larc.nasa.gov/)
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# ─── Path Setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ─── Page Config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Solar Industry AI Assistant",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://power.larc.nasa.gov/",
        "Report a bug": None,
        "About":        "Solar AI Assistant — Powered by NASA POWER data & ML",
    }
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: radial-gradient(circle at top left, #0e1526 0%, #07090f 100%) !important; color: #E8EAF0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(13, 21, 37, 0.4) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}
[data-testid="stSidebar"] .stMarkdown h2 { color: #F4A826; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(26, 58, 92, 0.3) 0%, rgba(13, 33, 55, 0.6) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    border-radius: 16px;
    padding: 3rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "☀️";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.1;
    animation: float 4s ease-in-out infinite;
    filter: drop-shadow(0 0 20px rgba(244,168,38,0.5));
}
@keyframes float { 0%,100%{transform:translateY(-50%) scale(1)} 50%{transform:translateY(-55%) scale(1.05)} }
.hero-title { font-size: 2.5rem; font-weight: 700; color: #F4A826; margin: 0 0 0.5rem; letter-spacing: -0.5px; }
.hero-sub   { font-size: 1.1rem; color: #9CA3AF; margin: 0; font-weight: 300; }

/* ── Metric Cards ── */
.metric-card {
    background: rgba(17, 24, 39, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover { 
    transform: translateY(-5px); 
    border-color: rgba(244, 168, 38, 0.3); 
    box-shadow: 0 8px 25px rgba(244, 168, 38, 0.15);
    background: rgba(17, 24, 39, 0.6);
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #F4A826; line-height: 1.2; }
.metric-label { font-size: 0.85rem; color: #9CA3AF; margin-top: 0.4rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-sub   { font-size: 0.75rem; color: #6B7280; margin-top: 0.2rem; }

/* ── Section Headings ── */
.section-header {
    font-size: 1.4rem; font-weight: 600; color: #E8EAF0;
    border-left: 4px solid #F4A826; padding-left: 0.75rem;
    margin: 2rem 0 1rem;
    letter-spacing: -0.2px;
}

/* ── Status Badges ── */
.badge {
    display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 999px; font-size: 0.8rem; font-weight: 600;
}
.badge-excellent { background: rgba(0,200,83,0.15);  color: #00C853; border: 1px solid rgba(0,200,83,0.3); }
.badge-average   { background: rgba(255,179,0,0.15); color: #FFB300; border: 1px solid rgba(255,179,0,0.3); }
.badge-poor      { background: rgba(255,61,0,0.15);  color: #FF3D00; border: 1px solid rgba(255,61,0,0.3); }

/* ── Chat UI ── */
.chat-msg-user {
    background: linear-gradient(135deg, rgba(31,58,92,0.6), rgba(26,37,64,0.6));
    border: 1px solid rgba(45,90,142,0.4); border-radius: 12px 12px 4px 12px;
    padding: 1rem 1.25rem; margin: 0.5rem 0; color: #E8EAF0;
    backdrop-filter: blur(8px);
}
.chat-msg-bot {
    background: linear-gradient(135deg, rgba(28,42,24,0.6), rgba(26,37,48,0.6));
    border: 1px solid rgba(45,90,62,0.4); border-radius: 12px 12px 12px 4px;
    padding: 1rem 1.25rem; margin: 0.5rem 0; color: #E8EAF0;
    backdrop-filter: blur(8px);
}

/* ── Info Box ── */
.info-box {
    background: rgba(66,165,245,0.05); border: 1px solid rgba(66,165,245,0.2);
    border-radius: 12px; padding: 1.25rem 1.5rem; margin: 0.75rem 0;
    backdrop-filter: blur(8px);
}
.success-box {
    background: rgba(0,200,83,0.05); border: 1px solid rgba(0,200,83,0.2);
    border-radius: 12px; padding: 1.25rem 1.5rem; margin: 0.75rem 0;
    backdrop-filter: blur(8px);
}
.warning-box {
    background: rgba(255,179,0,0.05); border: 1px solid rgba(255,179,0,0.2);
    border-radius: 12px; padding: 1.25rem 1.5rem; margin: 0.75rem 0;
    backdrop-filter: blur(8px);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #F4A826 0%, #FF6B35 100%) !important;
    color: #0A0E1A !important; 
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    border: 1px solid rgba(255,255,255,0.2) !important; 
    border-radius: 8px !important;
    padding: 0.7rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(244,168,38,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(244,168,38,0.4) !important;
    filter: brightness(1.1);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    color: #9CA3AF; font-weight: 500; padding: 0.75rem 1.5rem;
    transition: color 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #E8EAF0; }
.stTabs [aria-selected="true"] {
    color: #F4A826 !important; border-bottom-color: #F4A826 !important;
}

/* ── Inputs ── */
.stSelectbox > div > div, .stNumberInput > div > div, .stTextInput > div > div {
    background: rgba(17, 24, 39, 0.5) !important; 
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.stSelectbox > div > div:hover, .stNumberInput > div > div:hover, .stTextInput > div > div:hover {
    border-color: rgba(244, 168, 38, 0.5) !important;
}
.stSelectbox > div > div:focus-within, .stNumberInput > div > div:focus-within, .stTextInput > div > div:focus-within {
    border-color: #F4A826 !important;
    box-shadow: 0 0 0 1px #F4A826 !important;
}

/* ── Footer ── */
.footer {
    text-align: center; color: #4B5563; font-size: 0.85rem;
    padding: 2rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 3rem;
}

/* ── Mobile Responsive Adjustments ── */
@media (max-width: 768px) {
    .hero-title { font-size: 1.8rem; }
    .hero-sub { font-size: 0.95rem; }
    .hero-banner { padding: 2rem 1.5rem; }
    .metric-value { font-size: 1.8rem; }
    .metric-card { padding: 1.25rem; margin-bottom: 0.75rem; }
    .stButton > button { width: 100% !important; }
    .section-header { font-size: 1.2rem; }
    .info-box, .success-box, .warning-box { padding: 1rem 1.25rem; }
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Import Services ──────────────────────────────────────────────────────────
try:
    from src.location_service import get_location_info, get_india_cities, get_global_cities
    from src.weather_service  import get_weather_data, get_weather_description
    from src.preprocess       import (build_feature_vector, calculate_solar_suitability_score,
                                      classify_suitability, get_system_size)
    from src.predictor        import predict_generation, predict_all_months, is_model_loaded, get_model_metrics
    from src.roi_calculator   import full_financial_analysis
    from src.visualizer       import (gauge_suitability_score, chart_monthly_generation,
                                      chart_bill_comparison, chart_energy_sources,
                                      chart_location_map, chart_cost_breakdown,
                                      chart_savings_forecast, chart_roi_kpi, chart_score_radar)
    from src.ai_assistant     import SolarAIAssistant, build_context, get_suggested_questions
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    st.error(f"⚠ Import error: {e}. Please install: `pip install -r requirements.txt`")


# ─── Session State Init ───────────────────────────────────────────────────────
def init_session():
    defaults = {
        "analysis_done":   False,
        "financial_done":  False,
        "location":        None,
        "weather":         None,
        "features":        None,
        "generation":      None,
        "monthly_data":    None,
        "financial":       None,
        "suitability":     None,
        "score_breakdown": None,
        "chat_history":    [],
        "assistant":       None,
        "city_input":      "Jaipur",
        "active_tab":      0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.assistant is None and IMPORTS_OK:
        st.session_state.assistant = SolarAIAssistant()

init_session()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:3rem">☀️</div>
        <div style="font-size:1.2rem; font-weight:700; color:#F4A826">Solar AI</div>
        <div style="font-size:0.75rem; color:#6B7280">Industry Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.divider()

    # Model Status
    if IMPORTS_OK:
        model_loaded = is_model_loaded()
        if model_loaded:
            metrics = get_model_metrics()
            r2 = metrics.get("metrics", {}).get("generation_model", {}).get("r2", 0)
            st.markdown(f"""
            <div class="success-box">
                <b style="color:#00C853">✅ ML Model Active</b><br>
                <small style="color:#9CA3AF">R² = {r2:.3f} | NASA POWER Data</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <b style="color:#FFB300">⚠ Model Not Trained</b><br>
                <small style="color:#9CA3AF">Using physics-based estimates.<br>
                Run: <code>python src/train_model.py</code></small>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Quick Stats
    st.markdown("**📡 Data Source**")
    st.markdown("""
    <div style="font-size:0.8rem; color:#9CA3AF; line-height:1.7">
    🛸 NASA POWER API<br>
    📅 2000–2023 (24 years)<br>
    🏙 80 cities worldwide<br>
    🔬 Public Domain data<br>
    🔑 No API key needed
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Analysis Status
    if st.session_state.analysis_done:
        loc = st.session_state.location
        city_name = loc.get("city", "Unknown") if loc else "Unknown"
        score = st.session_state.suitability
        label, color = classify_suitability(score) if score else ("—", "#9CA3AF")
        st.markdown(f"""
        <div style="font-size:0.85rem; color:#9CA3AF">
        <b style="color:#F4A826">Last Analysis:</b><br>
        📍 {city_name}<br>
        🎯 Score: <b style="color:{color}">{score:.0f}/100 ({label})</b>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
def show_home():
    # Stats Row
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("80+", "Cities Worldwide", "India-first coverage"),
        ("24 Years", "Historical Data", "NASA POWER 2000–2023"),
        ("28,000+", "Training Records", "Real-world solar data"),
        ("90%+", "Model Accuracy", "R² on test set"),
    ]
    for col, (val, label, sub) in zip([col1,col2,col3,col4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🔬 How It Works</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <ol style="color:#E8EAF0; line-height:2">
            <li><b>Enter your location</b> — City name or coordinates</li>
            <li><b>Fetch live weather</b> — Open-Meteo real-time API</li>
            <li><b>Run ML models</b> — Trained on 24 years of NASA POWER data</li>
            <li><b>Get your score</b> — 0–100 solar suitability rating</li>
            <li><b>Financial analysis</b> — ROI, payback, 25-year forecast</li>
            <li><b>Chat with AI</b> — Personalized solar consultant</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">🌍 India Solar Advantage</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="success-box">
        <div style="color:#E8EAF0; line-height:2">
            🌞 <b>India ranks 5th globally</b> in solar capacity<br>
            ☀️ Avg. 5.5 kWh/m²/day across most of India<br>
            🏛 <b>PM Surya Ghar scheme</b>: 30% subsidy ≤3 kWp<br>
            📉 Solar costs down <b>90%</b> since 2010<br>
            🎯 Target: <b>500 GW</b> by 2030 (India)<br>
            💰 Typical payback: <b>5–8 years</b> in India<br>
            🌱 Each kWp offsets <b>~1.2 tonnes CO₂/year</b>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # India Solar Map preview
    st.markdown('<div class="section-header">🗺 India Solar Potential by Region</div>', unsafe_allow_html=True)
    india_data = [
        {"Region": "Rajasthan (Jaisalmer)", "GHI (kWh/m²/day)": 6.5, "Score": 95, "Typical Payback": "4–5 yrs"},
        {"Region": "Gujarat (Bhuj)",         "GHI (kWh/m²/day)": 6.2, "Score": 92, "Typical Payback": "5 yrs"},
        {"Region": "Maharashtra (Nagpur)",   "GHI (kWh/m²/day)": 5.8, "Score": 85, "Typical Payback": "5–6 yrs"},
        {"Region": "Karnataka (Bangalore)",  "GHI (kWh/m²/day)": 5.5, "Score": 80, "Typical Payback": "6 yrs"},
        {"Region": "Delhi",                  "GHI (kWh/m²/day)": 5.3, "Score": 78, "Typical Payback": "6–7 yrs"},
        {"Region": "Tamil Nadu (Chennai)",   "GHI (kWh/m²/day)": 5.4, "Score": 76, "Typical Payback": "6–7 yrs"},
        {"Region": "West Bengal (Kolkata)",  "GHI (kWh/m²/day)": 4.6, "Score": 65, "Typical Payback": "7–8 yrs"},
        {"Region": "Northeast (Shillong)",   "GHI (kWh/m²/day)": 3.8, "Score": 48, "Typical Payback": "9–10 yrs"},
    ]
    df_india = pd.DataFrame(india_data)
    st.dataframe(
        df_india,
        use_container_width=True,
        hide_index=True,
        column_config={
            "GHI (kWh/m²/day)": st.column_config.ProgressColumn(
                min_value=0, max_value=8, format="%.1f"
            ),
            "Score": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%d/100"
            ),
        }
    )

    st.markdown("""
    <div class="footer">
        ☀️ Solar AI Assistant | Data: NASA POWER (power.larc.nasa.gov) | 
        Built with Streamlit, XGBoost & Plotly | India-first, Global vision
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SOLAR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def show_analysis():
    st.markdown('<div class="section-header">📊 Solar Potential Analysis</div>', unsafe_allow_html=True)

    # ── Input Form ────────────────────────────────────────────────────────────
    with st.form("analysis_form", clear_on_submit=False):
        st.markdown("#### 📝 Enter Your Details")

        col1, col2 = st.columns(2)
        with col1:
            location_mode = st.selectbox(
                "Location Input Mode",
                ["📍 City Name", "🗺 Latitude & Longitude"],
                key="loc_mode"
            )

            if "City" in location_mode:
                india_cities = get_india_cities() if IMPORTS_OK else ["Jaipur", "Mumbai", "Delhi"]
                city_input = st.selectbox(
                    "Select City (or type any city name below)",
                    options=[""] + india_cities,
                    key="city_select"
                )
                city_custom = st.text_input("Or type any city name", placeholder="e.g. Coimbatore, Dubai, Phoenix")
                final_city = city_custom.strip() if city_custom.strip() else city_input
            else:
                final_city = None
                lat_input = st.number_input("Latitude", value=26.91, min_value=-90.0, max_value=90.0, step=0.01)
                lon_input = st.number_input("Longitude", value=75.79, min_value=-180.0, max_value=180.0, step=0.01)

            property_type = st.selectbox("Property Type", ["Residential", "Commercial"])

        with col2:
            monthly_bill = st.number_input(
                "Monthly Electricity Bill (₹)", min_value=100, max_value=500000,
                value=3000, step=100,
                help="Your average monthly electricity bill in INR"
            )
            monthly_consumption = st.number_input(
                "Monthly Electricity Consumption (kWh)", min_value=10, max_value=50000,
                value=300, step=10,
                help="Total kWh consumed per month"
            )
            roof_area = st.number_input(
                "Available Rooftop Area (m²)", min_value=5, max_value=10000,
                value=50, step=5,
                help="Usable area for solar panels"
            )
            budget = st.number_input(
                "Budget for Installation (₹)", min_value=10000, max_value=50000000,
                value=200000, step=10000,
                help="Your budget for the solar installation"
            )

        # Electricity rate
        elec_rate = st.slider(
            "Electricity Rate (₹/kWh)",
            min_value=3.0, max_value=15.0, value=8.0, step=0.5,
            help="Check your electricity bill for the per-unit rate"
        )

        submitted = st.form_submit_button(
            "🔍 Analyze Solar Potential",
            use_container_width=True,
        )

    # ── Analysis Logic ─────────────────────────────────────────────────────────
    if submitted:
        if not IMPORTS_OK:
            st.error("⚠ Dependencies not installed. Run: `pip install -r requirements.txt`")
            return

        with st.spinner("🛰 Fetching location & weather data..."):
            # Geocode
            if "City" in location_mode:
                if not final_city:
                    st.warning("⚠ Please select or type a city name.")
                    return
                loc = get_location_info(final_city)
            else:
                loc = {"lat": lat_input, "lon": lon_input, "city": f"({lat_input:.2f}°, {lon_input:.2f}°)", "display_name": "Custom Location"}

            if not loc:
                st.error(f"❌ Could not find location: '{final_city}'. Try a different spelling.")
                return

            st.session_state.location = loc

        with st.spinner("☀️ Fetching real-time solar & weather data..."):
            weather = get_weather_data(loc["lat"], loc["lon"])
            st.session_state.weather = weather

        with st.spinner("🤖 Running ML models..."):
            # Build feature vector
            features = build_feature_vector(
                lat                  = loc["lat"],
                lon                  = loc["lon"],
                ghi                  = weather.get("ghi", 5.0),
                temperature          = weather.get("temperature", 28.0),
                cloud_cover          = weather.get("cloud_cover", 30.0),
                humidity             = weather.get("humidity", 50.0),
                wind_speed           = weather.get("wind_speed", 3.0),
                clearness_index      = weather.get("clearness_index", 0.65),
                roof_area            = roof_area,
                monthly_consumption_kwh = monthly_consumption,
                property_type        = property_type,
            )
            st.session_state.features = features

            # Monthly generation prediction
            monthly_data = predict_all_months(features)
            st.session_state.monthly_data = monthly_data

            # Current month generation
            current_month = datetime.now().month
            generation = predict_generation(features)
            st.session_state.generation = generation

            # Suitability score
            score, breakdown = calculate_solar_suitability_score(
                ghi                  = weather.get("ghi", 5.0),
                cloud_cover          = weather.get("cloud_cover", 30.0),
                roof_area            = roof_area,
                budget               = budget,
                monthly_consumption_kwh = monthly_consumption,
                lat                  = loc["lat"],
            )
            st.session_state.suitability   = score
            st.session_state.score_breakdown = breakdown

            # Store user inputs for financial analysis
            st.session_state.user_inputs = {
                "monthly_bill":       monthly_bill,
                "monthly_consumption": monthly_consumption,
                "roof_area":          roof_area,
                "budget":             budget,
                "elec_rate":          elec_rate,
                "property_type":      property_type,
            }

            st.session_state.analysis_done = True

    # ── Results Display ────────────────────────────────────────────────────────
    if st.session_state.analysis_done:
        loc     = st.session_state.location
        weather = st.session_state.weather
        score   = st.session_state.suitability
        gen     = st.session_state.generation
        monthly = st.session_state.monthly_data
        inputs  = st.session_state.user_inputs

        label, color = classify_suitability(score)
        city_name = loc.get("city", "Unknown")

        st.divider()
        st.markdown(f"### 📍 Analysis Results for **{loc.get('display_name', city_name)}**")

        # ── Top KPI Row ──
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{score:.0f}</div>
                <div class="metric-label">Suitability Score</div>
                <div class="metric-sub"><span class="badge badge-{'excellent' if label=='Excellent' else 'average' if label=='Average' else 'poor'}">{label}</span></div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{weather.get('ghi', 0):.2f}</div>
                <div class="metric-label">GHI (kWh/m²/day)</div>
                <div class="metric-sub">Solar Irradiance</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{gen:.0f}</div>
                <div class="metric-label">Generation (kWh/mo)</div>
                <div class="metric-sub">ML Predicted</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            sys_size = get_system_size(inputs["roof_area"])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{sys_size:.1f}</div>
                <div class="metric-label">System Size (kWp)</div>
                <div class="metric-sub">{inputs['roof_area']} m² roof</div>
            </div>""", unsafe_allow_html=True)
        with k5:
            solar_pct = min(100, gen / max(1, inputs["monthly_consumption"]) * 100)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{solar_pct:.0f}%</div>
                <div class="metric-label">Bill Coverage</div>
                <div class="metric-sub">Solar fraction</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Weather Data ──
        w1, w2 = st.columns(2)
        with w1:
            st.markdown("**🌤 Current Weather & Solar Data**")
            weather_df = pd.DataFrame([{
                "Parameter":  p,
                "Value":      v,
                "Unit":       u,
            } for p, v, u in [
                ("Temperature",      f"{weather.get('temperature', 0):.1f}", "°C"),
                ("Humidity",         f"{weather.get('humidity', 0):.0f}", "%"),
                ("Cloud Cover",      f"{weather.get('cloud_cover', 0):.0f}", "%"),
                ("Wind Speed",       f"{weather.get('wind_speed', 0):.1f}", "m/s"),
                ("Solar GHI",        f"{weather.get('ghi', 0):.3f}", "kWh/m²/day"),
                ("Clearness Index",  f"{weather.get('clearness_index', 0):.2f}", "0-1"),
                ("Data Source",      weather.get("source", "unknown"), ""),
            ]])
            st.dataframe(weather_df, use_container_width=True, hide_index=True)

        with w2:
            st.markdown("**🎯 Suitability Score Breakdown**")
            bd = st.session_state.score_breakdown
            bd_df = pd.DataFrame([
                {"Factor": "Solar Irradiance",  "Score": bd.get("ghi_score", 0),    "Weight": "35%"},
                {"Factor": "Cloud Coverage",    "Score": bd.get("cloud_score", 0),  "Weight": "20%"},
                {"Factor": "Roof Area",         "Score": bd.get("roof_score", 0),   "Weight": "20%"},
                {"Factor": "Budget Adequacy",   "Score": bd.get("budget_score", 0), "Weight": "15%"},
                {"Factor": "Location Zone",     "Score": bd.get("loc_score", 0),    "Weight": "10%"},
            ])
            st.dataframe(
                bd_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f/100")
                }
            )

        st.divider()

        # ── Charts Row 1 ──
        c1, c2 = st.columns([1, 2])
        with c1:
            fig_gauge = gauge_suitability_score(score, label, color)
            st.plotly_chart(fig_gauge, use_container_width=True)
        with c2:
            if monthly:
                fig_monthly = chart_monthly_generation(monthly, inputs["monthly_consumption"])
                st.plotly_chart(fig_monthly, use_container_width=True)

        # ── Charts Row 2 ──
        c3, c4 = st.columns(2)
        with c3:
            bd = st.session_state.score_breakdown
            fig_radar = chart_score_radar(bd)
            st.plotly_chart(fig_radar, use_container_width=True)
        with c4:
            avg_annual_gen = (sum(d["generation_kwh"] for d in monthly) if monthly else gen * 12)
            fig_pie = chart_energy_sources(gen, inputs["monthly_consumption"])
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Location Map ──
        fig_map = chart_location_map(
            loc["lat"], loc["lon"],
            city_name,
            weather.get("ghi", 5.0),
            score,
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # ── CTA ──
        st.markdown("""
        <div class="success-box">
            ✅ Analysis complete! Go to <b>💰 Financial Report</b> for ROI calculations, 
            or <b>🤖 AI Assistant</b> for personalized advice.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FINANCIAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def show_financial():
    st.markdown('<div class="section-header">💰 Financial Report & ROI Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <b>Please run Solar Analysis first!</b><br>
            Go to <b>📊 Solar Analysis</b>, enter your details, and click Analyze.
        </div>
        """, unsafe_allow_html=True)
        return

    loc     = st.session_state.location
    weather = st.session_state.weather
    inputs  = st.session_state.user_inputs
    gen     = st.session_state.generation

    # Generate Financial Report
    with st.spinner("💹 Calculating financial projections..."):
        financial = full_financial_analysis(
            lat                    = loc["lat"],
            lon                    = loc["lon"],
            monthly_generation_kwh = gen,
            monthly_consumption_kwh= inputs["monthly_consumption"],
            electricity_rate       = inputs["elec_rate"],
            roof_area              = inputs["roof_area"],
            budget                 = inputs["budget"],
            property_type          = inputs["property_type"],
        )
        st.session_state.financial = financial

    fi   = financial
    cost = fi["cost_info"]
    sav  = fi["savings_info"]
    roi  = fi["roi_info"]
    city_name = loc.get("city", "Unknown")

    # ── ROI KPI Cards ──
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("₹" + f"{cost['net_cost']:,.0f}",       "Net Installation Cost",  f"₹{cost['subsidy_amount']:,.0f} subsidy applied", "#F4A826"),
        ("₹" + f"{sav['monthly_savings_avg']:,.0f}", "Monthly Savings",    f"₹{sav['total_annual_savings']:,.0f}/year", "#00C853"),
        (f"{roi['payback_years']} yrs",            "Payback Period",        "Including maintenance", "#42A5F5"),
        (f"{roi['roi_pct']:.0f}%",                 "25-Year ROI",           f"₹{roi['profit_25yr']:,.0f} profit", "#AB47BC"),
    ]
    for col, (val, label, sub, clr) in zip([k1,k2,k3,k4], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{clr}">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Environment Impact ──
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#00C853">{roi['co2_saved_tonnes']:.0f}t</div>
            <div class="metric-label">CO₂ Saved (25yr)</div></div>""", unsafe_allow_html=True)
    with e2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#00C853">{roi['trees_equivalent']:,}</div>
            <div class="metric-label">Tree Equivalent</div></div>""", unsafe_allow_html=True)
    with e3:
        irr_disp = f"{roi['irr_pct']:.1f}%" if roi.get("irr_pct") else "N/A"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#AB47BC">{irr_disp}</div>
            <div class="metric-label">IRR (Internal Rate of Return)</div></div>""", unsafe_allow_html=True)

    st.divider()

    # ── Financial Charts ──
    fc1, fc2 = st.columns(2)
    with fc1:
        fig_cost = chart_cost_breakdown(
            cost["cost_breakdown"],
            cost["subsidy_amount"],
            cost["net_cost"],
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with fc2:
        fig_bill = chart_bill_comparison(
            fi["monthly_bill_before"],
            fi["monthly_bill_after"],
            sav["monthly_savings_avg"],
        )
        st.plotly_chart(fig_bill, use_container_width=True)

    # Savings Forecast
    fig_savings = chart_savings_forecast(roi["yearly_data"], cost["net_cost"], years=15)
    st.plotly_chart(fig_savings, use_container_width=True)

    # ROI KPI indicators
    fig_kpi = chart_roi_kpi(
        roi["payback_years"],
        roi["roi_pct"],
        roi["profit_10yr"],
        roi.get("irr_pct"),
    )
    st.plotly_chart(fig_kpi, use_container_width=True)

    # ── Detailed Year-wise Table ──
    st.markdown('<div class="section-header">📅 Year-wise Projection</div>', unsafe_allow_html=True)
    yearly_df = pd.DataFrame(roi["yearly_data"][:20])
    yearly_df.columns = ["Year", "Gross Savings (₹)", "Maintenance (₹)",
                         "Net Savings (₹)", "Cumulative Savings (₹)", "Cumulative Profit (₹)"]
    st.dataframe(
        yearly_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cumulative Profit (₹)": st.column_config.NumberColumn(
                format="₹%.0f", help="Profit after recovering investment cost"
            ),
            "Cumulative Savings (₹)": st.column_config.NumberColumn(format="₹%.0f"),
        }
    )

    # ── Budget Warning ──
    if not fi["budget_sufficient"]:
        gap = fi["budget_gap"]
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ <b>Budget Gap:</b> Your budget is ₹{gap:,.0f} short of the net installation cost.<br>
            💡 <b>Options:</b> Apply for solar loan (SBI, Axis Bank offer green loans at 7–9%), 
            use government subsidy, or reduce system size to {fi['system_size_kwp']*0.7:.1f} kWp.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
def show_chat():
    st.markdown('<div class="section-header">🤖 AI Solar Assistant</div>', unsafe_allow_html=True)

    assistant = st.session_state.assistant
    if assistant is None:
        st.error("Assistant not initialized.")
        return

    # Load context if analysis done
    if st.session_state.analysis_done and st.session_state.financial:
        loc     = st.session_state.location
        weather = st.session_state.weather
        inputs  = st.session_state.user_inputs
        gen     = st.session_state.generation
        fi      = st.session_state.financial
        score   = st.session_state.suitability
        label, _ = classify_suitability(score)

        ctx = build_context(
            city                   = loc.get("city", "Unknown"),
            lat                    = loc.get("lat", 20),
            lon                    = loc.get("lon", 77),
            ghi                    = weather.get("ghi", 5),
            temperature            = weather.get("temperature", 28),
            cloud_cover            = weather.get("cloud_cover", 30),
            property_type          = inputs.get("property_type", "Residential"),
            roof_area              = inputs.get("roof_area", 50),
            monthly_consumption_kwh= inputs.get("monthly_consumption", 300),
            monthly_bill           = inputs.get("monthly_bill", 3000),
            budget                 = inputs.get("budget", 200000),
            system_size_kwp        = fi.get("system_size_kwp", 3),
            monthly_generation_kwh = gen,
            suitability_score      = score,
            suitability_label      = label,
            net_cost               = fi["cost_info"]["net_cost"],
            subsidy                = fi["cost_info"]["subsidy_amount"],
            monthly_savings        = fi["savings_info"]["monthly_savings_avg"],
            payback_years          = fi["roi_info"]["payback_years"],
            profit_10yr            = fi["roi_info"]["profit_10yr"],
            roi_pct                = fi["roi_info"]["roi_pct"],
            co2_tonnes             = fi["roi_info"]["co2_saved_tonnes"],
        )
        assistant.set_context(ctx)
        ai_mode = "🧠 AI-Powered" if assistant.is_ai_powered else "🔧 Smart Rule-Based"

        # Status bar
        st.markdown(f"""
        <div class="success-box">
            {ai_mode} | Loaded context for <b>{loc.get("city", "your location")}</b> | 
            Score: <b>{score:.0f}/100</b> | Monthly savings: <b>₹{fi['savings_info']['monthly_savings_avg']:,.0f}</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            ℹ️ Complete the <b>📊 Solar Analysis</b> and <b>💰 Financial Report</b> first 
            for personalized AI answers. Basic solar questions still work!
        </div>
        """, unsafe_allow_html=True)

    # ── Suggested Questions ──
    st.markdown("**💡 Suggested Questions** (click to ask):")
    questions = get_suggested_questions()
    cols = st.columns(4)
    for i, q in enumerate(questions):
        with cols[i % 4]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("🤔 Thinking..."):
                    response = assistant.chat(q)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.divider()

    # ── Chat History ──
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center; color:#6B7280; padding:2rem;">
                <div style="font-size:3rem">☀️</div>
                <div>Hello! I'm SolarBot, your AI solar consultant.</div>
                <div style="font-size:0.9rem">Ask me anything about solar energy!</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-msg-user">
                        <b style="color:#42A5F5">👤 You</b><br>{msg["content"]}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-msg-bot">
                        <b style="color:#F4A826">☀️ SolarBot</b><br>{msg["content"]}
                    </div>""", unsafe_allow_html=True)

    # ── Chat Input ──
    st.divider()
    col_input, col_btn, col_clr = st.columns([6, 1, 1])
    with col_input:
        user_msg = st.text_input(
            "Ask SolarBot...",
            placeholder="e.g. How much money will I save in 5 years?",
            label_visibility="collapsed",
            key="chat_input"
        )
    with col_btn:
        send = st.button("Send ➤", use_container_width=True)
    with col_clr:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            assistant.reset_chat()
            st.rerun()

    if send and user_msg.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_msg.strip()})
        with st.spinner("🤔 Thinking..."):
            response = assistant.chat(user_msg.strip())
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

# App Title & Logo Header
st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 2rem; margin-top: 1rem;">
    <div style="font-size: 3rem; filter: drop-shadow(0 0 15px rgba(244,168,38,0.4)); animation: float 4s ease-in-out infinite;">☀️</div>
    <div>
        <h1 style="margin: 0; padding: 0; font-size: 2.2rem; font-weight: 700; color: #E8EAF0; letter-spacing: -0.5px; line-height: 1.2;">Solar Industry <span style="color: #F4A826;">AI</span></h1>
        <p style="margin: 0; padding: 0; font-size: 0.95rem; color: #9CA3AF; font-weight: 400; letter-spacing: 0.5px;">SMART ROOFTOP INTELLIGENCE</p>
    </div>
</div>
""", unsafe_allow_html=True)

nav_options = ["Dashboard", "Solar Analysis", "Financial Report", "AI Assistant"]
icons = ["house", "bar-chart-line", "piggy-bank", "robot"]

selected_page = option_menu(
    menu_title=None,
    options=nav_options,
    icons=icons,
    default_index=0,
    orientation="horizontal",
    key="main_nav",
    styles={
        "container": {
            "display": "flex", "flex-wrap": "wrap", "justify-content": "center",
            "padding": "0!important", "background-color": "rgba(17, 24, 39, 0.4)", 
            "border": "1px solid rgba(255,255,255,0.05)", "border-radius": "12px", 
            "backdrop-filter": "blur(12px)", "margin-bottom": "2rem"
        },
        "icon": {"color": "#F4A826", "font-size": "1.1rem"}, 
        "nav-link": {
            "flex": "1 1 150px", "min-width": "150px", 
            "font-size": "1.05rem", "font-family": "Inter, sans-serif", 
            "text-align": "center", "margin": "0px", 
            "--hover-color": "rgba(244,168,38,0.1)", "color": "#E8EAF0"
        },
        "nav-link-selected": {
            "background-color": "rgba(244,168,38,0.2)", 
            "color": "#F4A826", "border": "1px solid rgba(244,168,38,0.3)"
        },
    }
)

if selected_page == "Dashboard":
    show_home()
elif selected_page == "Solar Analysis":
    show_analysis()
elif selected_page == "Financial Report":
    show_financial()
elif selected_page == "AI Assistant":
    show_chat()
