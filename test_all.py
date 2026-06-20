"""
Full Functional Test Suite for Solar AI Assistant
Run: python test_all.py
"""
import sys
import traceback

print("=" * 60)
print("  Solar AI Assistant - Full Functional Test Suite")
print("=" * 60)

passed = 0
failed = 0

def ok(n, msg):
    global passed
    passed += 1
    print(f"  [PASS {n:02d}] {msg}")

def fail(n, msg, err):
    global failed
    failed += 1
    print(f"  [FAIL {n:02d}] {msg}")
    print(f"           Error: {err}")


# ── 1. Preprocess ──────────────────────────────────────────────────────────────
try:
    from src.preprocess import (build_feature_vector, calculate_solar_suitability_score,
                                 classify_suitability, get_system_size, days_in_month, get_season)
    fv = build_feature_vector(
        lat=26.91, lon=75.79, ghi=5.8, temperature=28, cloud_cover=25,
        humidity=50, wind_speed=3, clearness_index=0.68, roof_area=50,
        monthly_consumption_kwh=300
    )
    assert len(fv) >= 20, f"Expected 20+ features, got {len(fv)}"
    ok(1, f"build_feature_vector: {len(fv)} features")
except Exception as e:
    fail(1, "build_feature_vector", e)
    fv = {}

try:
    score, bd = calculate_solar_suitability_score(5.8, 25, 50, 200000, 300, 26.91)
    assert 0 <= score <= 100
    assert "ghi_score" in bd
    ok(2, f"suitability_score: {score}/100")
except Exception as e:
    fail(2, "calculate_solar_suitability_score", e)
    score, bd = 75.0, {"ghi_score":80,"cloud_score":75,"roof_score":60,"budget_score":100,"loc_score":90}

try:
    label, color = classify_suitability(score)
    assert label in ("Excellent", "Average", "Poor")
    ok(3, f"classify_suitability: {label} ({color})")
except Exception as e:
    fail(3, "classify_suitability", e)
    label, color = "Excellent", "#00C853"

try:
    assert days_in_month(2) == 28
    assert days_in_month(1) == 31
    assert get_season(6, 26.91) == 2   # Summer for north India
    ok(4, "days_in_month and get_season: OK")
except Exception as e:
    fail(4, "days_in_month/get_season", e)

try:
    sz = get_system_size(50)
    assert 0 < sz < 50
    ok(5, f"get_system_size(50 m2) = {sz} kWp")
except Exception as e:
    fail(5, "get_system_size", e)


# ── 2. Location Service ────────────────────────────────────────────────────────
try:
    from src.location_service import get_location_info, get_india_cities, get_global_cities
    loc = get_location_info("Jaipur")
    assert loc is not None
    assert abs(loc["lat"] - 26.91) < 0.5
    ok(6, f"get_location_info(Jaipur): lat={loc['lat']}, lon={loc['lon']}")
except Exception as e:
    fail(6, "get_location_info", e)
    loc = {"lat": 26.91, "lon": 75.79, "city": "Jaipur", "display_name": "Jaipur, India"}

try:
    cities = get_india_cities()
    assert len(cities) >= 20
    ok(7, f"get_india_cities: {len(cities)} cities")
except Exception as e:
    fail(7, "get_india_cities", e)

try:
    all_cities = get_global_cities()
    assert len(all_cities) >= 50
    ok(8, f"get_global_cities: {len(all_cities)} cities")
except Exception as e:
    fail(8, "get_global_cities", e)


# ── 3. Predictor ───────────────────────────────────────────────────────────────
try:
    from src.predictor import predict_generation, predict_all_months, is_model_loaded, get_model_metrics
    loaded = is_model_loaded()
    ok(9, f"is_model_loaded: {loaded}")
except Exception as e:
    fail(9, "is_model_loaded", e)
    loaded = False

try:
    gen = predict_generation(fv)
    assert gen > 0
    ok(10, f"predict_generation: {gen:.1f} kWh/month")
except Exception as e:
    fail(10, "predict_generation", e)
    gen = 350.0

try:
    monthly = predict_all_months(fv)
    assert len(monthly) == 12
    assert all("generation_kwh" in m for m in monthly)
    assert all("month_name" in m for m in monthly)
    min_m = min(m["generation_kwh"] for m in monthly)
    max_m = max(m["generation_kwh"] for m in monthly)
    ok(11, f"predict_all_months: 12 months, range {min_m:.0f}-{max_m:.0f} kWh")
except Exception as e:
    fail(11, "predict_all_months", e)
    monthly = [{"month": i, "month_name": "Jan", "generation_kwh": 300, "kwh_per_kwp": 100, "ghi": 5.0, "cloud_cover": 30} for i in range(1,13)]

try:
    metrics = get_model_metrics()
    r2 = metrics.get("metrics", {}).get("generation_model", {}).get("r2", 0)
    ok(12, f"get_model_metrics: generation R2={r2:.4f}")
except Exception as e:
    fail(12, "get_model_metrics", e)


# ── 4. ROI Calculator ──────────────────────────────────────────────────────────
try:
    from src.roi_calculator import full_financial_analysis, calculate_installation_cost
    fi = full_financial_analysis(
        lat=26.91, lon=75.79,
        monthly_generation_kwh=gen,
        monthly_consumption_kwh=300,
        electricity_rate=8.0,
        roof_area=50,
        budget=200000,
        property_type="Residential",
    )
    assert "cost_info" in fi
    assert "roi_info" in fi
    assert "savings_info" in fi
    assert fi["cost_info"]["net_cost"] > 0
    ok(13, f"full_financial_analysis: net_cost=Rs{fi['cost_info']['net_cost']:,}, payback={fi['roi_info']['payback_years']}yr")
except Exception as e:
    fail(13, "full_financial_analysis", e)
    fi = {
        "cost_info": {"net_cost": 150000, "subsidy_amount": 45000, "gross_cost": 195000,
                      "cost_breakdown": {"Solar Panels": 87750, "Inverter": 35100,
                                        "Mounting Structure": 29250, "Wiring & BOS": 23400,
                                        "Installation Labor": 19500}, "cost_per_kwp": 55000},
        "savings_info": {"total_annual_savings": 21600, "monthly_savings_avg": 1800,
                         "annual_generation_kwh": 4200, "solar_self_consumed": 3600, "solar_exported": 600,
                         "savings_from_offset": 28800, "savings_from_export": 2400},
        "roi_info": {"payback_years": 7, "roi_pct": 85.0, "profit_10yr": 80000, "profit_25yr": 400000,
                     "co2_saved_tonnes": 15.0, "trees_equivalent": 675, "yearly_data": [],
                     "npv": 100000, "irr_pct": 15.0},
        "system_size_kwp": 7.5, "monthly_bill_before": 2400, "monthly_bill_after": 600,
        "bill_reduction_pct": 75.0, "budget_sufficient": True, "budget_gap": 0,
    }

try:
    cost_res = calculate_installation_cost(3.0, "Residential")
    assert cost_res["subsidy_pct"] == 30.0  # 30% subsidy for <=3kW
    ok(14, f"PM Surya Ghar subsidy: {cost_res['subsidy_pct']}% for 3kW residential")
except Exception as e:
    fail(14, "calculate_installation_cost subsidy", e)


# ── 5. Visualizer ─────────────────────────────────────────────────────────────
try:
    from src.visualizer import (gauge_suitability_score, chart_monthly_generation,
                                 chart_bill_comparison, chart_energy_sources,
                                 chart_cost_breakdown, chart_savings_forecast,
                                 chart_roi_kpi, chart_score_radar, chart_location_map)
    import plotly.graph_objects as go

    fig = gauge_suitability_score(score, label, color)
    assert isinstance(fig, go.Figure)
    ok(15, "gauge_suitability_score: Figure OK")

    fig = chart_monthly_generation(monthly, 300)
    assert isinstance(fig, go.Figure)
    ok(16, "chart_monthly_generation: Figure OK")

    fig = chart_bill_comparison(3000, 1200, 1800)
    assert isinstance(fig, go.Figure)
    ok(17, "chart_bill_comparison: Figure OK")

    fig = chart_energy_sources(gen, 300)
    assert isinstance(fig, go.Figure)
    ok(18, "chart_energy_sources: Figure OK")

    fig = chart_cost_breakdown(fi["cost_info"]["cost_breakdown"],
                               fi["cost_info"]["subsidy_amount"],
                               fi["cost_info"]["net_cost"])
    assert isinstance(fig, go.Figure)
    ok(19, "chart_cost_breakdown: Figure OK")

    fig = chart_savings_forecast(fi["roi_info"]["yearly_data"] or
                                  [{"year":i,"cumulative_savings":i*20000,"cum_profit":i*20000-150000,
                                    "gross_savings":20000,"maintenance_cost":2000,"net_savings":18000}
                                   for i in range(1,16)],
                                  fi["cost_info"]["net_cost"])
    assert isinstance(fig, go.Figure)
    ok(20, "chart_savings_forecast: Figure OK")

    fig = chart_roi_kpi(fi["roi_info"]["payback_years"], fi["roi_info"]["roi_pct"],
                        fi["roi_info"]["profit_10yr"], fi["roi_info"].get("irr_pct"))
    assert isinstance(fig, go.Figure)
    ok(21, "chart_roi_kpi: Figure OK")

    fig = chart_score_radar(bd)
    assert isinstance(fig, go.Figure)
    ok(22, "chart_score_radar: Figure OK")

    fig = chart_location_map(26.91, 75.79, "Jaipur", 5.8, score)
    assert isinstance(fig, go.Figure)
    ok(23, "chart_location_map: Figure OK")
except Exception as e:
    fail(15, "visualizer charts", e)
    traceback.print_exc()


# ── 6. AI Assistant ───────────────────────────────────────────────────────────
try:
    from src.ai_assistant import SolarAIAssistant, build_context, get_suggested_questions
    assistant = SolarAIAssistant()
    ctx = build_context(
        city="Jaipur", lat=26.91, lon=75.79, ghi=5.8, temperature=28,
        cloud_cover=25, monthly_generation_kwh=gen, suitability_score=score,
        suitability_label=label, net_cost=fi["cost_info"]["net_cost"],
        subsidy=fi["cost_info"]["subsidy_amount"],
        monthly_savings=fi["savings_info"]["monthly_savings_avg"],
        payback_years=fi["roi_info"]["payback_years"],
        profit_10yr=fi["roi_info"]["profit_10yr"],
        roi_pct=fi["roi_info"]["roi_pct"],
        co2_tonnes=fi["roi_info"]["co2_saved_tonnes"],
    )
    assistant.set_context(ctx)
    response = assistant.chat("Is solar worth it for me?")
    assert len(response) > 20
    ok(24, f"AI assistant response: '{response[:60]}...'")
except Exception as e:
    fail(24, "SolarAIAssistant.chat", e)

try:
    questions = get_suggested_questions()
    assert len(questions) >= 5
    ok(25, f"get_suggested_questions: {len(questions)} questions")
except Exception as e:
    fail(25, "get_suggested_questions", e)


# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  Results: {passed} PASSED  |  {failed} FAILED")
print("=" * 60)
if failed > 0:
    sys.exit(1)
