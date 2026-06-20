"""
ROI Calculator
==============
Financial analysis engine for solar installations.
Supports India-first pricing with global configurability.
"""

import numpy as np
from typing import Optional
import os

# ─── India Solar Market Constants (2024) ─────────────────────────────────────
INDIA_INSTALL_COST_PER_KWP = 55_000   # ₹ per kWp (on-grid, polycrystalline)
INDIA_PREMIUM_COST_PER_KWP = 75_000   # ₹ per kWp (monocrystalline PERC)
INDIA_GOVT_SUBSIDY_BELOW_3KW  = 0.30  # 30% subsidy (PM Surya Ghar scheme, ≤3kW)
INDIA_GOVT_SUBSIDY_3_TO_10KW  = 0.14  # 14% subsidy (3–10 kW)
ANNUAL_DEGRADATION_RATE       = 0.005 # 0.5% per year panel degradation
INFLATION_RATE                = 0.06  # 6% electricity price inflation (India avg)
MAINTENANCE_COST_PER_KWP_YR   = 500   # ₹ per kWp per year
CO2_KG_PER_KWH_GRID_INDIA     = 0.82  # kg CO2 per kWh (India grid emission factor)
DISCOUNT_RATE                  = 0.08  # 8% for NPV calculation

# Global defaults (USD)
GLOBAL_COST_PER_KWP_USD = 1_000  # ~$1000/kWp global average 2024
INR_USD_RATE            = 83.5   # Approximate exchange rate


def estimate_system_size(
    roof_area: float,
    panel_efficiency: float = 0.20,
    panel_coverage: float = 0.75,
) -> float:
    """Estimate installed capacity in kWp."""
    return round(roof_area * panel_coverage * panel_efficiency, 2)


def calculate_installation_cost(
    system_size_kwp: float,
    property_type: str = "Residential",
    panel_quality: str = "Standard",
    currency: str = "INR",
    apply_subsidy: bool = True,
) -> dict:
    """
    Calculate total installation cost with Indian subsidy scheme.
    
    Returns:
        cost_before_subsidy, subsidy_amount, net_cost, subsidy_pct, cost_breakdown
    """
    if currency == "INR":
        base_rate = (INDIA_PREMIUM_COST_PER_KWP
                     if panel_quality == "Premium"
                     else INDIA_INSTALL_COST_PER_KWP)
    else:
        base_rate = GLOBAL_COST_PER_KWP_USD * INR_USD_RATE

    # Commercial has slightly higher cost (mounting, wiring)
    if property_type == "Commercial":
        base_rate *= 1.15

    panel_cost     = system_size_kwp * base_rate * 0.45   # Panels ~45%
    inverter_cost  = system_size_kwp * base_rate * 0.18   # Inverter ~18%
    mounting_cost  = system_size_kwp * base_rate * 0.15   # Mounting ~15%
    wiring_cost    = system_size_kwp * base_rate * 0.12   # Wiring ~12%
    installation_labor = system_size_kwp * base_rate * 0.10 # Labor ~10%
    gross_cost     = panel_cost + inverter_cost + mounting_cost + wiring_cost + installation_labor

    # India PM Surya Ghar Subsidy (for residential)
    subsidy = 0.0
    subsidy_pct = 0.0
    if apply_subsidy and property_type == "Residential" and currency == "INR":
        if system_size_kwp <= 3:
            subsidy = gross_cost * INDIA_GOVT_SUBSIDY_BELOW_3KW
            subsidy_pct = INDIA_GOVT_SUBSIDY_BELOW_3KW * 100
        elif system_size_kwp <= 10:
            subsidy = gross_cost * INDIA_GOVT_SUBSIDY_3_TO_10KW
            subsidy_pct = INDIA_GOVT_SUBSIDY_3_TO_10KW * 100

    net_cost = max(0, gross_cost - subsidy)

    return {
        "gross_cost":       round(gross_cost),
        "subsidy_amount":   round(subsidy),
        "net_cost":         round(net_cost),
        "subsidy_pct":      round(subsidy_pct, 1),
        "cost_breakdown": {
            "Solar Panels":      round(panel_cost),
            "Inverter":          round(inverter_cost),
            "Mounting Structure":round(mounting_cost),
            "Wiring & BOS":      round(wiring_cost),
            "Installation Labor":round(installation_labor),
        },
        "cost_per_kwp":     round(base_rate),
    }


def calculate_annual_savings(
    monthly_generation_kwh: float,
    electricity_rate: float,
    grid_export_rate: float = None,
    annual_consumption_kwh: float = None,
) -> dict:
    """
    Calculate annual savings from solar.
    
    Args:
        monthly_generation_kwh: Average monthly solar output
        electricity_rate: Current electricity rate (₹/kWh)
        grid_export_rate: Feed-in tariff (if net metering); defaults to 50% of rate
        annual_consumption_kwh: Total annual consumption
    
    Returns: savings breakdown dict
    """
    if grid_export_rate is None:
        grid_export_rate = electricity_rate * 0.5  # 50% net metering typical in India

    annual_generation = monthly_generation_kwh * 12

    if annual_consumption_kwh:
        # How much solar covers consumption
        solar_self_consumed = min(annual_generation, annual_consumption_kwh)
        solar_exported      = max(0, annual_generation - annual_consumption_kwh)
    else:
        solar_self_consumed = annual_generation * 0.80  # 80% self-consumed
        solar_exported      = annual_generation * 0.20

    savings_from_offset  = solar_self_consumed * electricity_rate
    savings_from_export  = solar_exported * grid_export_rate
    annual_bill_savings  = savings_from_offset + savings_from_export

    return {
        "annual_generation_kwh": round(annual_generation, 1),
        "solar_self_consumed":   round(solar_self_consumed, 1),
        "solar_exported":        round(solar_exported, 1),
        "savings_from_offset":   round(savings_from_offset),
        "savings_from_export":   round(savings_from_export),
        "total_annual_savings":  round(annual_bill_savings),
        "monthly_savings_avg":   round(annual_bill_savings / 12),
    }


def calculate_roi(
    net_installation_cost: float,
    annual_savings: float,
    system_size_kwp: float,
    electricity_rate: float,
    years: int = 25,
    currency: str = "INR",
) -> dict:
    """
    Full ROI analysis over system lifetime.
    
    Returns yearly cash flow, payback period, NPV, IRR, 10/25-year profit.
    """
    annual_maintenance = system_size_kwp * MAINTENANCE_COST_PER_KWP_YR

    yearly_data = []
    cumulative_savings = 0.0
    payback_year = None
    npv = -net_installation_cost

    for year in range(1, years + 1):
        # Panel degradation reduces output each year
        degradation = (1 - ANNUAL_DEGRADATION_RATE) ** year

        # Electricity rate inflation increases savings value
        rate_inflation = (1 + INFLATION_RATE) ** (year - 1)

        year_savings    = annual_savings * degradation * rate_inflation
        year_cost       = annual_maintenance * (1 + 0.02) ** (year - 1)  # 2% maintenance inflation
        year_net        = year_savings - year_cost
        cumulative_savings += year_net

        # NPV: discount future cash flows
        discount_factor = (1 + DISCOUNT_RATE) ** year
        npv += year_net / discount_factor

        if payback_year is None and cumulative_savings >= net_installation_cost:
            payback_year = year

        yearly_data.append({
            "year":               year,
            "gross_savings":      round(year_savings),
            "maintenance_cost":   round(year_cost),
            "net_savings":        round(year_net),
            "cumulative_savings": round(cumulative_savings),
            "cum_profit":         round(cumulative_savings - net_installation_cost),
        })

    # IRR approximation (Newton's method on NPV)
    irr = _estimate_irr(net_installation_cost, [d["net_savings"] for d in yearly_data[:20]])

    profit_5yr  = yearly_data[4]["cumulative_savings"] - net_installation_cost if len(yearly_data) >= 5  else 0
    profit_10yr = yearly_data[9]["cumulative_savings"] - net_installation_cost if len(yearly_data) >= 10 else 0
    profit_25yr = yearly_data[-1]["cumulative_savings"] - net_installation_cost

    total_generation_25yr = sum(
        annual_savings / electricity_rate * (1 - ANNUAL_DEGRADATION_RATE) ** y
        for y in range(1, years + 1)
    )
    co2_saved_tonnes = (total_generation_25yr * CO2_KG_PER_KWH_GRID_INDIA) / 1000

    return {
        "payback_years":     payback_year or years,
        "npv":               round(npv),
        "irr_pct":           round(irr * 100, 1) if irr else None,
        "roi_pct":           round((profit_25yr / max(1, net_installation_cost)) * 100, 1),
        "profit_5yr":        round(profit_5yr),
        "profit_10yr":       round(profit_10yr),
        "profit_25yr":       round(profit_25yr),
        "total_savings_25yr":round(yearly_data[-1]["cumulative_savings"]),
        "co2_saved_tonnes":  round(co2_saved_tonnes, 1),
        "trees_equivalent":  round(co2_saved_tonnes * 45),  # 1 tree ≈ 22kg CO2/yr × 25yr
        "yearly_data":       yearly_data,
        "annual_maintenance":round(annual_maintenance),
    }


def full_financial_analysis(
    lat: float,
    lon: float,
    monthly_generation_kwh: float,
    monthly_consumption_kwh: float,
    electricity_rate: float,
    roof_area: float,
    budget: float,
    property_type: str = "Residential",
    panel_quality: str = "Standard",
    currency: str = "INR",
) -> dict:
    """
    Complete financial analysis combining all modules.
    Single entry point for the Streamlit app.
    """
    # System sizing
    from src.preprocess import get_system_size
    system_size = get_system_size(roof_area)

    # Installation cost
    cost_info = calculate_installation_cost(
        system_size_kwp = system_size,
        property_type   = property_type,
        panel_quality   = panel_quality,
        currency        = currency,
    )

    # Annual savings
    annual_consumption = monthly_consumption_kwh * 12
    savings_info = calculate_annual_savings(
        monthly_generation_kwh  = monthly_generation_kwh,
        electricity_rate        = electricity_rate,
        annual_consumption_kwh  = annual_consumption,
    )

    # ROI
    roi_info = calculate_roi(
        net_installation_cost = cost_info["net_cost"],
        annual_savings        = savings_info["total_annual_savings"],
        system_size_kwp       = system_size,
        electricity_rate      = electricity_rate,
    )

    # Bill reduction percentage
    monthly_bill = monthly_consumption_kwh * electricity_rate
    bill_reduction_pct = min(100, (monthly_generation_kwh / max(1, monthly_consumption_kwh)) * 100)
    new_bill = max(0, monthly_bill - savings_info["monthly_savings_avg"])

    return {
        "system_size_kwp":      system_size,
        "cost_info":            cost_info,
        "savings_info":         savings_info,
        "roi_info":             roi_info,
        "monthly_bill_before":  round(monthly_bill),
        "monthly_bill_after":   round(new_bill),
        "bill_reduction_pct":   round(bill_reduction_pct, 1),
        "budget_sufficient":    budget >= cost_info["net_cost"],
        "budget_gap":           max(0, cost_info["net_cost"] - budget),
    }


def _estimate_irr(initial_investment: float, cash_flows: list) -> Optional[float]:
    """Estimate IRR using Newton-Raphson method."""
    try:
        r = 0.10  # Initial guess
        for _ in range(100):
            npv = -initial_investment
            dnpv = 0
            for t, cf in enumerate(cash_flows, 1):
                npv  += cf / (1 + r) ** t
                dnpv -= t * cf / (1 + r) ** (t + 1)
            if abs(dnpv) < 1e-10:
                break
            r_new = r - npv / dnpv
            if abs(r_new - r) < 1e-6:
                r = r_new
                break
            r = max(-0.99, min(5.0, r_new))
        return r if 0 < r < 5 else None
    except Exception:
        return None
