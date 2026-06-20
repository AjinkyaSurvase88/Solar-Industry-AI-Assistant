"""
Plotly Visualizer
==================
All 9 interactive charts for the Solar AI Assistant dashboard.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional

# ─── Design Tokens ────────────────────────────────────────────────────────────
COLORS = {
    "solar_gold":   "#F4A826",
    "solar_orange": "#FF6B35",
    "solar_green":  "#00C853",
    "sky_blue":     "#42A5F5",
    "purple":       "#AB47BC",
    "dark_bg":      "#0A0E1A",
    "card_bg":      "#111827",
    "border":       "#1F2A44",
    "text_primary": "#E8EAF0",
    "text_muted":   "#9CA3AF",
    "red":          "#FF3D00",
    "amber":        "#FFB300",
}

# Base layout WITHOUT showlegend so each chart can set it individually
_BASE_LAYOUT = dict(
    paper_bgcolor=COLORS["dark_bg"],
    plot_bgcolor=COLORS["card_bg"],
    font=dict(family="Inter, sans-serif", color=COLORS["text_primary"]),
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(
        bgcolor=COLORS["card_bg"],
        bordercolor=COLORS["border"],
        borderwidth=1,
    ),
)


def _apply_base_layout(
    fig: go.Figure,
    title: str = "",
    height: int = 400,
    show_legend: bool = True,
) -> go.Figure:
    """Apply consistent dark theme layout to any figure."""
    fig.update_layout(
        **_BASE_LAYOUT,
        showlegend=show_legend,
        title=dict(
            text=title,
            font=dict(size=16, color=COLORS["solar_gold"]),
            x=0.01,
        ),
        height=height,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1: Solar Suitability Gauge
# ═══════════════════════════════════════════════════════════════════════════════
def gauge_suitability_score(score: float, label: str, color: str) -> go.Figure:
    """Animated gauge chart showing 0-100 suitability score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title=dict(text="Solar Suitability Score", font=dict(size=18, color=COLORS["text_primary"])),
        number=dict(suffix="", font=dict(size=48, color=color)),
        delta=dict(reference=50, valueformat=".1f"),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=2,
                tickcolor=COLORS["border"],
                tickfont=dict(color=COLORS["text_muted"]),
            ),
            bar=dict(color=color, thickness=0.85),
            bgcolor=COLORS["card_bg"],
            borderwidth=2,
            bordercolor=COLORS["border"],
            steps=[
                dict(range=[0,  40], color="#1C1010"),
                dict(range=[40, 70], color="#1C1800"),
                dict(range=[70, 100], color="#0D1C0D"),
            ],
            threshold=dict(
                line=dict(color=color, width=4),
                thickness=0.90,
                value=score,
            ),
        ),
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        showlegend=False,
        height=300,
        annotations=[dict(
            text=f"<b>{label}</b>",
            x=0.5, y=0.20, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=color),
        )],
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2: Monthly Energy Generation Line Chart
# ═══════════════════════════════════════════════════════════════════════════════
def chart_monthly_generation(monthly_data: list, monthly_consumption: float) -> go.Figure:
    """Line chart: 12-month solar generation vs consumption."""
    months = [d["month_name"] for d in monthly_data]
    gen    = [d["generation_kwh"] for d in monthly_data]
    cons   = [monthly_consumption] * len(months)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=months, y=gen,
        name="Solar Generation (kWh)",
        fill="tozeroy",
        fillcolor="rgba(244, 168, 38, 0.25)",
        line=dict(color=COLORS["solar_gold"], width=3),
        mode="lines+markers",
        marker=dict(size=8, color=COLORS["solar_gold"], symbol="circle"),
        hovertemplate="<b>%{x}</b><br>Generation: %{y:.0f} kWh<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=months, y=cons,
        name="Monthly Consumption (kWh)",
        line=dict(color=COLORS["sky_blue"], width=2, dash="dash"),
        mode="lines",
        hovertemplate="<b>%{x}</b><br>Consumption: %{y:.0f} kWh<extra></extra>",
    ))

    # Monsoon shading (June-Sept) — only annotate if months contain these labels
    if "Jun" in months and "Sep" in months:
        fig.add_vrect(
            x0="Jun", x1="Sep",
            fillcolor="rgba(66,165,245,0.06)", opacity=0.5, line_width=0,
            annotation_text="Monsoon", annotation_position="top left",
            annotation_font=dict(color=COLORS["sky_blue"], size=10),
        )

    fig = _apply_base_layout(fig, "Monthly Solar Generation Forecast", height=380)
    fig.update_xaxes(gridcolor=COLORS["border"], showgrid=True)
    fig.update_yaxes(gridcolor=COLORS["border"], showgrid=True, title_text="kWh")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3: Bill Before vs After Solar
# ═══════════════════════════════════════════════════════════════════════════════
def chart_bill_comparison(bill_before: float, bill_after: float, monthly_savings: float) -> go.Figure:
    """Grouped bar chart comparing electricity bills."""
    categories = ["Before Solar", "After Solar"]
    values     = [bill_before, max(0, bill_after)]
    bar_colors = [COLORS["red"], COLORS["solar_green"]]

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=bar_colors,
            line=dict(color=COLORS["border"], width=1),
            cornerradius=8,
        ),
        text=[f"Rs{v:,.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=15, color=COLORS["text_primary"]),
        hovertemplate="%{x}<br>Bill: Rs%{y:,.0f}<extra></extra>",
        width=0.4,
    ))

    if max(values) > 0:
        fig.add_annotation(
            x=0.5, y=max(values) * 0.5,
            text=f"<b>Monthly Savings<br>Rs{monthly_savings:,.0f}</b>",
            showarrow=False,
            font=dict(size=14, color=COLORS["solar_green"]),
            bgcolor=COLORS["card_bg"],
            bordercolor=COLORS["solar_green"],
            borderwidth=1,
            borderpad=8,
            xref="paper", yref="y",
        )

    fig = _apply_base_layout(fig, "Electricity Bill: Before vs After Solar",
                              height=360, show_legend=False)
    fig.update_yaxes(gridcolor=COLORS["border"], title_text="Monthly Bill (Rs)")
    fig.update_xaxes(showgrid=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4: Energy Source Pie / Donut Chart
# ═══════════════════════════════════════════════════════════════════════════════
def chart_energy_sources(solar_kwh: float, consumption_kwh: float) -> go.Figure:
    """Donut chart showing solar vs grid energy split."""
    consumption_kwh = max(1, consumption_kwh)
    solar_kwh       = max(0, solar_kwh)
    solar_pct       = min(100, (solar_kwh / consumption_kwh) * 100)
    exported_kwh    = max(0, solar_kwh - consumption_kwh)

    if exported_kwh > 0:
        labels = ["Solar (Self-Use)", "Solar (Grid Export)", "Grid Import"]
        vals   = [consumption_kwh, exported_kwh, 0.0]  # no grid import if solar > consumption
        clrs   = [COLORS["solar_gold"], COLORS["solar_orange"], COLORS["sky_blue"]]
    else:
        grid_kwh = max(0, consumption_kwh - solar_kwh)
        labels   = ["Solar Energy", "Grid Energy"]
        vals     = [solar_kwh, grid_kwh]
        clrs     = [COLORS["solar_gold"], COLORS["sky_blue"]]

    # Filter out zero values to avoid rendering issues
    filtered = [(l, v, c) for l, v, c in zip(labels, vals, clrs) if v > 0]
    if not filtered:
        filtered = [("No Data", 1, COLORS["border"])]
    labels, vals, clrs = zip(*filtered)

    fig = go.Figure(go.Pie(
        labels=list(labels),
        values=list(vals),
        hole=0.55,
        marker=dict(colors=list(clrs), line=dict(color=COLORS["dark_bg"], width=3)),
        textinfo="percent+label",
        textfont=dict(size=12),
        hovertemplate="%{label}<br>%{value:.0f} kWh (%{percent})<extra></extra>",
        pull=[0.05] + [0] * (len(vals) - 1),
    ))

    fig.add_annotation(
        text=f"<b>{solar_pct:.0f}%</b><br>Solar",
        x=0.5, y=0.5,
        font=dict(size=20, color=COLORS["solar_gold"]),
        showarrow=False,
    )

    fig = _apply_base_layout(fig, "Energy Source Distribution", height=380)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 5: Location Map with Solar Potential
# ═══════════════════════════════════════════════════════════════════════════════
def chart_location_map(lat: float, lon: float, city: str, ghi: float, score: float) -> go.Figure:
    """Scatter mapbox showing user location with solar potential."""
    fig = go.Figure(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode="markers+text",
        marker=dict(size=25, color=COLORS["solar_gold"], opacity=0.9),
        text=[city],
        textposition="top center",
        textfont=dict(color=COLORS["text_primary"], size=14),
        hovertemplate=(
            f"<b>{city}</b><br>"
            f"GHI: {ghi:.2f} kWh/m2/day<br>"
            f"Solar Score: {score:.0f}/100<br>"
            f"({lat:.3f}, {lon:.3f})"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        showlegend=False,
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=lat, lon=lon),
            zoom=8,
        ),
        height=350,
        title=dict(
            text=f"{city} - Solar Location",
            font=dict(size=16, color=COLORS["solar_gold"]),
            x=0.01,
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 6: Installation Cost Breakdown
# ═══════════════════════════════════════════════════════════════════════════════
def chart_cost_breakdown(cost_breakdown: dict, subsidy: float, net_cost: float) -> go.Figure:
    """Horizontal bar chart of cost components."""
    labels = list(cost_breakdown.keys())
    values = list(cost_breakdown.values())
    total  = sum(values) or 1  # avoid division by zero

    gradient_colors = [
        COLORS["solar_gold"],
        COLORS["solar_orange"],
        COLORS["sky_blue"],
        COLORS["purple"],
        COLORS["solar_green"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker=dict(
            color=gradient_colors[:len(labels)],
            line=dict(color=COLORS["dark_bg"], width=1),
            cornerradius=5,
        ),
        text=[f"Rs{v:,.0f} ({v/total*100:.0f}%)" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["text_primary"], size=12),
        hovertemplate="%{y}<br>Cost: Rs%{x:,.0f}<extra></extra>",
    ))

    if subsidy > 0 and values:
        fig.add_annotation(
            text=f"Govt Subsidy: -Rs{subsidy:,.0f}<br>Net Cost: Rs{net_cost:,.0f}",
            x=max(values) * 0.5,
            y=len(labels) - 0.5,
            showarrow=False,
            font=dict(size=13, color=COLORS["solar_green"]),
            bgcolor=COLORS["card_bg"],
            bordercolor=COLORS["solar_green"],
            borderwidth=1,
            borderpad=6,
        )

    fig = _apply_base_layout(fig, "Installation Cost Breakdown",
                              height=380, show_legend=False)
    fig.update_xaxes(title_text="Cost (Rs)", gridcolor=COLORS["border"])
    fig.update_yaxes(showgrid=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 7: Cumulative Savings Forecast
# ═══════════════════════════════════════════════════════════════════════════════
def chart_savings_forecast(yearly_data: list, net_cost: float, years: int = 15) -> go.Figure:
    """Line chart showing cumulative profit over time with payback highlighted."""
    if not yearly_data:
        yearly_data = [
            {"year": y, "cumulative_savings": y * 20000,
             "cum_profit": y * 20000 - net_cost}
            for y in range(1, years + 1)
        ]

    data = yearly_data[:years]
    yrs  = [d["year"] for d in data]
    cum  = [d["cumulative_savings"] for d in data]
    prof = [d["cum_profit"] for d in data]

    fig = go.Figure()

    fig.add_hline(y=0, line=dict(color=COLORS["border"], width=1, dash="dot"))

    if net_cost > 0:
        fig.add_hline(
            y=net_cost,
            line=dict(color=COLORS["amber"], width=1, dash="dash"),
            annotation_text=f"Investment: Rs{net_cost:,.0f}",
            annotation_font=dict(color=COLORS["amber"]),
        )

    fig.add_trace(go.Scatter(
        x=yrs, y=cum,
        name="Cumulative Savings",
        fill="tozeroy",
        fillcolor="rgba(244,168,38,0.15)",
        line=dict(color=COLORS["solar_gold"], width=3),
        mode="lines+markers",
        marker=dict(size=7),
        hovertemplate="Year %{x}<br>Savings: Rs%{y:,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=yrs, y=prof,
        name="Net Profit",
        line=dict(color=COLORS["solar_green"], width=2, dash="dash"),
        mode="lines",
        hovertemplate="Year %{x}<br>Profit: Rs%{y:,.0f}<extra></extra>",
    ))

    for d in data:
        if d["cum_profit"] >= 0:
            fig.add_vline(
                x=d["year"],
                line=dict(color=COLORS["solar_green"], width=2, dash="dot"),
                annotation_text=f"Payback (Yr {d['year']})",
                annotation_font=dict(color=COLORS["solar_green"]),
                annotation_position="top right",
            )
            break

    fig = _apply_base_layout(fig, f"{years}-Year Financial Forecast", height=400)
    fig.update_xaxes(title_text="Year", gridcolor=COLORS["border"], dtick=2)
    fig.update_yaxes(title_text="Amount (Rs)", gridcolor=COLORS["border"],
                     tickformat=",.0f")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 8: ROI Dashboard (multi-panel KPI)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_roi_kpi(
    payback_years: float,
    roi_pct: float,
    profit_10yr: float,
    irr_pct: Optional[float],
) -> go.Figure:
    """4-panel KPI indicator chart."""
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}] * 4],
    )

    kpis = [
        ("Payback Period", float(payback_years), " yrs", COLORS["solar_gold"]),
        ("Total ROI",      float(roi_pct),        "%",    COLORS["solar_green"]),
        ("10-Year Profit", float(profit_10yr) / 1000, "K Rs", COLORS["sky_blue"]),
        ("IRR",            float(irr_pct or 0),   "%",    COLORS["purple"]),
    ]

    for i, (title, val, suf, color) in enumerate(kpis, 1):
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=val,
                number=dict(
                    suffix=suf,
                    font=dict(size=34, color=color),
                    valueformat=".1f",
                ),
                title=dict(
                    text=f"<b>{title}</b>",
                    font=dict(size=14, color=COLORS["text_muted"]),
                ),
            ),
            row=1, col=i,
        )

    fig.update_layout(
        **_BASE_LAYOUT,
        showlegend=False,
        height=200,
        title=dict(
            text="Financial KPIs",
            font=dict(size=16, color=COLORS["solar_gold"]),
            x=0.01,
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 9: Score Breakdown Radar
# ═══════════════════════════════════════════════════════════════════════════════
def chart_score_radar(score_breakdown: dict) -> go.Figure:
    """Radar chart showing suitability score components."""
    categories = ["GHI Score", "Cloud Score", "Roof Score", "Budget Score", "Location"]
    values = [
        float(score_breakdown.get("ghi_score", 0)),
        float(score_breakdown.get("cloud_score", 0)),
        float(score_breakdown.get("roof_score", 0)),
        float(score_breakdown.get("budget_score", 0)),
        float(score_breakdown.get("loc_score", 0)),
    ]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed     = values + [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(244,168,38,0.25)",
        line=dict(color=COLORS["solar_gold"], width=2),
        marker=dict(color=COLORS["solar_gold"], size=8),
        name="Score",
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        showlegend=False,            # set here, NOT in _BASE_LAYOUT
        polar=dict(
            bgcolor=COLORS["card_bg"],
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color=COLORS["text_muted"],
                gridcolor=COLORS["border"],
                tickfont=dict(size=9, color=COLORS["text_muted"]),
            ),
            angularaxis=dict(color=COLORS["text_primary"]),
        ),
        height=350,
        title=dict(
            text="Score Breakdown",
            font=dict(size=16, color=COLORS["solar_gold"]),
            x=0.01,
        ),
    )
    return fig
