"""
AI Solar Assistant (LLM Integration)
=======================================
Conversational solar consultant powered by Google Gemini.
Falls back to a smart rule-based engine when no API key is set.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

# Fallback to Streamlit Secrets for cloud deployment
if not GEMINI_KEY or GEMINI_KEY == "your_gemini_api_key_here":
    try:
        import streamlit as st
        GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", GEMINI_KEY)
    except Exception:
        pass

try:
    import google.generativeai as genai
    HAS_GEMINI = bool(GEMINI_KEY and GEMINI_KEY != "your_gemini_api_key_here")
    if HAS_GEMINI:
        genai.configure(api_key=GEMINI_KEY)
except ImportError:
    HAS_GEMINI = False

# ─── System Prompt Template ───────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """
You are SolarBot, an expert AI solar energy consultant specializing in rooftop solar installations.
You are friendly, professional, and give practical, actionable advice.

USER'S SOLAR ANALYSIS CONTEXT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 Location: {city} ({lat:.2f}°N, {lon:.2f}°E)
☀️ Solar Irradiance (GHI): {ghi:.2f} kWh/m²/day
🌡 Temperature: {temperature:.1f}°C  |  ☁️ Cloud Cover: {cloud_cover:.0f}%

🏠 Property: {property_type}
📐 Roof Area: {roof_area} m²
⚡ Monthly Consumption: {monthly_consumption_kwh} kWh
💰 Monthly Bill: ₹{monthly_bill:,.0f}
🎯 Budget: ₹{budget:,.0f}

🔋 SYSTEM DETAILS:
   System Size: {system_size_kwp:.2f} kWp
   Monthly Solar Generation: {monthly_generation_kwh:.0f} kWh
   Solar Suitability Score: {suitability_score:.0f}/100 ({suitability_label})

💵 FINANCIAL ANALYSIS:
   Installation Cost: ₹{net_cost:,.0f} (after ₹{subsidy:,.0f} govt subsidy)
   Monthly Savings: ₹{monthly_savings:,.0f}
   Payback Period: {payback_years} years
   10-Year Profit: ₹{profit_10yr:,.0f}
   ROI: {roi_pct:.1f}%
   CO₂ Saved (25yr): {co2_tonnes:.1f} tonnes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GUIDELINES:
- Always reference the user's specific data in your answers
- Give practical, India-specific advice (PM Surya Ghar scheme, DISCOM policies, etc.)
- Mention government subsidies and net metering where relevant
- Keep responses concise (3-5 sentences) unless asked for detail
- Use emojis sparingly for readability
- If asked about global context, compare India's solar policies internationally
- Never make up data - use the context provided above
"""

# ─── Rule-Based Fallback ──────────────────────────────────────────────────────
RULE_BASED_RESPONSES = {
    "benefit": lambda ctx: (
        f"Based on your analysis, solar installation is {'highly' if ctx['suitability_score'] >= 70 else 'moderately'} "
        f"beneficial for you! ☀️\n\n"
        f"With a solar suitability score of **{ctx['suitability_score']:.0f}/100** in {ctx['city']}, "
        f"your {ctx['system_size_kwp']:.1f} kWp system will generate approximately "
        f"**{ctx['monthly_generation_kwh']:.0f} kWh/month**, covering "
        f"**{min(100, ctx['monthly_generation_kwh']/max(1,ctx['monthly_consumption_kwh'])*100):.0f}%** "
        f"of your electricity needs. You'll save **₹{ctx['monthly_savings']:,.0f}/month** and recover "
        f"your investment in just **{ctx['payback_years']} years**."
    ),
    "savings": lambda ctx: (
        f"Here's your 10-year savings breakdown 💰:\n\n"
        f"- **Monthly savings**: ₹{ctx['monthly_savings']:,.0f}\n"
        f"- **Annual savings**: ₹{ctx['monthly_savings']*12:,.0f}\n"
        f"- **10-year profit**: ₹{ctx['profit_10yr']:,.0f}\n"
        f"- **ROI**: {ctx['roi_pct']:.1f}%\n\n"
        f"With electricity prices rising ~6% annually in India, your savings will grow "
        f"significantly over time. By year 10, your monthly savings could be "
        f"**Rs{ctx['monthly_savings'] * (1.06 ** 10):,.0f}**!"
    ),
    "size": lambda ctx: (
        f"For your {ctx['roof_area']} m² roof and {ctx['monthly_consumption_kwh']} kWh/month consumption:\n\n"
        f"✅ **Recommended System**: **{ctx['system_size_kwp']:.1f} kWp**\n"
        f"- Panels needed: ~{int(ctx['system_size_kwp']/0.4)} panels (400W each)\n"
        f"- Area used: ~{int(ctx['system_size_kwp']/0.4 * 1.7)} m² (of your {ctx['roof_area']} m² roof)\n"
        f"- Inverter: {ctx['system_size_kwp']:.1f} kW hybrid inverter recommended\n\n"
        f"This system will cover **{min(100, ctx['monthly_generation_kwh']/max(1,ctx['monthly_consumption_kwh'])*100):.0f}%** "
        f"of your electricity needs."
    ),
    "maintenance": lambda ctx: (
        f"Solar panels require minimal maintenance 🔧:\n\n"
        f"- **Monthly**: Clean panels with water (dust reduces output by 10-25% in India)\n"
        f"- **Quarterly**: Check inverter status lights and connections\n"
        f"- **Annually**: Professional inspection (~₹{ctx['system_size_kwp']*500:,.0f}/year for your system)\n"
        f"- **Lifetime**: Panels degrade ~0.5%/year; inverter replacement in year 10-12 (~₹25,000)\n\n"
        f"Your estimated annual maintenance cost is ₹{int(ctx['system_size_kwp']*500):,}, "
        f"already factored into your financial projections."
    ),
    "subsidy": lambda ctx: (
        f"Great news on subsidies! 🏛️\n\n"
        f"Under the **PM Surya Ghar Muft Bijli Yojana** scheme:\n"
        f"- Systems up to 3 kWp: **30% central subsidy**\n"
        f"- Systems 3-10 kWp: **14% central subsidy**\n"
        f"- Additional state subsidies may apply\n\n"
        f"For your {ctx['system_size_kwp']:.1f} kWp system, you're eligible for approximately "
        f"**₹{ctx['subsidy']:,.0f}** in subsidies, bringing your net cost to "
        f"**₹{ctx['net_cost']:,.0f}** (down from ₹{ctx['net_cost']+ctx['subsidy']:,.0f}).\n\n"
        f"Apply through your local DISCOM or the National Portal for Rooftop Solar."
    ),
    "default": lambda ctx: (
        f"I'm here to help with all your solar energy questions! ☀️ "
        f"Based on your analysis for {ctx['city']}, your solar score is **{ctx['suitability_score']:.0f}/100** "
        f"and you could save **₹{ctx['monthly_savings']:,.0f}/month**. "
        f"Feel free to ask me about savings, system sizing, maintenance, subsidies, or anything solar-related!"
    ),
}


def _classify_question(question: str) -> str:
    """Simple keyword classifier for rule-based routing."""
    q = question.lower()
    if any(w in q for w in ["benefit", "worth", "good", "should i", "recommend"]):
        return "benefit"
    elif any(w in q for w in ["save", "saving", "money", "10 year", "profit", "earn"]):
        return "savings"
    elif any(w in q for w in ["size", "panel", "how many", "kw", "kwp", "watt"]):
        return "size"
    elif any(w in q for w in ["maintenance", "clean", "repair", "service", "maintain"]):
        return "maintenance"
    elif any(w in q for w in ["subsidy", "government", "scheme", "pm surya", "mnre"]):
        return "subsidy"
    return "default"


def _rule_based_response(question: str, context: dict) -> str:
    """Generate intelligent rule-based response using context."""
    category = _classify_question(question)
    handler  = RULE_BASED_RESPONSES.get(category, RULE_BASED_RESPONSES["default"])
    return handler(context)


def build_context(
    city: str = "Unknown",
    lat: float = 20.0,
    lon: float = 77.0,
    ghi: float = 5.0,
    temperature: float = 28.0,
    cloud_cover: float = 30.0,
    property_type: str = "Residential",
    roof_area: float = 50.0,
    monthly_consumption_kwh: float = 250.0,
    monthly_bill: float = 2000.0,
    budget: float = 200000.0,
    system_size_kwp: float = 3.0,
    monthly_generation_kwh: float = 400.0,
    suitability_score: float = 70.0,
    suitability_label: str = "Excellent",
    net_cost: float = 150000.0,
    subsidy: float = 45000.0,
    monthly_savings: float = 1800.0,
    payback_years: int = 7,
    profit_10yr: float = 100000.0,
    roi_pct: float = 85.0,
    co2_tonnes: float = 15.0,
) -> dict:
    """Build the context dictionary for the AI assistant."""
    return {k: v for k, v in locals().items()}


class SolarAIAssistant:
    """Conversational AI assistant for solar energy consultation."""

    def __init__(self):
        self.history = []
        self.context = {}
        self.model   = None
        self._init_gemini()

    def _init_gemini(self):
        if HAS_GEMINI:
            try:
                self.model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=512,
                    ),
                )
            except Exception:
                self.model = None

    def set_context(self, context: dict):
        """Load user analysis context into the assistant."""
        self.context = context

    def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        if self.model and HAS_GEMINI and self.context:
            return self._gemini_response(user_message)
        else:
            return self._fallback_response(user_message)

    def _gemini_response(self, message: str) -> str:
        """Get response from Gemini API."""
        try:
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(**self.context)

            # Build conversation with history
            if not self.history:
                # First message: include system prompt
                full_prompt = f"{system_prompt}\n\nUser: {message}"
                chat = self.model.start_chat(history=[])
                response = chat.send_message(full_prompt)
            else:
                # Continue conversation
                chat = self.model.start_chat(history=self.history)
                response = chat.send_message(message)
                self.history = chat.history

            text = response.text
            self.history = getattr(chat, 'history', self.history)
            return text

        except Exception as e:
            print(f"Gemini API Error: {e}")
            return self._fallback_response(message)

    def _fallback_response(self, message: str) -> str:
        """Rule-based response when Gemini is unavailable."""
        if not self.context:
            return ("Hello! I'm SolarBot ☀️ Please complete the Solar Analysis first "
                    "so I can give you personalized recommendations!")
        return _rule_based_response(message, self.context)

    def reset_chat(self):
        """Clear conversation history."""
        self.history = []

    @property
    def is_ai_powered(self) -> bool:
        return HAS_GEMINI and self.model is not None


def get_suggested_questions() -> list[str]:
    """Return a list of suggested questions for the chatbot UI."""
    return [
        "Is solar installation beneficial for my home?",
        "How much money will I save in 10 years?",
        "Which solar panel size should I choose?",
        "What are the maintenance requirements?",
        "Am I eligible for government subsidies?",
        "How does net metering work in India?",
        "What's the best time to install solar panels?",
        "How do I compare different solar brands?",
    ]
