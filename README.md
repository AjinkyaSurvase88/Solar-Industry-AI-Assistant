# Solar AI Assistant

A professional AI-powered solar feasibility analysis tool built for India, with a global vision.

## Data Source
- **NASA POWER API** (https://power.larc.nasa.gov/) — Public Domain, no key required
- 80 cities worldwide (50 India + 30 global), 2000–2023 (24 years)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build dataset from NASA POWER API (~10-15 min, runs once)
python data/build_dataset.py

# 3. Train ML models (~2-3 min)
python src/train_model.py

# 4. Launch the app
streamlit run app.py
```

## Optional: Add API Keys
Copy `.env` and fill in:
- `GEMINI_API_KEY` — for AI chatbot (free at aistudio.google.com)
- `OPENWEATHER_API_KEY` — for enhanced live weather (free tier)

The app works fully without any API keys using Open-Meteo + NASA POWER fallbacks.

## Project Structure
```
solar_prediction/
├── app.py                  # Main Streamlit app
├── data/
│   ├── build_dataset.py    # NASA POWER data pipeline
│   ├── solar_dataset.csv   # Generated training data
│   └── city_coordinates.csv
├── models/
│   ├── solar_prediction.pkl
│   └── roi_prediction.pkl
├── src/
│   ├── location_service.py
│   ├── weather_service.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predictor.py
│   ├── roi_calculator.py
│   ├── visualizer.py
│   └── ai_assistant.py
└── requirements.txt
```

## Contributors
- **Ajinkya Survase** - [ajinkyasurvase13@gmail.com](mailto:ajinkyasurvase13@gmail.com)
- **Gajanan Vishwamitre** - [vishwamitregajanan@gmail.com](mailto:vishwamitregajanan@gmail.com)
