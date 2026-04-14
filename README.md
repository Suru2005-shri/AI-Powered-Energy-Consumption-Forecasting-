# ⚡ AI-Powered Energy Consumption Forecasting System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> Predict hourly electricity consumption using machine learning to support smart cities, grid operators, and building energy managers.

---

## Problem Statement

Power grids worldwide struggle to balance supply and demand in real time:

| Problem | Impact | This Project's Solution |
|---------|--------|------------------------|
| Unpredictable demand | Blackouts / wastage | 24h AI forecast |
| Peak-hour overload | High cost penalties | Pattern recognition |
| Manual monitoring | Slow, error-prone | Automated pipeline |
| Carbon emissions | Climate damage | Optimised dispatch |

---

## Industry Relevance

Companies actively working on AI energy forecasting:

**Product-based:** Google DeepMind (AlphaFold for energy), Tesla Autobidder, Siemens EnergyIP, GE Digital, Schneider Electric EcoStruxure

**Service-based:** TCS Smart Grid solutions, Infosys Energy Practice, Wipro EnergyCentral, Accenture Energy Analytics, L&T Technology Services

---

## Architecture

```
Raw Data → Preprocess → Feature Engineering → Train Model
                                                    ↓
Forecast Charts ← Visualise ← Evaluate ← Predict
```

See `docs/architecture.md` for the full system diagram.

---

## Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| ML Model | scikit-learn RandomForestRegressor |
| Visualisation | Matplotlib, Seaborn |
| Persistence | Joblib |
| Notebook | Jupyter |

---

## Dataset

**UCI Household Electric Power Consumption**
- 2+ million minute-level readings (2006–2010)
- Resampled to hourly (≈ 35,000 rows)
- Download: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption) or [Kaggle](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)

**No dataset?** Run the synthetic generator:
```bash
python main.py --generate
```

---

## Project Structure

```
ai-energy-forecasting/
├── data/
│   ├── raw/          ← original dataset (gitignored)
│   └── processed/    ← clean_data.csv, features.csv
├── src/
│   ├── __init__.py
│   ├── generate_data.py   ← synthetic dataset generator
│   ├── preprocess.py      ← load & clean
│   ├── features.py        ← feature engineering
│   ├── model.py           ← train & save model
│   ├── evaluate.py        ← metrics & feature importance
│   ├── forecast.py        ← future predictions
│   └── visualize.py       ← chart generation
├── models/
│   └── rf_model.pkl       ← saved model
├── outputs/
│   ├── metrics.txt
│   ├── predictions.csv
│   └── images/            ← 5 PNG charts
├── notebooks/
│   └── EDA_and_Demo.ipynb
├── docs/
│   └── architecture.md
├── main.py
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-energy-forecasting.git
cd ai-energy-forecasting

# 2. Create virtual environment
python -m venv venv

# 3. Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline

```bash
# With synthetic data (no download needed)
python main.py --generate

# With UCI dataset (place in data/raw/ first)
python main.py

# 7-day forecast
python main.py --hours 168

# Skip chart generation
python main.py --no-plots
```

### Interactive notebook

```bash
jupyter notebook notebooks/EDA_and_Demo.ipynb
```

---

## Results

| Metric | Value |
|--------|-------|
| RMSE | ~0.30 kWh |
| MAE | ~0.21 kWh |
| R² Score | ~0.88 |
| Forecast horizon | 24 hours |
| Top feature | lag_1h (consumption 1 hr ago) |

---

## Output Charts

| Chart | Description |
|-------|-------------|
| `eda_overview.png` | Consumption by hour / day / month |
| `actual_vs_predicted.png` | Model accuracy over 2-week test window |
| `feature_importance.png` | Which features drive the model |
| `residuals.png` | Error distribution + predicted vs actual scatter |
| `forecast_24h.png` | Next 24h demand forecast with confidence band |

---

## Features Engineered

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0–23) |
| `day_of_week` | Day number (0=Mon, 6=Sun) |
| `month` | Month (1–12) |
| `is_weekend` | Binary weekend flag |
| `lag_1h` | Power consumption 1 hour ago |
| `lag_24h` | Power consumption 24 hours ago (same time yesterday) |
| `lag_168h` | Power consumption 168 hours ago (same time last week) |
| `rolling_mean_24h` | Rolling 24h average |
| `rolling_std_24h` | Rolling 24h standard deviation |

---

## Learning Outcomes

- Time-series data handling and hourly resampling
- Lag and rolling feature engineering for temporal data
- Chronological train/test splitting (critical for time-series!)
- Random Forest training, persistence, and inference
- Iterative multi-step forecasting
- Evaluation metrics: RMSE, MAE, R², MAPE
- Professional Python project structure
- GitHub-ready documentation and commit strategy

---

## Why This Project Stands Out

- **Real industry simulation** — mimics how grid operators use 24h ahead forecasting
- **Clean modular code** — each pipeline stage in its own module
- **Two data modes** — works with real UCI data or synthetic fallback
- **5 publication-quality charts** — ready for resume and LinkedIn
- **Beginner-friendly** — every function is fully commented

---

## License

MIT © 2024

---

## Author

Built as a portfolio project demonstrating applied ML in the energy / smart-city domain.
