# Energy Market Analyzer


**A Kedro-based pipeline for Belgian electricity market regime detection, price prediction, and procurement alert generation.**

---

## 1. Why This Project Matters

**The Price Volatility Problem.**
Belgian electricity prices swung from ~40 €/MWh (2019) to over 700 €/MWh (August 2022 energy crisis), then spiked again to ~150 €/MWh (March 2026, Iran-USA conflict and Strait of Hormuz closure). Industrial consumers purchasing ~100 GWh/year on the spot market absorb these shocks without any anticipation tool.

**The Anticipation Problem.**
The official day-ahead price is published by EPEX SPOT at 12:45 CET. Any prediction made *after* publication has zero value. This pipeline produces its morning report at 11:00 CET — giving the procurement team 1h45 of actionable lead time before the market reveals the price.

**The Context Problem.**
A price of 150 €/MWh is perfectly normal during an energy crisis but highly anomalous during a calm market. Traditional threshold-based alerts ignore this. This pipeline detects the current market regime (calm / volatile / crisis) and contextualizes every alert within its regime — a 150 €/MWh hour in a calm regime triggers a critical alert, the same price in a crisis regime is flagged as normal.

**The Trust Problem.**
ML models are black boxes for procurement teams. This pipeline explains every prediction using SHAP: "price is high because TTF gas rose +12% and wind production dropped 40%." The team acts on reasons, not on opaque scores.

---

## 2. Problem Statement

A Belgian industrial client consuming ~100 GWh/year purchases electricity on the European day-ahead spot market. The client has a steelmaking profile: 10% of consumption (arc furnaces, grinders) is flexible and can be shifted within a ±4-hour window without impacting production.

**Business KPI:** € saved per year by shifting 10% of flexible consumption from critical-alert hours to calm-regime hours.

**Target performance (validated by backtest):**
- Model A (day-ahead hourly, J+1): MAE ... €/MWh
- Model B (daily average, J+2 to J+4): gradual degradation
- Estimated annual savings: ~... k€/year for the reference client

---

## 3. Solution Overview

The pipeline delivers five capabilities:

1. **Regime Detection** — Identifies the current market state (calm / volatile / crisis) using a Hidden Markov Model with monthly expanding window.
2. **Anomaly Detection** — Flags abnormal prices *within their regime context* using per-regime Isolation Forests.
3. **Price Prediction** — Forecasts 24 hourly prices for J+1 (Model A) and daily averages for J+2 to J+4 (Model B) using 4 XGBoost variants with LightGBM quantile confidence intervals.
4. **Interpretation** — Explains each prediction with SHAP TreeExplainer: global importance, per-regime importance, and force plots on 4 key economic moments (COVID, 2022 crisis, calm return, Iran 2026).
5. **Actionable Alerts** — Generates procurement signals combining alert level (normal / high / critical), confidence (from CI width), and top-3 SHAP drivers per hour.

**Morning report example (produced at 11:00 CET):**
```
Date: Monday 11:00
Predictions for Tuesday (J+1):
  Average price: 87 €/MWh (interval 72–103)
  00h–06h: 65 €/MWh  → CHEAPEST
  12h–18h: 105 €/MWh → MOST EXPENSIVE
  Top 8 hours to shift to: 03h, 04h, 05h, 02h, 06h, 01h, 23h, 00h
  Top 4 hours to avoid: 17h, 18h, 16h, 19h
  Regime: VOLATILE
  Alert: critical — 180 €/MWh (155–205) — high confidence
         drivers: TTF +40%, wind -60%, temperature -8%
```

---

## 4. Pipeline Architecture

```
ENTSO-E API ──┐
yfinance ─────┤── Ingestion ── Features ── HMM ── IF ── XGBoost ── SHAP ── Alerts
Open-Meteo ───┘       │           │         │      │       │         │        │
                  master.parquet  26 feat  3 reg  anom  4 var+LGB  explain  report
```

Built with **Kedro**. Each box is an independent sub-pipeline:

```
kedro run                          # full pipeline
kedro run --pipeline=ingestion     # only ingestion
kedro run --pipeline=inference     # production: predict + alert
kedro viz                          # interactive DAG visualization
```

---

## 5. Data Sources

| Source | Tool | Data | Resolution | Period | API key |
|--------|------|------|------------|--------|---------|
| ENTSO-E Transparency Platform | `entsoe-py` | Day-ahead prices, actual load, day-ahead load forecast, generation by production type, physical cross-border flows with FR, NL, DE and GB (both directions) | 15 min or hourly | 2018–2026 | Yes (free) |
| Yahoo Finance | `yfinance` | Dutch TTF natural gas, front-month future (`TTF=F`), EUR/MWh | Daily | 2018–2026 | No |
| EEX | `requests` | EU allowance (EUA) primary auction clearing price, EUR/t | Auction days | 2018–2026 | No |
| Open-Meteo (ERA5) | `requests` | Observed temperature, wind speed at 100 m, solar radiation, cloud cover (Brussels) | Hourly | 2018–2026 | No |

All data is free and downloaded by the pipeline: no manual download.

---

## 6. Models

### Model A — Day-ahead hourly prediction (J+1)

Predicts the 24 hourly prices for the next day, produced before the EPEX SPOT publication at 12:45 CET.

4 XGBoost variants compared:

| Variant | Strategy | When it works best |
|---------|----------|-------------------|
| Global | Single model on all data | Baseline — stable performance |
| Weighted | Global + sample weighting by regime | Better on rare regimes (crisis) |
| Per-regime | 3 specialized models (one per regime) | Best within each regime |
| Hybrid | Bayesian weighting by HMM confidence | Best during regime transitions |

### Model B — Daily average prediction (J+2 to J+4)

One parameterizable function `train_horizon_model(horizon=h)` producing 3 instances:
- h=2 → J+2 daily average
- h=3 → J+3 daily average
- h=4 → J+4 daily average


### Model C — Retrospective analysis

Not a prediction model. HMM detects regimes and Isolation Forest detects anomalies on the full historical dataset. Used for market understanding, not for alerts.

---

## 7. Key Results

> ** Results will be added after model training and backtesting.**

---

## 8. How to Run

### Prerequisites

- Python 3.12
- A free ENTSO-E API key: create an account on the
  [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) and request
  access to the Restful API. The key (a UUID) then appears in your account settings.

### Installation

```bash
# 1. Clone
git clone https://github.com/Zekhayoub/energy-market-analyzer.git
cd energy-market-analyzer

# 2. Create a virtual environment and install the exact versions
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Install the git hooks (ruff and the secret scan run on every commit)
pre-commit install
```

### ENTSO-E API key

Create the file `conf/local/credentials.yml` (git ignores everything in
`conf/local/`) with a single line:

```yaml
entsoe_api_key: "<your key>"
```

Never paste the key anywhere else (notebook, log, issue, chat). The pipeline
masks it in every log and error message, and gitleaks blocks any commit that
contains it.

### Run

```bash
kedro run --pipeline=ingestion     # download the sources, build data/03_primary/master.parquet
pytest                             # unit tests, no network call
```

---

## 9. Limitations

This section lists every known limitation of the project. Each point is a deliberate documentation choice, not an oversight.

---
