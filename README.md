# Energy Market Analyzer

**A Kedro pipeline that forecasts Belgian day-ahead electricity prices at 11:00, the morning before the auction, explains every forecast, and turns it into a load-shifting plan for an industrial buyer.**

> Status: pipelines, tests and results are final. The six notebooks (figures) are in progress; this README will then be extended.

---

## 1. Why This Project Matters

**The price volatility problem.**
Belgian electricity prices swung from ~40 €/MWh (2019) to over 700 €/MWh (August 2022 energy crisis), then spiked again to ~150 €/MWh (March 2026, Iran-USA conflict and Strait of Hormuz closure). Industrial consumers buying ~100 GWh/year on the spot market absorb these shocks without any anticipation tool.

**The anticipation problem.**
The day-ahead auction closes at 12:00 and its prices are published around 12:45. A forecast made after that has no value. Every forecast here is made at **11:00 on the day before delivery, with only the information available at 11:00**: this is enforced in the code and checked by tests that inject leaks on purpose.

**The context problem.**
150 €/MWh is ordinary in a crisis and exceptional in a calm market. The pipeline detects the market regime (calm / volatile / crisis) and judges alerts and anomalies against the recent history of that regime.

**The trust problem.**
Every published forecast is explained with exact SHAP values: the procurement team sees which known facts (today's prices, gas and carbon, load forecast, hour, calendar) moved tomorrow's price, in €/MWh.

---

## 2. Problem Statement

A Belgian industrial client buys 100 GWh/year on the day-ahead market. 10% of its consumption is flexible: at 11:00 it can move energy within the next day, by at most 4 hours, at a cost of 5 €/MWh moved.

**Business KPI:** euros saved per year by shifting the flexible load, measured against what the client does without the project (a free naive rule) and against perfect foresight (the ceiling).

**Evaluation protocol:** monthly walk-forward (each month predicted by models trained only on earlier delivered days), 69 development months (2020-01 to 2025-09) and a **12-month reserved period (2025-10 to 2026-09) never used for any choice**. Every rule (model choice, number of trees, interval calibration, alert classes) was written before the run and applied on the development period only.

---

## 3. Solution Overview

1. **Regime detection** — daily Hidden Markov Model (3 states), refitted monthly. The *filtered* regime (past only) feeds the models; the *smoothed* regime is kept for retrospective analysis.
2. **Anomaly detection** — one Isolation Forest per regime, refitted monthly on the previous 24 months; flagged hours are grouped into events.
3. **Price forecasting** — Model A: the 24 hourly prices of J+1. Model B: the daily mean of J+2 to J+4. XGBoost predicts the gap to today's mean price; candidates with and without the regime are compared, and the code keeps the simplest one that is not significantly worse (Diebold-Mariano test with HAC variance). 80% intervals: LightGBM quantiles recalibrated daily by conformal prediction.
4. **Alerts** — `high` / `critical` when the forecast exceeds the 80th / 95th percentile of the last 24 months in the current regime, with a probability and a confidence class; each hour is ranked within its day.
5. **Backtest** — exact daily optimizer (linear programming) of the flexible load, compared with the naive rule and perfect foresight.
6. **Explanations and morning report** — exact TreeSHAP of every published forecast, split into the level of the day and the shape of the day; a morning report that can be replayed for any past day.

**Morning report (real output, Saturday 26 September 2026, 11:00):**
```
Market today: regime crisis; mean price 152.9 €/MWh (19.5 to 236.8)
Tomorrow (Sunday): mean predicted price 117.4 €/MWh
  = 152.9 (mean price today) + 0.3 (expected gap)
    - 20.5 (mean price today) + 9.0 (same hour today) - 7.2 (load forecast) - 17.1 (other)
Alerts: 18:00 to 23:00 high (e.g. 19:00: 204.7 €/MWh, 157.3 to 226.3, medium confidence)
Cheapest hours: 14:00, 13:00, 12:00 (12.0 to 21.5 €/MWh: solar)
Schedule: 14.8 MWh moved; differs from the naive schedule at 18:00 and 23:00
```

---

## 4. Pipeline Architecture

```
ENTSO-E ────┐
Yahoo ──────┤
EEX ────────┤── ingestion ── features ──┬── regime ── anomaly
Open-Meteo ─┘                           └── prediction ── backtest ── interpretability
```

Seven Kedro pipelines, each with its own tests. Every report has a `choices_before_holdout` section (the only one used for decisions) and a `holdout_descriptive_only` section.

```bash
kedro run                                   # full chain (ingestion downloads ~1 h)
kedro run --pipeline=prediction             # one part
kedro run --pipeline=interpretability --nodes=morning_report_node \
  --params="interpretability.morning_report.decision_day=2026-01-15"   # replay a morning
```

---

## 5. Data Sources

| Source | Tool | Data | Resolution | Period | API key |
|--------|------|------|------------|--------|---------|
| ENTSO-E Transparency Platform | `entsoe-py` | Day-ahead prices, actual load, day-ahead load forecast, generation by type, cross-border flows with FR, NL, DE and GB | 15 min or hourly | 2018–2026 | Yes (free) |
| Yahoo Finance | `yfinance` | Dutch TTF natural gas, front-month future (`TTF=F`), EUR/MWh | Daily | 2018–2026 | No |
| EEX | `requests` | EU allowance (EUA) primary auction price, EUR/t | Auction days | 2018–2026 | No |
| Open-Meteo (ERA5) | `requests` | Observed temperature, wind, radiation, cloud cover (Brussels) | Hourly | 2018–2026 | No |

76,679 hours from 1 January 2018 to 30 September 2026. Observed weather is only used for the retrospective analysis: it is not known at 11:00 the day before.

---

## 6. Key Results

All figures below are on the **12-month reserved period** (October 2025 to September 2026), read after every choice was frozen.

**Forecasts**

| | Model | Naive baseline | Improvement |
|---|---|---|---|
| Model A, hourly J+1 (MAE) | 22.9 €/MWh | 28.5 €/MWh (same hour today / a week before) | **−19.5%** (p < 0.001) |
| Model B, daily J+2 to J+4 (MAE) | 21.5 €/MWh | 32.0 €/MWh (today's mean) | **−33%** |

- Calibrated 80% intervals cover **79.4%** of the realized prices (raw LightGBM quantiles: 65% for model A, 73% for model B).
- Development period (2020-01 to 2025-09): Model A beats the naive baseline by 23% and a ridge regression on the same features.
- Regimes: the regime-aware variants do not beat the simpler model for A (the selection keeps `no_regime`); they help B a little (`global`).

**Business value (backtest, 100 GWh/year client)**

| Strategy | Savings over 12 months | Share of perfect foresight |
|---|---|---|
| Optimizer on the model's forecast | **218,734 €** | 90.8% |
| Optimizer on the naive forecast | 215,378 € | 89.4% |
| Perfect foresight (ceiling) | 240,987 € | 100% |

- The value comes from **shifting the load itself** (~220 k€/year, 2.2% of the client's bill).
- The model's own contribution is small: +3,365 €/year with the optimizer (not significant, p = 0.054), **+11,386 €/year** with the simpler alert rule (p < 0.001), the same gain as in development. Today's prices already reveal most of tomorrow's shape.

**Explanations (SHAP, development period)**

- Level of tomorrow vs today: driven by the clean spark spread and today's mean price (prices revert towards the level set by gas and carbon costs).
- Shape of the day: driven by the hour, today's price at the same hour and the load forecast.
- During the 2021-2022 crisis, prices and fuels dominate; since 2023, the hour of day (the solar shape).

---

## 7. How to Run

### Prerequisites

- Python 3.12
- A free ENTSO-E API key: create an account on the
  [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) and request
  access to the Restful API. The key (a UUID) then appears in your account settings.

### Installation

```bash
git clone https://github.com/Zekhayoub/energy-market-analyzer.git
cd energy-market-analyzer
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install                 # ruff and the secret check run on every commit
```

### ENTSO-E API key

Create `conf/local/credentials.yml` (git ignores everything in `conf/local/`) with a single line:

```yaml
entsoe_api_key: "<your key>"
```

Never paste the key anywhere else (notebook, log, issue, chat). The pipeline masks it in every log and error message, and a pre-commit hook blocks any commit that contains it.

### Run

```bash
pytest          # 68 tests, no network call
kedro run       # full chain: ~1 h of download, then ~40 min of computation
```

---

## 8. Limitations

- **No wind or solar forecast.** They are only guaranteed after 11:00 the day before, so the models do not use them. The forecasts are a floor; the models over-predict by 8 to 10 €/MWh around midday, when solar pushes prices down.
- **Data revisions.** ENTSO-E keeps the latest version of the load forecast and realized data, not the one available at 11:00.
- **Monthly retraining.** A model refitted every day would be the standard; daily recalibrated intervals partly compensate.
- **Hourly decisions.** Since October 2025 prices are quarter-hourly; the project works on hourly means.
- **Gas prices from Yahoo Finance.** Free but unofficial; production would need licensed data (ICE Endex, EEX).
- **Regime labels before the first crisis** are relative to the calm past: in 2021, "crisis" means "the most agitated days seen so far".
