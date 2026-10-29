# WellPulse

**A soft sensor that estimates downhole pressure from surface and operational data, with an interactive dashboard.**

Downhole conditions are hard to measure directly, so engineers estimate key variables from operating data. WellPulse trains a gradient-boosting model on the public Equinor Volve dataset to estimate bottom-hole pressure from wellhead measurements, choke settings, production volumes and well history. It flags days the model cannot explain, and presents everything in a Streamlit dashboard.

> Independent portfolio project built on public data. It is not an entry to any competition.

<!-- Add your screenshots to the screenshots/ folder, then uncomment: -->
<!-- ![Well explorer](screenshots/well_explorer.png) -->
<!-- ![What-if prediction](screenshots/what_if.png) -->

## Results

Target: `AVG_DOWNHOLE_PRESSURE` (bar). Model: `HistGradientBoostingRegressor` with default settings (no hyperparameter tuning).

| Evaluation | MAE (bar) | RMSE (bar) | R² |
|---|---|---|---|
| Per-well time split (first 80% of days train, last 20% test) | 8.09 | 11.13 | 0.75 |
| Leave-one-well-out (mean over 5 wells) | n/a | 16.5 | n/a |

Baselines on the same time split: always predicting the global mean gives RMSE 28.6 bar, predicting each well's training mean gives 21.4 bar.

Feature history matters. Adding well history features (days on production, cumulative oil and water, water cut) improved the time-split R² from 0.63 to 0.75 and the leave-one-well-out RMSE from 20.5 to 16.5 bar.

**What these numbers do and do not show.** The model estimates pressure on later days of wells it has seen, and generalises partially to a new well. Results vary by well (leave-one-well-out RMSE ranges from about 13 to 22 bar), and with only five wells and a single split they should be read as indicative, not as a guarantee.

## Data

[Equinor Volve Data Village](https://www.equinor.com/energy/volve-data-sharing), daily production sheet: 15,634 rows, 7 wells (5 production, 2 injection), 2007 to 2016. After cleaning, 5,949 rows from 5 production wells remain.

The raw data is not included in this repository. Download it from the official Equinor page (Equinor Open Data Licence) and place `Volve_production_data.xlsx` in `data/`.

Cleaning rules: keep days with `ON_STREAM_HRS > 0`, a real gauge reading (`AVG_DOWNHOLE_PRESSURE > 0`), no negative water volume, and no missing values in target or features. Injection wells drop out because they have no downhole pressure reading.

## Method

**Features.** Wellhead pressure and temperature, choke size, hours on stream, allocated oil, gas and water volumes, plus history features: days since first production, cumulative oil, cumulative water, water cut.

**Leakage control.** Two columns are excluded on purpose:
- `AVG_DP_TUBING` is the difference between downhole and wellhead pressure (correlation with `AVG_DOWNHOLE_PRESSURE − AVG_WHP_P` is 0.9996), so using it would leak the target.
- `AVG_DOWNHOLE_TEMPERATURE` comes from the same downhole gauge as the target and would not be available when that gauge fails.

**Validation.** Random train/test splits are misleading here because pressure changes slowly (median day-to-day change 0.39 bar) and neighbouring days are nearly identical. Pressure also drifts upward over the years. Two stricter checks are used: a per-well chronological split, and leave-one-well-out.

**Anomaly flags.** Out-of-fold predictions are computed with five consecutive time blocks per well. A day is flagged when its residual is more than 3 robust standard deviations (MAD-based, 7.29 bar) from the median residual. 381 days (6.4%) are flagged. The flags overlap with known data problems: for example, 75 days with wellhead pressure above 100 bar and 60 days with a daily jump above 36 bar were checked, and about 37% of them are flagged, against 6.4% overall.

**Model interpretation (hypothesis, not proof).** Permutation importance ranks water cut, cumulative oil and wellhead pressure highest. Higher water cut makes the fluid column heavier, which is consistent with higher downhole pressure, but this has not been tested beyond the importance ranking.

## Dashboard

- **Well explorer:** measured vs predicted pressure per well, flagged days, residual plot.
- **What-if prediction:** set operating conditions, get a pressure estimate.
- **Metrics:** the evaluation table above.
- **Limitations:** summary of the caveats below.

## Limitations

- Only five production wells. Generalisation to a new well is partial and uneven.
- Oil, gas and water volumes are **allocated**, not directly measured, and the history features are built from them.
- History features need the well's production record from the first day.
- Pressure drifts over the years, so later periods are harder to predict.
- Anomaly flags are residual-based and unlabelled, so precision and recall cannot be reported. A flag means "the model cannot explain this day". It is not a fault diagnosis: the cause may be a sensor error, a transient, a real event, or model weakness. F-12 H has a high flag rate (15.8%), most likely because its usable record is short.
- Single train/test split, untuned models. The blocked out-of-fold metrics (RMSE 12.33, R² 0.71) are used only for dashboard plots, since the blocks complement each other and do not represent a true forecast.
- The What-if tab does not extrapolate reliably outside the ranges seen in training.

## Run it locally

```bash
git clone https://github.com/<your-username>/wellpulse.git
cd wellpulse
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. download the Volve production data and put it in data/Volve_production_data.xlsx
# 2. train the model and build dashboard files
python -m src.train
# 3. launch the dashboard
streamlit run app.py
```

Training prints the metrics above and writes `models/hgb_v2.joblib`, `models/app_data.csv` and `models/metrics.json`. `app_data.csv` contains Volve values and is git-ignored.

## Project structure

```
wellpulse/
├── app.py               # Streamlit dashboard
├── requirements.txt
├── src/
│   ├── pipeline.py      # loading, cleaning, features, split, metrics
│   └── train.py         # training, out-of-fold residuals, anomaly flags
├── models/              # generated artifacts (metrics.json is tracked)
├── data/                # place the Volve file here (not tracked)
└── screenshots/
```

## Tech stack

Python, pandas, scikit-learn, Plotly, Streamlit, joblib.

## Acknowledgements

Data: Equinor Volve dataset, used under the Equinor Open Data Licence. The data is provided as is and may contain errors.
