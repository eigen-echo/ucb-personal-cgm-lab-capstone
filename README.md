# personal-cgm-lab

A personal glucose modeling project built on my own CGM data, South Indian meal log, and walking records. UC Berkeley ML/AI capstone, n=1.

---

## Why I built this

I'm a Type 2 diabetic. Ten months of continuous glucose monitoring and intermittent fasting brought my HbA1c from 10 down to 6.7. I want to reach 6.0, and I've found that generic diabetes apps don't model me well. My diet is rice, lentils, dosa, sambar, and curries. The standard glycemic-index tables were built around a Western plate.

So I built my own. This repo is the data scaffolding for that push: raw glucose readings joined with my food log, walk timing, medication schedule, and fasting windows, with a regression model on top that can actually answer the questions I have at dinner time.

It's n=1 by design. The methods could generalize; the numbers won't.

---

## What I'm trying to figure out

Four questions come up for me every day:

1. **Walk dose-response.** If a meal is going to push me over 180 mg/dL, is 25 minutes of walking enough or do I actually need 45?
2. **Dinner given lunch.** If lunch was rice-heavy, what does the data say I should eat for dinner to keep my daily average on target?
3. **Stable vs spiky dishes.** Which foods reliably keep me in range, and which ones blow me up even in small portions?
4. **A1C trajectory.** What combination of food choices and walk habits most consistently moves my daily mean glucose toward an A1C of 6.0 (roughly 125 mg/dL average)?

---

## Where the data comes from

**CGM.** I use the [Stelo by Dexcom](https://www.stelo.com) continuous glucose monitor, an over-the-counter sensor that runs about $45 per month. It records a glucose reading every five minutes and syncs to the Stelo app on my phone. The raw export from Stelo's web portal is the starting point for everything here.

**Walking.** I wear a Google Pixel Watch 2 throughout the day. It tracks activity through Fitbit and syncs that data to Google Health Connect on my phone. Stelo reads activity from Health Connect, so walk events show up alongside glucose readings in a single export. I cross-reference those with watch timestamps to get walk start time and duration.

**Meals.** This is the weakest part of the data. About 24 meals were logged directly in the Stelo app at the time I ate using their AI analyzer feature (take a photo and it identifies dish with an estimate which was likely an experimental feature that was shut down). The remaining ~270 were logged in a spreadsheet. It is imperfect and I know it.

**Medications.** Jardiance 10mg, Glipizide 5mg x2, and Crestor 10mg around 10 AM daily. Glipizide 5mg x2 again around 4 PM. The medication log is auto-generated as scheduled rows and I manually edit the record when I shift or miss a dose.

**Fasting windows.** I follow a roughly 12-hour overnight fast, 7 PM to 7 AM. On days when I skip breakfast and push the first meal to lunch, the window gets flagged as an intermittent fast automatically based on meal timing.

**Food nutrition.** I built a custom lookup table for the South Indian dishes I actually eat. Mainstream nutrition APIs don't have granular entries for dosa, rasam, or veg biryani as I prepare them, so I assembled estimates from a combination of published sources and portion-size notes.

---

## Status

Three months of data: February 1 through April 28, 2026. 22,178 five-minute readings.

| metric | value |
|---|---|
| mean glucose | 175.9 mg/dL |
| time-in-range (70-180) | 56.3% |
| GMI estimate | 7.52 |
| logged walks | 48 |
| meal events | 295 |

That GMI of 7.52 corresponds to an estimated HbA1c around 7.5. The goal is 6.0. The model needs to find the levers.

---

## Repo layout

```
.
├── data/
│   ├── raw/                        source CSVs, version-controlled
│   │   ├── cgm_5min.csv            5-min glucose readings
│   │   ├── meals_v2.csv            logged meal events
│   │   ├── activities.csv          walk events with duration
│   │   ├── medications.csv         scheduled doses, one row per dose
│   │   ├── fasting_windows.csv     overnight and IF windows
│   │   ├── food_lookup.csv         custom South Indian dish nutrition table
│   │   └── raw_export_stelo.csv    original Stelo export (not committed)
│   └── processed/                  feature matrices, rebuilt from raw
│       ├── per_meal_features.csv   one row per meal (target: peak_delta)
│       └── daily_features.csv      one row per physiological day (4 AM boundary)
├── scripts/
│   └── 00-build-features.py        reads data/raw/, writes data/processed/
├── notebooks/
│   ├── 00-exploratory-data-analysis.ipynb
│   └── 01-regression-model.ipynb
├── models/
│   ├── per_meal_model.joblib        fitted Ridge + GBM pipelines
│   └── per_meal_metrics.json        train/test RMSE, MAE, R2, walk dose-response
└── README.md
```

The CSVs in `data/raw/` are the editable layer. The processed feature matrices are fully reproducible from a single script run.

For a full column-by-column description of `per_meal_features` and `daily_features`, see [docs/data-dictionary.md](docs/data-dictionary.md).

---

## Quick start

```bash
# Build the feature matrices from raw CSVs
python scripts/00-build-features.py
```

Dependencies: pandas, numpy, scikit-learn, shap, matplotlib, seaborn, pyarrow (for parquet).

Then open the notebooks in order:

- [00-exploratory-data-analysis.ipynb](notebooks/00-exploratory-data-analysis.ipynb) — data quality checks, EDA, feature validation, stationarity
- [01-regression-model.ipynb](notebooks/01-regression-model.ipynb) — Ridge + GBM, SHAP, walk ROI counterfactuals, confounder-adjusted dish ranking

---

## Methods

**Feature engineering.** The build script joins CGM readings, activity, medications, and fasting windows to each meal event by timestamp proximity. Key derived features include 30 and 60-minute pre-meal glucose averages, glucose velocity in the 30 minutes before eating, walk minutes in 60/90/180-minute windows after the meal, minutes since the last Glipizide dose, and whether the meal broke a fasting window.

**Daily features** are aggregated on a physiological-day boundary (4 AM) so that an overnight fast stays on a single date rather than splitting across two calendar days. Lag and rolling window features are built with a deliberate shift to avoid leaking tomorrow's glucose into today's training features.

**Regression.** I fit two models on a temporal train/test split, holding out the last 21 days as a test set. No random shuffling; CGM readings are autocorrelated and shuffling would leak future context into training. A Ridge baseline with time-series cross-validation, and a HistGradientBoostingRegressor tuned with GridSearchCV. SHAP values from the gradient booster show which features are actually driving predictions.

**Counterfactuals for walk ROI.** For each held-out meal, I re-predict at fixed walk_min_within_90 values of 0, 15, 25, 30, 45, and 60 minutes while holding everything else constant. The resulting dose-response curve answers the "how many minutes do I actually need" question with an estimate from my own data rather than population averages.

**Confounder-adjusted dish ranking.** Raw peak_delta averages by dish are confounded: chai appears at breakfast when cortisol is rising, rice at lunch when I'm already walking more. The model controls for time-of-day, pre-meal glucose, fasting state, and recent walks. Ranking dishes by their mean residual (actual minus predicted) removes those confounders and gives a cleaner read on which foods are specifically driving spikes versus which ones happen to be eaten in high-spike contexts.

---

## What's next: SARIMAX on daily glucose

The per-meal regression answers the meal-level questions, but it can't tell me whether my overall trajectory is actually moving toward 6.0. That needs a time-series model on the daily data.

The plan is a SARIMAX model on `mean_glucose` (or `time_in_range_pct`) from `daily_features.csv`, with the behavioral columns as exogenous regressors:

- `total_walk_min` — total minutes walked that day
- `total_carbs` — sum of carb estimates across all meals
- `is_if_day` — whether I skipped breakfast and extended the fast to lunch
- `overnight_fast_hours` — length of the previous night's fast
- `doses_taken` — medication compliance for the day

The EDA notebook already checks stationarity (ADF and KPSS tests on the `mean_glucose` series) and confirms the 4 AM physiological-day boundary is the right aggregation to use. The series has enough structure that a seasonal component is worth testing, since weekends tend to look different from weekdays in my eating and walking patterns.

The goal isn't just to fit the series. I want to use the fitted model to run scenarios: given a week where I walk 45 minutes every day after lunch and keep total carbs under 80g, what does the model predict for my 7-day mean? That's the A1C trajectory question answered in a way that accounts for the autocorrelation in daily glucose rather than treating each day as independent.

---

## What I learned

After 5 months in the course, the modeling was the straightforward part. Getting the data into a shape the model could actually use was where most of the real work happened.

I came into this project thinking the hard problem was picking the right algorithm. For this first submission, it turned out the hard problem was much earlier: the raw Stelo export arrives in a format designed for their app, not for analysis. Timestamps are in one format, dates in another. Activity data comes through Health Connect with its own schema. Meal records are partially logged in the app and partially maintained outside in a ledger. Medication records needed to be generated from scratch and maintained manually. None of these sources were designed to be joined to each other.

The discipline that forced was something I wouldn't have learned from a clean classroom dataset off of kaggle. Every modeling decision downstream traces back to a data decision upstream. The 4 AM physiological-day boundary exists because I noticed overnight fasts were splitting across calendar dates and distorting the daily averages. The `carbs_g_final` priority chain exists because three different sources each have partial coverage and none is authoritative on its own. The `taken` column bug (boolean `True` stored as a string `'TRUE'` in one path, actual bool in another) was silent — the model ran fine, it just treated every meal as if no medication had ever been taken.

Sourcing your own data removes the safety net of a pre-cleaned dataset. There is no answer key. When something looks wrong, you have to decide whether the data is bad, the feature engineering is wrong, or the biology is actually doing something unexpected. That judgment call is most of what data work actually is, and I don't think I would have developed it the same way working on someone else's almost-already-tidy CSV.

---

## Honest limitations

**n=1.** Nothing here generalizes to other people. The methods are reusable; the coefficients are not.

**Sparse walk data.** Only 48 walks in three months. The dose-response curve above 30 minutes is largely extrapolation. I need more walks at varied durations before I can trust those estimates.

**Selection bias on walks.** I tend to walk after meals where I expect a spike, not randomly. Even controlling for carbs and pre-meal glucose, unmeasured expectation is correlated with the walk decision. A few weeks of coin-flip randomization would produce cleaner evidence than any regression on this dataset.

**Confounders not captured:** stress, sleep quality, hydration, exact medication onset time relative to eating. Their absence shows up as residual variance in the model.

---

## Glossary

**Clinical / CGM terms**

| acronym | full term | meaning |
|---|---|---|
| CGM | Continuous Glucose Monitor | A wearable sensor that measures interstitial glucose every few minutes without fingersticks |
| TIR | Time In Range | Percentage of CGM readings between 70 and 180 mg/dL; the standard target is >70% |
| GMI | Glucose Management Indicator | An estimated HbA1c derived from CGM mean glucose: `3.31 + 0.02392 × mean_mg_dL` (Bergenstal 2018) |
| A1C / HbA1c | Hemoglobin A1c | A lab test measuring average blood glucose over the past ~3 months; target for well-controlled T2D is <7% |
| IF | Intermittent Fasting | An eating pattern that restricts food to a defined window; here used to mean skipping breakfast and eating first at lunch |
| mg/dL | milligrams per deciliter | The glucose concentration unit used in the US |
| peak_delta | — | The rise in glucose from pre-meal baseline to peak within ~3 hours; the primary modeling target |

**Machine learning / statistics terms**

| acronym | full term | meaning |
|---|---|---|
| EDA | Exploratory Data Analysis | Initial investigation of a dataset through summary statistics and visualizations |
| RMSE | Root Mean Squared Error | Square root of the average squared prediction error; penalises large misses more than MAE |
| MAE | Mean Absolute Error | Average absolute difference between predicted and actual values, in the original unit (mg/dL here) |
| R² | Coefficient of determination | Proportion of variance in the target explained by the model; 1.0 is perfect, 0 means the model does no better than predicting the mean |
| SHAP | SHapley Additive exPlanations | A method from cooperative game theory that assigns each feature a contribution value for each individual prediction |
| GBM | Gradient Boosting Machine | An ensemble of decision trees built sequentially; here specifically `HistGradientBoostingRegressor` from scikit-learn |
| SARIMAX | Seasonal AutoRegressive Integrated Moving Average with eXogenous variables | A time-series forecasting model that accounts for trend, seasonality, autocorrelation, and external regressors |
| ACF | AutoCorrelation Function | Measures correlation between a time series and its own lagged values; used to identify the MA order |
| PACF | Partial AutoCorrelation Function | Like ACF but removes indirect correlations through intermediate lags; used to identify the AR order |
| ADF | Augmented Dickey-Fuller test | A statistical test for whether a time series is stationary (no unit root); p < 0.05 means stationary |
| AR / MA | AutoRegressive / Moving Average | The two core components of ARIMA: AR uses past values, MA uses past forecast errors |
| CV | Cross-Validation | Technique for estimating model performance by splitting data into multiple train/validation folds |
| DOW | Day of Week | |

---

## Disclaimer

This is a personal research project for an academic capstone. It is not medical advice, not a clinical tool, and not validated for use by anyone other than me. If you are diabetic, talk to your endocrinologist.

---

## License

MIT (code). The data files in `data/*` are personal health records belonging to the author and are not shared under any open data license. Including it here for completeness of my capstone project. If you want to use this, please email the author to seek permission. 
