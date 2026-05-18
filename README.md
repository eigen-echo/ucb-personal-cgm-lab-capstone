# personal-cgm-lab

A personal glucose modeling project built on my own CGM data, South Indian meal log, and walking records. UC Berkeley ML/AI capstone, n=1.

---

## Contents

- [Why I built this](#why-i-built-this)
- [What I'm trying to figure out](#what-im-trying-to-figure-out)
- [Where the data comes from](#where-the-data-comes-from)
- [Status](#status)
- [Key findings](#key-findings)
- [Repo layout](#repo-layout)
- [Quick start](#quick-start)
- [Methods](#methods)
- [SARIMAX time-series experiments](#sarimax-time-series-experiments)
- [Why not just run a plain regression on the hourly data?](#why-not-just-run-a-plain-regression-on-the-hourly-data)
- [What this actually means for me](#what-this-actually-means-for-me-plain-language)
- [What I learned](#what-i-learned)
- [Honest limitations](#honest-limitations)
- [Glossary](docs/glossary.md)
- [Disclaimer](#disclaimer)

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

## Key findings

These are the headline results from the per-meal regression model, fitted on 218 training meals and evaluated on 77 held-out meals from the last 21 days.

**Model performance.** The Ridge regression generalizes the best. On the held-out test set it achieves an RMSE of 34 mg/dL and an MAE of 26 mg/dL, explaining about 15% of variance in post-meal glucose excursions (R² = 0.15). The gradient booster fits training data tightly (train R² = 0.91) but does not generalize — its test R² is 0.10, nearly identical to Ridge on absolute error while badly overfitting. With 218 training rows and 24 features, that gap is expected. More data, especially more walked meals across varied carb loads, is the fix.

**Walk dose-response.** A 25-minute walk within 90 minutes of a meal reduces the predicted peak excursion by about 3.8 mg/dL compared to no walk, holding carb load and pre-meal glucose constant. Extending to 45 minutes adds no further predicted benefit — both models produce the same response curve above 25 minutes. Given only 22 walked meals in training, the dose-response above 30 minutes is extrapolation rather than a firm estimate. The directional answer to my dinner-time question is: 25 minutes appears to be the threshold, and more time doesn't buy more suppression in this dataset.

**Pre-meal glucose is the dominant predictor.** Both the Ridge coefficients and the SHAP summary agree: where my glucose starts before a meal is the strongest single driver of how high it peaks. Carb load and time of day follow behind it. Walk minutes matter but rank lower, which reflects the sparse walk data.

**Dish ranking (adjusted).** Using the model residuals to rank dishes removes the confounding from when and how I tend to eat each food. Chai and dosa tend to spike less than their raw averages suggest - they often appear at breakfast when pre-meal glucose happens to be lower. Meals logged as "salad" or veg biryani carry positive residuals, meaning they spike more than the model expects given their context — likely a logging artifact (hidden carbs in dressings, larger portions than assumed) rather than a property of the dish itself.

**Hourly SARIMAX.** The carb signal that was invisible at daily resolution became identifiable at hourly: `carb_load_decayed` (tau=75 min exponential decay) has a coefficient of +0.137 mg/dL per g-equivalent (p < 0.001). Multi-step RMSE across the 21-day holdout is 41 mg/dL vs 48 for the naive mean baseline. Walk effect is correctly signed but not significant given the sparse walk data (48 events in 86 days). The medication variable (`glipizide_active`) is confounded with the diurnal glucose pattern — Glipizide is taken during waking hours when glucose is naturally elevated — and cannot be cleanly identified. See the "SARIMAX time-series experiments" section below for the full analysis.

**Plain regression on hourly data is a trap.** A behavior-only regression barely beats guessing the mean (test RMSE 46 vs 48) with severely autocorrelated residuals. Adding "glucose one hour ago" inflates apparent accuracy (RMSE 24) but collapses to worse-than-baseline (RMSE 64) once it must forecast forward without the true recent reading. A gradient booster does no better and draws 80% of its predictions from the lag, not behavior. The headline takeaway for the individual: glucose is momentum-dominated, carbs are the one consistently trustworthy lever, and the per-meal model is the right tool for actual decisions. See "Why not just run a plain regression on the hourly data?" and "What this actually means for me" below.

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
│       ├── daily_features.csv      one row per physiological day (4 AM boundary)
│       └── hourly_features.csv     one row per hour (target: glucose_hourly_mean)
├── scripts/
│   ├── 00-build-features.py        reads data/raw/, writes per_meal + daily features
│   └── 01-build-hourly.py          reads data/raw/, writes hourly_features.csv
├── notebooks/
│   ├── 00-exploratory-data-analysis.ipynb
│   ├── 01-regression-model.ipynb
│   ├── 02-sarimax-model.ipynb      daily SARIMAX (contrast case)
│   ├── 03-sarimax-hourly.ipynb     hourly SARIMAX with glipizide comparison
│   └── 04-regression-hourly.ipynb  Ridge/GBM on hourly data (autocorrelation pitfall analysis)
├── models/
│   ├── per_meal_model.joblib        fitted Ridge + GBM pipelines
│   ├── per_meal_metrics.json        train/test RMSE, MAE, R2, walk dose-response
│   ├── sarimax_hourly.pkl           fitted SARIMAX(1,0,2)(1,0,1,24) model
│   ├── sarimax_hourly_meta.pkl      best-model metadata (order, AIC, column lists)
│   └── sarimax_hourly_metrics.json  hourly SARIMAX train/test RMSE, MAE, 21-day counterfactuals
└── README.md
```

The CSVs in `data/raw/` are the editable layer. The processed feature matrices are fully reproducible from a single script run.

For a full column-by-column description of all three feature matrices, see [docs/data-dictionary.md](docs/data-dictionary.md).

---

## Quick start

```bash
# Build the feature matrices from raw CSVs
python scripts/00-build-features.py
```

Dependencies: pandas, numpy, scikit-learn, shap, matplotlib, seaborn, pyarrow (for parquet).

Then open the notebooks in order:

- [00-exploratory-data-analysis.ipynb](notebooks/00-exploratory-data-analysis.ipynb) — data quality checks, EDA, feature validation, stationarity tests, hourly feature visualizations
- [01-regression-model.ipynb](notebooks/01-regression-model.ipynb) — Ridge + GBM, SHAP, walk ROI counterfactuals, confounder-adjusted dish ranking
- [02-sarimax-model.ipynb](notebooks/02-sarimax-model.ipynb) — daily SARIMAX (contrast case; shows why daily resolution fails)
- [03-sarimax-hourly.ipynb](notebooks/03-sarimax-hourly.ipynb) — hourly SARIMAX, carb signal recovery, glipizide confounding experiment, counterfactual scenarios
- [04-regression-hourly.ipynb](notebooks/04-regression-hourly.ipynb) — Ridge/GBM on hourly data; demonstrates the autocorrelation pitfall (no-lag vs lag, oracle vs recursive)

Build the hourly feature matrix before running notebook 03:
```bash
python scripts/01-build-hourly.py
```

---

## Methods

**Feature engineering.** The build script joins CGM readings, activity, medications, and fasting windows to each meal event by timestamp proximity. Key derived features include 30 and 60-minute pre-meal glucose averages, glucose velocity in the 30 minutes before eating, walk minutes in 60/90/180-minute windows after the meal, minutes since the last Glipizide dose, and whether the meal broke a fasting window.

**Daily features** are aggregated on a physiological-day boundary (4 AM) so that an overnight fast stays on a single date rather than splitting across two calendar days. Lag and rolling window features are built with a deliberate shift to avoid leaking tomorrow's glucose into today's training features.

**Regression.** I fit two models on a temporal train/test split, holding out the last 21 days as a test set. No random shuffling; CGM readings are autocorrelated and shuffling would leak future context into training. A Ridge baseline with time-series cross-validation, and a HistGradientBoostingRegressor tuned with GridSearchCV. SHAP values from the gradient booster show which features are actually driving predictions.

**Counterfactuals for walk ROI.** For each held-out meal, I re-predict at fixed walk_min_within_90 values of 0, 15, 25, 30, 45, and 60 minutes while holding everything else constant. The resulting dose-response curve answers the "how many minutes do I actually need" question with an estimate from my own data rather than population averages.

**Confounder-adjusted dish ranking.** Raw peak_delta averages by dish are confounded: chai appears at breakfast when cortisol is rising, rice at lunch when I'm already walking more. The model controls for time-of-day, pre-meal glucose, fasting state, and recent walks. Ranking dishes by their mean residual (actual minus predicted) removes those confounders and gives a cleaner read on which foods are specifically driving spikes versus which ones happen to be eaten in high-spike contexts.

---

## SARIMAX time-series experiments

The per-meal regression answers meal-level questions. Understanding whether my overall glucose trajectory is actually moving toward 6.0 requires a time-series model that captures how today's behaviors affect tomorrow's readings through the autocorrelation structure of daily glucose.

### Why daily resolution failed

The first attempt fit a SARIMAX on the 84 daily means in `daily_features.csv`. It collapsed immediately: a 30-minute walk occupies 2% of a day and is indistinguishable from noise at that granularity. The model produced a flat forecast that was barely better than the training mean. The carb and walk coefficients had correct signs but were nowhere near significant. The series did not have enough time steps to support seasonal estimation either — 84 daily observations trying to learn a 7-day seasonal pattern left fewer than 12 full cycles.

### Rebuilding at hourly resolution

`scripts/01-build-hourly.py` resamples the same source data to a complete hourly grid (2,076 hours). The key feature is `carb_load_decayed`: a decay-weighted carb sum over the preceding 6 hours, `Σ carbs × exp(−δ_min / 75)`, where tau=75 minutes reflects the post-meal glucose peak at 45–90 minutes. At exact meal time the feature equals the meal's carb content; by 2 hours it has decayed to 20% of its peak value. This matches the shape of the actual glucose excursion closely enough that the model can identify it.

The fitted model is SARIMAX(1,0,2)(1,0,1,24): p=1, q=2, seasonal AR and MA at lag 24. Seasonal period of 24 is justified by the data — the intraday profile shows a ~30 mg/dL peak-to-trough swing (31.96% of total variance) driven by the fasting/feeding cycle.

**Hourly SARIMAX performance:**

| forecast type | RMSE | MAE |
|---|---|---|
| 1-step-ahead (in-sample anchor) | 22.4 mg/dL | 16.8 mg/dL |
| Multi-step (504-hour holdout) | 41.4 mg/dL | 33.6 mg/dL |
| Naive (train mean) | 48.0 mg/dL | 39.9 mg/dL |

Multi-step RMSE of 41 vs naive 48 is a modest improvement. That reflects the nature of the problem: glucose 3 weeks out is primarily determined by behavior choices that the model can't observe — the exog regressors must be specified as hypothetical counterfactuals.

**What the hourly model recovered:** The carb coefficient is +0.137 mg/dL per g-equivalent (p < 0.001, correct sign). This is the main payoff from moving to hourly resolution: the carb signal was invisible at daily granularity and identifiable at hourly. A 50g carb meal produces a predicted glucose response of +6.9 mg/dL above baseline in the SARIMAX framework (the actual excursion is larger; the per-meal Ridge model captures the full peak better at +33 mg/dL per 50g).

Walk is correctly signed (−0.046 mg/dL per minute) but not significant (p≈0.4). With only 48 walks across 86 days, most hours have `walk_min_last_2h = 0`. The feature is too sparse for the SARIMAX to separate its effect from the hour-to-hour noise.

### The glipizide confounding experiment

I take Glipizide with 2 of 3 meals per day (meal-contingent, physician recommendation), not on a fixed 12-hour clock. Jardiance (SGLT2i) is once-daily with ~24-hour renal glucose excretion — it is always active and its effect is a fixed background already absorbed into the intercept. The `glipizide_active` binary flag (1 if within 12 hours of a taken dose) is therefore a crude proxy.

Initial SARIMAX fits (before adding explicit time-of-day controls) produced a `glipizide_active` coefficient of +137 mg/dL — flagrantly wrong sign, highly significant. The problem is structural: Glipizide is taken at 10 AM and 4 PM, so `glipizide_active=1` covers roughly 8 AM–8 PM — the same waking hours when glucose is naturally elevated by the diurnal feeding cycle. The medication flag was acting as a proxy for time-of-day, not measuring drug effect.

Adding `hour_sin` and `hour_cos` as explicit exogenous regressors partially corrects this (coefficient drops from +137 to +109), but confounding is not fully resolved. The Glipizide window tracks daytime hours and `is_fasting` tracks nighttime hours — they are nearly anti-correlated along the diurnal cycle.

Refitting without `glipizide_active` reveals the entanglement: the `is_fasting` coefficient collapses from −4.26 to +0.015 (essentially zero) when Glipizide is dropped. They were sharing the daytime/nighttime split signal between them. The no-medication model's AIC is 12698.6 vs 12686.1 for the full model (delta = +12.6 in favor of keeping it), but the counterfactual scenarios flatten completely — all three behavioral scenarios produce ~176 mg/dL mean forecast, indistinguishable from the status quo. The differentiation in the full model (B/C scenarios predicting ~144 mg/dL vs A at 175) was driven by the collinear fasting effect, not causal identification of fasting benefit.

**Conclusion:** The hourly SARIMAX can cleanly identify the carb signal. Walk and fasting are underpowered and entangled with the diurnal pattern. The per-meal Ridge model remains the better tool for behavioral what-ifs (walk dose-response, meal carb targets) — it works at the right granularity and has a cleaner separation between pre-meal state and post-meal intervention. The SARIMAX's value is in capturing the glucose autocorrelation structure for multi-day trajectory projections.

---

## Why not just run a plain regression on the hourly data?

If SARIMAX is finicky, the obvious question is: why not fit an ordinary regression — or a gradient-boosted tree — directly on the 2,076 hourly rows? Notebook 04 builds exactly that, three models, specifically to demonstrate why the obvious approach is a trap on autocorrelated data.

**Model A — regression with behavior features only (no recent-glucose input).** Predict each hour's glucose from carbs, walks, fasting, medication, and time of day, nothing else. It barely beats guessing the average: test RMSE 46 mg/dL vs 48 for a flat predict-the-mean baseline. The residual diagnostic (Durbin-Watson 0.51, where 2.0 is healthy and below ~1.0 is severe) confirms the model leaves almost all the structure unexplained. Glucose this hour is mostly determined by glucose last hour, and a model that cannot see last hour is flying blind.

**Model B — the same regression, plus "glucose one hour ago" as an input.** On paper this is transformative: test RMSE drops to 24 mg/dL, R² to 0.74. But it is cheating. To forecast next week you do not have next week's readings to feed in. When the model is forced to run forward on its own predictions (the realistic recursive mode), error blows up to **64 mg/dL — worse than just guessing the average (48)**. A model that looked twice as good as the baseline is actually a third worse once it has to stand on its own. Adding the lag also distorted the behavior coefficients unevenly — the walk coefficient shrank by 73% and stayed wrong-signed, the carb coefficient drifted further from the SARIMAX estimate — so the model is also worse for inference, not just forecasting.

**Model C — gradient-boosted trees, same inputs as B.** The nonlinear model lands at 24 mg/dL test RMSE — statistically identical to the simple linear regression — while massively overfitting the training data (train R² 0.99). A breakdown of what the tree relies on (permutation importance) shows 80% of its predictive weight comes from recent-glucose lag features, not behavior. The extra modeling power buys nothing; the carb × time-of-day interactions it was meant to capture contribute less than noise.

**The unifying lesson.** Glucose is dominated by momentum — where it just was. Any model handed the recent reading looks brilliant and learns nothing about behavior; any model denied it looks useless. This is precisely why the per-meal model (notebook 01) is the workhorse for "what should I do" questions: it sidesteps the momentum problem by measuring each meal's *rise from its own starting point* rather than the absolute glucose level.

---

## What this actually means for me (plain language)

Stripping out the statistics, here is what three months of my own data is telling me.

**"How good is the model?" decoded.** When I say a model explains 7% or 74% of the variation, the plain version is this: the dumbest possible forecast is "tomorrow will equal my three-month average." A model explaining 7% is barely better than that dumb guess. A model explaining 74% sounds great — but that number only appears when the model is quietly handed my actual recent reading. Take that away (which is the situation whenever I am planning ahead) and every model I built falls back to roughly the dumb-guess line, or worse.

**The single biggest finding: glucose has momentum.** By far the strongest predictor of my glucose in any hour is my glucose the hour before. Food, walking, fasting, and medication timing all matter, but their effect is small next to sheer momentum and my daily rhythm (higher through the day, lower overnight). There is no behavioral magic lever in this data that swamps that momentum, because one does not exist.

**What I can actually trust:**

- **Carbs are the one lever that shows up clean every single time.** Across every model and every variation, more carb load → higher glucose, correct direction, never ambiguous. This is the most reliable, actionable result in the whole project.
- **The per-meal view is the trustworthy one.** When I ask "what will *this* meal do to me," the per-meal model gives an answer I can rely on, because it measures the *jump* from wherever I started, not the absolute number. That is the model to consult at the table — not the hourly forecasts.
- **Walking helps, but I cannot put a precise number on it.** Every hourly model got the walk effect muddied, partly because I tend to walk *after* meals I expect to spike, so the data makes walking look associated with high glucose. The per-meal estimate (a ~25-minute walk ≈ a few mg/dL off the peak) is the best I have, and even that is soft.

**Practical recommendations for myself:**

1. **Treat carbs as the primary dial.** It is the one input the data agrees on without exception. For pushing toward A1C 6.0, consistent carb moderation pays off more predictably than any other single change.
2. **Use the per-meal model for decisions, not the trajectory forecasts.** "Should I walk after this dinner?" → per-meal model, good answer. "What will my average be in three weeks if I do X?" → no model here answers that reliably; do not over-trust any number that claims to.
3. **Consistency beats heroics.** Because glucose carries so much momentum, a steady run of moderate days moves my average more than occasional dramatic interventions wedged between ordinary ones. Smoothing out the bad days matters more than perfecting the good ones.
4. **To learn the walk effect properly, I would have to randomize it.** Walking only when I feel I need to permanently contaminates the data. A few weeks of deciding by coin-flip whether to walk after a meal would teach me more than any further modeling of what I already have.
5. **None of this replaces my endocrinologist.** These are day-to-day decision aids, not a treatment plan.

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

Clinical and ML/statistics terms are defined in [docs/glossary.md](docs/glossary.md).

---

## Disclaimer

This is a personal research project for an academic capstone. It is not medical advice, not a clinical tool, and not validated for use by anyone other than me. If you are diabetic, talk to your endocrinologist.

---

## License

MIT (code). The data files in `data/*` are personal health records belonging to the author and are not shared under any open data license. Including it here for completeness of my capstone project. If you want to use this, please email the author to seek permission. 
