# personal-cgm-lab

A personal glucose modeling project built on my own CGM data, South Indian meal log, and walking records. UC Berkeley ML/AI capstone, n=1.

---

## Contents

- [Why I built this](#why-i-built-this)
- [What I'm trying to figure out](#what-im-trying-to-figure-out)
- [Where the data comes from](#where-the-data-comes-from)
- [The web app: built with AI, used every day](#the-web-app-built-with-ai-used-every-day)
- [Status](#status)
- [Key findings](#key-findings)
- [Repo layout](#repo-layout)
- [Quick start](#quick-start)
- [Methods](#methods)
- [SARIMAX time-series experiments](#sarimax-time-series-experiments)
- [Deep learning: teaching glucose that it has a memory](#deep-learning-teaching-glucose-that-it-has-a-memory)
- [Why not just run a plain regression on the hourly data?](#why-not-just-run-a-plain-regression-on-the-hourly-data)
- [What this actually means for me](#what-this-actually-means-for-me-plain-language)
- [What I learned](#what-i-learned)
- [Honest limitations](#honest-limitations)
- [What comes next](#what-comes-next)
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

**Meals.** This is the weakest part of the data. About 24 meals were logged directly in the Stelo app at the time I ate using their AI analyzer feature (take a photo and it identifies dish with an estimate which was likely an experimental feature that was shut down). The remaining ~320 were logged initially in a spreadsheet, then migrated to the web app described below. It is imperfect and I know it.

**Medications.** Jardiance 10mg, Glipizide 5mg x2, and Crestor 10mg around 10 AM daily. Glipizide 5mg x2 again around 4 PM. The medication log is auto-generated as scheduled rows and I manually edit the record when I shift or miss a dose.

**Fasting windows.** I follow a roughly 12-hour overnight fast, 7 PM to 7 AM. On days when I skip breakfast and push the first meal to lunch, the window gets flagged as an intermittent fast automatically based on meal timing.

**Food nutrition.** I built a custom lookup table for the South Indian dishes I actually eat. Mainstream nutrition APIs don't have granular entries for dosa, rasam, or veg biryani as I prepare them, so I assembled estimates from a combination of published sources and portion-size notes.

---

## The web app: built with AI, used every day

Partway through the project it became clear that logging meals in a spreadsheet was friction I would eventually stop tolerating. So I built a small personal tracking web app - FastAPI backend, SQLite database, served from a Docker container on a machine at home - to replace the spreadsheet entirely.

The app itself is outside the scope of this capstone but it played a role. The app handles meal logging (dish name, carbs, meal type, fasting state), activity logging, CGM data import from the Stelo export, medication tracking, and fasting window detection. It has a weekly glucose dashboard that overlays CGM traces day by day, a spike review queue that flags unattributed glucose peaks for triage, and a training pipeline that exports all logged data to CSVs, rebuilds the feature matrices, and refits the models on demand - all from a browser.

I built the app using generative AI assistance guiding it to through the scripts already used in the notebooks. 

The reason this is in the README rather than a footnote: the app directly contributed to the results. Having a low-friction way to log a meal immediately after eating - phone browser, thirty seconds - meant I actually logged every meal in the last two weeks closer to real time collecting more details. More complete, more timely data. That is part of why the recent numbers look better. The model got better inputs, and I got better feedback. These are not separable.

![CGM Tracker weekly dashboard](images/09-web-app.png)

---

## Status

Nearly four months of data: February 1 through May 26, 2026. 29,557 five-minute readings.

| metric | value |
|---|---|
| mean glucose (full period) | 170.3 mg/dL |
| 7-day mean glucose | 132.2 mg/dL |
| time-in-range (70–180, full period) | 61.0% |
| GMI estimate (full period) | ~7.4 |
| logged walks | 72 |
| meal events | 343 |

![Dashboard: 129 mg/dL 7-day mean, 88.4% time-in-range](images/09-outcome.png)

The 7-day mean of 132 mg/dL is worth calling out separately. After the first model submission, I started walking more than 40 minutes after every meal - partly to test the model's prediction, partly to collect better walk data. My overall glucose average has been tracking close to 130 for the most recent weeks. The target is ~125 mg/dL (A1C 6.0). I don't have enough post-intervention data to model this properly yet - a few weeks of consistent behaviour doesn't constitute a new training regime - but the direction is exactly right and I intend to keep collecting.

---

## Key findings

These are the headline results across all models, fitted on data through late April 2026 and evaluated on the last 21 days as a held-out test set.

### All-model comparison

| Model | Test RMSE | Test R² | Notes |
|---|---|---|---|
| Ridge (per-meal) | 33.86 mg/dL | 0.152 | Best generalising tabular model |
| Random Forest (per-meal) | 31.58 mg/dL | 0.262 | Best single per-meal model overall |
| GBM (per-meal) | 34.80 mg/dL | 0.104 | Badly overfit; train R²=0.91 |
| SARIMAX hourly | 41.4 mg/dL | - | Beats naive (48.0); carb signal clean |
| LSTM per-meal | 33.97 ± 0.3 mg/dL | ~0.170 | Competitive with Ridge; sequence input |
| LSTM hourly | 23.31 ± 0.19 mg/dL | ~0.77 | Beats persistence baseline (27.71) |
| LSTM 5-min | 4.19 ± 0.02 mg/dL | ~0.993 | Beats persistence (4.84); near real-time |

**Walk dose-response.** A 25-minute walk within 90 minutes of a meal reduces the predicted peak excursion by about 3.8 mg/dL compared to no walk, holding carb load and pre-meal glucose constant. Extending to 45 minutes adds no further predicted benefit in the training data - but that estimate is soft, since only 48 walked meals were in the original training set. Post-submission I have been walking 40+ minutes after every meal, and the accumulating data will eventually test whether that plateau was a data artefact or physiology.

**Pre-meal glucose is the dominant predictor.** Both the Ridge coefficients and the SHAP summary agree: where my glucose starts before a meal is the strongest single driver of how high it peaks. Carb load and time of day follow behind it. Walk minutes matter but rank lower, which reflects the sparse walk data in the original training window.

**Dish ranking (adjusted).** Using the model residuals to rank dishes removes the confounding from when and how I tend to eat each food. Chai and dosa tend to spike less than their raw averages suggest - they often appear at breakfast when pre-meal glucose happens to be lower. Meals logged as "salad" or veg biryani carry positive residuals, meaning they spike more than the model expects given their context - likely a logging artefact (hidden carbs in dressings, larger portions than assumed) rather than a property of the dish itself.

**Hourly SARIMAX.** The carb signal that was invisible at daily resolution became identifiable at hourly: `carb_load_decayed` (tau=75 min exponential decay) has a coefficient of +0.137 mg/dL per g-equivalent (p < 0.001). Multi-step RMSE across the 21-day holdout is 41 mg/dL vs 48 for the naive mean baseline. Walk effect is correctly signed but not significant given the sparse walk data. The medication variable (`glipizide_active`) is confounded with the diurnal glucose pattern and cannot be cleanly identified. See the "SARIMAX time-series experiments" section below for the full analysis.

**LSTM hourly.** Beats the persistence baseline (23.31 vs 27.71 RMSE, ~0.77 R²) with the advantage concentrated specifically during post-meal windows - which is the clinically meaningful window. Persistence says "glucose stays the same"; the LSTM has learned that it doesn't, especially in the 60 minutes after eating.

**LSTM 5-min.** At five-minute resolution, the persistence baseline is extremely strong (RMSE 4.84, R²=0.990) - glucose really doesn't change much in five minutes most of the time. The LSTM still beats it (RMSE 4.19, R²=0.993), and its advantage again concentrates post-meal. The permutation importance result is humbling: current glucose contributes +59 mg/dL to RMSE when permuted; carbs, walks, and everything else combined contribute about +0.35 mg/dL. A 170:1 ratio. The model mostly learned "glucose will be close to what it is now." That turns out to be almost always correct. The remaining 1 part in 170 is where the actionable features live.

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
│       ├── hourly_features.csv     one row per hour (target: glucose_hourly_mean)
│       └── cgm_5min_features.csv   one row per 5-min reading (target: next glucose)
├── scripts/
│   ├── 00-build-features.py        reads data/raw/, writes per_meal + daily features
│   ├── 01-build-hourly.py          reads data/raw/, writes hourly_features.csv
│   └── 02-build-5min.py            reads data/raw/, writes cgm_5min_features.csv
├── notebooks/
│   ├── 00-exploratory-data-analysis.ipynb
│   ├── 01-regression-model.ipynb
│   ├── 02-sarimax-model.ipynb      daily SARIMAX (contrast case)
│   ├── 03-sarimax-hourly.ipynb     hourly SARIMAX with glipizide comparison
│   ├── 04-regression-hourly.ipynb  Ridge/GBM on hourly data (autocorrelation pitfall)
│   ├── 05-lstm-per-meal.ipynb      LSTM on pre-meal CGM sequence -> peak_delta
│   ├── 06-lstm-hourly.ipynb        LSTM next-hour forecasting; beats persistence
│   └── 07-lstm-5min.ipynb          LSTM next-5-min forecasting; feature importance
├── models/
│   ├── 
│   └──                             models saved here
└── README.md
```

The CSVs in `data/raw/` are the editable layer. The processed feature matrices are fully reproducible from a single script run.

For a full column-by-column description of all three feature matrices, see [docs/data-dictionary.md](docs/data-dictionary.md).

---

## Quick start

```bash
# Build the per-meal and daily feature matrices
python scripts/00-build-features.py

# Build the hourly feature matrix (needed for notebooks 03, 04, 06)
python scripts/01-build-hourly.py

# Build the 5-min feature matrix (needed for notebook 07)
python scripts/02-build-5min.py
```

Dependencies: pandas, numpy, scikit-learn, shap, matplotlib, seaborn, pyarrow, torch (CPU build is fine for all LSTM notebooks).

Then open the notebooks in order:

- [00-exploratory-data-analysis.ipynb](notebooks/00-exploratory-data-analysis.ipynb) - data quality checks, EDA, feature validation, stationarity tests, hourly feature visualizations
- [01-regression-model.ipynb](notebooks/01-regression-model.ipynb) - Ridge + GBM, SHAP, walk ROI counterfactuals, confounder-adjusted dish ranking
- [02-sarimax-model.ipynb](notebooks/02-sarimax-model.ipynb) - daily SARIMAX (contrast case; shows why daily resolution fails)
- [03-sarimax-hourly.ipynb](notebooks/03-sarimax-hourly.ipynb) - hourly SARIMAX, carb signal recovery, glipizide confounding experiment, counterfactual scenarios
- [04-regression-hourly.ipynb](notebooks/04-regression-hourly.ipynb) - Ridge/GBM on hourly data; demonstrates the autocorrelation pitfall (no-lag vs lag, oracle vs recursive)
- [05-lstm-per-meal.ipynb](notebooks/05-lstm-per-meal.ipynb) - LSTM on 2-hour pre-meal CGM sequence; data augmentation experiment; hybrid LSTM+tabular architecture
- [06-lstm-hourly.ipynb](notebooks/06-lstm-hourly.ipynb) - LSTM next-hour forecasting with 17 exogenous features; beats persistence; post-meal window analysis
- [07-lstm-5min.ipynb](notebooks/07-lstm-5min.ipynb) - LSTM next-5-min forecasting; 19 features; actionable vs physiological feature importance

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

The first attempt fit a SARIMAX on the 84 daily means in `daily_features.csv`. It collapsed immediately: a 30-minute walk occupies 2% of a day and is indistinguishable from noise at that granularity. The model produced a flat forecast that was barely better than the training mean. The carb and walk coefficients had correct signs but were nowhere near significant. The series did not have enough time steps to support seasonal estimation either - 84 daily observations trying to learn a 7-day seasonal pattern left fewer than 12 full cycles.

### Rebuilding at hourly resolution

`scripts/01-build-hourly.py` resamples the same source data to a complete hourly grid (2,076 hours). The key feature is `carb_load_decayed`: a decay-weighted carb sum over the preceding 6 hours, `Σ carbs × exp(−δ_min / 75)`, where tau=75 minutes reflects the post-meal glucose peak at 45–90 minutes. At exact meal time the feature equals the meal's carb content; by 2 hours it has decayed to 20% of its peak value. This matches the shape of the actual glucose excursion closely enough that the model can identify it.

The fitted model is SARIMAX(1,0,2)(1,0,1,24): p=1, q=2, seasonal AR and MA at lag 24. Seasonal period of 24 is justified by the data - the intraday profile shows a ~30 mg/dL peak-to-trough swing (31.96% of total variance) driven by the fasting/feeding cycle.

**Hourly SARIMAX performance:**

| forecast type | RMSE | MAE |
|---|---|---|
| 1-step-ahead (in-sample anchor) | 22.4 mg/dL | 16.8 mg/dL |
| Multi-step (504-hour holdout) | 41.4 mg/dL | 33.6 mg/dL |
| Naive (train mean) | 48.0 mg/dL | 39.9 mg/dL |

Multi-step RMSE of 41 vs naive 48 is a modest improvement. That reflects the nature of the problem: glucose 3 weeks out is primarily determined by behavior choices that the model can't observe - the exog regressors must be specified as hypothetical counterfactuals.

**What the hourly model recovered:** The carb coefficient is +0.137 mg/dL per g-equivalent (p < 0.001, correct sign). This is the main payoff from moving to hourly resolution: the carb signal was invisible at daily granularity and identifiable at hourly. A 50g carb meal produces a predicted glucose response of +6.9 mg/dL above baseline in the SARIMAX framework (the actual excursion is larger; the per-meal Ridge model captures the full peak better at +33 mg/dL per 50g).

Walk is correctly signed (−0.046 mg/dL per minute) but not significant (p≈0.4). With only 48 walks across 86 days, most hours have `walk_min_last_2h = 0`. The feature is too sparse for the SARIMAX to separate its effect from the hour-to-hour noise.

### The glipizide confounding experiment

I take Glipizide with 2 of 3 meals per day (meal-contingent, physician recommendation), not on a fixed 12-hour clock. Jardiance (SGLT2i) is once-daily with ~24-hour renal glucose excretion - it is always active and its effect is a fixed background already absorbed into the intercept. The `glipizide_active` binary flag (1 if within 12 hours of a taken dose) is therefore a crude proxy.

Initial SARIMAX fits (before adding explicit time-of-day controls) produced a `glipizide_active` coefficient of +137 mg/dL - flagrantly wrong sign, highly significant. The problem is structural: Glipizide is taken at 10 AM and 4 PM, so `glipizide_active=1` covers roughly 8 AM–8 PM - the same waking hours when glucose is naturally elevated by the diurnal feeding cycle. The medication flag was acting as a proxy for time-of-day, not measuring drug effect.

Adding `hour_sin` and `hour_cos` as explicit exogenous regressors partially corrects this (coefficient drops from +137 to +109), but confounding is not fully resolved. The Glipizide window tracks daytime hours and `is_fasting` tracks nighttime hours - they are nearly anti-correlated along the diurnal cycle.

Refitting without `glipizide_active` reveals the entanglement: the `is_fasting` coefficient collapses from −4.26 to +0.015 (essentially zero) when Glipizide is dropped. They were sharing the daytime/nighttime split signal between them is my interpretation. The no-medication model's AIC is 12698.6 vs 12686.1 for the full model (delta = +12.6 in favor of keeping it), but the counterfactual scenarios flatten completely - all three behavioral scenarios produce ~176 mg/dL mean forecast, indistinguishable from the status quo. The differentiation in the full model (B/C scenarios predicting ~144 mg/dL vs A at 175) was driven by the collinear fasting effect, not causal identification of fasting benefit.

**Conclusion:** The hourly SARIMAX can cleanly identify the carb signal. Walk and fasting are underpowered and entangled with the diurnal pattern. The per-meal Ridge model remains the better tool for behavioral what-ifs (walk dose-response, meal carb targets) - it works at the right granularity and has a cleaner separation between pre-meal state and post-meal intervention. The SARIMAX's value is in capturing the glucose autocorrelation structure for multi-day trajectory projections.

---

## Deep learning: teaching glucose that it has a memory

The regression results were real but modest - Ridge explained about 15% of test variance, Random Forest about 26%. That's not nothing, but it's also not a model you'd bet a dinner on. The natural question after SARIMAX was: what architecture actually matches the structure of the problem?

Glucose is not stateless. The plateau from a big rice lunch is still pulling up my post-dinner reading eight hours later. The overnight fast slowly unwinds that effect, hour by hour. This is temporal memory - not just "what I ate" but "what I ate, and when, and what happened to it since." SARIMAX captures some of this through its AR terms, but it assumes a fixed linear autocorrelation structure. What if the memory is nonlinear, or depends on context?

A course visualization from my learning facilitator showing how LSTM cell state carries information through long sequences made the connection click. LSTMs were designed for exactly this - sequences where the relevant signal from several steps ago still matters now. I skipped RNN entirely and went straight to LSTM; purely on hunch (a theoretical study on RNN is due).

Three notebooks explore this:

### Notebook 05 - LSTM per-meal (pre-meal CGM sequence -> peak delta)

Instead of compressing the 2-hour pre-meal CGM history into three scalar summary statistics, this model reads all 24 five-minute readings as a sequence and predicts the post-meal spike. The hypothesis was that the *shape* of the pre-meal trajectory carries information the scalars discard - a rising glucose entering a meal behaves differently than a flat one at the same level.

Results: RMSE 33.97 (±0.3 across 5 seeds), competitive with Ridge (33.86) but behind Random Forest (31.58). The tabular models' summary statistics apparently captured most of what the sequence shape adds. One notable experiment: data augmentation - adding small Gaussian noise to training sequences to artificially multiply the dataset - made the model *worse* at every multiplier tested (×2, ×3, ×10). More synthetic data did not help; the model was not limited by quantity, it was limited by the variation that actually exists in ~200 real meals.

### Notebook 06 - LSTM hourly (next-hour glucose forecasting)

Predict next-hour glucose from a 7-hour rolling window of 17 features: CGM history, decayed carb load, walk minutes, medication timing, fasting state, and time-of-day.

Test RMSE: **23.31 ± 0.19 mg/dL** vs persistence baseline of **27.71 mg/dL** (R² ~0.77). The improvement over persistence concentrates specifically in post-meal windows - the 0–60 minutes after eating, when glucose is moving fastest and the "it'll stay the same" assumption breaks down hardest. This is the clinically relevant window. Outside of meal windows the LSTM and persistence are nearly indistinguishable, which is actually correct: stable glucose really is mostly persistent.

### Notebook 07 - LSTM 5-min (next-reading forecasting)

At 5-minute resolution, the persistence baseline is almost unbeatable - glucose typically changes less than 2 mg/dL between readings during stable periods. Persistence RMSE: 4.84 mg/dL (R²=0.990). The LSTM reaches **4.19 ± 0.02 mg/dL** (R²=0.993), a real improvement even if the absolute numbers look small.

The permutation feature importance result is the most interesting finding: current glucose contributes +59.24 mg/dL to RMSE when its values are shuffled; carbs, walks, medication, and every other feature combined contribute about +0.35 mg/dL. A ratio of roughly 170:1. The model has learned, correctly, that glucose five minutes from now is almost always close to glucose right now. The actionable features - carb load, activity, medication - live in the remaining 1 part in 170, but they are real, they are correctly signed, and they are the only part a behavioral intervention can actually move.

![Actionable vs physiological: what can a patient influence?](images/07-lstm-explantory.png)

![LSTM 5-min: predicted vs actual scatter and 3-day trace](images/07-lstm-test-result.png)

---

## Why not just run a plain regression on the hourly data?

If SARIMAX is finicky, the obvious question is: why not fit an ordinary regression - or a gradient-boosted tree - directly on the 2,076 hourly rows? Notebook 04 builds exactly that, three models, specifically to demonstrate why the obvious approach is a trap on autocorrelated data.

**Model A - regression with behavior features only (no recent-glucose input).** Predict each hour's glucose from carbs, walks, fasting, medication, and time of day, nothing else. It barely beats guessing the average: test RMSE 46 mg/dL vs 48 for a flat predict-the-mean baseline. The residual diagnostic (Durbin-Watson 0.51, where 2.0 is healthy and below ~1.0 is severe) confirms the model leaves almost all the structure unexplained. Glucose this hour is mostly determined by glucose last hour, and a model that cannot see last hour is flying blind.

**Model B - the same regression, plus "glucose one hour ago" as an input.** On paper this is transformative: test RMSE drops to 24 mg/dL, R² to 0.74. But it is cheating. To forecast next week you do not have next week's readings to feed in. When the model is forced to run forward on its own predictions (the realistic recursive mode), error blows up to **64 mg/dL - worse than just guessing the average (48)**. A model that looked twice as good as the baseline is actually a third worse once it has to stand on its own. Adding the lag also distorted the behavior coefficients unevenly - the walk coefficient shrank by 73% and stayed wrong-signed, the carb coefficient drifted further from the SARIMAX estimate - so the model is also worse for inference, not just forecasting.

**Model C - gradient-boosted trees, same inputs as B.** The nonlinear model lands at 24 mg/dL test RMSE - statistically identical to the simple linear regression - while massively overfitting the training data (train R² 0.99). A breakdown of what the tree relies on (permutation importance) shows 80% of its predictive weight comes from recent-glucose lag features, not behavior. The extra modeling power buys nothing; the carb × time-of-day interactions it was meant to capture contribute less than noise.

**The unifying lesson.** Glucose is dominated by momentum - where it just was. Any model handed the recent reading looks brilliant and learns nothing about behavior; any model denied it looks useless. This is precisely why the per-meal model (notebook 01) is the workhorse for "what should I do" questions: it sidesteps the momentum problem by measuring each meal's *rise from its own starting point* rather than the absolute glucose level.

---

## What this actually means for me (plain language)

Stripping out the statistics, here is what four months of my own data is telling me.

**"How good is the model?" decoded.** When I say a model explains 7% or 74% of the variation, the plain version is this: the dumbest possible forecast is "tomorrow will equal my three-month average." A model explaining 7% is barely better than that dumb guess. A model explaining 74% sounds great - but that number only appears when the model is quietly handed my actual recent reading. Take that away (which is the situation whenever I am planning ahead) and every model I built falls back to roughly the dumb-guess line, or worse.

**The single biggest finding: glucose has momentum.** By far the strongest predictor of my glucose in any hour is my glucose the hour before. Food, walking, fasting, and medication timing all matter, but their effect is small next to sheer momentum and my daily rhythm (higher through the day, lower overnight). There is no behavioral single magic lever in this data that swamps that momentum. Intermittent Fasting helps here and walks immediately after meals keeps the post meal spikes contained. 

**What I can actually trust:**

- **Carbs are the one lever that shows up clean every single time.** Across every model and every variation, more carb load -> higher glucose, correct direction, never ambiguous. This is the most reliable, actionable result in the whole project.
- **The per-meal view is the trustworthy one.** When I ask "what will *this* meal do to me," the per-meal model gives an answer I can rely on, because it measures the *jump* from wherever I started, not the absolute number. That is the model to consult at the table - not the hourly forecasts.
- **Walking helps, but I cannot put a precise number on it yet.** Every hourly model got the walk effect muddied, partly because I tend to walk *after* meals I expect to spike, so the data makes walking look associated with high glucose. The per-meal estimate (a ~25-minute walk ≈ a few mg/dL off the peak) is the best I have, and even that is soft.

**The result that matters most: I started walking more.** After the first model submission, I took the walk dose-response estimate seriously and started walking more than 40 minutes after every meal - more consistently and for longer than before. The data since then tells the story: my 7-day glucose average is now around 132 mg/dL, compared to 170 mg/dL across the full period. The target for A1C 6.0 is roughly 125 mg/dL. I am not there yet, and I do not have enough post-intervention data to model this properly - but the direction is exactly right.

There is something both satisfying and slightly absurd about this outcome. I built a fairly involved ML pipeline, ran eight notebooks, tried five different modelling approaches, and the most important result is: walking consistently after meals works. Every doctor has said this. I came here looking for a magical answer and the beauty is that the lack of data guided me to it. The data just made me actually do it.

**Practical recommendations for myself:**

1. **Treat carbs as the primary dial.** It is the one input the data agrees on without exception. For pushing toward A1C 6.0, consistent carb moderation pays off more predictably than any other single change. Seeing a factor of 170x is real. 
2. **Use the per-meal model for decisions, not the trajectory forecasts.** "Should I walk after this dinner?" -> per-meal model, good answer. "What will my average be in three weeks if I do X?" -> no model here answers that reliably; do not over-trust any number that claims to.
3. **Consistency beats heroics.** Because glucose carries so much momentum, a steady run of moderate days moves my average more than occasional dramatic interventions wedged between ordinary ones. Smoothing out the bad days matters more than perfecting the good ones.
4. **To learn the walk effect properly, I would have to randomize it.** Walking only when I feel I need to permanently contaminates the data. A few weeks of deciding by coin-flip whether to walk after a meal would teach me more than any further modeling of what I already have.
5. **None of this replaces my endocrinologist.** These are day-to-day decision aids, not a treatment plan.

---

## What I learned

After 5 months in the course, the modeling was the straightforward part. Getting the data into a shape the model could actually use was where most of the real work happened.

I came into this project thinking the hard problem was picking the right algorithm. For the first submission, it turned out the hard problem was much earlier: the raw Stelo export arrives in a format designed for their app, not for analysis. Timestamps are in one format, dates in another. Activity data comes through Health Connect with its own schema. Meal records are partially logged in the app and partially maintained outside in a ledger. Medication records needed to be generated from scratch and maintained manually. None of these sources were designed to be joined to each other.

The discipline that forced was something I wouldn't have learned from a clean classroom dataset off of Kaggle. Every modeling decision downstream traces back to a data decision upstream. The 4 AM physiological-day boundary exists because I noticed overnight fasts were splitting across calendar dates and distorting the daily averages. The `carbs_g_final` priority chain exists because three different sources each have partial coverage and none is authoritative on its own. The `taken` column bug (boolean `True` stored as a string `'TRUE'` in one path, actual bool in another) was silent - the model ran fine, it just treated every meal as if no medication had ever been taken.

The second thing I learned was about choosing the right architecture for the right question. SARIMAX captured the carb signal that plain regression missed entirely, but it struggled with the sparse walk data and the Glipizide confounding. Moving to LSTM was driven by a specific observation: glucose has temporal memory, and architectures designed for sequential data should respect that. Seeing how LSTM cell state carries context across long sequences - not just "what happened last step" but "what happened 20 steps ago and is still relevant now" - made the switch feel principled rather than just trying the next thing on the list. I skipped RNN and went straight to LSTM because the vanishing gradient problem for longer sequences was a known limitation I did not want to work around.

Sourcing your own data removes the safety net of a pre-cleaned dataset. There is no answer key. When something looks wrong, you have to decide whether the data is bad, the feature engineering is wrong, or the biology is actually doing something unexpected. That judgment call is most of what data work actually is, and I don't think I would have developed it the same way working on someone else's almost-already-tidy CSV.

---

## Honest limitations

**n=1.** Nothing here generalizes to other people. The methods are reusable; the coefficients are not.

**Sparse walk data (improving).** The original training window had only 48 walks. Post-submission, walking 40+ minutes after every meal has increased the activity log to 72 events and counting. The dose-response curve above 30 minutes was largely extrapolation in the first model; a retrained model on the expanded dataset will test whether that plateau holds.

**Selection bias on walks.** I tend to walk after meals where I expect a spike, not randomly. Even controlling for carbs and pre-meal glucose, unmeasured expectation is correlated with the walk decision. A few weeks of coin-flip randomization would produce cleaner evidence than any regression on this dataset.

**Small n for deep learning.** The LSTM per-meal model trains on ~200 sequences. That is genuinely small for a neural network, and the multi-seed evaluation in notebook 05 shows the variance is non-trivial. The hourly and 5-min LSTMs have more sequences (1,500+ and 20,000+ respectively) and are more stable. With more months of data the per-meal LSTM will be worth revisiting.

**Confounders not captured:** stress, sleep quality, hydration, exact medication onset time relative to eating. Their absence shows up as residual variance in the model.

---

## What comes next

### 1. More data, better models

The most important thing I can do is keep collecting. Seven more months of consistent logging - meals, walks, CGM - would roughly triple the dataset and bring the per-meal LSTM into a range where its results are actually trustworthy. With more data, the walk dose-response curve above 30 minutes stops being extrapolation and becomes something I can actually rely on at dinner time.

On the modeling side, two papers are on the reading list. Namazi & Shakeri's *"From Prediction to Practice: A Task-Aware Evaluation Framework for Blood Glucose Forecasting"* ([arXiv:2605.00645](https://arxiv.org/abs/2605.00645)) reframes the evaluation problem entirely - rather than minimizing aggregate RMSE, it asks whether a forecasting model is actually useful for specific clinical tasks like hypoglycemia early warning. That framing fits this project well: the question was never "what RMSE can I achieve" but "what should I eat tonight." A second paper on LSTM architectures for CGM forecasting ([medRxiv:2024.02.08.24302542](https://www.medrxiv.org/content/10.1101/2024.02.08.24302542v1)) is on the list for techniques that might improve the per-meal sequence model specifically.

### 2. Reach out for more data

The hardest constraint in this project is n=1. A paper by researchers studying glucose dynamics in a South Asian dietary context ([preprints.org:202309.0755](https://www.preprints.org/manuscript/202309.0755)) covers a population whose food patterns overlap significantly with mine - rice-based meals, lentils, the glycemic profile of a South Indian plate rather than a Western one. I intend to contact the authors to ask whether their dataset is available for modeling work, or whether there is appetite for a collaboration. Even a small multi-subject dataset with similar dietary patterns would let me test whether any of the coefficients here generalize beyond one person.

### 3. Close the logging loop with direct API integration

Right now the data flow has two manual steps: exporting from the Stelo app and uploading the CSV, and manually entering meals into the web app. Both are low friction but not zero friction. The Stelo/Dexcom and Google Health Connect APIs exist; wiring them up would make the pipeline fully automatic - CGM readings and walk data land in the database as they happen, with no export or upload step. That would also enable near-real-time spike detection and feedback rather than end-of-day review.

### 4. Stretch - mobile logging with image-based food recognition

The remaining friction is meal entry. Typing "low gi basmati rice, 60g" is fine when I'm at a desk; less so mid-meal. A lightweight mobile companion - either a React Native app or a progressive web app - that lets me photograph a plate and have a vision model estimate the dish and carb content would close that gap. The custom South Indian food lookup table I built is a head start: the image recognition output just needs to be mapped onto that table rather than a generic Western nutrition database. This is a stretch goal that depends on how well the core data collection holds up first, but the pieces are already mostly in place.

---

## Glossary

Clinical and ML/statistics terms are defined in [docs/glossary.md](docs/glossary.md).

## Disclaimer

This is a personal research project for an academic capstone. It is not medical advice, not a clinical tool, and not validated for use by anyone other than me. If you are diabetic, talk to your endocrinologist.

## License

MIT (code). The data files in `data/*` are personal health records belonging to the author and are not shared under any open data license. Including it here for completeness of my capstone project. If you want to use this, please email the author to seek permission.
