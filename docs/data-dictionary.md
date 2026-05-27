# Data Dictionary: processed feature matrices

These files live in `data/processed/` and are rebuilt by running `scripts/00-build-features.py`.
Neither file is a primary source; both are fully derived from the CSVs in `data/raw/` that is outside the scope of this excercise to define. 

---

## per_meal_features.csv 

One row per meal event (295 rows as of April 2026). The primary modeling table.
Target variable: **`peak_delta`**.

### Identifiers and context

| column | type | description |
|---|---|---|
| `meal_id` | str | Unique meal identifier (e.g. `M0001`) |
| `ts` | datetime | Meal timestamp, ISO 8601 |
| `date` | date | Calendar date of the meal |
| `dow` | str | Day of week (Monday through Sunday) |
| `time_bucket` | str | Coarse time-of-day label (morning, midday, evening, night) |
| `meal_type` | str | Breakfast, lunch, dinner, or snack |
| `dish_names` | str | Free-text dish description as entered |
| `food_match` | str | Matched key from `food_lookup.csv`, null if no match |

### Targets

| column | type | description |
|---|---|---|
| `peak_delta` | int | **Primary target.** Peak glucose rise above pre-meal level (mg/dL). Computed as `peak_glucose - pre_meal_glucose` over a ~3-hour post-meal window. |
| `peak_glucose` | int | Maximum CGM reading in the ~3-hour post-meal window (mg/dL) |
| `time_to_peak_min` | int | Minutes from meal start to peak glucose |
| `out_of_range_180` | int | 1 if `peak_glucose > 180 mg/dL`, 0 otherwise |

### Pre-meal glucose state

These are all safe to use as features; they are computed from CGM readings that precede the meal.

| column | type | description |
|---|---|---|
| `pre_meal_glucose` | int | CGM reading closest to meal time (mg/dL) |
| `glucose_30min_before_mean` | float | Mean CGM glucose in the 30 minutes before the meal |
| `glucose_60min_before_mean` | float | Mean CGM glucose in the 60 minutes before the meal |
| `glucose_velocity_30min` | float | `(glucose_now - glucose_30min_ago) / 30`. Positive = glucose was rising into the meal (mg/dL per minute) |

### Nutrition

`carbs_g_final` is the recommended column for modeling. The three source columns are kept for traceability.

| column | type | description |
|---|---|---|
| `carbs_g_final` | float | Best available carb estimate. Priority: `carbs_grams_logged` > `carbs_grams_estimated` > `carbs_g_lookup` |
| `carbs_grams_logged` | float | Carbs logged at the time of eating (Stelo app or manual entry) |
| `carbs_grams_estimated` | float | Carbs estimated during post-hoc backfill |
| `carbs_g_lookup` | float | Carbs from `food_lookup.csv` for the matched dish |
| `protein_g_lookup` | float | Protein from `food_lookup.csv` (g) |
| `fat_g_lookup` | float | Fat from `food_lookup.csv` (g) |
| `gl_lookup` | str | Glycemic load category from `food_lookup.csv` (low / medium / high) |

### Fasting context

| column | type | description |
|---|---|---|
| `fasted_meal` | bool | True if the meal followed a fasting window, as recorded in `meals_v2.csv` |
| `broke_if_window` | bool | True if this meal ended an intermittent-fasting window (derived from `fasting_windows.csv`) |
| `broke_window_type` | str | Type of window broken: `intermittent_skip_breakfast` or `overnight` |
| `prev_fast_hours` | float | Duration of the most recently completed fasting window before this meal (hours) |

### Post-meal activity

These are observable after the meal and are safe for post-event analysis. For prospective
predictions, treat them as interventions you can set to a counterfactual value.

| column | type | description |
|---|---|---|
| `walk_min_within_60` | float | Total walk duration within 60 minutes of meal start (minutes) |
| `walk_min_within_90` | float | Total walk duration within 90 minutes of meal start (minutes) |
| `walk_min_within_180` | float | Total walk duration within 180 minutes of meal start (minutes) |
| `minutes_to_first_walk` | float | Minutes from meal start to the first walk. Null if no walk within 3 hours |

### Medication timing

All three columns are null for any meal that has no prior medication record.

| column | type | description |
|---|---|---|
| `minutes_since_last_dose` | float | Minutes from the last taken dose (any drug) to meal timestamp |
| `last_drug` | str | Name of the drug from that last dose |
| `minutes_since_last_glipizide` | float | Minutes since the last Glipizide dose specifically |

### Time encoding

| column | type | description |
|---|---|---|
| `hour` | int | Hour of day (0-23) |
| `hour_sin` | float | `sin(2π × hour / 24)`, cyclical encoding of time of day |
| `hour_cos` | float | `cos(2π × hour / 24)`, cyclical encoding of time of day |
| `is_weekend` | int | 1 if Saturday or Sunday, 0 otherwise |
| `day_index` | int | Days since the first meal in the dataset (0-indexed). Useful as a linear trend term |

---

## daily_features.csv 

One row per **physiological day** (35 columns). The daily modeling and SARIMAX table.

A physiological day runs from 4:00 AM to 3:59 AM the next calendar day. This keeps overnight
fasting glucose on the same date as the meal that broke the fast, which matters for the
fast-window vs feeding-window decomposition.

### Identifiers

| column | type | description |
|---|---|---|
| `date` | date | Physiological day start (4 AM boundary) |
| `dow` | str | Day of week |
| `is_weekend` | int | 1 if Saturday or Sunday |
| `day_index` | int | Days since the first date in the dataset (0-indexed) |

### Primary targets

| column | type | description |
|---|---|---|
| `mean_glucose` | float | **Primary target.** Mean CGM glucose for the physiological day (mg/dL) |
| `time_in_range_pct` | float | Percentage of readings between 70 and 180 mg/dL |
| `gmi_estimate` | float | Glucose Management Indicator: `3.31 + 0.02392 × mean_glucose` (Bergenstal 2018). Approximates HbA1c from CGM mean |
| `time_above_180_pct` | float | Percentage of readings above 180 mg/dL |
| `time_below_70_pct` | float | Percentage of readings below 70 mg/dL |
| `mean_glucose_calendar` | float | Mean glucose by calendar day (midnight boundary). Kept for reporting to an endocrinologist who thinks in calendar days, not physiological days |

### Fast / feeding window decomposition

CGM readings are split into those that fall inside a logged fasting window and those outside.
Both sides are summarized separately to isolate overnight glucose from post-meal glucose.

| column | type | description |
|---|---|---|
| `fast_window_mean_glucose` | float | Mean glucose during fasting windows (mg/dL) |
| `feeding_window_mean_glucose` | float | Mean glucose outside fasting windows (mg/dL) |
| `fast_window_tir_pct` | float | Time-in-range during fasting windows (%) |
| `feeding_window_tir_pct` | float | Time-in-range during feeding windows (%) |
| `fast_window_n_readings` | int | CGM readings that fell inside a fasting window |
| `feeding_window_n_readings` | int | CGM readings that fell outside a fasting window |

### Glucose distribution

| column | type | description |
|---|---|---|
| `min_glucose` | int | Minimum CGM reading for the day (mg/dL) |
| `max_glucose` | int | Maximum CGM reading for the day (mg/dL) |
| `std_glucose` | float | Standard deviation of CGM readings (mg/dL). Proxy for glucose variability |
| `n_readings` | int | Total CGM readings for the day. Days with fewer than ~288 readings had sensor gaps |

### Behavior: exogenous regressors

Zero-filled when missing (no walk, no logged meal, no dose). Use as exogenous regressors in
SARIMAX, or as predictors in a daily regression.

| column | type | description |
|---|---|---|
| `n_walks` | int | Number of walk events logged |
| `total_walk_min` | float | Total minutes walked (0 on days with no logged walks) |
| `max_walk_min` | float | Duration of the longest single walk |
| `n_meals` | int | Number of meal events |
| `total_carbs` | float | Sum of `carbs_g_final` across all meals (0 if none recorded) |
| `avg_peak_delta` | float | Mean `peak_delta` across meals with a computed value. Null if no peak_delta was available |
| `max_peak_delta` | float | Largest single-meal `peak_delta` for the day |
| `is_if_day` | int | 1 if an intermittent-fasting window was observed this day |
| `if_duration_hours` | float | Duration of the IF window (hours). Null on non-IF days |
| `overnight_fast_hours` | float | Duration of the overnight fast. Null if not logged |
| `doses_taken` | int | Total medication doses taken (0 if none) |

### Lag and rolling features

These follow a strict leakage policy: `_prior` and `_lag1` columns are shifted so they
reflect only past data and are safe to use as features when the target is `mean_glucose`
or `time_in_range_pct`. The `_descriptive` column includes the current day and must not
be used as a model input.

| column | type | safe as feature? | description |
|---|---|---|---|
| `mean_glucose_lag1` | float | yes | Yesterday's `mean_glucose` |
| `total_walk_min_lag1` | float | yes | Yesterday's `total_walk_min` |
| `total_carbs_lag1` | float | yes | Yesterday's `total_carbs` |
| `mean_glucose_7d_avg_prior` | float | yes | 7-day rolling mean of `mean_glucose`, ending yesterday (min 3 days). Covers [N-7, N-1] |
| `mean_glucose_7d_avg_descriptive` | float | **no** | 7-day rolling mean including today. For smoothed trend overlays in plots only. Using this as a feature leaks the target |

---

---

## hourly_features.csv

One row per hour (2,076 rows as of April 2026, covering 2026-02-01 to 2026-04-28). Built by
`scripts/01-build-hourly.py`. The primary modeling table for the hourly SARIMAX (notebook 03).
Target variable: **`glucose_hourly_mean`**.

The hourly grid is **complete and regular** - every hour from the first to the last CGM reading
is present, including hours with no sensor data. This is required for SARIMAX's evenly-spaced
assumption. Hours with no sensor data are represented as NaN in `glucose_hourly_mean`.

### Timestamp

| column | type | description |
|---|---|---|
| `ts` | datetime | Hour-start timestamp (UTC). Each row covers `[ts, ts + 1h)` |

### Target

| column | type | description |
|---|---|---|
| `glucose_hourly_mean` | float | **Primary target.** Mean of all CGM readings in `[ts, ts+1h)` (mg/dL). NaN for hours with no readings after gap handling (see below) |

### CGM diagnostics (not model inputs)

| column | type | description |
|---|---|---|
| `glucose_n_readings` | int | CGM readings in this hour (0–12 for 5-min sensor). 0 for gap hours; 0 with non-null mean for interpolated hours |
| `glucose_hourly_std` | float | Standard deviation of CGM readings in this hour (mg/dL). NaN if fewer than 2 readings |
| `glucose_hourly_min` | float | Minimum CGM reading in this hour (mg/dL) |
| `glucose_hourly_max` | float | Maximum CGM reading in this hour (mg/dL) |

**Sensor-gap policy:** Hours with zero CGM readings are NaN by default. Gaps of **≤ 2 consecutive
hours** are linearly interpolated (3 such hours in this dataset). Longer gaps remain NaN and are
handled by statsmodels' Kalman filter natively. The longest gap is 63 hours (sensor replacement).

### Carb-load features

All carb windows look strictly backward (no leakage). A meal at time `t` is included in
`carbs_last_1h` for hour `t` - this is intentional and not leakage, because at
counterfactual time the meal is a planned input.

Carb values use the same priority chain as the per-meal build:
`carbs_grams_logged > carbs_grams_estimated > carbs_g_lookup`. Meals with no carb estimate
contribute 0 to window sums and 0 to `carb_load_decayed`.

| column | type | description |
|---|---|---|
| `carbs_last_1h` | float | Sum of `carbs_g_final` for meals with `ts` in `(t-1h, t]` (g) |
| `carbs_last_2h` | float | Same over `(t-2h, t]` (g) |
| `carbs_last_3h` | float | Same over `(t-3h, t]` (g) |
| `carb_load_decayed` | float | Decay-weighted carb load: `sum(carbs_g_final * exp(-delta_min / 75))` over meals in the preceding 6h. tau=75 min reflects digestion kinetics (peak ~45-90 min post-meal). Primary carb feature for SARIMAX |
| `minutes_since_last_meal` | float | Minutes since the most recent meal, clipped at 720 min |
| `carbs_last_meal` | float | `carbs_g_final` of the most recent meal (g). Null if no carb value recorded for that meal |

### Activity features

| column | type | description |
|---|---|---|
| `walk_min_last_2h` | float | Sum of `duration_min` for activities in `(t-2h, t]` (minutes) |
| `walk_min_last_3h` | float | Same over `(t-3h, t]` (minutes) |
| `minutes_since_last_walk` | float | Minutes since the most recent walk activity, clipped at 720 min |

### Medication features

| column | type | description |
|---|---|---|
| `minutes_since_last_glipizide` | float | Minutes since the most recent **taken** Glipizide dose. NaN if no prior dose |
| `glipizide_active` | int | 1 if within the 12-hour glucose-lowering window of a taken Glipizide dose (IR formulation), else 0 |

### Fasting features

| column | type | description |
|---|---|---|
| `is_fasting` | int | 1 if hour `t` falls inside any logged fasting window (`fasting_windows.csv`), else 0 |
| `hours_into_fast` | float | Hours elapsed since the active fasting window started. 0 for non-fasting hours |

### Time encoding

| column | type | description |
|---|---|---|
| `hour` | int | Hour of day (0–23) |
| `hour_sin` | float | `sin(2pi * hour / 24)`, cyclical encoding of time of day |
| `hour_cos` | float | `cos(2pi * hour / 24)`, cyclical encoding of time of day |
| `is_weekend` | int | 1 if Saturday or Sunday, 0 otherwise |
| `day_index` | int | Days since the first date in the dataset (0-indexed). Useful as a linear trend term |

---

## Notes

**Null values.** Most nutrition and medication columns are partially null by design. The imputation strategy used in `01-regression-model.ipynb` is median imputation for numerics and most-frequent for categoricals, applied inside a sklearn pipeline fitted only on training data.

**Physio-day vs calendar-day.** When joining `daily_features` to an external calendar (e.g. for medication refill dates), remember that a row dated 2026-02-10 covers 2026-02-10 04:00 through 2026-02-11 03:59. The `mean_glucose_calendar` column is there specifically for comparisons that need the midnight-to-midnight view.

**`carbs_g_final` coverage.** Not every meal has a carb value. Meals that were never labeled with a dish name will have null `carbs_g_final`. Check
`carbs_grams_logged.notna()` vs `carbs_g_lookup.notna()` to understand how many rows fall back to each source.
