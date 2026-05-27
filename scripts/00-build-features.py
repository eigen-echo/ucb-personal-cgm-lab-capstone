"""
Build processed feature matrices directly from raw CSVs.

Reads from  data/raw/
Writes to   data/processed/

Outputs:
  per_meal_features.csv (.parquet)  -- one row per meal; target = peak_delta
  daily_features.csv    (.parquet)  -- one row per physio-day (4 AM boundary)

Idempotent: re-running overwrites existing outputs.

Usage:
    python scripts/build_features.py
"""

import os
import re

import numpy as np
import pandas as pd

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.abspath(os.path.join(HERE, '..'))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
OUT_DIR = os.path.join(ROOT, 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

PHYSIO_DAY_HOUR = 4


def to_physio(ts_series):
    return (ts_series - pd.Timedelta(hours=PHYSIO_DAY_HOUR)).dt.normalize()


def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', (s or '').lower()).strip('_')


# ============================================================== load raw CSVs
cgm   = pd.read_csv(os.path.join(RAW_DIR, 'cgm_5min.csv'),
                    parse_dates=['ts'])
meals = pd.read_csv(os.path.join(RAW_DIR, 'meals_v2.csv'),
                    parse_dates=['ts'])
acts  = pd.read_csv(os.path.join(RAW_DIR, 'activities.csv'),
                    parse_dates=['ts'])
meds  = pd.read_csv(os.path.join(RAW_DIR, 'medications.csv'),
                    parse_dates=['scheduled_ts'])
fast  = pd.read_csv(os.path.join(RAW_DIR, 'fasting_windows.csv'),
                    parse_dates=['start_ts', 'end_ts'])
food  = pd.read_csv(os.path.join(RAW_DIR, 'food_lookup.csv'))

print(f"Loaded: cgm={len(cgm)}, meals={len(meals)}, acts={len(acts)}, "
      f"meds={len(meds)}, fast={len(fast)}, food={len(food)}")

cgm   = cgm.sort_values('ts').reset_index(drop=True)
meals = meals.sort_values('ts').reset_index(drop=True)
acts  = acts.sort_values('ts').reset_index(drop=True)


# ============================================================== food lookup
food['key_norm']  = food['dish_key'].map(_norm)
food['name_norm'] = food['display_name'].map(_norm)
food_idx = {}
for _, r in food.iterrows():
    food_idx.setdefault(r['key_norm'], r)
    food_idx.setdefault(r['name_norm'], r)


def lookup_food(dish_names):
    if pd.isna(dish_names) or not str(dish_names).strip():
        return pd.Series({'carbs_g_lookup':   np.nan,
                          'protein_g_lookup': np.nan,
                          'fat_g_lookup':     np.nan,
                          'gl_lookup':        np.nan,
                          'food_match':       None})
    r = food_idx.get(_norm(str(dish_names)))
    if r is None:
        return pd.Series({'carbs_g_lookup':   np.nan,
                          'protein_g_lookup': np.nan,
                          'fat_g_lookup':     np.nan,
                          'gl_lookup':        np.nan,
                          'food_match':       None})
    return pd.Series({'carbs_g_lookup':   r['carbs_g'],
                      'protein_g_lookup': r['protein_g'],
                      'fat_g_lookup':     r['fat_g'],
                      'gl_lookup':        r['gl'],
                      'food_match':       r['dish_key']})


meals = pd.concat([meals, meals['dish_names'].apply(lookup_food)], axis=1)


# ============================================================== CGM rolling features per meal
cgm_idx = cgm.set_index('ts')['glucose_mg_dl'].sort_index()
cgm_idx = cgm_idx[~cgm_idx.index.duplicated(keep='first')]


def compute_meal_outcome(meal_ts):
    """Compute pre_meal_glucose, peak_glucose, peak_delta, time_to_peak_min from CGM data."""
    # Pre-meal: last reading within 10 min before (or at) meal time
    pre_window = cgm_idx.loc[meal_ts - pd.Timedelta(minutes=10) : meal_ts]
    pre_meal_g = float(pre_window.iloc[-1]) if len(pre_window) else np.nan

    # Post-meal: CGM readings up to 3 hours after meal
    post_window = cgm_idx.loc[meal_ts : meal_ts + pd.Timedelta(hours=3)]
    if len(post_window) < 3 or pd.isna(pre_meal_g):
        return pd.Series({
            'pre_meal_glucose': pre_meal_g,
            'peak_glucose':     np.nan,
            'peak_delta':       np.nan,
            'time_to_peak_min': np.nan,
        })

    peak_idx = post_window.idxmax()
    peak_g   = float(post_window.max())
    return pd.Series({
        'pre_meal_glucose': pre_meal_g,
        'peak_glucose':     peak_g,
        'peak_delta':       max(0.0, peak_g - pre_meal_g),
        'time_to_peak_min': float((peak_idx - meal_ts).total_seconds() / 60),
    })


outcome_df = meals['ts'].apply(compute_meal_outcome)
meals = pd.concat([meals, outcome_df], axis=1)
print(f"  Meal outcomes computed: "
      f"{meals['peak_delta'].notna().sum()}/{len(meals)} with valid peak_delta")


def cgm_window_mean(ts, minutes):
    start = ts - pd.Timedelta(minutes=minutes)
    s = cgm_idx.loc[(cgm_idx.index > start) & (cgm_idx.index <= ts)]
    return float(s.mean()) if len(s) else np.nan


def cgm_value_near(ts, minutes_before):
    target = ts - pd.Timedelta(minutes=minutes_before)
    window = cgm_idx.loc[
        (cgm_idx.index >= target - pd.Timedelta(minutes=5))
        & (cgm_idx.index <= target + pd.Timedelta(minutes=5))]
    return float(window.iloc[0]) if len(window) else np.nan


pre30, pre60, vel30 = [], [], []
for ts in meals['ts']:
    pre30.append(cgm_window_mean(ts, 30))
    pre60.append(cgm_window_mean(ts, 60))
    v0  = cgm_value_near(ts, 0)
    v30 = cgm_value_near(ts, 30)
    vel30.append((v0 - v30) / 30.0
                 if (not np.isnan(v0) and not np.isnan(v30)) else np.nan)
meals['glucose_30min_before_mean'] = pre30
meals['glucose_60min_before_mean'] = pre60
meals['glucose_velocity_30min']    = vel30


# ============================================================== activity per meal
def walk_minutes_in_window(meal_ts, minutes_after):
    end  = meal_ts + pd.Timedelta(minutes=minutes_after)
    mask = (acts['ts'] >= meal_ts) & (acts['ts'] <= end)
    return float(acts.loc[mask, 'duration_min'].fillna(0).sum())


def minutes_to_first_walk(meal_ts, minutes_after=180):
    end  = meal_ts + pd.Timedelta(minutes=minutes_after)
    mask = (acts['ts'] >= meal_ts) & (acts['ts'] <= end)
    sub  = acts.loc[mask, 'ts']
    if len(sub) == 0:
        return np.nan
    return float((sub.iloc[0] - meal_ts).total_seconds() / 60)


meals['walk_min_within_60']    = meals['ts'].apply(lambda t: walk_minutes_in_window(t, 60))
meals['walk_min_within_90']    = meals['ts'].apply(lambda t: walk_minutes_in_window(t, 90))
meals['walk_min_within_180']   = meals['ts'].apply(lambda t: walk_minutes_in_window(t, 180))
meals['minutes_to_first_walk'] = meals['ts'].apply(minutes_to_first_walk)


# ============================================================== medication per meal
meds_taken = meds[pd.to_numeric(meds['taken'], errors='coerce').fillna(0).astype(bool)].sort_values('scheduled_ts').reset_index(drop=True)


def last_dose_features(meal_ts):
    prior = meds_taken[meds_taken['scheduled_ts'] <= meal_ts]
    if prior.empty:
        return pd.Series({'minutes_since_last_dose':       np.nan,
                          'last_drug':                     None,
                          'minutes_since_last_glipizide':  np.nan})
    last = prior.iloc[-1]
    glip = prior[prior['drug'] == 'Glipizide']
    glip_min = (float((meal_ts - glip.iloc[-1]['scheduled_ts']).total_seconds() / 60)
                if not glip.empty else np.nan)
    return pd.Series({
        'minutes_since_last_dose':      float(
            (meal_ts - last['scheduled_ts']).total_seconds() / 60),
        'last_drug':                    last['drug'],
        'minutes_since_last_glipizide': glip_min,
    })


meals = pd.concat([meals, meals['ts'].apply(last_dose_features)], axis=1)


# ============================================================== fasting per meal
def fasting_features(row):
    meal_ts = row['ts']
    window  = fast[(fast['end_ts'] >= meal_ts - pd.Timedelta(minutes=5))
                   & (fast['end_ts'] <= meal_ts + pd.Timedelta(minutes=5))]
    if window.empty:
        before = fast[fast['end_ts'] < meal_ts].sort_values('end_ts')
        if before.empty:
            return pd.Series({'broke_if_window':   False,
                              'broke_window_type': None,
                              'prev_fast_hours':   np.nan})
        last = before.iloc[-1]
        return pd.Series({'broke_if_window':   False,
                          'broke_window_type': last['window_type'],
                          'prev_fast_hours':   float(last['duration_hours'])
                          if pd.notna(last['duration_hours']) else np.nan})
    w = window.iloc[0]
    return pd.Series({
        'broke_if_window':   w['window_type'] == 'intermittent_skip_breakfast',
        'broke_window_type': w['window_type'],
        'prev_fast_hours':   float(w['duration_hours'])
        if pd.notna(w['duration_hours']) else np.nan,
    })


meals = pd.concat([meals, meals.apply(fasting_features, axis=1)], axis=1)


# ============================================================== meal time / cyclic / carbs
meals['hour']       = meals['ts'].dt.hour
meals['hour_sin']   = np.sin(2 * np.pi * meals['hour'] / 24)
meals['hour_cos']   = np.cos(2 * np.pi * meals['hour'] / 24)
meals['is_weekend'] = meals['dow'].isin(['Saturday', 'Sunday']).astype(int)
meals['day_index']  = (meals['ts'].dt.normalize()
                       - meals['ts'].dt.normalize().min()).dt.days


def best_carbs(row):
    for col in ('carbs_grams_logged', 'carbs_grams_estimated', 'carbs_g_lookup'):
        v = row.get(col)
        if pd.notna(v) and v != '':
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return np.nan


meals['carbs_g_final']    = meals.apply(best_carbs, axis=1)
meals['out_of_range_180'] = (
    pd.to_numeric(meals['peak_glucose'], errors='coerce') > 180).astype('Int64')


# ============================================================== write per-meal matrix
PER_MEAL_COLS = [
    'meal_id', 'ts', 'date', 'dow', 'time_bucket', 'meal_type',
    'dish_names', 'food_match',
    'peak_delta', 'peak_glucose', 'time_to_peak_min', 'out_of_range_180',
    'pre_meal_glucose', 'glucose_30min_before_mean',
    'glucose_60min_before_mean', 'glucose_velocity_30min',
    'fasted_meal',
    'carbs_g_final', 'carbs_grams_logged', 'carbs_grams_estimated',
    'carbs_g_lookup', 'protein_g_lookup', 'fat_g_lookup', 'gl_lookup',
    'walk_min_within_60', 'walk_min_within_90', 'walk_min_within_180',
    'minutes_to_first_walk',
    'minutes_since_last_dose', 'last_drug', 'minutes_since_last_glipizide',
    'broke_if_window', 'broke_window_type', 'prev_fast_hours',
    'hour', 'hour_sin', 'hour_cos', 'is_weekend', 'day_index',
]
per_meal = meals[PER_MEAL_COLS].copy()
per_meal_path = os.path.join(OUT_DIR, 'per_meal_features.csv')
per_meal.to_csv(per_meal_path, index=False)

# ============================================================== DAILY MATRIX
print("\nBuilding daily feature matrix...")

cgm['date_calendar'] = cgm['ts'].dt.normalize()
cgm['date_physio']   = to_physio(cgm['ts'])
cgm['date']          = cgm['date_physio']

fast_clean = fast.dropna(subset=['start_ts', 'end_ts']).copy()
intervals  = pd.IntervalIndex.from_arrays(
    fast_clean['start_ts'].values, fast_clean['end_ts'].values, closed='both')
cgm['is_fasting'] = cgm['ts'].apply(
    lambda t: bool(intervals.contains(t).any()))
print(f"  CGM points flagged is_fasting=True: "
      f"{int(cgm['is_fasting'].sum())} / {len(cgm)} "
      f"({100 * cgm['is_fasting'].mean():.1f}%)")


def tir_pct(s):
    return 100 * ((s >= 70) & (s <= 180)).mean()


daily_glucose = cgm.groupby('date').agg(
    mean_glucose=('glucose_mg_dl', 'mean'),
    min_glucose=('glucose_mg_dl', 'min'),
    max_glucose=('glucose_mg_dl', 'max'),
    std_glucose=('glucose_mg_dl', 'std'),
    n_readings=('glucose_mg_dl', 'count'),
).reset_index()
daily_glucose['mean_glucose'] = daily_glucose['mean_glucose'].round(2)

cal_means = (cgm.groupby('date_calendar')['glucose_mg_dl']
             .mean().round(2)
             .rename('mean_glucose_calendar')
             .reset_index()
             .rename(columns={'date_calendar': 'date'}))
daily_glucose = daily_glucose.merge(cal_means, on='date', how='left')

fast_means = (cgm[cgm['is_fasting']].groupby('date')['glucose_mg_dl']
              .agg(['mean', 'count'])
              .rename(columns={'mean':  'fast_window_mean_glucose',
                               'count': 'fast_window_n_readings'})
              .reset_index())
fast_means['fast_window_mean_glucose'] = fast_means['fast_window_mean_glucose'].round(2)

feed_means = (cgm[~cgm['is_fasting']].groupby('date')['glucose_mg_dl']
              .agg(['mean', 'count'])
              .rename(columns={'mean':  'feeding_window_mean_glucose',
                               'count': 'feeding_window_n_readings'})
              .reset_index())
feed_means['feeding_window_mean_glucose'] = feed_means['feeding_window_mean_glucose'].round(2)

daily_glucose = (daily_glucose
                 .merge(fast_means, on='date', how='left')
                 .merge(feed_means, on='date', how='left'))

fast_tir = (cgm[cgm['is_fasting']].groupby('date')['glucose_mg_dl']
            .apply(tir_pct).round(1)
            .rename('fast_window_tir_pct').reset_index())
feed_tir = (cgm[~cgm['is_fasting']].groupby('date')['glucose_mg_dl']
            .apply(tir_pct).round(1)
            .rename('feeding_window_tir_pct').reset_index())
daily_glucose = (daily_glucose
                 .merge(fast_tir, on='date', how='left')
                 .merge(feed_tir, on='date', how='left'))


def daily_tir(g):
    return pd.Series({
        'time_in_range_pct':  round(100 * ((g >= 70) & (g <= 180)).mean(), 1),
        'time_above_180_pct': round(100 * (g > 180).mean(), 1),
        'time_below_70_pct':  round(100 * (g < 70).mean(),  1),
    })


tir = cgm.groupby('date')['glucose_mg_dl'].apply(daily_tir).unstack().reset_index()
daily_glucose = daily_glucose.merge(tir, on='date', how='left')

daily_glucose['gmi_estimate'] = (
    3.31 + 0.02392 * daily_glucose['mean_glucose']).round(2)


acts['date'] = to_physio(acts['ts'])
daily_act = acts.groupby('date').agg(
    n_walks=('activity_id', 'count'),
    total_walk_min=('duration_min', 'sum'),
    max_walk_min=('duration_min', 'max'),
).reset_index()

meals_d = meals.copy()
meals_d['date_phys'] = to_physio(meals_d['ts'])
daily_meals = (meals_d.groupby('date_phys').agg(
    n_meals=('meal_id', 'count'),
    total_carbs=('carbs_g_final', 'sum'),
    avg_peak_delta=('peak_delta', 'mean'),
    max_peak_delta=('peak_delta', 'max'),
).reset_index()
.rename(columns={'date_phys': 'date'}))

fast_d  = fast.copy()
fast_d['date'] = pd.to_datetime(fast_d['date'])
if_days   = (fast_d[fast_d['window_type'] == 'intermittent_skip_breakfast']
             [['date', 'duration_hours']]
             .rename(columns={'duration_hours': 'if_duration_hours'}))
overnight = (fast_d[fast_d['window_type'] == 'overnight']
             [['date', 'duration_hours']]
             .rename(columns={'duration_hours': 'overnight_fast_hours'}))

meds['date'] = to_physio(meds['scheduled_ts'])
daily_doses  = (meds[pd.to_numeric(meds['taken'], errors='coerce').fillna(0).astype(bool)]
                .groupby('date').agg(doses_taken=('med_id', 'count'))
                .reset_index())

daily = (daily_glucose
         .merge(daily_act,   on='date', how='left')
         .merge(daily_meals, on='date', how='left')
         .merge(if_days,     on='date', how='left')
         .merge(overnight,   on='date', how='left')
         .merge(daily_doses, on='date', how='left'))

for c in ['n_walks', 'total_walk_min', 'max_walk_min',
          'n_meals', 'total_carbs', 'doses_taken']:
    if c in daily.columns:
        daily[c] = daily[c].fillna(0)

daily['is_if_day']  = daily['if_duration_hours'].notna().astype(int)
daily['dow']        = daily['date'].dt.day_name()
daily['is_weekend'] = daily['dow'].isin(['Saturday', 'Sunday']).astype(int)
daily['day_index']  = (daily['date'] - daily['date'].min()).dt.days

# ---- Lag / rolling features ----
# Leakage policy: shift(1) before rolling so the window covers [N-7, N-1],
# not [N-6, N]. _descriptive columns include today and are for plotting only.
daily = daily.sort_values('date').reset_index(drop=True)

daily['mean_glucose_lag1']   = daily['mean_glucose'].shift(1)
daily['total_walk_min_lag1'] = daily['total_walk_min'].shift(1)
daily['total_carbs_lag1']    = daily['total_carbs'].shift(1)

daily['mean_glucose_7d_avg_prior'] = (
    daily['mean_glucose'].shift(1).rolling(7, min_periods=3).mean())

# Trailing 7-day avg including today - for descriptive plots only, not a feature.
daily['mean_glucose_7d_avg_descriptive'] = (
    daily['mean_glucose'].rolling(7, min_periods=3).mean())


DAILY_COLS = [
    'date', 'dow', 'is_weekend', 'day_index',
    # Primary targets (physio-day, 04:00 boundary)
    'mean_glucose', 'time_in_range_pct', 'gmi_estimate',
    'time_above_180_pct', 'time_below_70_pct',
    # Calendar-day mean for endocrinologist-comparable reporting
    'mean_glucose_calendar',
    # Fast vs feeding window decomposition
    'fast_window_mean_glucose', 'feeding_window_mean_glucose',
    'fast_window_tir_pct',     'feeding_window_tir_pct',
    'fast_window_n_readings',  'feeding_window_n_readings',
    # Glucose distribution
    'min_glucose', 'max_glucose', 'std_glucose', 'n_readings',
    # Behavior (exogenous regressors)
    'n_walks', 'total_walk_min', 'max_walk_min',
    'n_meals', 'total_carbs',
    'avg_peak_delta', 'max_peak_delta',
    'is_if_day', 'if_duration_hours', 'overnight_fast_hours',
    'doses_taken',
    # Lag / rolling (_prior excludes today; _descriptive includes today).
    'mean_glucose_lag1',
    'mean_glucose_7d_avg_prior', 'mean_glucose_7d_avg_descriptive',
    'total_walk_min_lag1', 'total_carbs_lag1',
]
daily = daily[DAILY_COLS]

daily_path = os.path.join(OUT_DIR, 'daily_features.csv')
daily.to_csv(daily_path, index=False)

# ============================================================== summary
print("\n=== Per-meal feature matrix ===")
print(f"  shape: {per_meal.shape}")
print(f"  meals with carbs:        "
      f"{per_meal['carbs_g_final'].notna().sum()} / {len(per_meal)}")
print(f"  meals with food match:   "
      f"{per_meal['food_match'].notna().sum()} / {len(per_meal)}")
print(f"  meals followed by walk:  "
      f"{(per_meal['walk_min_within_90'] > 0).sum()}")
print(f"  meals breaking IF:       {per_meal['broke_if_window'].sum()}")
print(f"  target peak_delta non-null: "
      f"{pd.to_numeric(per_meal['peak_delta'], errors='coerce').notna().sum()}")

print("\n=== Daily feature matrix (physio-day, 04:00 boundary) ===")
print(f"  shape: {daily.shape}")
print(f"  dates: {daily['date'].min().date()} -> {daily['date'].max().date()}")
print(f"  IF days: {daily['is_if_day'].sum()}")
print(f"  mean glucose (physio):       {daily['mean_glucose'].mean():.1f}")
print(f"  mean glucose (calendar):     {daily['mean_glucose_calendar'].mean():.1f}")
print(f"  mean TIR:                    {daily['time_in_range_pct'].mean():.1f}%")
print(f"  mean GMI:                    {daily['gmi_estimate'].mean():.2f}")
print(f"  fast-window mean glucose:    "
      f"{daily['fast_window_mean_glucose'].mean():.1f}")
print(f"  feeding-window mean glucose: "
      f"{daily['feeding_window_mean_glucose'].mean():.1f}")
print(f"  fast-window TIR:    {daily['fast_window_tir_pct'].mean():.1f}%")
print(f"  feeding-window TIR: {daily['feeding_window_tir_pct'].mean():.1f}%")

print(f"\nDone. Outputs in {OUT_DIR}/")
