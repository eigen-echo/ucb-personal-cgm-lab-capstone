"""
Build hourly feature matrix for SARIMAX time-series modeling.

Reads from  data/raw/
Writes to   data/processed/hourly_features.csv

One row per hour (complete regular grid, s=24 friendly).
Target = glucose_hourly_mean (NaN for long sensor gaps).
Idempotent: re-running overwrites existing output.

Usage:
    python scripts/01-build-hourly.py
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

CARB_DECAY_TAU_MIN  = 75.0   # exponential decay time-constant (minutes)
GLIPIZIDE_WINDOW_H  = 12     # glucose-lowering duration of IR Glipizide (hours)
CLIP_MINUTES        = 720    # cap for "minutes since last X" features


# ============================================================== load raw CSVs
cgm   = pd.read_csv(os.path.join(RAW_DIR, 'cgm_5min.csv'),       parse_dates=['ts'])
meals = pd.read_csv(os.path.join(RAW_DIR, 'meals_v2.csv'),        parse_dates=['ts'])
acts  = pd.read_csv(os.path.join(RAW_DIR, 'activities.csv'),      parse_dates=['ts'])
meds  = pd.read_csv(os.path.join(RAW_DIR, 'medications.csv'),     parse_dates=['scheduled_ts'])
fast  = pd.read_csv(os.path.join(RAW_DIR, 'fasting_windows.csv'), parse_dates=['start_ts', 'end_ts'])
food  = pd.read_csv(os.path.join(RAW_DIR, 'food_lookup.csv'))

print(f"Loaded: cgm={len(cgm)}, meals={len(meals)}, acts={len(acts)}, "
      f"meds={len(meds)}, fast={len(fast)}")

cgm   = cgm.sort_values('ts').reset_index(drop=True)
meals = meals.sort_values('ts').reset_index(drop=True)
acts  = acts.sort_values('ts').reset_index(drop=True)
meds  = meds.sort_values('scheduled_ts').reset_index(drop=True)


# ============================================================== carb priority chain
# Same logic as 00-build-features.py
def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', (s or '').lower()).strip('_')

food['key_norm']  = food['dish_key'].map(_norm)
food['name_norm'] = food['display_name'].map(_norm)
food_idx = {}
for _, row in food.iterrows():
    food_idx.setdefault(row['key_norm'], row)
    food_idx.setdefault(row['name_norm'], row)

def _lookup_carbs(dish_names):
    if pd.isna(dish_names) or not str(dish_names).strip():
        return np.nan
    r = food_idx.get(_norm(str(dish_names)))
    return float(r['carbs_g']) if r is not None else np.nan

def _best_carbs(row):
    for col in ('carbs_grams_logged', 'carbs_grams_estimated'):
        v = row.get(col)
        if pd.notna(v) and v != '':
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return _lookup_carbs(row.get('dish_names'))

meals['carbs_g_final'] = meals.apply(_best_carbs, axis=1)


# ============================================================== hourly CGM grid
hour_min = cgm['ts'].min().floor('h')
hour_max = cgm['ts'].max().floor('h')
hourly_index = pd.date_range(hour_min, hour_max, freq='h')
print(f"Hourly grid: {hour_min} -> {hour_max}  ({len(hourly_index)} hours)")

cgm['hour_bucket'] = cgm['ts'].dt.floor('h')
hourly_glucose = cgm.groupby('hour_bucket')['glucose_mg_dl'].agg(
    glucose_hourly_mean='mean',
    glucose_hourly_std='std',
    glucose_hourly_min='min',
    glucose_hourly_max='max',
    glucose_n_readings='count',
).reindex(hourly_index)
hourly_glucose.index.name = 'ts'
hourly_glucose['glucose_n_readings'] = (
    hourly_glucose['glucose_n_readings'].fillna(0).astype(int)
)

# Sensor-gap handling: linear-interpolate gaps of ≤2 consecutive missing hours;
# leave longer gaps as NaN (statsmodels Kalman filter handles missing endog).
missing = hourly_glucose['glucose_hourly_mean'].isna()
gap_id  = (missing != missing.shift()).cumsum()
gap_len = missing.groupby(gap_id).transform('sum')
small_gap = missing & (gap_len <= 2)
interpolated = hourly_glucose['glucose_hourly_mean'].interpolate(method='linear')
hourly_glucose.loc[small_gap, 'glucose_hourly_mean'] = interpolated[small_gap]

n_filled  = int(small_gap.sum())
n_nan     = int(hourly_glucose['glucose_hourly_mean'].isna().sum())
print(f"  Gaps filled (<=2h): {n_filled}   NaN remaining (longer gaps): {n_nan}")

hourly = hourly_glucose.reset_index()


# ============================================================== helper: numpy ts arrays
meal_ts    = meals['ts'].values.astype('datetime64[ns]')
meal_carbs = meals['carbs_g_final'].values.astype(float)

act_ts  = acts['ts'].values.astype('datetime64[ns]')
act_dur = acts['duration_min'].values.astype(float)

# Glipizide: taken doses only - taken is exported as int 1/0
taken_mask = pd.to_numeric(meds['taken'], errors='coerce').fillna(0).astype(bool)
meds_glip  = meds[(meds['drug'] == 'Glipizide') & taken_mask].copy()
meds_glip  = meds_glip.sort_values('scheduled_ts').reset_index(drop=True)
glip_ts   = meds_glip['scheduled_ts'].values.astype('datetime64[ns]')

fast_clean  = fast.dropna(subset=['start_ts', 'end_ts']).copy()
fast_starts = fast_clean['start_ts'].values.astype('datetime64[ns]')
fast_ends   = fast_clean['end_ts'].values.astype('datetime64[ns]')

n = len(hourly)
ts_arr = hourly['ts'].values.astype('datetime64[ns]')


# ============================================================== carb features
carbs_last_1h           = np.zeros(n)
carbs_last_2h           = np.zeros(n)
carbs_last_3h           = np.zeros(n)
carb_load_decayed       = np.zeros(n)
minutes_since_last_meal = np.full(n, float(CLIP_MINUTES))
carbs_last_meal         = np.full(n, np.nan)

_1h  = np.timedelta64(60,  'm')
_2h  = np.timedelta64(120, 'm')
_3h  = np.timedelta64(180, 'm')
_6h  = np.timedelta64(360, 'm')

print("Building carb features...")
for i, t in enumerate(ts_arr):
    # Index of first meal AFTER t (all prior meals are at indices < hi)
    hi = int(np.searchsorted(meal_ts, t, side='right'))

    # Most recent meal (any time before t)
    if hi > 0:
        recent_delta_s = float((t - meal_ts[hi - 1]) / np.timedelta64(1, 's'))
        minutes_since_last_meal[i] = min(recent_delta_s / 60.0, CLIP_MINUTES)
        carbs_last_meal[i] = meal_carbs[hi - 1]

    # Meals within 3h window: (t-3h, t]
    lo_3h = int(np.searchsorted(meal_ts, t - _3h, side='right'))
    if lo_3h < hi:
        sub_c = meal_carbs[lo_3h:hi]
        sub_d = (t - meal_ts[lo_3h:hi]) / np.timedelta64(1, 's') / 60.0  # minutes
        carbs_last_1h[i] = float(np.nansum(sub_c[sub_d <= 60.0]))
        carbs_last_2h[i] = float(np.nansum(sub_c[sub_d <= 120.0]))
        carbs_last_3h[i] = float(np.nansum(sub_c))

    # Decay window: 6h (≈4.8 tau; contribution < 0.1% of τ=75min beyond)
    lo_6h = int(np.searchsorted(meal_ts, t - _6h, side='right'))
    if lo_6h < hi:
        sub_c6 = meal_carbs[lo_6h:hi]
        sub_d6 = (t - meal_ts[lo_6h:hi]) / np.timedelta64(1, 's') / 60.0
        valid  = np.where(np.isnan(sub_c6), 0.0, sub_c6)
        carb_load_decayed[i] = float(np.sum(valid * np.exp(-sub_d6 / CARB_DECAY_TAU_MIN)))

hourly['carbs_last_1h']           = carbs_last_1h
hourly['carbs_last_2h']           = carbs_last_2h
hourly['carbs_last_3h']           = carbs_last_3h
hourly['carb_load_decayed']       = np.round(carb_load_decayed, 3)
hourly['minutes_since_last_meal'] = np.round(minutes_since_last_meal, 1)
hourly['carbs_last_meal']         = carbs_last_meal
print("  done.")


# ============================================================== activity features
walk_min_last_2h        = np.zeros(n)
walk_min_last_3h        = np.zeros(n)
minutes_since_last_walk = np.full(n, float(CLIP_MINUTES))

print("Building activity features...")
for i, t in enumerate(ts_arr):
    hi = int(np.searchsorted(act_ts, t, side='right'))

    # Most recent activity (any time before t)
    if hi > 0:
        recent_delta_s = float((t - act_ts[hi - 1]) / np.timedelta64(1, 's'))
        minutes_since_last_walk[i] = min(recent_delta_s / 60.0, CLIP_MINUTES)

    # Activities within 3h window
    lo_3h = int(np.searchsorted(act_ts, t - _3h, side='right'))
    if lo_3h < hi:
        sub_dur = act_dur[lo_3h:hi]
        sub_d   = (t - act_ts[lo_3h:hi]) / np.timedelta64(1, 's') / 60.0
        walk_min_last_2h[i] = float(np.nansum(sub_dur[sub_d <= 120.0]))
        walk_min_last_3h[i] = float(np.nansum(sub_dur))

hourly['walk_min_last_2h']        = walk_min_last_2h
hourly['walk_min_last_3h']        = walk_min_last_3h
hourly['minutes_since_last_walk'] = np.round(minutes_since_last_walk, 1)
print("  done.")


# ============================================================== medication features
minutes_since_last_glipizide = np.full(n, np.nan)
glipizide_active             = np.zeros(n, dtype=int)

print("Building medication features...")
if len(glip_ts) > 0:
    for i, t in enumerate(ts_arr):
        hi = int(np.searchsorted(glip_ts, t, side='right'))
        if hi > 0:
            delta_s   = float((t - glip_ts[hi - 1]) / np.timedelta64(1, 's'))
            delta_min = delta_s / 60.0
            minutes_since_last_glipizide[i] = delta_min
            if delta_min <= GLIPIZIDE_WINDOW_H * 60:
                glipizide_active[i] = 1
else:
    print("  No taken Glipizide doses found - columns will be NaN/0.")

hourly['minutes_since_last_glipizide'] = np.round(minutes_since_last_glipizide, 1)
hourly['glipizide_active']             = glipizide_active
print("  done.")


# ============================================================== fasting features
is_fasting      = np.zeros(n, dtype=int)
hours_into_fast = np.zeros(n)

print("Building fasting features...")
for i, t in enumerate(ts_arr):
    for j in range(len(fast_starts)):
        if fast_starts[j] <= t <= fast_ends[j]:
            is_fasting[i] = 1
            hours_into_fast[i] = float(
                (t - fast_starts[j]) / np.timedelta64(1, 's') / 3600.0
            )
            break

hourly['is_fasting']      = is_fasting
hourly['hours_into_fast'] = np.round(hours_into_fast, 2)
print("  done.")


# ============================================================== time encoding
hourly['hour']       = hourly['ts'].dt.hour
hourly['hour_sin']   = np.sin(2 * np.pi * hourly['hour'] / 24)
hourly['hour_cos']   = np.cos(2 * np.pi * hourly['hour'] / 24)
hourly['is_weekend'] = hourly['ts'].dt.dayofweek.isin([5, 6]).astype(int)
hourly['day_index']  = (
    hourly['ts'].dt.normalize() - hourly['ts'].dt.normalize().min()
).dt.days


# ============================================================== write output
HOURLY_COLS = [
    'ts',
    # Target
    'glucose_hourly_mean',
    # Diagnostic / audit (not model inputs)
    'glucose_n_readings', 'glucose_hourly_std', 'glucose_hourly_min', 'glucose_hourly_max',
    # Carb features
    'carbs_last_1h', 'carbs_last_2h', 'carbs_last_3h',
    'carb_load_decayed', 'minutes_since_last_meal', 'carbs_last_meal',
    # Activity features
    'walk_min_last_2h', 'walk_min_last_3h', 'minutes_since_last_walk',
    # Medication features
    'minutes_since_last_glipizide', 'glipizide_active',
    # Fasting features
    'is_fasting', 'hours_into_fast',
    # Time encoding
    'hour', 'hour_sin', 'hour_cos', 'is_weekend', 'day_index',
]
hourly = hourly[HOURLY_COLS]

out_path = os.path.join(OUT_DIR, 'hourly_features.csv')
hourly.to_csv(out_path, index=False)

print(f"\n=== Hourly feature matrix ===")
print(f"  shape:                       {hourly.shape}")
print(f"  ts:                          {hourly['ts'].min()} -> {hourly['ts'].max()}")
print(f"  glucose non-null rows:       "
      f"{hourly['glucose_hourly_mean'].notna().sum()} / {len(hourly)}")
print(f"  is_fasting=1 hours:          {hourly['is_fasting'].sum()}")
print(f"  glipizide_active=1 hours:    {hourly['glipizide_active'].sum()}")
print(f"  hours with carbs (last 3h):  {(hourly['carbs_last_3h'] > 0).sum()}")
print(f"  hours with walk  (last 3h):  {(hourly['walk_min_last_3h'] > 0).sum()}")
print(f"\nDone. Written to {out_path}")
