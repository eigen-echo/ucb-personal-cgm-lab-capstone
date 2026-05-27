"""
Build 5-minute feature matrix for LSTM time-series modeling.

Reads from  data/raw/
Writes to   data/processed/5min_features.csv

One row per 5-minute slot (complete regular grid).
Target = glucose_mg_dl (NaN for sensor gaps > 10 min).
Idempotent: re-running overwrites existing output.

Note: CGM timestamps are NOT on 5-min boundaries (e.g., 11:47:10).
      Readings are snapped to the nearest 5-min floor bucket.

Usage:
    python scripts/02-build-5min.py
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


# ============================================================== 5-min CGM grid
# Snap CGM timestamps to 5-min floor buckets (raw timestamps are not aligned)
cgm['ts_5min'] = cgm['ts'].dt.floor('5min')

# Keep one reading per 5-min bucket (average if multiple fall in same bucket)
cgm_snapped = (
    cgm.groupby('ts_5min')['glucose_mg_dl']
    .mean()
    .rename('glucose_mg_dl')
)

grid_min = cgm['ts_5min'].min()
grid_max = cgm['ts_5min'].max()
grid_5min = pd.date_range(grid_min, grid_max, freq='5min')
print(f"5-min grid: {grid_min} -> {grid_max}  ({len(grid_5min)} slots)")

# Reindex to complete grid (NaN where no CGM reading)
glucose_series = cgm_snapped.reindex(grid_5min)
glucose_series.index.name = 'ts'

# Sensor-gap handling: linear-interpolate gaps of <=2 consecutive missing slots
# (<=10 min); leave longer gaps as NaN.
missing  = glucose_series.isna()
gap_id   = (missing != missing.shift()).cumsum()
gap_len  = missing.groupby(gap_id).transform('sum')
small_gap = missing & (gap_len <= 2)
interpolated = glucose_series.interpolate(method='linear')
glucose_series[small_gap] = interpolated[small_gap]

n_filled = int(small_gap.sum())
n_nan    = int(glucose_series.isna().sum())
print(f"  Gaps filled (<=2 slots / <=10 min): {n_filled}   "
      f"NaN remaining (longer gaps): {n_nan}")

df = glucose_series.reset_index()
df.columns = ['ts', 'glucose_mg_dl']


# ============================================================== helper: numpy ts arrays
meal_ts    = meals['ts'].values.astype('datetime64[ns]')
meal_carbs = meals['carbs_g_final'].values.astype(float)

act_ts  = acts['ts'].values.astype('datetime64[ns]')
act_dur = acts['duration_min'].values.astype(float)

# Glipizide: taken doses only
taken_col = meds['taken']
if taken_col.dtype == bool:
    meds_glip = meds[(meds['drug'] == 'Glipizide') & (taken_col)].copy()
else:
    meds_glip = meds[(meds['drug'] == 'Glipizide') & (taken_col.astype(str) == 'TRUE')].copy()
meds_glip = meds_glip.sort_values('scheduled_ts').reset_index(drop=True)
glip_ts   = meds_glip['scheduled_ts'].values.astype('datetime64[ns]')

fast_clean  = fast.dropna(subset=['start_ts', 'end_ts']).copy()
fast_starts = fast_clean['start_ts'].values.astype('datetime64[ns]')
fast_ends   = fast_clean['end_ts'].values.astype('datetime64[ns]')

n      = len(df)
ts_arr = df['ts'].values.astype('datetime64[ns]')


# ============================================================== carb features
carbs_last_30min        = np.zeros(n)
carbs_last_1h           = np.zeros(n)
carbs_last_2h           = np.zeros(n)
carbs_last_3h           = np.zeros(n)
carb_load_decayed       = np.zeros(n)
minutes_since_last_meal = np.full(n, float(CLIP_MINUTES))
carbs_last_meal         = np.full(n, np.nan)

_30m = np.timedelta64(30,  'm')
_1h  = np.timedelta64(60,  'm')
_2h  = np.timedelta64(120, 'm')
_3h  = np.timedelta64(180, 'm')
_6h  = np.timedelta64(360, 'm')

print("Building carb features...")
for i, t in enumerate(ts_arr):
    hi = int(np.searchsorted(meal_ts, t, side='right'))

    if hi > 0:
        recent_delta_s = float((t - meal_ts[hi - 1]) / np.timedelta64(1, 's'))
        minutes_since_last_meal[i] = min(recent_delta_s / 60.0, CLIP_MINUTES)
        carbs_last_meal[i] = meal_carbs[hi - 1]

    # Meals within 3h window: (t-3h, t]
    lo_3h = int(np.searchsorted(meal_ts, t - _3h, side='right'))
    if lo_3h < hi:
        sub_c = meal_carbs[lo_3h:hi]
        sub_d = (t - meal_ts[lo_3h:hi]) / np.timedelta64(1, 's') / 60.0  # minutes
        carbs_last_30min[i] = float(np.nansum(sub_c[sub_d <= 30.0]))
        carbs_last_1h[i]    = float(np.nansum(sub_c[sub_d <= 60.0]))
        carbs_last_2h[i]    = float(np.nansum(sub_c[sub_d <= 120.0]))
        carbs_last_3h[i]    = float(np.nansum(sub_c))

    # Decay window: 6h
    lo_6h = int(np.searchsorted(meal_ts, t - _6h, side='right'))
    if lo_6h < hi:
        sub_c6 = meal_carbs[lo_6h:hi]
        sub_d6 = (t - meal_ts[lo_6h:hi]) / np.timedelta64(1, 's') / 60.0
        valid  = np.where(np.isnan(sub_c6), 0.0, sub_c6)
        carb_load_decayed[i] = float(np.sum(valid * np.exp(-sub_d6 / CARB_DECAY_TAU_MIN)))

df['carbs_last_30min']        = carbs_last_30min
df['carbs_last_1h']           = carbs_last_1h
df['carbs_last_2h']           = carbs_last_2h
df['carbs_last_3h']           = carbs_last_3h
df['carb_load_decayed']       = np.round(carb_load_decayed, 3)
df['minutes_since_last_meal'] = np.round(minutes_since_last_meal, 1)
df['carbs_last_meal']         = carbs_last_meal
print("  done.")


# ============================================================== activity features
walk_min_last_1h        = np.zeros(n)
walk_min_last_2h        = np.zeros(n)
walk_min_last_3h        = np.zeros(n)
minutes_since_last_walk = np.full(n, float(CLIP_MINUTES))

print("Building activity features...")
for i, t in enumerate(ts_arr):
    hi = int(np.searchsorted(act_ts, t, side='right'))

    if hi > 0:
        recent_delta_s = float((t - act_ts[hi - 1]) / np.timedelta64(1, 's'))
        minutes_since_last_walk[i] = min(recent_delta_s / 60.0, CLIP_MINUTES)

    lo_3h = int(np.searchsorted(act_ts, t - _3h, side='right'))
    if lo_3h < hi:
        sub_dur = act_dur[lo_3h:hi]
        sub_d   = (t - act_ts[lo_3h:hi]) / np.timedelta64(1, 's') / 60.0
        walk_min_last_1h[i] = float(np.nansum(sub_dur[sub_d <= 60.0]))
        walk_min_last_2h[i] = float(np.nansum(sub_dur[sub_d <= 120.0]))
        walk_min_last_3h[i] = float(np.nansum(sub_dur))

df['walk_min_last_1h']        = walk_min_last_1h
df['walk_min_last_2h']        = walk_min_last_2h
df['walk_min_last_3h']        = walk_min_last_3h
df['minutes_since_last_walk'] = np.round(minutes_since_last_walk, 1)
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
    print("  No taken Glipizide doses found -- columns will be NaN/0.")

df['minutes_since_last_glipizide'] = np.round(minutes_since_last_glipizide, 1)
df['glipizide_active']             = glipizide_active
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

df['is_fasting']      = is_fasting
df['hours_into_fast'] = np.round(hours_into_fast, 2)
print("  done.")


# ============================================================== time encoding
df['hour']       = df['ts'].dt.hour
df['minute']     = df['ts'].dt.minute
df['hour_frac']  = df['hour'] + df['minute'] / 60.0          # fractional hour (0–24)
df['hour_sin']   = np.sin(2 * np.pi * df['hour_frac'] / 24)
df['hour_cos']   = np.cos(2 * np.pi * df['hour_frac'] / 24)
df['is_weekend'] = df['ts'].dt.dayofweek.isin([5, 6]).astype(int)
df['day_index']  = (
    df['ts'].dt.normalize() - df['ts'].dt.normalize().min()
).dt.days
df['slot_index'] = (
    (df['ts'] - df['ts'].min()) / pd.Timedelta('5min')
).astype(int)


# ============================================================== write output
COLS_5MIN = [
    'ts',
    # Target
    'glucose_mg_dl',
    # Carb features
    'carbs_last_30min', 'carbs_last_1h', 'carbs_last_2h', 'carbs_last_3h',
    'carb_load_decayed', 'minutes_since_last_meal', 'carbs_last_meal',
    # Activity features
    'walk_min_last_1h', 'walk_min_last_2h', 'walk_min_last_3h',
    'minutes_since_last_walk',
    # Medication features
    'minutes_since_last_glipizide', 'glipizide_active',
    # Fasting features
    'is_fasting', 'hours_into_fast',
    # Time encoding
    'hour', 'minute', 'hour_frac', 'hour_sin', 'hour_cos',
    'is_weekend', 'day_index', 'slot_index',
]
df = df[COLS_5MIN]

out_path = os.path.join(OUT_DIR, '5min_features.csv')
df.to_csv(out_path, index=False)

print(f"\n=== 5-min feature matrix ===")
print(f"  shape:                        {df.shape}")
print(f"  ts:                           {df['ts'].min()} -> {df['ts'].max()}")
print(f"  glucose non-null rows:        "
      f"{df['glucose_mg_dl'].notna().sum()} / {len(df)}")
print(f"  is_fasting=1 slots:           {df['is_fasting'].sum()}")
print(f"  glipizide_active=1 slots:     {df['glipizide_active'].sum()}")
print(f"  slots with carbs (last 1h):   {(df['carbs_last_1h'] > 0).sum()}")
print(f"  slots with carbs (last 30m):  {(df['carbs_last_30min'] > 0).sum()}")
print(f"  slots with walk  (last 1h):   {(df['walk_min_last_1h'] > 0).sum()}")
print(f"\nDone. Written to {out_path}")
