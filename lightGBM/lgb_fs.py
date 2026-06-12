#!/usr/bin/env python3
"""
LightGBM 실험 C — 직접 분류 (Focal Loss + SMOTE)
  • WGRADE_CLASS      (3 클래스: A/B/C → 0/1/2)
  • QUALITY_GRADE_CLASS (5 클래스: 1++/1+/1/2/3 → 0/1/2/3/4)
  • LAST_GRADE = QUALITY_GRADE + WGRADE 조합 (15 클래스)
구조는 실험 B와 동일, Focal Loss + SMOTE만 추가
"""

import os
import sys
import warnings
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
MAX_SAMPLES = 300_000

# ──────────────────────────────────────────────
# 섹션 1: 상수 정의 (실험 B와 완전 동일)
# ──────────────────────────────────────────────
DATA_DIR = '../data/'
TRAIN_FILE = os.path.join(DATA_DIR, 'hanwoo_train_merged.parquet')
TEST_FILE = os.path.join(DATA_DIR, 'hanwoo_test_merged.parquet')
WEATHER_FILE = os.path.join(DATA_DIR, 'hanwoo_weather_imputed.csv')

QUALITY_GRADE_MAP = {'1++': 0, '1+': 1, '1': 2, '2': 3, '3': 4}
QUALITY_GRADE_INV = {v: k for k, v in QUALITY_GRADE_MAP.items()}

WGRADE_MAP = {'A': 0, 'B': 1, 'C': 2}
WGRADE_INV = {v: k for k, v in WGRADE_MAP.items()}

ALL_TARGETS = ['WGRADE_CLASS', 'QUALITY_GRADE_CLASS']
NUM_CLASSES = {'WGRADE_CLASS': 3, 'QUALITY_GRADE_CLASS': 5}

MISSING_TOKENS = ['MISSING', '-99', '-99.0', '',
                  'kluWj1LiM8I6nYWfDenO7q4tJySB2AVV8z9cMqweuXA=',
                  'gQagjD++POKUI4kyvXKUoA==',
                  '2XwK0r9Ij2yaHcePqO7Bwg==']

EXCLUDE_COLS = [
    'DATA_ROW_ID', 'CATTLE_NO', 'FARM_UNIQUE_NO',
    'LINEAGE_FATHER_NO', 'LINEAGE_MOTHER_NO', 'FARM_AREA',
    'PAST_SLAUGHTER_SAMPLE_COUNT', 'PAST_SLAUGHTER_HISTORY_YEARS',
    'PAST_SLAUGHTER_AVAILABLE',
    'BACKFAT', 'REA',
    'INSFAT', 'YUKSAK', 'FATSAK', 'TISSUE', 'GROWTH',
    'INSFAT_GRADE_CLASS', 'YUKSAK_GRADE_CLASS',
    'FATSAK_GRADE_CLASS', 'TISSUE_GRADE_CLASS',
    'GROWTH_STATUS_CLASS',
    'WINDEX', 'WGRADE', 'LAST_GRADE',
    'WGRADE_CLASS', 'QUALITY_GRADE_CLASS',
    'ABATT_DATE', 'JUDGE_DATE', 'BIRTH_YMD',
]

CATEGORICAL_COLS = ['sido', 'sigungu', 'eupmyeondong', 'stn',
                    'JUDGE_SEX', 'ABATT_SEASON', 'BIRTH_SEASON']

FLAG_COLS = ['FATHER_OFFSPRING_IMPUTED', 'MOTHER_OFFSPRING_IMPUTED',
             'FARM_BIOSECURITY_STATUS', 'FARM_HEALTH_STATUS',
             'FARM_ACCIDENT_STATUS']

WEATHER_VARS = ['ta_max_mean', 'ta_min_mean', 'rn_day_sum',
                'rhm_avg_mean', 'ws_davg_mean', 'temp_range_mean']

# ──────────────────────────────────────────────
# 섹션 2: 데이터 로드 및 타깃 파싱
# ──────────────────────────────────────────────
print("\n[섹션 2] 데이터 로드 및 타깃 파싱...")

train_raw = pd.read_parquet(TRAIN_FILE)
print(f"  Train raw: {train_raw.shape}")

test_df = None
if os.path.exists(TEST_FILE):
    test_df = pd.read_parquet(TEST_FILE)
    print(f"  Test raw: {test_df.shape}")

# 결측 토큰 → NaN
def clean_missing(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(MISSING_TOKENS, np.nan)
        try:
            numeric = pd.to_numeric(df[col], errors='coerce')
            if numeric.notna().any():
                df[col] = numeric
                df.loc[df[col] == -99, col] = np.nan
                df.loc[df[col] == -99.0, col] = np.nan
        except Exception:
            pass
    return df

train_raw = clean_missing(train_raw)
if test_df is not None:
    test_df = clean_missing(test_df)

# LAST_GRADE에서 타깃 파싱
train_raw['QUALITY_GRADE_CLASS'] = train_raw['LAST_GRADE'].astype(str).str[:-1]
train_raw['WGRADE_CLASS'] = train_raw['LAST_GRADE'].astype(str).str[-1]

train_raw['QUALITY_GRADE_CLASS'] = train_raw['QUALITY_GRADE_CLASS'].map(QUALITY_GRADE_MAP)
train_raw['WGRADE_CLASS'] = train_raw['WGRADE_CLASS'].map(WGRADE_MAP)

before = len(train_raw)
train_raw = train_raw.dropna(subset=ALL_TARGETS).reset_index(drop=True)
for col in ALL_TARGETS:
    train_raw[col] = train_raw[col].astype(int)
print(f"  타깃 파싱: {before} → {len(train_raw)} rows")

for target in ALL_TARGETS:
    counts = train_raw[target].value_counts().sort_index()
    print(f"  {target}: {NUM_CLASSES[target]} classes, dist={dict(counts)}")

# ──────────────────────────────────────────────
# 섹션 3: 피처 전처리 (실험 B와 동일)
# ──────────────────────────────────────────────
print("\n[섹션 3] 피처 전처리...")

def preprocess_features(df, freq_maps=None, is_train=True):
    df = df.copy()

    for dcol in ['ABATT_DATE', 'JUDGE_DATE', 'BIRTH_YMD']:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors='coerce')

    if 'JUDGE_SEX' in df.columns:
        df['JUDGE_SEX_orig'] = df['JUDGE_SEX'].copy()

    new_freq_maps = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        freq_col = f'{col}_freq'
        if is_train:
            fmap = df[col].value_counts(normalize=True).to_dict()
            new_freq_maps[col] = fmap
        else:
            fmap = freq_maps.get(col, {}) if freq_maps else {}
        df[freq_col] = df[col].map(fmap).fillna(0.0).astype(float)

    for col in FLAG_COLS:
        if col not in df.columns:
            continue
        flag_col = f'{col}_flag'
        df[flag_col] = df[col].map({'YES': 1, 'NO': 0}).astype(float)

    return df, new_freq_maps

train_df, freq_maps = preprocess_features(train_raw, is_train=True)
if test_df is not None:
    test_df, _ = preprocess_features(test_df, freq_maps=freq_maps, is_train=False)

def get_static_real_cols(df):
    exclude = set(EXCLUDE_COLS + ALL_TARGETS + CATEGORICAL_COLS + FLAG_COLS)
    exclude.update([c for c in df.columns if df[c].dtype == object])
    exclude.update([c for c in df.columns if 'datetime' in str(df[c].dtype)])
    exclude.update([c for c in df.columns if c.endswith('_orig')])

    static_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.endswith('_freq') or col.endswith('_flag'):
            static_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            static_cols.append(col)
    return sorted(static_cols)

static_real_cols = get_static_real_cols(train_df)
print(f"  정적 피처 수: {len(static_real_cols)}")
print(f"  정적 피처: {static_real_cols[:10]}...")

# 결측치 median 대체
median_dict = {}
for col in static_real_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    median_val = train_df[col].median()
    if pd.isna(median_val):
        median_val = 0.0
    median_dict[col] = median_val
    train_df[col] = train_df[col].fillna(median_val)
    if test_df is not None:
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
            test_df[col] = test_df[col].fillna(median_val)
        else:
            test_df[col] = median_val

print(f"  전처리 완료: train {train_df.shape}")

# ──────────────────────────────────────────────
# 섹션 4: 날씨 데이터 처리 (실험 B와 동일)
# ──────────────────────────────────────────────
print("\n[섹션 4] 날씨 데이터 로드 및 개체별 요약 통계량 생성...")

weather_raw = pd.read_csv(WEATHER_FILE)
print(f"  날씨 원본: {weather_raw.shape}")

weather_raw['date'] = pd.to_datetime(weather_raw['date'].astype(str), format='%Y%m%d', errors='coerce')
weather_raw = weather_raw.dropna(subset=['date'])
weather_raw['stn'] = weather_raw['stn'].astype(int)

# -99 결측치 처리
weather_cols = ['ta_max', 'ta_min', 'rn_day', 'rhm_avg', 'ws_davg']
for col in weather_cols:
    weather_raw[col] = pd.to_numeric(weather_raw[col], errors='coerce')
weather_raw[weather_cols] = weather_raw[weather_cols].replace(-99.0, np.nan)
weather_raw = weather_raw.sort_values(['stn', 'date']).reset_index(drop=True)
weather_raw[weather_cols] = weather_raw.groupby('stn')[weather_cols].transform(
    lambda x: x.interpolate(method='linear', limit_direction='both')
)

weather_raw['temp_range'] = weather_raw['ta_max'] - weather_raw['ta_min']

# 주간 집계
weather_raw['year_week'] = weather_raw['date'].dt.isocalendar().year.astype(str) + \
                           '-W' + weather_raw['date'].dt.isocalendar().week.astype(str).str.zfill(2)

weather_weekly = weather_raw.groupby(['stn', 'year_week']).agg(
    ta_max_mean=('ta_max', 'mean'),
    ta_min_mean=('ta_min', 'mean'),
    rn_day_sum=('rn_day', 'sum'),
    rhm_avg_mean=('rhm_avg', 'mean'),
    ws_davg_mean=('ws_davg', 'mean'),
    temp_range_mean=('temp_range', 'mean'),
    week_start=('date', 'min')
).reset_index()

weather_weekly['week_start'] = pd.to_datetime(weather_weekly['week_start'])
print(f"  주간 집계 완료: {len(weather_weekly)} rows")

weather_global_median = {}
for v in WEATHER_VARS:
    weather_global_median[v] = weather_weekly[v].median()

def compute_weather_summary(df, weather_weekly_df):
    """stn별 chunk 처리로 메모리 절약"""
    valid = df.copy()
    valid['_orig_idx'] = valid.index

    if 'stn' in valid.columns:
        valid['stn_int'] = pd.to_numeric(valid['stn'], errors='coerce')
    else:
        valid['stn_int'] = np.nan

    mask = valid['BIRTH_YMD'].notna() & valid['ABATT_DATE'].notna() & valid['stn_int'].notna()
    valid_sub = valid.loc[mask, ['_orig_idx', 'BIRTH_YMD', 'ABATT_DATE', 'stn_int']].copy()
    valid_sub['stn_int'] = valid_sub['stn_int'].astype(int)

    print(f"    벡터화 날씨 요약: {len(valid_sub)} 개체")

    ww = weather_weekly_df[['stn', 'week_start'] + WEATHER_VARS].copy()
    ww = ww.rename(columns={'stn': 'stn_int'})
    ww['stn_int'] = ww['stn_int'].astype(int)

    agg_dict = {}
    for v in WEATHER_VARS:
        agg_dict[f'weather_{v}_mean'] = (v, 'mean')
        agg_dict[f'weather_{v}_std'] = (v, 'std')
        agg_dict[f'weather_{v}_min'] = (v, 'min')
        agg_dict[f'weather_{v}_max'] = (v, 'max')

    unique_stns = valid_sub['stn_int'].unique()
    total_stns = len(unique_stns)
    print(f"    stn별 chunk 처리: {total_stns}개 관측소")

    results = []
    processed_entities = 0

    for i, stn in enumerate(unique_stns):
        if (i + 1) % 20 == 0 or (i + 1) == total_stns:
            print(f"      처리 중: {i+1}/{total_stns} 관측소", end='\r')

        sub_entities = valid_sub[valid_sub['stn_int'] == stn]
        sub_weather = ww[ww['stn_int'] == stn]

        if len(sub_entities) == 0 or len(sub_weather) == 0:
            continue

        merged = sub_entities.merge(sub_weather, on='stn_int', how='left')
        merged = merged[
            (merged['week_start'] >= merged['BIRTH_YMD']) &
            (merged['week_start'] <= merged['ABATT_DATE'])
        ]

        if len(merged) == 0:
            continue

        chunk_summary = merged.groupby('_orig_idx').agg(**agg_dict).reset_index()
        results.append(chunk_summary)
        processed_entities += len(chunk_summary)

    print()

    if results:
        summary = pd.concat(results, ignore_index=True)
    else:
        cols = ['_orig_idx'] + [k for k in agg_dict.keys()]
        summary = pd.DataFrame(columns=cols)

    std_cols = [c for c in summary.columns if c.endswith('_std')]
    summary[std_cols] = summary[std_cols].fillna(0.0)

    summary = summary.set_index('_orig_idx').reindex(df.index)

    weather_feature_cols_local = [c for c in summary.columns if c.startswith('weather_')]
    for col in weather_feature_cols_local:
        parts = col.replace('weather_', '').rsplit('_', 1)
        base_var = parts[0]
        if base_var in weather_global_median:
            summary[col] = summary[col].fillna(weather_global_median[base_var])
        else:
            summary[col] = summary[col].fillna(0.0)

    summary = summary.reset_index()
    summary = summary.rename(columns={'index': '_orig_idx'})

    print(f"    요약 완료: {summary.shape}")
    return summary

# Train 날씨 요약
print("\n  Train 데이터 날씨 요약 생성...")
train_weather_summary = compute_weather_summary(train_df, weather_weekly)
weather_feature_cols = [c for c in train_weather_summary.columns if c.startswith('weather_')]
print(f"  날씨 요약 피처 수: {len(weather_feature_cols)}")

train_weather_summary = train_weather_summary.set_index('_orig_idx')
for col in weather_feature_cols:
    train_df[col] = train_weather_summary[col].reindex(train_df.index).values

weather_summary_medians = {}
for col in weather_feature_cols:
    med = train_df[col].median()
    if pd.isna(med):
        med = 0.0
    weather_summary_medians[col] = med
    train_df[col] = train_df[col].fillna(med)

# Test 날씨 요약
if test_df is not None:
    print("\n  Test 데이터 날씨 요약 생성...")
    test_weather_summary = compute_weather_summary(test_df, weather_weekly)

    test_weather_summary = test_weather_summary.set_index('_orig_idx')
    for col in weather_feature_cols:
        if col in test_weather_summary.columns:
            test_df[col] = test_weather_summary[col].reindex(test_df.index).values
        else:
            test_df[col] = 0.0

    for col in weather_feature_cols:
        test_df[col] = test_df[col].fillna(weather_summary_medians.get(col, 0.0))

print("  날씨 요약 병합 완료")

# ──────────────────────────────────────────────
# 섹션 5: 학습 데이터 구성
# ──────────────────────────────────────────────
print("\n[섹션 5] 학습 데이터 구성...")

valid_mask = train_df['BIRTH_YMD'].notna() & train_df['ABATT_DATE'].notna()
if 'stn' in train_df.columns:
    stn_numeric = pd.to_numeric(train_df['stn'], errors='coerce')
    valid_mask = valid_mask & stn_numeric.notna()
valid_mask = valid_mask & (train_df['ABATT_DATE'] > train_df['BIRTH_YMD'])

train_valid = train_df[valid_mask].reset_index(drop=True)

train_valid['n_weeks'] = ((train_valid['ABATT_DATE'] - train_valid['BIRTH_YMD']).dt.days // 7).clip(lower=0)
train_valid = train_valid[train_valid['n_weeks'] >= 3].reset_index(drop=True)
print(f"  유효 개체 (≥3주): {len(train_valid)}")

if len(train_valid) > MAX_SAMPLES:
    train_valid = train_valid.sample(n=MAX_SAMPLES, random_state=SEED).reset_index(drop=True)
    print(f"  샘플링: {MAX_SAMPLES}")

feature_cols = static_real_cols + weather_feature_cols
feature_cols = [f for f in feature_cols if f in train_valid.columns]
print(f"  총 피처 수: {len(feature_cols)} (정적 {len(static_real_cols)} + 날씨요약 {len(weather_feature_cols)})")

X_all = train_valid[feature_cols].values.astype(np.float32)
y_wgrade = train_valid['WGRADE_CLASS'].values.astype(int)
y_quality = train_valid['QUALITY_GRADE_CLASS'].values.astype(int)

print(f"  X shape: {X_all.shape}")
print(f"  WGRADE 분포: {dict(pd.Series(y_wgrade).value_counts().sort_index())}")
print(f"  QUALITY_GRADE 분포: {dict(pd.Series(y_quality).value_counts().sort_index())}")

# ──────────────────────────────────────────────
# 섹션 6: Train/Val 분할 (8:2)
# ──────────────────────────────────────────────
print("\n[섹션 6] Train/Val 분할 (8:2)...")

n_total = len(train_valid)
indices = np.arange(n_total)
np.random.shuffle(indices)
split_idx = int(n_total * 0.8)

train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

X_train = X_all[train_indices]
X_val = X_all[val_indices]
y_wgrade_train, y_wgrade_val = y_wgrade[train_indices], y_wgrade[val_indices]
y_quality_train, y_quality_val = y_quality[train_indices], y_quality[val_indices]

y_train_dict = {'WGRADE_CLASS': y_wgrade_train, 'QUALITY_GRADE_CLASS': y_quality_train}
y_val_dict = {'WGRADE_CLASS': y_wgrade_val, 'QUALITY_GRADE_CLASS': y_quality_val}

print(f"  Train: {len(train_indices)} 개체")
print(f"  Val: {len(val_indices)} 개체")

# ──────────────────────────────────────────────
# 섹션 7: Focal Loss 정의
# ──────────────────────────────────────────────
print("\n[섹션 7] Focal Loss 정의...")

def focal_loss_objective(y_true, y_pred, n_classes, class_weights, gamma=2.0):
    y_pred = y_pred.reshape(-1, n_classes, order='F')

    y_pred_max = y_pred.max(axis=1, keepdims=True)
    exp_pred = np.exp(y_pred - y_pred_max)
    softmax = exp_pred / exp_pred.sum(axis=1, keepdims=True)

    y_onehot = np.zeros_like(softmax)
    y_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0

    sample_w = np.array([class_weights[int(y)] for y in y_true])

    pt = (softmax * y_onehot).sum(axis=1, keepdims=True)
    pt = np.clip(pt, 1e-7, 1.0)
    focal_weight = (1 - pt) ** gamma

    grad = focal_weight * sample_w[:, None] * (softmax - y_onehot)
    hess = focal_weight * sample_w[:, None] * softmax * (1 - softmax)
    hess = np.maximum(hess, 1e-7)

    return grad.flatten(order='F'), hess.flatten(order='F')


def focal_loss_eval(y_true, y_pred, n_classes, class_weights, gamma=2.0):
    y_pred = y_pred.reshape(-1, n_classes, order='F')

    y_pred_max = y_pred.max(axis=1, keepdims=True)
    exp_pred = np.exp(y_pred - y_pred_max)
    softmax = exp_pred / exp_pred.sum(axis=1, keepdims=True)

    pt = softmax[np.arange(len(y_true)), y_true.astype(int)]
    pt = np.clip(pt, 1e-7, 1.0)

    sample_w = np.array([class_weights[int(y)] for y in y_true])
    focal_weight = (1 - pt) ** gamma
    loss = -focal_weight * sample_w * np.log(pt)

    return 'focal_loss', loss.mean(), False


# ──────────────────────────────────────────────
# 섹션 8: LightGBM 학습
# ──────────────────────────────────────────────
print("\n[섹션 8] LightGBM 학습...")

# ── WGRADE: 내장 multiclass + 역빈도 sample_weight (실험 B 동일) ──
print("\n  --- WGRADE_CLASS 학습 (내장 multiclass) ---")

def compute_sample_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    n_samples = len(y)
    class_weights = np.array([
        n_samples / (n_classes * max(counts[i], 1)) for i in range(n_classes)
    ])
    sample_weights = class_weights[y]
    return sample_weights, class_weights

wgrade_sample_weights, wgrade_class_weights = compute_sample_weights(
    y_train_dict['WGRADE_CLASS'], NUM_CLASSES['WGRADE_CLASS']
)
print(f"    클래스 가중치: {[f'{w:.4f}' for w in wgrade_class_weights]}")

lgb_train_wgrade = lgb.Dataset(
    X_train, label=y_train_dict['WGRADE_CLASS'],
    weight=wgrade_sample_weights,
    feature_name=feature_cols,
    free_raw_data=False
)
lgb_val_wgrade = lgb.Dataset(
    X_val, label=y_val_dict['WGRADE_CLASS'],
    feature_name=feature_cols,
    reference=lgb_train_wgrade,
    free_raw_data=False
)

wgrade_params = {
    'objective': 'multiclass',
    'num_class': NUM_CLASSES['WGRADE_CLASS'],
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'verbose': -1,
    'seed': SEED,
    'n_jobs': -1,
}

models = {}
models['WGRADE_CLASS'] = lgb.train(
    wgrade_params,
    lgb_train_wgrade,
    num_boost_round=700,
    valid_sets=[lgb_train_wgrade, lgb_val_wgrade],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]
)
print(f"    WGRADE best iteration: {models['WGRADE_CLASS'].best_iteration}")

# ── QUALITY_GRADE: Focal Loss + SMOTE ──
def train_model_with_focal(X_tr, y_tr, X_v, y_v, n_classes, target_name,
                           num_boost_round=700, gamma=2.0):

    class_counts = np.bincount(y_tr, minlength=n_classes)
    min_count = class_counts[class_counts > 0].min()
    print(f"\n  --- {target_name} 학습 (Focal Loss + SMOTE) ---")

    if min_count < 1000:
        median_count = int(np.median(class_counts[class_counts > 0]))
        target_count = max(1000, int(median_count * 0.1))
        sampling_strategy = {}
        for cls_id in range(n_classes):
            if class_counts[cls_id] < target_count:
                sampling_strategy[cls_id] = target_count

        if sampling_strategy:
            k_neighbors = min(5, min_count - 1) if min_count > 1 else 1
            k_neighbors = max(1, k_neighbors)
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=SEED
            )
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
            new_counts = np.bincount(y_tr, minlength=n_classes)
            print(f"    SMOTE 적용: {dict(enumerate(class_counts))} → {dict(enumerate(new_counts))}")
        else:
            print(f"    SMOTE 불필요 (최소 클래스 {min_count}개)")
    else:
        print(f"    SMOTE 불필요 (최소 클래스 {min_count}개)")

    class_counts_after = np.bincount(y_tr, minlength=n_classes).astype(float)
    total = class_counts_after.sum()
    cw = total / (n_classes * class_counts_after)
    cw = cw / cw.sum() * n_classes
    print(f"    클래스 가중치: {[f'{w:.4f}' for w in cw]}")

    def fobj(preds, dataset):
        y_true = dataset.get_label()
        return focal_loss_objective(y_true, preds, n_classes, cw, gamma)

    def feval(preds, dataset):
        y_true = dataset.get_label()
        return focal_loss_eval(y_true, preds, n_classes, cw, gamma)

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_v, label=y_v, reference=lgb_train)

    params = {
        'objective': fobj,
        'num_class': n_classes,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': -1,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1,
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        feval=feval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    return model

models['QUALITY_GRADE_CLASS'] = train_model_with_focal(
    X_train.copy(), y_train_dict['QUALITY_GRADE_CLASS'].copy(),
    X_val, y_val_dict['QUALITY_GRADE_CLASS'],
    NUM_CLASSES['QUALITY_GRADE_CLASS'], 'QUALITY_GRADE_CLASS',
    num_boost_round=700, gamma=2.0
)

# ──────────────────────────────────────────────
# 섹션 9: 검증 평가
# ──────────────────────────────────────────────
print("\n[섹션 9] 검증 평가...")

target_scores = {}

wgrade_names = ['A', 'B', 'C']
quality_names = ['1++', '1+', '1', '2', '3']

for target in ALL_TARGETS:
    raw_pred = models[target].predict(X_val)
    y_pred = raw_pred.argmax(axis=1)
    y_true = y_val_dict[target]

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    target_scores[target] = {'f1': macro_f1, 'acc': acc}

    if target == 'WGRADE_CLASS':
        labels = wgrade_names
    else:
        labels = quality_names

    print(f"\n  [{target}]")
    print(f"    Macro F1: {macro_f1:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=labels))

# LAST_GRADE 조합
wgrade_pred = models['WGRADE_CLASS'].predict(X_val).argmax(axis=1)
quality_pred = models['QUALITY_GRADE_CLASS'].predict(X_val).argmax(axis=1)

pred_last = []
true_last = []
for i in range(len(y_val_dict['WGRADE_CLASS'])):
    q_pred = QUALITY_GRADE_INV[quality_pred[i]]
    w_pred = WGRADE_INV[wgrade_pred[i]]
    pred_last.append(f"{q_pred}{w_pred}")

    q_true = QUALITY_GRADE_INV[y_val_dict['QUALITY_GRADE_CLASS'][i]]
    w_true = WGRADE_INV[y_val_dict['WGRADE_CLASS'][i]]
    true_last.append(f"{q_true}{w_true}")

all_grade_labels = []
for q in ['1++', '1+', '1', '2', '3']:
    for w in ['A', 'B', 'C']:
        all_grade_labels.append(f"{q}{w}")

last_f1 = f1_score(true_last, pred_last, average='macro', labels=all_grade_labels, zero_division=0)
last_acc = accuracy_score(true_last, pred_last)

print(f"\n  [LAST_GRADE (조합)]")
print(f"    Macro F1: {last_f1:.4f}")
print(f"    Accuracy: {last_acc:.4f}")
print(classification_report(true_last, pred_last, labels=all_grade_labels, zero_division=0))

# ──────────────────────────────────────────────
# 섹션 10: 피처 중요도
# ──────────────────────────────────────────────
print("\n[섹션 10] 피처 중요도 (Top 20)...")

for target in ALL_TARGETS:
    importance = models[target].feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  [{target}] Top 20:")
    for rank, (feat, imp) in enumerate(feat_imp[:20], 1):
        print(f"    {rank:2d}. {feat:<40s} {imp:>12,.0f}")

# ──────────────────────────────────────────────
# 섹션 11: 테스트 추론
# ──────────────────────────────────────────────
print("\n[섹션 11] 테스트 데이터 추론...")

os.makedirs('checkpoints', exist_ok=True)

if test_df is not None:
    for col in weather_feature_cols:
        if col not in test_df.columns:
            test_df[col] = weather_summary_medians.get(col, 0.0)

    X_test = test_df[feature_cols].values.astype(np.float32)
    print(f"  테스트 X shape: {X_test.shape}")

    wgrade_test_pred = models['WGRADE_CLASS'].predict(X_test).argmax(axis=1)
    quality_test_pred = models['QUALITY_GRADE_CLASS'].predict(X_test).argmax(axis=1)

    wgrade_labels = [WGRADE_INV[p] for p in wgrade_test_pred]
    quality_labels = [QUALITY_GRADE_INV[p] for p in quality_test_pred]
    last_labels = [f"{q}{w}" for q, w in zip(quality_labels, wgrade_labels)]

    result_df = pd.DataFrame({
        'WGRADE': wgrade_labels,
        'QUALITY_GRADE': quality_labels,
        'LAST_GRADE': last_labels,
    })

    out_path = 'checkpoints/test_predictions_lgbm_expC.csv'
    result_df.to_csv(out_path, index=False)
    print(f"  저장: {out_path} ({len(result_df)} rows)")

    print(f"\n  WGRADE 분포:\n{result_df['WGRADE'].value_counts().sort_index()}")
    print(f"\n  QUALITY_GRADE 분포:\n{result_df['QUALITY_GRADE'].value_counts().sort_index()}")
    print(f"\n  LAST_GRADE 분포:\n{result_df['LAST_GRADE'].value_counts().sort_index()}")
else:
    print("  테스트 파일 없음, 스킵")

# ──────────────────────────────────────────────
# 섹션 12: 모델 저장
# ──────────────────────────────────────────────
print("\n[섹션 12] 모델 저장...")

for target in ALL_TARGETS:
    model_path = f'checkpoints/lgbm_{target.lower()}_expC.txt'
    models[target].save_model(model_path)
    print(f"  {model_path}")

# ──────────────────────────────────────────────
# 섹션 13: 비교 요약
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("실험 B vs 실험 C 비교")
print("=" * 70)

wf1 = target_scores['WGRADE_CLASS']['f1']
wacc = target_scores['WGRADE_CLASS']['acc']
qf1 = target_scores['QUALITY_GRADE_CLASS']['f1']
qacc = target_scores['QUALITY_GRADE_CLASS']['acc']

print(f"  {'지표':<30s} {'실험B':>10s} {'실험C':>10s}")
print(f"  {'-' * 50}")
print(f"  {'WGRADE Macro F1':<30s} {'0.4494':>10s} {wf1:>10.4f}")
print(f"  {'WGRADE Accuracy':<30s} {'0.4521':>10s} {wacc:>10.4f}")
print(f"  {'QUALITY_GRADE Macro F1':<30s} {'0.3991':>10s} {qf1:>10.4f}")
print(f"  {'QUALITY_GRADE Accuracy':<30s} {'0.3922':>10s} {qacc:>10.4f}")
print(f"  {'LAST_GRADE Macro F1':<30s} {'0.1829':>10s} {last_f1:>10.4f}")
print(f"  {'LAST_GRADE Accuracy':<30s} {'0.1789':>10s} {last_acc:>10.4f}")

print("\n" + "=" * 70)
print("LightGBM 실험 C 완료!")
print("=" * 70)
