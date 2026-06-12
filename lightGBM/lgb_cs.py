# =============================================================================
# lgbm_expB.py — 한우 LightGBM 파이프라인 (실험 B 구조: 직접 분류 방식)
# 태스크: WGRADE_CLASS(A/B/C) + QUALITY_GRADE_CLASS(1++/1+/1/2/3)
# LAST_GRADE = QUALITY_GRADE + WGRADE 단순 조합
# TFT 실험 B와 동일한 데이터, 전처리, 평가 — 모델만 LightGBM으로 교체
# =============================================================================

import os
import warnings
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    f1_score, classification_report, accuracy_score
)

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# 섹션 1: 설정
# =============================================================================
DATA_DIR = '../data/'
TRAIN_FILE = os.path.join(DATA_DIR, 'hanwoo_train_merged.parquet')
TEST_FILE = os.path.join(DATA_DIR, 'hanwoo_test_merged.parquet')
WEATHER_FILE = os.path.join(DATA_DIR, 'hanwoo_weather_imputed.csv')

ALL_TARGETS = ['WGRADE_CLASS', 'QUALITY_GRADE_CLASS']

EXCLUDE_COLS = [
    'DATA_ROW_ID', 'CATTLE_NO', 'FARM_UNIQUE_NO',
    'LINEAGE_FATHER_NO', 'LINEAGE_MOTHER_NO', 'FARM_AREA',
    'PAST_SLAUGHTER_SAMPLE_COUNT', 'PAST_SLAUGHTER_HISTORY_YEARS',
    'PAST_SLAUGHTER_AVAILABLE',
    'PAST_WEIGHT_VARIANCE', 'PAST_BACKFAT_VARIANCE',
    'PAST_REA_VARIANCE',
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

MAX_SAMPLES = 300_000

QUALITY_GRADE_MAP = {'1++': 0, '1+': 1, '1': 2, '2': 3, '3': 4}
QUALITY_GRADE_INV = {v: k for k, v in QUALITY_GRADE_MAP.items()}
WGRADE_MAP = {'A': 0, 'B': 1, 'C': 2}
WGRADE_INV = {v: k for k, v in WGRADE_MAP.items()}

print("=" * 70)
print("한우 LightGBM 파이프라인 — 실험 B 구조 (직접 분류 방식)")
print("태스크: WGRADE(A/B/C) + QUALITY_GRADE(1++/1+/1/2/3) 2개만")
print("LAST_GRADE = QUALITY_GRADE + WGRADE 조합")
print("TFT 실험 B와 동일 조건 — 모델만 LightGBM")
print("=" * 70)

# =============================================================================
# 섹션 2: 데이터 로드 및 결측 처리
# =============================================================================
print("\n[섹션 2] 데이터 로드...")

train_raw = pd.read_parquet(TRAIN_FILE)
print(f"  Train raw: {train_raw.shape}")

has_test = os.path.exists(TEST_FILE)
if has_test:
    test_raw = pd.read_parquet(TEST_FILE)
    print(f"  Test raw: {test_raw.shape}")
else:
    test_raw = None
    print("  Test 파일 없음 — 학습/검증만 진행")

MISSING_TOKENS = ['MISSING', '-99', '-99.0', '',
                  'kluWj1LiM8I6nYWfDenO7q4tJySB2AVV8z9cMqweuXA=',
                  'gQagjD++POKUI4kyvXKUoA==',
                  '2XwK0r9Ij2yaHcePqO7Bwg==']

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
if test_raw is not None:
    test_raw = clean_missing(test_raw)

# ★ LAST_GRADE에서 타깃 파싱 (실험 B와 동일)
print("\n  [타깃 생성: LAST_GRADE 파싱]")
train_raw['QUALITY_GRADE_CLASS'] = train_raw['LAST_GRADE'].astype(str).str[:-1]
train_raw['WGRADE_CLASS'] = train_raw['LAST_GRADE'].astype(str).str[-1]

train_raw['QUALITY_GRADE_CLASS'] = train_raw['QUALITY_GRADE_CLASS'].map(QUALITY_GRADE_MAP)
train_raw['WGRADE_CLASS'] = train_raw['WGRADE_CLASS'].map(WGRADE_MAP)

before = len(train_raw)
train_raw = train_raw.dropna(subset=['QUALITY_GRADE_CLASS', 'WGRADE_CLASS']).reset_index(drop=True)
train_raw['QUALITY_GRADE_CLASS'] = train_raw['QUALITY_GRADE_CLASS'].astype(int)
train_raw['WGRADE_CLASS'] = train_raw['WGRADE_CLASS'].astype(int)
print(f"  LAST_GRADE 파싱: {before} → {len(train_raw)} rows")

# 클래스 분포 확인
NUM_CLASSES = {}
print("\n  [클래스 분포]")
for col in ALL_TARGETS:
    counts = train_raw[col].value_counts().sort_index()
    n_classes = int(train_raw[col].max()) + 1
    NUM_CLASSES[col] = n_classes
    print(f"    {col}: {n_classes} classes, dist={dict(counts)}")

# =============================================================================
# 섹션 3: 피처 전처리 (실험 B와 동일)
# =============================================================================
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
if test_raw is not None:
    test_df, _ = preprocess_features(test_raw, freq_maps=freq_maps, is_train=False)
else:
    test_df = None

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

# 결측치 median 대체 (실험 B와 동일)
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

# =============================================================================
# 섹션 4: 날씨 데이터 주간 집계 → 개체별 요약 통계량
# =============================================================================
print("\n[섹션 4] 날씨 데이터 로드 및 개체별 요약 통계량 생성...")

weather_raw = pd.read_csv(WEATHER_FILE)
print(f"  날씨 원본: {weather_raw.shape}")

weather_raw['date'] = pd.to_datetime(weather_raw['date'].astype(str), format='%Y%m%d', errors='coerce')
weather_raw = weather_raw.dropna(subset=['date'])
weather_raw['stn'] = weather_raw['stn'].astype(int)

# ★ -99 결측치 처리: NaN 변환 후 관측소별 선형 보간
weather_cols = ['ta_max', 'ta_min', 'rn_day', 'rhm_avg', 'ws_davg']
for col in weather_cols:
    weather_raw[col] = pd.to_numeric(weather_raw[col], errors='coerce')
weather_raw[weather_cols] = weather_raw[weather_cols].replace(-99.0, np.nan)
weather_raw = weather_raw.sort_values(['stn', 'date']).reset_index(drop=True)
weather_raw[weather_cols] = weather_raw.groupby('stn')[weather_cols].transform(
    lambda x: x.interpolate(method='linear', limit_direction='both')
)

weather_raw['temp_range'] = weather_raw['ta_max'] - weather_raw['ta_min']

# 주간 집계 (TFT와 동일)
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

def compute_weather_summary(df, weather_weekly):
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
    
    ww = weather_weekly[['stn', 'week_start'] + WEATHER_VARS].copy()
    ww = ww.rename(columns={'stn': 'stn_int'})
    ww['stn_int'] = ww['stn_int'].astype(int)
    
    # ★ stn별 chunk 처리 (메모리 절약)
    agg_dict = {}
    for v in WEATHER_VARS:
        agg_dict[f'weather_{v}_mean'] = (v, 'mean')
        agg_dict[f'weather_{v}_std'] = (v, 'std')
        agg_dict[f'weather_{v}_min'] = (v, 'min')
        agg_dict[f'weather_{v}_max'] = (v, 'max')
    
    unique_stns = valid_sub['stn_int'].unique()
    print(f"    stn별 chunk 처리: {len(unique_stns)}개 관측소")
    
    results = []
    for i, stn in enumerate(unique_stns):
        if (i + 1) % 20 == 0 or (i + 1) == len(unique_stns):
            print(f"      처리 중: {i+1}/{len(unique_stns)} 관측소", end='\r')
        
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
    
    print()  # 줄바꿈
    
    if results:
        summary = pd.concat(results, ignore_index=True)
    else:
        cols = ['_orig_idx'] + [k for k in agg_dict.keys()]
        summary = pd.DataFrame(columns=cols)
    
    # std NaN → 0 (단일 주차인 경우)
    std_cols = [c for c in summary.columns if c.endswith('_std')]
    summary[std_cols] = summary[std_cols].fillna(0.0)
    
    # 전체 인덱스에 맞춰 reindex (매칭 안 된 개체는 NaN)
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


# ── Train 날씨 요약 ──
print("\n  Train 데이터 날씨 요약 생성...")
train_weather_summary = compute_weather_summary(train_df, weather_weekly)
print(f"  Train 날씨 요약: {train_weather_summary.shape}")

weather_feature_cols = [c for c in train_weather_summary.columns if c.startswith('weather_')]
print(f"  날씨 요약 피처 수: {len(weather_feature_cols)}")

# ★ 벡터화 병합 (기존 for 루프 제거)
print("  Train 날씨 요약 병합...")
train_weather_summary = train_weather_summary.set_index('_orig_idx')
for col in weather_feature_cols:
    train_df[col] = train_weather_summary[col].reindex(train_df.index).values

# 날씨 요약 결측 채우기
weather_summary_medians = {}
for col in weather_feature_cols:
    med = train_df[col].median()
    if pd.isna(med):
        med = 0.0
    weather_summary_medians[col] = med
    train_df[col] = train_df[col].fillna(med)

# ── Test 날씨 요약 ──
if test_df is not None:
    print("\n  Test 데이터 날씨 요약 생성...")
    test_weather_summary = compute_weather_summary(test_df, weather_weekly)
    print(f"  Test 날씨 요약: {test_weather_summary.shape}")
    
    # ★ 벡터화 병합
    print("  Test 날씨 요약 병합...")
    test_weather_summary = test_weather_summary.set_index('_orig_idx')
    for col in weather_feature_cols:
        test_df[col] = test_weather_summary[col].reindex(test_df.index).values
    
    for col in weather_feature_cols:
        test_df[col] = test_df[col].fillna(weather_summary_medians[col])

print("  날씨 요약 병합 완료")

# =============================================================================
# 섹션 5: 학습 데이터 구성
# =============================================================================
print("\n[섹션 5] 학습 데이터 구성...")

# ★ TFT 실험 B와 동일한 개체 필터 (BIRTH_YMD, ABATT_DATE 존재, stn 존재)
valid_mask = train_df['BIRTH_YMD'].notna() & train_df['ABATT_DATE'].notna()
if 'stn' in train_df.columns:
    stn_numeric = pd.to_numeric(train_df['stn'], errors='coerce')
    valid_mask = valid_mask & stn_numeric.notna()

# ABATT_DATE > BIRTH_YMD 필터
valid_mask = valid_mask & (train_df['ABATT_DATE'] > train_df['BIRTH_YMD'])

train_valid = train_df[valid_mask].reset_index(drop=True)

# 시퀀스 길이 ≥ 3주 필터 (TFT와 동일)
def compute_weeks(row):
    try:
        weeks = pd.date_range(start=row['BIRTH_YMD'], end=row['ABATT_DATE'], freq='W-MON')
        return len(weeks)
    except:
        return 0

print("  시퀀스 길이 계산 중...")
train_valid['n_weeks'] = ((train_valid['ABATT_DATE'] - train_valid['BIRTH_YMD']).dt.days // 7).clip(lower=0)
train_valid = train_valid[train_valid['n_weeks'] >= 3].reset_index(drop=True)
print(f"  유효 개체 (≥3주): {len(train_valid)}")

# MAX_SAMPLES 제한 (TFT와 동일)
if len(train_valid) > MAX_SAMPLES:
    train_valid = train_valid.sample(n=MAX_SAMPLES, random_state=SEED).reset_index(drop=True)
    print(f"  샘플링: {MAX_SAMPLES}")

# 피처 컬럼 = 정적 피처 + 날씨 요약 피처
feature_cols = static_real_cols + weather_feature_cols
print(f"  총 피처 수: {len(feature_cols)} (정적 {len(static_real_cols)} + 날씨요약 {len(weather_feature_cols)})")

X_all = train_valid[feature_cols].values.astype(np.float32)
y_wgrade = train_valid['WGRADE_CLASS'].values.astype(int)
y_quality = train_valid['QUALITY_GRADE_CLASS'].values.astype(int)

print(f"  X shape: {X_all.shape}")
print(f"  WGRADE 분포: {dict(pd.Series(y_wgrade).value_counts().sort_index())}")
print(f"  QUALITY_GRADE 분포: {dict(pd.Series(y_quality).value_counts().sort_index())}")

# =============================================================================
# 섹션 6: Train/Val 분할 (8:2, 개체 단위 — TFT와 동일 비율)
# =============================================================================
print("\n[섹션 6] Train/Val 분할 (8:2)...")

n_total = len(train_valid)
indices = np.arange(n_total)
np.random.shuffle(indices)
split_idx = int(n_total * 0.8)

train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

X_train, X_val = X_all[train_indices], X_all[val_indices]
y_wgrade_train, y_wgrade_val = y_wgrade[train_indices], y_wgrade[val_indices]
y_quality_train, y_quality_val = y_quality[train_indices], y_quality[val_indices]

print(f"  Train: {len(train_indices)} 개체")
print(f"  Val: {len(val_indices)} 개체")

# =============================================================================
# 섹션 7: 클래스 가중치 계산 (Cost-Sensitive 버전)
# =============================================================================
print("\n[섹션 7] 클래스 가중치 계산 (Cost-Sensitive)...")

def compute_sample_weights(y, n_classes, power=1.0):
    """
    역빈도 기반 샘플 가중치 + power 스케일링
    - power=1.0: 기존과 동일 (역빈도 그대로)
    - power=1.5~2.0: 소수 클래스 가중치 강화
    - power=0.5: 소수 클래스 가중치 완화
    """
    counts = np.bincount(y, minlength=n_classes)
    n_samples = len(y)
    class_weights = np.array([
        n_samples / (n_classes * max(counts[i], 1)) for i in range(n_classes)
    ])
    # ★ power 스케일링 적용
    class_weights = class_weights ** power
    # 정규화 (평균 1.0으로)
    class_weights = class_weights / class_weights.mean()
    
    sample_weights = class_weights[y]
    return sample_weights, class_weights

# ★ WGRADE는 불균형이 크지 않으므로 기존과 동일 (power=1.0)
wgrade_sample_weights, wgrade_class_weights = compute_sample_weights(
    y_wgrade_train, NUM_CLASSES['WGRADE_CLASS'], power=1.0
)

# ★ QUALITY_GRADE는 소수 클래스 강화 (power=1.5)
#   power 값을 1.0 / 1.5 / 2.0 / 2.5 로 바꿔가며 실험
quality_sample_weights, quality_class_weights = compute_sample_weights(
    y_quality_train, NUM_CLASSES['QUALITY_GRADE_CLASS'], power=1.5
)

print(f"  WGRADE class weights (power=1.0): {[f'{w:.4f}' for w in wgrade_class_weights]}")
print(f"  QUALITY_GRADE class weights (power=1.5): {[f'{w:.4f}' for w in quality_class_weights]}")

# =============================================================================
# 섹션 8: LightGBM 모델 학습
# =============================================================================
print("\n[섹션 8] LightGBM 학습...")

# ★ WGRADE 모델 (3클래스 분류)
print("\n  --- WGRADE_CLASS 학습 ---")
lgb_train_wgrade = lgb.Dataset(
    X_train, label=y_wgrade_train,
    weight=wgrade_sample_weights,
    feature_name=feature_cols,
    free_raw_data=False
)
lgb_val_wgrade = lgb.Dataset(
    X_val, label=y_wgrade_val,
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

wgrade_model = lgb.train(
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
print(f"  WGRADE best iteration: {wgrade_model.best_iteration}")

# ★ QUALITY_GRADE 모델 (5클래스 분류)
print("\n  --- QUALITY_GRADE_CLASS 학습 ---")
lgb_train_quality = lgb.Dataset(
    X_train, label=y_quality_train,
    weight=quality_sample_weights,
    feature_name=feature_cols,
    free_raw_data=False
)
lgb_val_quality = lgb.Dataset(
    X_val, label=y_quality_val,
    feature_name=feature_cols,
    reference=lgb_train_quality,
    free_raw_data=False
)

quality_params = {
    'objective': 'multiclass',
    'num_class': NUM_CLASSES['QUALITY_GRADE_CLASS'],
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

quality_model = lgb.train(
    quality_params,
    lgb_train_quality,
    num_boost_round=1000,
    valid_sets=[lgb_train_quality, lgb_val_quality],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]
)
print(f"  QUALITY_GRADE best iteration: {quality_model.best_iteration}")

# =============================================================================
# 섹션 9: 검증 평가
# =============================================================================
print("\n[섹션 9] 검증 평가...")

# 예측
wgrade_val_proba = wgrade_model.predict(X_val, num_iteration=wgrade_model.best_iteration)
wgrade_val_pred = np.argmax(wgrade_val_proba, axis=1)

quality_val_proba = quality_model.predict(X_val, num_iteration=quality_model.best_iteration)
quality_val_pred = np.argmax(quality_val_proba, axis=1)

# 태스크별 평가
print("\n" + "=" * 50)
print("분류 평가")
print("=" * 50)

wgrade_names = ['A', 'B', 'C']
quality_names = ['1++', '1+', '1', '2', '3']

wgrade_f1 = f1_score(y_wgrade_val, wgrade_val_pred, average='macro', zero_division=0)
wgrade_acc = accuracy_score(y_wgrade_val, wgrade_val_pred)
print(f"\n  WGRADE_CLASS (가중치 적용): Macro F1={wgrade_f1:.4f}, Accuracy={wgrade_acc:.4f}")
print(classification_report(y_wgrade_val, wgrade_val_pred, target_names=wgrade_names, zero_division=0))

quality_f1 = f1_score(y_quality_val, quality_val_pred, average='macro', zero_division=0)
quality_acc = accuracy_score(y_quality_val, quality_val_pred)
print(f"\n  QUALITY_GRADE_CLASS (가중치 적용): Macro F1={quality_f1:.4f}, Accuracy={quality_acc:.4f}")
print(classification_report(y_quality_val, quality_val_pred, target_names=quality_names, zero_division=0))

# LAST_GRADE 평가
print("\n" + "=" * 50)
print("LAST_GRADE 평가")
print("=" * 50)

pred_last = []
true_last = []

for i in range(len(y_wgrade_val)):
    q_pred = QUALITY_GRADE_INV[quality_val_pred[i]]
    w_pred = WGRADE_INV[wgrade_val_pred[i]]
    pred_last.append(f"{q_pred}{w_pred}")
    
    q_true = QUALITY_GRADE_INV[y_quality_val[i]]
    w_true = WGRADE_INV[y_wgrade_val[i]]
    true_last.append(f"{q_true}{w_true}")

all_grade_labels = []
for q in ['1++', '1+', '1', '2', '3']:
    for w in ['A', 'B', 'C']:
        all_grade_labels.append(f"{q}{w}")

last_f1 = f1_score(true_last, pred_last, average='macro', labels=all_grade_labels, zero_division=0)
last_acc = accuracy_score(true_last, pred_last)
print(f"\n  LAST_GRADE: Macro F1={last_f1:.4f}, Accuracy={last_acc:.4f}")
try:
    print(classification_report(true_last, pred_last, labels=all_grade_labels, zero_division=0))
except Exception as e:
    print(f"    classification_report 오류: {e}")

# =============================================================================
# 섹션 10: Feature Importance
# =============================================================================
print("\n" + "=" * 50)
print("Feature Importance (Top 20)")
print("=" * 50)

for model_name, model_obj in [('WGRADE', wgrade_model), ('QUALITY_GRADE', quality_model)]:
    importance = model_obj.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n  --- {model_name} ---")
    for _, row in feat_imp.head(20).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:.2f}")

# =============================================================================
# 섹션 11: 테스트 데이터 추론
# =============================================================================
if has_test:
    print("\n[섹션 11] 테스트 데이터 추론...")
    os.makedirs('checkpoints', exist_ok=True)
    
    # 테스트 데이터에도 동일 필터 적용
    test_valid_mask = test_df['BIRTH_YMD'].notna() & test_df['ABATT_DATE'].notna()
    if 'stn' in test_df.columns:
        stn_numeric_test = pd.to_numeric(test_df['stn'], errors='coerce')
        test_valid_mask = test_valid_mask & stn_numeric_test.notna()
    test_valid_mask = test_valid_mask & (test_df['ABATT_DATE'] > test_df['BIRTH_YMD'])
    
    test_valid = test_df[test_valid_mask].reset_index(drop=True)
    
    # 날씨 요약 피처가 없는 컬럼 채우기
    for col in weather_feature_cols:
        if col not in test_valid.columns:
            test_valid[col] = weather_summary_medians.get(col, 0.0)
    
    X_test = test_valid[feature_cols].values.astype(np.float32)
    print(f"  테스트 유효 개체: {len(test_valid)}, X shape: {X_test.shape}")
    
    # 예측
    wgrade_test_proba = wgrade_model.predict(X_test, num_iteration=wgrade_model.best_iteration)
    wgrade_test_pred = np.argmax(wgrade_test_proba, axis=1)
    
    quality_test_proba = quality_model.predict(X_test, num_iteration=quality_model.best_iteration)
    quality_test_pred = np.argmax(quality_test_proba, axis=1)
    
    result_df = pd.DataFrame()
    result_df['WGRADE'] = [WGRADE_INV[p] for p in wgrade_test_pred]
    result_df['QUALITY_GRADE'] = [QUALITY_GRADE_INV[p] for p in quality_test_pred]
    result_df['LAST_GRADE'] = result_df['QUALITY_GRADE'] + result_df['WGRADE']
    
    result_df.to_csv('checkpoints/test_predictions_lgbm_expB.csv', index=False)
    print(f"\n  테스트 예측 저장: checkpoints/test_predictions_lgbm_expB.csv ({len(result_df)} rows)")
    print(f"\n  WGRADE 분포:\n{result_df['WGRADE'].value_counts().sort_index()}")
    print(f"\n  QUALITY_GRADE 분포:\n{result_df['QUALITY_GRADE'].value_counts().sort_index()}")
    print(f"\n  LAST_GRADE 분포:\n{result_df['LAST_GRADE'].value_counts().sort_index()}")
else:
    print("\n[섹션 11] 테스트 파일 없음 — 건너뜀")

# =============================================================================
# 섹션 12: 모델 저장
# =============================================================================
print("\n[섹션 12] 모델 저장...")
os.makedirs('checkpoints', exist_ok=True)
wgrade_model.save_model('checkpoints/lgbm_wgrade_expB.txt')
quality_model.save_model('checkpoints/lgbm_quality_expB.txt')
print(f"  저장 완료: checkpoints/lgbm_wgrade_expB.txt")
print(f"  저장 완료: checkpoints/lgbm_quality_expB.txt")

# =============================================================================
# 섹션 13: TFT vs LightGBM 비교 요약
# =============================================================================
print("\n" + "=" * 70)
print("TFT 실험 B vs LightGBM 실험 B 비교")
print("=" * 70)
print(f"  {'지표':<25s} {'TFT':>10s} {'LightGBM':>10s}")
print(f"  {'-'*45}")
print(f"  {'WGRADE Macro F1':<25s} {'0.3724':>10s} {f'{wgrade_f1:.4f}':>10s}")
print(f"  {'WGRADE Accuracy':<25s} {'0.3794':>10s} {f'{wgrade_acc:.4f}':>10s}")
print(f"  {'QUALITY_GRADE Macro F1':<25s} {'0.3687':>10s} {f'{quality_f1:.4f}':>10s}")
print(f"  {'QUALITY_GRADE Accuracy':<25s} {'0.3758':>10s} {f'{quality_acc:.4f}':>10s}")
print(f"  {'LAST_GRADE Macro F1':<25s} {'0.1393':>10s} {f'{last_f1:.4f}':>10s}")
print(f"  {'LAST_GRADE Accuracy':<25s} {'0.1393':>10s} {f'{last_acc:.4f}':>10s}")

print("\n" + "=" * 70)
print("LightGBM 실험 B 완료!")
print("=" * 70)
