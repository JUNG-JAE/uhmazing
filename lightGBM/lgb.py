import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, classification_report, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 설정
# ============================================================
SEED = 42
N_FOLDS = 5
DATA_DIR = 'hanwoo/'

EXCLUDE_FROM_INPUT = [
    'BACKFAT', 'REA', 'WINDEX', 'WGRADE',
    'INSFAT', 'YUKSAK', 'FATSAK', 'TISSUE', 'GROWTH',
    'LAST_GRADE', 'COST_AMT',
    'FARM_UNIQUE_NO', 'CATTLE_NO', 'date',
    'sido', 'sigungu', 'eupmyeondong', 'stn',
    'JUDGE_SEX',
    'KPN_NO', 'FATHER_CATTLE_NO', 'MOTHER_ANIMAL_NO',
    'F_GMOTHER_ANIMAL_NO', 'F_GFATHER_CATTLE_NO',
    'M_GMOTHER_ANIMAL_NO', 'M_GFATHER_CATTLE_NO',
    'ABATT_DATE', 'JUDGE_DATE', 'BIRTH_YMD',
]

# ============================================================
# 1. 데이터 로드
# ============================================================
def load_and_clean(path, encoding='utf-8-sig'):
    df = pd.read_csv(path, encoding=encoding)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace(-99, np.nan)
    return df

print("데이터 로드 중...")
train   = load_and_clean(DATA_DIR + 'hanwoo_train.csv')
area    = load_and_clean(DATA_DIR + 'hanwoo_area.csv')
death   = load_and_clean(DATA_DIR + 'hanwoo_death.csv')
lineage = load_and_clean(DATA_DIR + 'hanwoo_lineage.csv')
weather = load_and_clean(DATA_DIR + 'hanwoo_weather.csv')
print(f"train: {train.shape}, area: {area.shape}, death: {death.shape}, "
      f"lineage: {lineage.shape}, weather: {weather.shape}")

# ============================================================
# 2-1. Area 전처리
# ============================================================
count_cols = ['C2023', 'C2024', 'C2025']

# 중복 FARM_UNIQUE_NO 합산
area_counts = area.groupby('FARM_UNIQUE_NO')[count_cols].sum(min_count=1)
area_area   = area.groupby('FARM_UNIQUE_NO')['AREA'].first()
farm_df     = area_counts.join(area_area).reset_index()

# 인접 연도 보간: 양쪽 있으면 평균
m = farm_df['C2024'].isna() & farm_df['C2023'].notna() & farm_df['C2025'].notna()
farm_df.loc[m, 'C2024'] = (farm_df.loc[m, 'C2023'] + farm_df.loc[m, 'C2025']) / 2
print(f"[Area] C2024 보간: {m.sum()}건")

# 외삽: 한쪽만 없으면 선형 외삽
m = farm_df['C2023'].isna() & farm_df['C2024'].notna() & farm_df['C2025'].notna()
farm_df.loc[m, 'C2023'] = (2 * farm_df.loc[m, 'C2024'] - farm_df.loc[m, 'C2025']).clip(lower=0)
print(f"[Area] C2023 외삽: {m.sum()}건")

m = farm_df['C2025'].isna() & farm_df['C2023'].notna() & farm_df['C2024'].notna()
farm_df.loc[m, 'C2025'] = (2 * farm_df.loc[m, 'C2024'] - farm_df.loc[m, 'C2023']).clip(lower=0)
print(f"[Area] C2025 외삽: {m.sum()}건")

# 하나만 있으면 나머지를 같은 값으로
for col in count_cols:
    others = [c for c in count_cols if c != col]
    mask = farm_df[col].notna() & farm_df[others[0]].isna() & farm_df[others[1]].isna()
    for oc in others:
        farm_df.loc[mask, oc] = farm_df.loc[mask, col]

farm_df['all_C_missing'] = farm_df[count_cols].isna().all(axis=1)
farm_df['C_mean']     = farm_df[count_cols].mean(axis=1)
farm_df['C_max']      = farm_df[count_cols].max(axis=1)
farm_df['C_min']      = farm_df[count_cols].min(axis=1)
farm_df['C_std']      = farm_df[count_cols].std(axis=1)
farm_df['C_growth']   = ((farm_df['C2025'] - farm_df['C2023']) / farm_df['C2023']).replace([np.inf, -np.inf], np.nan)
farm_df['AREA_per_head'] = (farm_df['AREA'] / farm_df['C_mean']).replace([np.inf, -np.inf], np.nan)
farm_df['has_area']      = farm_df['AREA'].notna().astype(int)
farm_df['missing_C_count'] = farm_df[count_cols].isna().sum(axis=1)

# ============================================================
# 2-2. Death 전처리
# ============================================================
death['BIRTH_YMD'] = pd.to_datetime(death['BIRTH_YMD'], errors='coerce')
death['DEAD_YMD']  = pd.to_datetime(death['DEAD_YMD'], errors='coerce')
death['death_age_days'] = (death['DEAD_YMD'] - death['BIRTH_YMD']).dt.days
death['dead_year'] = death['DEAD_YMD'].dt.year

# 농가별 전체 통계
death_agg = death.groupby('FARM_UNIQUE_NO').agg(
    death_count       = ('DEAD_YMD', 'count'),
    death_age_mean    = ('death_age_days', 'mean'),
    death_age_median  = ('death_age_days', 'median'),
    death_age_std     = ('death_age_days', 'std'),
    death_age_min     = ('death_age_days', 'min'),
    death_age_max     = ('death_age_days', 'max'),
    early_death_count = ('death_age_days', lambda x: (x <= 90).sum()),
).reset_index()
death_agg['early_death_ratio'] = death_agg['early_death_count'] / death_agg['death_count']

# 연도별 폐사 건수
death_yearly = death.groupby(['FARM_UNIQUE_NO', 'dead_year']).size().unstack(fill_value=0)
death_yearly.columns = [f'death_{int(y)}' for y in death_yearly.columns]
death_yearly = death_yearly.reset_index()

# 사육두수 기반 폐사율
death_yearly = death_yearly.merge(
    farm_df[['FARM_UNIQUE_NO', 'C2023', 'C2024', 'C2025']],
    on='FARM_UNIQUE_NO', how='left'
)

rate_cols = []
for year in [2023, 2024, 2025]:
    d_col = f'death_{year}'
    c_col = f'C{year}'
    r_col = f'death_rate_{year}'
    if d_col in death_yearly.columns:
        death_yearly[r_col] = (death_yearly[d_col] / death_yearly[c_col]).replace([np.inf, -np.inf], np.nan)
        rate_cols.append(r_col)

# 결측 연도 → 나머지 연도 평균으로 보간
death_yearly['death_rate_mean_temp'] = death_yearly[rate_cols].mean(axis=1)
for rc in rate_cols:
    death_yearly[rc] = death_yearly[rc].fillna(death_yearly['death_rate_mean_temp'])
death_yearly.drop(columns='death_rate_mean_temp', inplace=True)

# 폐사율 파생변수
death_yearly['death_rate_mean']  = death_yearly[rate_cols].mean(axis=1)
death_yearly['death_rate_std']   = death_yearly[rate_cols].std(axis=1)
death_yearly['death_rate_max']   = death_yearly[rate_cols].max(axis=1)
death_yearly['death_rate_min']   = death_yearly[rate_cols].min(axis=1)
if 'death_rate_2023' in death_yearly.columns and 'death_rate_2025' in death_yearly.columns:
    death_yearly['death_rate_growth'] = death_yearly['death_rate_2025'] - death_yearly['death_rate_2023']

# 불필요한 컬럼 제거 후 병합
drop_from_yearly = ['C2023', 'C2024', 'C2025']
death_yearly.drop(columns=[c for c in drop_from_yearly if c in death_yearly.columns], inplace=True)
death_final = death_agg.merge(death_yearly, on='FARM_UNIQUE_NO', how='outer')
print(f"[Death] 폐사 기록 농가: {len(death_final):,}")

# ============================================================
# 2-3. Lineage
# ============================================================
lineage_fe = lineage.copy()
freq_cols = ['KPN_NO', 'FATHER_CATTLE_NO', 'F_GMOTHER_ANIMAL_NO',
             'F_GFATHER_CATTLE_NO', 'M_GFATHER_CATTLE_NO']
high_card = ['MOTHER_ANIMAL_NO', 'M_GMOTHER_ANIMAL_NO']
for col in freq_cols + high_card:
    fmap = lineage_fe[col].value_counts().to_dict()
    lineage_fe[f'{col}_freq'] = lineage_fe[col].map(fmap)

lineage_fe['lineage_completeness'] = lineage_fe[freq_cols + high_card].notna().sum(axis=1)
lineage_fe['same_father_mother'] = (lineage_fe['FATHER_CATTLE_NO'] == lineage_fe['MOTHER_ANIMAL_NO']).astype(int)
lineage_fe['same_paternal_gp']   = (lineage_fe['F_GMOTHER_ANIMAL_NO'] == lineage_fe['F_GFATHER_CATTLE_NO']).astype(int)
lineage_fe['same_maternal_gp']   = (lineage_fe['M_GMOTHER_ANIMAL_NO'] == lineage_fe['M_GFATHER_CATTLE_NO']).astype(int)

keep_lineage = ['CATTLE_NO'] + [f'{c}_freq' for c in freq_cols + high_card] + \
               ['lineage_completeness', 'same_father_mother', 'same_paternal_gp', 'same_maternal_gp']
lineage_features = lineage_fe[keep_lineage]

# ============================================================
# 2-4. Weather
# ============================================================
weather['date'] = pd.to_datetime(weather['date'], errors='coerce')
weather['temp_range'] = weather['ta_max'] - weather['ta_min']

def compute_weather_features(weather_df, windows=[7, 14, 30, 90]):
    ws = weather_df.sort_values(['stn', 'date']).copy()
    ws = ws.reset_index(drop=True)
    weather_vars = ['ta_max', 'ta_min', 'rn_day', 'rhm_avg', 'ws_davg', 'temp_range']

    all_feats = ws[['stn', 'date']].copy()
    for w in windows:
        for v in weather_vars:
            grp = ws.groupby('stn')[v]
            all_feats[f'{v}_mean_w{w}'] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
            all_feats[f'{v}_std_w{w}']  = grp.transform(lambda x: x.rolling(w, min_periods=1).std())
            all_feats[f'{v}_max_w{w}']  = grp.transform(lambda x: x.rolling(w, min_periods=1).max())
            all_feats[f'{v}_min_w{w}']  = grp.transform(lambda x: x.rolling(w, min_periods=1).min())
        rain_flag = (ws['rn_day'] > 0).astype(float)
        all_feats[f'rain_ratio_w{w}'] = rain_flag.groupby(ws['stn']).transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
    return all_feats

print("날씨 피처 생성 중... (시간 소요)")
weather_features = compute_weather_features(weather)
print(f"[Weather] 피처 수: {weather_features.shape[1] - 2}")

# ============================================================
# 3. 타깃 생성
# ============================================================
train['date'] = pd.to_datetime(
    train['ABATT_DATE'] if 'ABATT_DATE' in train.columns else train.get('date'),
    errors='coerce'
)

def insfat_to_grade(v):
    if pd.isna(v): return np.nan
    v = int(v)
    if v >= 7: return 0
    if v == 6: return 1
    if v >= 4: return 2
    if v >= 2: return 3
    return 4

def yuksak_to_grade(v):
    if pd.isna(v): return np.nan
    v = int(v)
    if v in [3, 4, 5]: return 0
    if v in [2, 6]:    return 1
    if v == 1:         return 2
    if v == 7:         return 3
    return 4

def fatsak_to_grade(v):
    if pd.isna(v): return np.nan
    v = int(v)
    if v <= 4: return 0
    if v == 5: return 1
    if v == 6: return 2
    if v == 7: return 3
    return 4

def tissue_to_grade(v):
    if pd.isna(v): return np.nan
    v = int(v)
    if v == 1: return 0
    if v == 2: return 1
    if v == 3: return 2
    if v == 4: return 3
    return 4

train['y_insfat'] = train['INSFAT'].apply(insfat_to_grade)
train['y_yuksak'] = train['YUKSAK'].apply(yuksak_to_grade)
train['y_fatsak'] = train['FATSAK'].apply(fatsak_to_grade)
train['y_tissue'] = train['TISSUE'].apply(tissue_to_grade)
train['y_growth'] = train['GROWTH'].apply(
    lambda v: np.nan if pd.isna(v) else (1 if int(v) >= 8 else 0)
)
train['y_wgrade'] = train['WGRADE'].map({'A': 0, 'B': 1, 'C': 2})
train['y_backfat'] = train['BACKFAT']
train['y_rea']     = train['REA']

# ============================================================
# 4. 피처 병합
# ============================================================
print("피처 병합 중...")
df = train.copy()
df = df.merge(farm_df, on='FARM_UNIQUE_NO', how='left')
df = df.merge(death_final, on='FARM_UNIQUE_NO', how='left')

death_fill = [c for c in death_final.columns if c != 'FARM_UNIQUE_NO']
df[death_fill] = df[death_fill].fillna(0)

if 'CATTLE_NO' in df.columns:
    df = df.merge(lineage_features, on='CATTLE_NO', how='left')

if 'stn' in df.columns and 'date' in df.columns:
    df['stn'] = df['stn'].astype(str)
    weather_features['stn'] = weather_features['stn'].astype(str)
    weather_features['date'] = pd.to_datetime(weather_features['date'])
    df = df.merge(weather_features, on=['stn', 'date'], how='left')

if 'AGE' in df.columns:
    df['AGE_sq'] = df['AGE'] ** 2
if 'date' in df.columns:
    df['month']     = df['date'].dt.month
    df['year']      = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['season']    = df['month'].map({12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3})
if 'BIRTH_YMD' in df.columns:
    bdt = pd.to_datetime(df['BIRTH_YMD'], format='%Y%m%d', errors='coerce')
    df['birth_year']  = bdt.dt.year
    df['birth_month'] = bdt.dt.month

for col in ['sido', 'sigungu', 'eupmyeondong']:
    if col in df.columns:
        fmap = df[col].value_counts().to_dict()
        df[f'{col}_freq'] = df[col].map(fmap)
if 'JUDGE_SEX' in df.columns:
    le_sex = LabelEncoder()
    df['JUDGE_SEX_enc'] = le_sex.fit_transform(df['JUDGE_SEX'].astype(str))

target_y_cols = ['y_insfat', 'y_yuksak', 'y_fatsak', 'y_tissue',
                 'y_growth', 'y_wgrade', 'y_backfat', 'y_rea', 'all_C_missing']
all_drop = list(set(EXCLUDE_FROM_INPUT + target_y_cols))
X_all = df.drop(columns=[c for c in all_drop if c in df.columns], errors='ignore')
X_all = X_all.select_dtypes(include=[np.number])

groups = df['FARM_UNIQUE_NO']

print(f"\n입력 피처 수: {X_all.shape[1]}")
print(f"전체 데이터 행 수: {X_all.shape[0]:,}")
print(f"피처 목록: {list(X_all.columns)[:20]} ... (총 {X_all.shape[1]}개)")

# ============================================================
# 5. 모델 파라미터
# ============================================================
def make_params(task_type, num_class=None):
    p = {
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 3000,
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1,
        'max_bin': 63,
        'feature_pre_filter': True,
    }
    if task_type == 'regression':
        p['objective'] = 'regression'
        p['metric'] = 'mae'
    elif task_type == 'multiclass':
        p['objective'] = 'multiclass'
        p['metric'] = 'multi_logloss'
        p['num_class'] = num_class
    elif task_type == 'binary':
        p['objective'] = 'binary'
        p['metric'] = 'binary_logloss'
    return p

# ============================================================
# 6. 태스크 정의 (cls_wgrade 제거)
# ============================================================
tasks = {
    'reg_backfat': {
        'type': 'regression', 'y_col': 'y_backfat',
        'params': make_params('regression'),
    },
    'reg_rea': {
        'type': 'regression', 'y_col': 'y_rea',
        'params': make_params('regression'),
    },
    'cls_insfat': {
        'type': 'multiclass', 'y_col': 'y_insfat', 'n_class': 5,
        'class_names': ['1++', '1+', '1', '2', '3'],
        'params': make_params('multiclass', 5),
    },
    'cls_yuksak': {
        'type': 'multiclass', 'y_col': 'y_yuksak', 'n_class': 5,
        'class_names': ['1++', '1+', '1', '2', '3'],
        'params': make_params('multiclass', 5),
    },
    'cls_fatsak': {
        'type': 'multiclass', 'y_col': 'y_fatsak', 'n_class': 5,
        'class_names': ['1++', '1+', '1', '2', '3'],
        'params': make_params('multiclass', 5),
    },
    'cls_tissue': {
        'type': 'multiclass', 'y_col': 'y_tissue', 'n_class': 5,
        'class_names': ['1++', '1+', '1', '2', '3'],
        'params': make_params('multiclass', 5),
    },
    'cls_growth': {
        'type': 'binary', 'y_col': 'y_growth',
        'class_names': ['정상', '결격'],
        'params': make_params('binary'),
    },
}

# ============================================================
# 7. 태스크별 독립 학습
# ============================================================
all_results = {}

for task_name, task_cfg in tasks.items():
    print(f"\n{'='*60}")
    print(f"태스크: {task_name}")
    print(f"{'='*60}")

    y_col     = task_cfg['y_col']
    task_type = task_cfg['type']
    params    = task_cfg['params']

    valid_mask = df[y_col].notna()
    X_task = X_all.loc[valid_mask].reset_index(drop=True)
    y_task = df.loc[valid_mask, y_col].reset_index(drop=True)
    groups_task = groups.loc[valid_mask].reset_index(drop=True)

    if task_type != 'regression':
        y_task = y_task.astype(int)
    else:
        y_task = y_task.astype(float)

    print(f"유효 데이터: {len(X_task):,}행")
    if task_type != 'regression':
        print(f"클래스 분포:\n{y_task.value_counts().sort_index().to_string()}")

    if task_type == 'regression':
        y_bin = pd.qcut(y_task, q=10, labels=False, duplicates='drop')
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(sgkf.split(X_task, y_bin, groups_task))
    else:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(sgkf.split(X_task, y_task, groups_task))

    if task_type == 'multiclass':
        oof_preds = np.full((len(X_task), task_cfg['n_class']), np.nan)
    else:
        oof_preds = np.full(len(X_task), np.nan)

    models = []
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va = X_task.iloc[tr_idx], X_task.iloc[va_idx]
        y_tr, y_va = y_task.iloc[tr_idx], y_task.iloc[va_idx]

        if task_type == 'regression':
            model = lgb.LGBMRegressor(**params)
        else:
            model = lgb.LGBMClassifier(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
        )

        if task_type == 'regression':
            oof_preds[va_idx] = model.predict(X_va)
            score = mean_absolute_error(y_va, oof_preds[va_idx])
            print(f"  Fold {fold+1} MAE: {score:.4f}")
        elif task_type == 'multiclass':
            oof_preds[va_idx] = model.predict_proba(X_va)
            pred_labels = np.argmax(oof_preds[va_idx], axis=1)
            score = f1_score(y_va, pred_labels, average='macro')
            print(f"  Fold {fold+1} Macro F1: {score:.4f}")
        else:
            proba = model.predict_proba(X_va)[:, 1]
            oof_preds[va_idx] = proba
            pred_labels = (proba > 0.5).astype(int)
            score = f1_score(y_va, pred_labels, average='macro')
            print(f"  Fold {fold+1} Macro F1: {score:.4f}")

        fold_scores.append(score)
        models.append(model)

    print(f"\n--- {task_name} 전체 OOF ---")
    if task_type == 'regression':
        mae  = mean_absolute_error(y_task, oof_preds)
        rmse = np.sqrt(mean_squared_error(y_task, oof_preds))
        r2   = r2_score(y_task, oof_preds)
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        all_results[task_name] = {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'models': models, 'oof_preds': oof_preds, 'valid_mask': valid_mask,
        }
    elif task_type == 'multiclass':
        oof_labels = np.argmax(oof_preds, axis=1)
        mf1 = f1_score(y_task, oof_labels, average='macro')
        print(f"Macro F1: {mf1:.4f}")
        print(classification_report(y_task, oof_labels, target_names=task_cfg['class_names']))
        all_results[task_name] = {
            'macro_f1': mf1, 'models': models, 'oof_preds': oof_preds, 'valid_mask': valid_mask,
        }
    else:
        oof_labels = (oof_preds > 0.5).astype(int)
        mf1 = f1_score(y_task, oof_labels, average='macro')
        print(f"Macro F1: {mf1:.4f}")
        print(classification_report(y_task, oof_labels, target_names=task_cfg['class_names']))
        all_results[task_name] = {
            'macro_f1': mf1, 'models': models, 'oof_preds': oof_preds, 'valid_mask': valid_mask,
        }

# ============================================================
# 8. 최종 등급 결정 (후처리)
# ============================================================
print(f"\n{'='*60}")
print("최종 등급 결정")
print(f"{'='*60}")

grade_names = ['1++', '1+', '1', '2', '3']
yield_names = ['A', 'B', 'C']

def map_oof_to_df(task_name, task_type):
    res = all_results[task_name]
    valid_mask = res['valid_mask']
    valid_indices = df.index[valid_mask]
    result = pd.Series(np.nan, index=df.index, dtype=float)
    if task_type == 'multiclass':
        labels = np.argmax(res['oof_preds'], axis=1)
    elif task_type == 'binary':
        labels = (res['oof_preds'] > 0.5).astype(int)
    else:
        labels = res['oof_preds']
    result.loc[valid_indices] = labels
    return result

pred_insfat = map_oof_to_df('cls_insfat', 'multiclass')
pred_yuksak = map_oof_to_df('cls_yuksak', 'multiclass')
pred_fatsak = map_oof_to_df('cls_fatsak', 'multiclass')
pred_tissue = map_oof_to_df('cls_tissue', 'multiclass')
pred_growth = map_oof_to_df('cls_growth', 'binary')

# 육질등급 결정 (기존과 동일)
quality_preds = pd.concat([pred_insfat, pred_yuksak, pred_fatsak, pred_tissue], axis=1)
quality_grade = quality_preds.max(axis=1)
quality_final = quality_grade.copy()
downgrade = (pred_growth == 1) & (quality_grade < 4)
quality_final[downgrade] = quality_grade[downgrade] + 1

# --- 육량등급: 회귀 예측값 + 성별별 공식으로 결정 ---
pred_backfat = map_oof_to_df('reg_backfat', 'regression')
pred_rea     = map_oof_to_df('reg_rea', 'regression')
weight       = df['WEIGHT']
sex          = df['JUDGE_SEX']

def compute_yield_index(backfat, rea, w, s):
    if s == '암':
        return (6.90137 - 0.94460 * backfat + 0.31805 * rea + 0.54952 * w) / w * 100
    elif s == '수':
        return (0.20103 - 2.18525 * backfat + 0.29275 * rea + 0.64099 * w) / w * 100
    elif s == '거세':
        return (11.06398 - 1.25149 * backfat + 0.28293 * rea + 0.56781 * w) / w * 100
    else:
        return np.nan

yield_index = pd.Series(
    [compute_yield_index(bf, r, w, s)
     for bf, r, w, s in zip(pred_backfat, pred_rea, weight, sex)],
    index=df.index
)

def yield_index_to_grade(yi, s):
    if pd.isna(yi) or pd.isna(s):
        return np.nan
    if s == '암':
        if yi >= 61.83: return 0
        elif yi >= 59.70: return 1
        else: return 2
    elif s == '수':
        if yi >= 68.45: return 0
        elif yi >= 66.32: return 1
        else: return 2
    elif s == '거세':
        if yi >= 62.52: return 0
        elif yi >= 60.40: return 1
        else: return 2
    else:
        return np.nan

pred_wgrade = pd.Series(
    [yield_index_to_grade(yi, s) for yi, s in zip(yield_index, sex)],
    index=df.index
)

print(f"\n--- 육량등급 (성별별 공식 기반) ---")
print(f"육량지수 통계: mean={yield_index.mean():.2f}, std={yield_index.std():.2f}, "
      f"min={yield_index.min():.2f}, max={yield_index.max():.2f}")
for s in ['암', '수', '거세']:
    mask = sex == s
    print(f"  [{s}] mean={yield_index[mask].mean():.2f}, "
          f"A={int((pred_wgrade[mask]==0).sum()):,}, "
          f"B={int((pred_wgrade[mask]==1).sum()):,}, "
          f"C={int((pred_wgrade[mask]==2).sum()):,}")

# --- 최종 등급 조합 ---
valid_all = (
    df['y_insfat'].notna() & df['y_yuksak'].notna() &
    df['y_fatsak'].notna() & df['y_tissue'].notna() &
    df['y_growth'].notna() & pred_wgrade.notna() &
    df['WGRADE'].notna()
)
valid_idx = df.index[valid_all]
print(f"유효 행: {len(valid_idx):,}")

fq = quality_final[valid_idx].astype(int)
fy = pred_wgrade[valid_idx].astype(int)
final_grade = [f"{grade_names[q]}{yield_names[y]}" for q, y in zip(fq, fy)]

actual = df.loc[valid_idx, 'LAST_GRADE'].values

match_total = sum(a == p for a, p in zip(actual, final_grade))
print(f"최종 등급 일치율: {match_total:,} / {len(final_grade):,} = {match_total/len(final_grade):.4f}")

q_map = {'1++': 0, '1+': 1, '1': 2, '2': 3, '3': 4}
actual_q = pd.Series(actual).str.extract(r'^(1\+\+|1\+|1|2|3)')[0].map(q_map).values
q_match = np.nansum(fq.values == actual_q)
print(f"육질등급 일치율: {q_match:,} / {len(final_grade):,} = {q_match/len(final_grade):.4f}")

y_map = {'A': 0, 'B': 1, 'C': 2}
actual_y = pd.Series(actual).str.extract(r'([ABC])$')[0].map(y_map).values
y_match = np.nansum(fy.values == actual_y)
print(f"육량등급 일치율: {y_match:,} / {len(final_grade):,} = {y_match/len(final_grade):.4f}")

# [참고] 실제 측정값으로 성별별 공식 적용 시 육량등급 일치율 (상한선)
actual_yi = pd.Series(
    [compute_yield_index(bf, r, w, s)
     for bf, r, w, s in zip(df['BACKFAT'], df['REA'], df['WEIGHT'], df['JUDGE_SEX'])],
    index=df.index
)
actual_wg = pd.Series(
    [yield_index_to_grade(yi, s) for yi, s in zip(actual_yi, df['JUDGE_SEX'])],
    index=df.index
)
actual_wgrade_label = df['WGRADE'].map({'A': 0, 'B': 1, 'C': 2})
formula_valid = valid_idx[actual_wg[valid_idx].notna() & actual_wgrade_label[valid_idx].notna()]
formula_match = (actual_wg[formula_valid] == actual_wgrade_label[formula_valid]).mean()
print(f"[참고] 실제 측정값 + 성별별 공식 적용 시 육량등급 일치율: {formula_match:.4f}")

# ============================================================
# 9. 모델 저장
# ============================================================
save_dict = {
    'results': {},
    'models': {},
    'feature_names': list(X_all.columns),
}
for k, v in all_results.items():
    save_dict['results'][k] = {kk: vv for kk, vv in v.items() if kk not in ['models', 'oof_preds', 'valid_mask']}
    save_dict['models'][k] = v['models']

with open('lgb_multitask_models.pkl', 'wb') as f:
    pickle.dump(save_dict, f)

print("\n모델 저장 완료: lgb_multitask_models.pkl")
