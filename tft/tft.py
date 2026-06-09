import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import TorchNormalizer, MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import CrossEntropy, MultiLoss
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import copy
import random

pl.seed_everything(42)

SEED = 42
DATA_DIR = 'hanwoo/'

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
print(f"train: {train.shape}, weather: {weather.shape}")

# ============================================================
# 2. 타깃 생성 (5개 멀티태스크 분류)
# ===================================================================================================================

# -99 → NaN 처리 (숫자 컬럼)
for col in ['INSFAT', 'YUKSAK', 'FATSAK', 'TISSUE', 'GROWTH']:
    train[col] = train[col].replace(-99, np.nan)
# WGRADE는 문자열이므로 별도 처리
train['WGRADE'] = train['WGRADE'].replace('-99', np.nan).replace(-99, np.nan)

# --- INSFAT: 근내지방도 5등급 ---
def insfat_to_grade(v):
    if pd.isna(v): return np.nan
    v = float(v)
    if v >= 7: return 0    # 1++ (7,8,9)
    if v == 6: return 1    # 1+
    if v >= 4: return 2    # 1  (4,5)
    if v >= 2: return 3    # 2  (2,3)
    return 4               # 3  (1)

train['target_insfat'] = train['INSFAT'].apply(insfat_to_grade)

# --- YUKSAK: 육색등급 3그룹 ---
# 1~3 → 0 (밝음, 희소), 4~5 → 1 (정상, 90%), 6~8 → 2 (어두움)
def yuksak_to_grade(v):
    if pd.isna(v): return np.nan
    v = float(v)
    if v <= 3: return 0
    if v <= 5: return 1
    return 2

train['target_yuksak'] = train['YUKSAK'].apply(yuksak_to_grade)

# --- FATSAK: 지방색등급 3그룹 ---
# 1~2 → 0 (흰색), 3 → 1 (정상, 91%), 4~8 → 2 (황색)
def fatsak_to_grade(v):
    if pd.isna(v): return np.nan
    v = float(v)
    if v <= 2: return 0
    if v == 3: return 1
    return 2

train['target_fatsak'] = train['FATSAK'].apply(fatsak_to_grade)

# --- TISSUE: 조직감등급 5등급 (1~5 → 0~4) ---
def tissue_to_grade(v):
    if pd.isna(v): return np.nan
    v = float(v)
    return int(v) - 1  # 1→0, 2→1, 3→2, 4→3, 5→4

train['target_tissue'] = train['TISSUE'].apply(tissue_to_grade)

# --- GROWTH: 성장등급 3그룹 ---
# 1~3 → 0 (상위, 62%), 4~6 → 1 (중위, 18%), 7~9 → 2 (하위, 20%)
def growth_to_grade(v):
    if pd.isna(v): return np.nan
    v = float(v)
    if v <= 3: return 0
    if v <= 6: return 1
    return 2

train['target_growth'] = train['GROWTH'].apply(growth_to_grade)

# 타깃 컬럼 (WGRADE는 제외 — 별도 추가 가능)
TARGET_COLS = ['target_insfat', 'target_yuksak', 'target_fatsak',
               'target_tissue', 'target_growth']

# 모든 타깃이 유효한 행만 유지
train = train.dropna(subset=TARGET_COLS).reset_index(drop=True)
for col in TARGET_COLS:
    train[col] = train[col].astype(int)

print(f"타깃 유효 행: {len(train):,}")
for col in TARGET_COLS:
    n_cls = train[col].nunique()
    print(f"  {col}: {n_cls}개 클래스, 분포: {dict(train[col].value_counts().sort_index())}")

# 각 태스크별 클래스 수 저장
NUM_CLASSES = {}
for col in TARGET_COLS:
    NUM_CLASSES[col] = train[col].nunique()
print(f"태스크별 클래스 수: {NUM_CLASSES}")

# ============================================================
# 3. 날짜 파싱 및 월 단위 기상 집계
# ============================================================
print("날짜 파싱 중...")
train['BIRTH_YMD'] = pd.to_datetime(train['BIRTH_YMD'].astype(str), format='%Y%m%d', errors='coerce')
train['ABATT_DATE'] = pd.to_datetime(train['ABATT_DATE'], errors='coerce')
train = train.dropna(subset=['BIRTH_YMD', 'ABATT_DATE', 'stn']).reset_index(drop=True)
train['stn'] = train['stn'].astype(str)

weather['date'] = pd.to_datetime(weather['date'], errors='coerce')
weather['stn'] = weather['stn'].astype(str)
weather['year_month'] = weather['date'].dt.to_period('M')

print("월별 기상 집계 중...")
weather_monthly = weather.groupby(['stn', 'year_month']).agg(
    ta_max_mean   = ('ta_max', 'mean'),
    ta_min_mean   = ('ta_min', 'mean'),
    rn_day_sum    = ('rn_day', 'sum'),
    rhm_avg_mean  = ('rhm_avg', 'mean'),
    ws_davg_mean  = ('ws_davg', 'mean'),
).reset_index()
weather_monthly['temp_range_mean'] = (
    weather_monthly['ta_max_mean'] - weather_monthly['ta_min_mean']
)
weather_monthly['year_month_str'] = weather_monthly['year_month'].astype(str)
print(f"월별 기상 행: {len(weather_monthly):,}")

# ============================================================
# 4. 정적(static) 피처 준비
# ============================================================
print("정적 피처 준비 중...")

# --- Area ---
count_cols = ['C2023', 'C2024', 'C2025']
area_counts = area.groupby('FARM_UNIQUE_NO')[count_cols].sum(min_count=1)
area_area   = area.groupby('FARM_UNIQUE_NO')['AREA'].first()
farm_df     = area_counts.join(area_area).reset_index()
farm_df['C_mean'] = farm_df[count_cols].mean(axis=1)
farm_df['AREA_per_head'] = (farm_df['AREA'] / farm_df['C_mean']).replace(
    [np.inf, -np.inf], np.nan
)

# --- Death ---
death['DEAD_YMD'] = pd.to_datetime(death['DEAD_YMD'], errors='coerce')
death['BIRTH_YMD_d'] = pd.to_datetime(death['BIRTH_YMD'], errors='coerce')
death['death_age_days'] = (death['DEAD_YMD'] - death['BIRTH_YMD_d']).dt.days
death_agg = death.groupby('FARM_UNIQUE_NO').agg(
    death_count       = ('DEAD_YMD', 'count'),
    death_age_mean    = ('death_age_days', 'mean'),
    early_death_count = ('death_age_days', lambda x: (x <= 90).sum()),
).reset_index()
death_agg['early_death_ratio'] = (
    death_agg['early_death_count'] / death_agg['death_count']
)

# --- Lineage ---
lineage_fe = lineage.copy()
freq_cols_lin = [
    'KPN_NO', 'FATHER_CATTLE_NO', 'F_GMOTHER_ANIMAL_NO',
    'F_GFATHER_CATTLE_NO', 'M_GFATHER_CATTLE_NO',
    'MOTHER_ANIMAL_NO', 'M_GMOTHER_ANIMAL_NO',
]
for col in freq_cols_lin:
    fmap = lineage_fe[col].value_counts().to_dict()
    lineage_fe[f'{col}_freq'] = lineage_fe[col].map(fmap)

lineage_fe['same_father_mother'] = (
    lineage_fe['FATHER_CATTLE_NO'] == lineage_fe['MOTHER_ANIMAL_NO']
).astype(int)
lineage_fe['same_paternal_gp'] = (
    lineage_fe['F_GMOTHER_ANIMAL_NO'] == lineage_fe['F_GFATHER_CATTLE_NO']
).astype(int)
lineage_fe['same_maternal_gp'] = (
    lineage_fe['M_GMOTHER_ANIMAL_NO'] == lineage_fe['M_GFATHER_CATTLE_NO']
).astype(int)

keep_lin = ['CATTLE_NO'] + [f'{c}_freq' for c in freq_cols_lin] + [
    'same_father_mother', 'same_paternal_gp', 'same_maternal_gp'
]
lineage_features = lineage_fe[keep_lin]

# --- 성별 인코딩 ---
le_sex = LabelEncoder()
train['sex_enc'] = le_sex.fit_transform(train['JUDGE_SEX'].astype(str))

# --- 지역 빈도 ---
for col in ['sido', 'sigungu', 'eupmyeondong']:
    if col in train.columns:
        fmap = train[col].value_counts().to_dict()
        train[f'{col}_freq'] = train[col].map(fmap)

# --- 정적 피처 병합 ---
train = train.merge(farm_df[['FARM_UNIQUE_NO', 'C_mean', 'AREA', 'AREA_per_head']],
                     on='FARM_UNIQUE_NO', how='left')
train = train.merge(death_agg, on='FARM_UNIQUE_NO', how='left')
train = train.merge(lineage_features, on='CATTLE_NO', how='left')

# 폐사 없는 농장은 0
for c in ['death_count', 'death_age_mean', 'early_death_count', 'early_death_ratio']:
    train[c] = train[c].fillna(0)

static_real_cols = [
    'AGE', 'sex_enc', 'C_mean', 'AREA', 'AREA_per_head',
    'death_count', 'death_age_mean', 'early_death_ratio',
    'KPN_NO_freq', 'FATHER_CATTLE_NO_freq',
    'F_GMOTHER_ANIMAL_NO_freq', 'F_GFATHER_CATTLE_NO_freq',
    'M_GFATHER_CATTLE_NO_freq', 'MOTHER_ANIMAL_NO_freq',
    'M_GMOTHER_ANIMAL_NO_freq',
    'same_father_mother', 'same_paternal_gp', 'same_maternal_gp',
    'sido_freq', 'sigungu_freq', 'eupmyeondong_freq',
]

# 결측 채우기
for c in static_real_cols:
    if c in train.columns:
        train[c] = train[c].fillna(train[c].median() if train[c].notna().any() else 0)

print(f"정적 피처 준비 완료. 유효 행: {len(train):,}")

# ============================================================
# 5. 소 1마리 = 1시퀀스 (월별 기상 시계열) 생성
# ============================================================
print("시퀀스 데이터 생성 중... (시간 소요)")

MAX_SAMPLES = 300_000
if len(train) > MAX_SAMPLES:
    train_sampled = train.sample(n=MAX_SAMPLES, random_state=SEED).reset_index(drop=True)
    print(f"샘플링: {len(train):,} → {MAX_SAMPLES:,}")
else:
    train_sampled = train.reset_index(drop=True)

# weather_monthly를 dict로 변환
weather_monthly_dict = {}
for _, row in weather_monthly.iterrows():
    key = (row['stn'], row['year_month_str'])
    weather_monthly_dict[key] = {
        'ta_max_mean': row['ta_max_mean'],
        'ta_min_mean': row['ta_min_mean'],
        'rn_day_sum': row['rn_day_sum'],
        'rhm_avg_mean': row['rhm_avg_mean'],
        'ws_davg_mean': row['ws_davg_mean'],
        'temp_range_mean': row['temp_range_mean'],
    }

weather_vars = [
    'ta_max_mean', 'ta_min_mean', 'rn_day_sum',
    'rhm_avg_mean', 'ws_davg_mean', 'temp_range_mean'
]

rows_list = []
skipped = 0

for idx in range(len(train_sampled)):
    row = train_sampled.iloc[idx]
    birth = row['BIRTH_YMD']
    abatt = row['ABATT_DATE']
    stn   = str(row['stn'])
    cattle_id = idx

    if pd.isna(birth) or pd.isna(abatt):
        skipped += 1
        continue

    months = pd.period_range(start=birth.to_period('M'),
                             end=abatt.to_period('M'), freq='M')

    if len(months) < 2:
        skipped += 1
        continue

    for t, ym in enumerate(months):
        ym_str = str(ym)
        key = (stn, ym_str)
        w_data = weather_monthly_dict.get(key, {})

        r = {
            'cattle_idx': cattle_id,
            'time_idx': t,
        }
        # 타깃 5개
        for tc in TARGET_COLS:
            r[tc] = row[tc]

        # 시계열 피처 (기상)
        for wv in weather_vars:
            r[wv] = w_data.get(wv, np.nan)

        # 정적 피처
        for sc in static_real_cols:
            r[sc] = row[sc]

        # 그룹 (CV용)
        r['farm_id'] = row['FARM_UNIQUE_NO']

        rows_list.append(r)

    if idx % 50000 == 0 and idx > 0:
        print(f"  {idx:,}/{len(train_sampled):,} 처리 완료 "
              f"(누적 행: {len(rows_list):,})")

print(f"스킵: {skipped:,}, 생성 행: {len(rows_list):,}")

ts_df = pd.DataFrame(rows_list)

# 결측 보간
for wv in weather_vars:
    ts_df[wv] = ts_df[wv].fillna(ts_df[wv].median())
for sc in static_real_cols:
    ts_df[sc] = ts_df[sc].fillna(ts_df[sc].median() if ts_df[sc].notna().any() else 0)

ts_df['cattle_idx'] = ts_df['cattle_idx'].astype(str)

# 타깃을 float으로
for tc in TARGET_COLS:
    ts_df[tc] = ts_df[tc].astype(float)

print(f"시계열 DataFrame: {ts_df.shape}")
print(f"고유 소: {ts_df['cattle_idx'].nunique():,}")
print(f"시점 범위: {ts_df['time_idx'].min()} ~ {ts_df['time_idx'].max()}")

# ============================================================
# 6. 시퀀스 길이 통계 및 패딩 설정
# ============================================================
seq_lens = ts_df.groupby('cattle_idx')['time_idx'].max() + 1
print(f"\n시퀀스 길이 통계:")
print(f"  평균: {seq_lens.mean():.1f}, 중앙값: {seq_lens.median():.0f}, "
      f"최소: {seq_lens.min()}, 최대: {seq_lens.max()}")

MAX_ENCODER_LENGTH = min(int(seq_lens.quantile(0.95)), 60)
MAX_PREDICTION_LENGTH = 1

print(f"  MAX_ENCODER_LENGTH: {MAX_ENCODER_LENGTH}")
print(f"  MAX_PREDICTION_LENGTH: {MAX_PREDICTION_LENGTH}")

min_len = MAX_PREDICTION_LENGTH + 2
valid_cattle = seq_lens[seq_lens >= min_len].index
ts_df = ts_df[ts_df['cattle_idx'].isin(valid_cattle)].reset_index(drop=True)
print(f"최소 {min_len} 시점 이상 개체: {len(valid_cattle):,}")

# ============================================================
# 7. Train / Validation 분할
# ============================================================
last_rows = ts_df.groupby('cattle_idx').tail(1).reset_index(drop=True)
cattle_ids = last_rows['cattle_idx'].values
# stratify 기준은 insfat (가장 중요한 태스크)
targets_for_strat = last_rows['target_insfat'].astype(int).values
farms = last_rows['farm_id'].values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_cattle_idx, val_cattle_idx = next(sgkf.split(cattle_ids, targets_for_strat, farms))

train_cattle = set(cattle_ids[train_cattle_idx])
val_cattle   = set(cattle_ids[val_cattle_idx])

ts_train = ts_df[ts_df['cattle_idx'].isin(train_cattle)].reset_index(drop=True)
ts_val   = ts_df[ts_df['cattle_idx'].isin(val_cattle)].reset_index(drop=True)

print(f"\nTrain 소: {len(train_cattle):,}, Val 소: {len(val_cattle):,}")
print(f"Train 행: {len(ts_train):,}, Val 행: {len(ts_val):,}")

# ============================================================
# 8. TimeSeriesDataSet 생성 (멀티타깃)
# ============================================================
print("TimeSeriesDataSet 생성 중...")

time_varying_known_reals = weather_vars.copy()
static_reals = [c for c in static_real_cols if c in ts_df.columns]

# 멀티타깃: 각 타깃은 분류이므로 NaNLabelEncoder 사용
# MultiNormalizer로 묶어준다
target_normalizers = [
    TorchNormalizer(method='identity', center=False, transformation=None)
    for _ in TARGET_COLS
]

training = TimeSeriesDataSet(
    ts_train,
    time_idx='time_idx',
    target=TARGET_COLS,
    group_ids=['cattle_idx'],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    min_encoder_length=2,
    static_reals=static_reals,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=TARGET_COLS,
    target_normalizer=MultiNormalizer(target_normalizers),
    categorical_encoders={'cattle_idx': NaNLabelEncoder(add_nan=True)},
    add_relative_time_idx=True,
    add_target_scales=False,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

validation = TimeSeriesDataSet.from_dataset(training, ts_val, stop_randomization=True)

BATCH_SIZE = 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=4
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=4
)

print(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

# ============================================================
# 9. Kendall Uncertainty Weighting 모듈
# ============================================================
class UncertaintyWeightedLoss(nn.Module):
    """
    Kendall et al. (CVPR 2018)
    loss = Σ_i [ 1/(2·σ_i²) · L_i + log(1 + σ_i²) ]
    σ_i는 학습 가능한 파라미터
    """
    def __init__(self, num_tasks):
        super().__init__()
        # log(σ²) 파라미터로 초기화 (log space → 안정적)
        self.log_sigma_sq = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        losses: list of scalar tensors, len == num_tasks
        returns: weighted total loss (scalar)
        """
        total = 0.0
        for i, loss in enumerate(losses):
            sigma_sq = torch.exp(self.log_sigma_sq[i])
            total += 0.5 / sigma_sq * loss + 0.5 * torch.log(1 + sigma_sq)
        return total

    def get_weights(self):
        """현재 각 태스크 가중치 반환 (1/(2σ²))"""
        with torch.no_grad():
            sigma_sq = torch.exp(self.log_sigma_sq)
            return (0.5 / sigma_sq).cpu().numpy()


# ============================================================
# 10. PCGrad 구현
# ============================================================
class PCGrad:
    """
    Yu et al. (NeurIPS 2020) - Gradient Surgery for Multi-Task Learning
    태스크 간 gradient가 충돌(내적 < 0)할 때 conflicting component를 제거
    """
    def __init__(self, optimizer, reduction='mean'):
        self._optim = optimizer
        self._reduction = reduction

    @property
    def optimizer(self):
        return self._optim

    @property
    def param_groups(self):
        return self._optim.param_groups

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        return self._optim.load_state_dict(state_dict)

    def pc_backward(self, objectives):
        """
        objectives: list of per-task loss tensors
        각 태스크의 gradient를 구한 후 PCGrad 수정 적용
        """
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)

    def _project_conflicting(self, grads, has_grads):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad = copy.deepcopy(grads)
        num_task = len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2 + 1e-8)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction == 'mean':
            merged_grad[shared] = torch.stack(
                [g[shared] for g in pc_grad]
            ).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack(
                [g[shared] for g in pc_grad]
            ).sum(dim=0)
        merged_grad[~shared] = torch.stack(
            [g[~shared] for g in pc_grad]
        ).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        return torch.cat([g.flatten() for g in grads])

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                else:
                    shape.append(p.grad.shape)
                    grad.append(p.grad.clone())
                    has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


# ============================================================
# 11. 멀티태스크 TFT 모델 (커스텀 LightningModule)
# ============================================================
class MultiTaskTFT(pl.LightningModule):
    """
    공유 TFT 인코더 + 태스크별 분류 헤드
    Kendall Uncertainty Weighting + PCGrad 적용
    """
    def __init__(
        self,
        tft_kwargs: dict,
        dataset: TimeSeriesDataSet,
        num_classes_per_task: dict,
        target_cols: list,
        learning_rate: float = 1e-3,
        use_pcgrad: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset'])
        self.target_cols = target_cols
        self.num_tasks = len(target_cols)
        self.num_classes_per_task = num_classes_per_task
        self.lr = learning_rate
        self.use_pcgrad = use_pcgrad

        # 기본 TFT 생성 (첫 번째 태스크 기준, 내부 인코더만 활용)
        # output_size를 임시로 [각 태스크 클래스 수] 합으로 설정
        total_output = sum(num_classes_per_task[tc] for tc in target_cols)
        tft_kwargs['output_size'] = [num_classes_per_task[tc] for tc in target_cols]
        tft_kwargs['loss'] = MultiLoss(
            [CrossEntropy() for _ in target_cols]
        )
        self.tft = TemporalFusionTransformer.from_dataset(
            dataset, **tft_kwargs
        )

        # TFT의 output_layer를 태스크별 개별 헤드로 교체
        hidden_size = tft_kwargs.get('hidden_size', 64)
        self.task_heads = nn.ModuleDict()
        for tc in target_cols:
            n_cls = num_classes_per_task[tc]
            self.task_heads[tc] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, n_cls),
            )

        # 태스크별 CrossEntropy loss
        self.task_losses = nn.ModuleDict()
        for tc in target_cols:
            self.task_losses[tc] = nn.CrossEntropyLoss()

        # Kendall Uncertainty Weighting
        self.uncertainty_loss = UncertaintyWeightedLoss(self.num_tasks)

        # 자동 최적화 끔 (PCGrad 수동 적용 위해)
        self.automatic_optimization = False

    def forward(self, x):
        """TFT 인코더 통과 후 태스크별 헤드 적용"""
        # TFT 내부 forward를 호출해서 hidden state를 얻어야 함
        # 하지만 TFT의 내부 구조에 직접 접근이 필요
        # → TFT의 encode/decode 과정을 수행하고 최종 hidden을 추출

        # TFT forward 호출
        tft_output = self.tft(x)
        return tft_output

    def _extract_hidden_and_predict(self, x):
        """
        TFT 인코더를 통과시켜 hidden representation을 얻고,
        각 태스크별 헤드로 예측값을 반환
        """
        # TFT의 forward를 통해 내부 output 가져오기
        # 내부적으로 encode → decode → output_layer 순서로 진행됨
        # 우리는 output_layer 직전의 hidden state가 필요

        # TFT 내부 메서드 접근
        tft = self.tft

        # 인코더 입력 준비
        encoder_lengths = x['encoder_lengths']
        decoder_lengths = x['decoder_lengths']

        # TFT의 내부 forward 일부를 직접 실행
        # (pytorch-forecasting TFT 구현에 맞춤)
        embeddings = tft.input_embeddings(x)

        # 인코더
        encoder_output = tft.encode(x)

        # 디코더 + attention
        # 전체 forward를 호출하되 output을 raw dict로 받기
        raw_output = tft.forward(x)

        # TFT 내부의 최종 hidden (attention output)을 추출
        # raw_output.prediction은 이미 output_layer를 통과한 결과
        # 우리는 그 직전 hidden이 필요 → TFT 내부 hook이 필요

        # 대안: TFT의 output을 무시하고, tft.output_layer 직전 hook 사용
        # 더 실용적인 방법: TFT forward를 수정해서 hidden을 반환

        return raw_output

    def _get_per_task_predictions(self, batch):
        """
        배치에서 각 태스크별 prediction + target을 추출
        TFT의 내장 multi-target 처리를 활용
        """
        x, y = batch
        # y는 (target_list, weight) 또는 target_list
        # multi-target의 경우 y[0]은 list of tensors

        # TFT forward
        output = self.tft(x)

        # output['prediction']은 list (multi-target)
        predictions = output.prediction  # list of [batch, pred_len, n_classes]

        if isinstance(y, (tuple, list)):
            targets = y[0]  # list of target tensors
        else:
            targets = y

        return predictions, targets

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        x, y = batch
        # TFT forward
        output = self.tft(x)
        predictions = output.prediction

        # y 추출
        if isinstance(y, (tuple, list)):
            targets = y[0]
        else:
            targets = y

        # 타깃이 리스트가 아닌 경우 대비
        if not isinstance(targets, (list, tuple)):
            targets = [targets]
        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

        # 태스크별 loss 계산
        task_losses = []
        for i, tc in enumerate(self.target_cols):
            pred = predictions[i]  # [batch, pred_len, n_classes]
            tgt = targets[i]       # [batch, pred_len] 또는 [batch]

            # 차원 정리
            if pred.dim() == 3:
                pred = pred[:, -1, :]  # 마지막 시점만
            if tgt.dim() == 2:
                tgt = tgt[:, -1]

            n_cls = self.num_classes_per_task[tc]
            tgt = tgt.long().clamp(0, n_cls - 1)
            loss_i = self.task_losses[tc](pred, tgt)
            task_losses.append(loss_i)

        if self.use_pcgrad:
            # PCGrad: 각 태스크 loss에 대해 개별 backward + gradient surgery
            # 먼저 Kendall weighting으로 개별 가중치 적용
            weighted_losses = []
            sigma_sq = torch.exp(self.uncertainty_loss.log_sigma_sq)
            for i, loss_i in enumerate(task_losses):
                w_loss = 0.5 / sigma_sq[i] * loss_i + 0.5 * torch.log(
                    1 + sigma_sq[i]
                )
                weighted_losses.append(w_loss)

            # PCGrad backward
            # PCGrad 래퍼가 내부적으로 backward를 호출하므로,
            # 여기서는 수동으로 구현
            all_grads = []
            all_shapes = []
            all_has_grads = []
            params = list(self.parameters())

            for w_loss in weighted_losses:
                optimizer.zero_grad()
                w_loss.backward(retain_graph=True)
                grad_list = []
                shape_list = []
                has_grad_list = []
                for p in params:
                    if p.grad is not None:
                        grad_list.append(p.grad.clone().flatten())
                        has_grad_list.append(torch.ones_like(p.grad).flatten())
                    else:
                        grad_list.append(torch.zeros(p.numel(), device=p.device))
                        has_grad_list.append(torch.zeros(p.numel(), device=p.device))
                    shape_list.append(p.shape)
                all_grads.append(torch.cat(grad_list))
                all_has_grads.append(torch.cat(has_grad_list))
                all_shapes.append(shape_list)

            # Project conflicting gradients
            shared = torch.stack(all_has_grads).prod(0).bool()
            pc_grads = [g.clone() for g in all_grads]

            for i in range(len(pc_grads)):
                shuffled_indices = list(range(len(all_grads)))
                random.shuffle(shuffled_indices)
                for j in shuffled_indices:
                    if i == j:
                        continue
                    dot = torch.dot(pc_grads[i], all_grads[j])
                    if dot < 0:
                        pc_grads[i] -= dot * all_grads[j] / (
                            all_grads[j].norm() ** 2 + 1e-8
                        )

            # Merge gradients (mean)
            merged = torch.zeros_like(all_grads[0])
            merged[shared] = torch.stack(
                [g[shared] for g in pc_grads]
            ).mean(dim=0)
            merged[~shared] = torch.stack(
                [g[~shared] for g in pc_grads]
            ).sum(dim=0)

            # Set merged gradient
            optimizer.zero_grad()
            idx = 0
            for p in params:
                numel = p.numel()
                p.grad = merged[idx:idx + numel].view(p.shape)
                idx += numel

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
            optimizer.step()

            total_loss = sum(task_losses).detach()
        else:
            # PCGrad 없이 Kendall weighting만 사용
            optimizer.zero_grad()
            total_loss = self.uncertainty_loss(task_losses)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
            optimizer.step()
            total_loss = total_loss.detach()

        # 스케줄러 step
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step(total_loss)

        # 로깅
        self.log('train_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        for i, tc in enumerate(self.target_cols):
            self.log(f'train_loss_{tc}', task_losses[i].detach(), on_epoch=True)

        # Kendall 가중치 로깅
        weights = self.uncertainty_loss.get_weights()
        for i, tc in enumerate(self.target_cols):
            self.log(f'kendall_w_{tc}', weights[i], on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.tft(x)
        predictions = output.prediction

        if isinstance(y, (tuple, list)):
            targets = y[0]
        else:
            targets = y

        if not isinstance(targets, (list, tuple)):
            targets = [targets]
        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

        task_losses = []
        task_preds = []
        task_targets = []

        for i, tc in enumerate(self.target_cols):
            pred = predictions[i]
            tgt = targets[i]

            if pred.dim() == 3:
                pred = pred[:, -1, :]
            if tgt.dim() == 2:
                tgt = tgt[:, -1]

            n_cls = self.num_classes_per_task[tc]
            tgt = tgt.long().clamp(0, n_cls - 1)
            loss_i = self.task_losses[tc](pred, tgt)
            task_losses.append(loss_i)

            pred_labels = pred.argmax(dim=-1)
            task_preds.append(pred_labels)
            task_targets.append(tgt)

        total_loss = self.uncertainty_loss(task_losses)
        self.log('val_loss', total_loss, prog_bar=True, on_epoch=True)

        for i, tc in enumerate(self.target_cols):
            self.log(f'val_loss_{tc}', task_losses[i], on_epoch=True)

        return {
            'val_loss': total_loss,
            'preds': {tc: task_preds[i].cpu() for i, tc in enumerate(self.target_cols)},
            'targets': {tc: task_targets[i].cpu() for i, tc in enumerate(self.target_cols)},
        }

    def on_validation_epoch_end(self):
        pass  # Lightning이 자동으로 val_loss를 집계

    def configure_optimizers(self):
        # 모델 파라미터 + Kendall 파라미터를 함께 최적화
        optimizer = torch.optim.Adam([
            {'params': self.tft.parameters(), 'lr': self.lr},
            {'params': self.task_heads.parameters(), 'lr': self.lr},
            {'params': self.uncertainty_loss.parameters(), 'lr': self.lr, 'weight_decay': 0},
        ])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
            }
        }


# ============================================================
# 12. 모델 생성
# ============================================================
print("멀티태스크 TFT 모델 생성 중...")

tft_kwargs = dict(
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
    log_interval=100,
    reduce_on_plateau_patience=5,
)

model = MultiTaskTFT(
    tft_kwargs=tft_kwargs,
    dataset=training,
    num_classes_per_task=NUM_CLASSES,
    target_cols=TARGET_COLS,
    learning_rate=1e-3,
    use_pcgrad=True,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"모델 파라미터 수: {total_params/1e6:.2f}M")
print(f"태스크 수: {model.num_tasks}")
print(f"태스크별 클래스 수: {NUM_CLASSES}")

# ============================================================
# 13. 학습
# ============================================================
print("학습 시작...")

trainer = pl.Trainer(
    max_epochs=30,
    accelerator='gpu',
    devices=1,
    gradient_clip_val=0.0,  # PCGrad 내부에서 직접 클리핑하므로 0
    callbacks=[
        EarlyStopping(
            monitor='val_loss', patience=5, mode='min'
        ),
        ModelCheckpoint(
            monitor='val_loss', mode='min',
            filename='tft_multitask_best'
        ),
        LearningRateMonitor(),
    ],
    enable_progress_bar=True,
    log_every_n_steps=50,
)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# ============================================================
# 14. 평가
# ============================================================
print("\n평가 중...")

best_model_path = trainer.checkpoint_callback.best_model_path
best_model = MultiTaskTFT.load_from_checkpoint(
    best_model_path,
    tft_kwargs=tft_kwargs,
    dataset=training,
    num_classes_per_task=NUM_CLASSES,
    target_cols=TARGET_COLS,
)
best_model.eval()
best_model.to('cuda')

# 전체 val set에 대해 예측 수집
all_preds = {tc: [] for tc in TARGET_COLS}
all_targets = {tc: [] for tc in TARGET_COLS}

with torch.no_grad():
    for batch in val_dataloader:
        x, y = batch
        # GPU로 이동
        x = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in x.items()}

        output = best_model.tft(x)
        predictions = output.prediction

        if isinstance(y, (tuple, list)):
            targets = y[0]
        else:
            targets = y

        if not isinstance(targets, (list, tuple)):
            targets = [targets]
        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

        for i, tc in enumerate(TARGET_COLS):
            pred = predictions[i]
            tgt = targets[i]

            if pred.dim() == 3:
                pred = pred[:, -1, :]
            if tgt.dim() == 2:
                tgt = tgt[:, -1]

            pred_labels = pred.argmax(dim=-1).cpu().numpy()
            tgt_np = tgt.cpu().numpy().astype(int)

            all_preds[tc].append(pred_labels)
            all_targets[tc].append(tgt_np)

# 결과 출력
TASK_CLASS_NAMES = {
    'target_insfat': ['1++', '1+', '1', '2', '3'],
    'target_yuksak': None,  # 실제 클래스 수에 맞게 자동
    'target_fatsak': None,
    'target_tissue': None,
    'target_growth': ['Good', 'Bad'],
}

print(f"\n{'='*70}")
print(f"멀티태스크 TFT 분류 결과 (Kendall Uncertainty + PCGrad)")
print(f"{'='*70}")

for tc in TARGET_COLS:
    preds = np.concatenate(all_preds[tc])
    targets = np.concatenate(all_targets[tc])

    min_len_eval = min(len(preds), len(targets))
    preds = preds[:min_len_eval]
    targets = targets[:min_len_eval]

    macro_f1 = f1_score(targets, preds, average='macro')
    class_names = TASK_CLASS_NAMES.get(tc)

    print(f"\n--- {tc} ---")
    print(f"Macro F1: {macro_f1:.4f}")
    if class_names and len(class_names) == NUM_CLASSES[tc]:
        print(classification_report(targets, preds, target_names=class_names))
    else:
        print(classification_report(targets, preds))

# Kendall 가중치 최종값
print(f"\n--- Kendall Uncertainty 최종 가중치 ---")
weights = best_model.uncertainty_loss.get_weights()
for i, tc in enumerate(TARGET_COLS):
    sigma_sq = torch.exp(best_model.uncertainty_loss.log_sigma_sq[i]).item()
    print(f"  {tc}: weight={weights[i]:.4f}, σ²={sigma_sq:.4f}")

# ============================================================
# 15. 모델 저장
# ============================================================
trainer.save_checkpoint('tft_multitask_model.ckpt')
print("\n모델 저장: tft_multitask_model.ckpt")
print("학습 완료.")
