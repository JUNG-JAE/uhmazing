# =============================================================================
# tft.py — 한우 멀티태스크 TFT 파이프라인 (분류 전용)
# 분류 6개: INSFAT/YUKSAK/FATSAK/TISSUE/GROWTH 등급 + WGRADE(A/B/C)
# Kendall Uncertainty Weighting + 클래스 가중치
# 후처리: 육질등급 + WGRADE → LAST_GRADE
# =============================================================================

import os
import warnings
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, classification_report, accuracy_score
)

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# 섹션 1: 설정
# =============================================================================
DATA_DIR = '../data/'
TRAIN_FILE = os.path.join(DATA_DIR, 'hanwoo_train_merged.parquet')
TEST_FILE = os.path.join(DATA_DIR, 'hanwoo_test_merged.parquet')
WEATHER_FILE = os.path.join(DATA_DIR, 'hanwoo_weather_augmented.csv')

# ★ 회귀 제거, WGRADE 분류 추가
CLASSIFICATION_TARGETS = [
    'INSFAT_GRADE_CLASS', 'YUKSAK_GRADE_CLASS',
    'FATSAK_GRADE_CLASS', 'TISSUE_GRADE_CLASS',
    'GROWTH_STATUS_CLASS', 'WGRADE_CLASS'
]
ALL_TARGETS = CLASSIFICATION_TARGETS

# 클래스 가중치 적용 대상 (GROWTH 제외)
WEIGHTED_TARGETS = ['INSFAT_GRADE_CLASS', 'YUKSAK_GRADE_CLASS',
                    'FATSAK_GRADE_CLASS', 'TISSUE_GRADE_CLASS',
                    'WGRADE_CLASS']

EXCLUDE_COLS = [
    'DATA_ROW_ID', 'CATTLE_NO', 'FARM_UNIQUE_NO',
    'LINEAGE_FATHER_NO', 'LINEAGE_MOTHER_NO', 'FARM_AREA',
    'PAST_SLAUGHTER_SAMPLE_COUNT', 'PAST_SLAUGHTER_HISTORY_YEARS',
    'PAST_SLAUGHTER_AVAILABLE',
    'PAST_WEIGHT_VARIANCE', 'PAST_BACKFAT_VARIANCE',
    'PAST_REA_VARIANCE',
    'INSFAT', 'YUKSAK', 'FATSAK', 'TISSUE', 'GROWTH',
    'BACKFAT', 'REA',
    'WINDEX', 'WGRADE', 'LAST_GRADE',
    'ABATT_DATE', 'JUDGE_DATE', 'BIRTH_YMD',
]

CATEGORICAL_COLS = ['sido', 'sigungu', 'eupmyeondong', 'stn',
                    'JUDGE_SEX', 'ABATT_SEASON', 'BIRTH_SEASON']

FLAG_COLS = ['FATHER_OFFSPRING_IMPUTED', 'MOTHER_OFFSPRING_IMPUTED',
             'FARM_BIOSECURITY_STATUS', 'FARM_HEALTH_STATUS',
             'FARM_ACCIDENT_STATUS']

WEATHER_VARS = ['ta_max_mean', 'ta_min_mean', 'rn_day_sum',
                'rhm_avg_mean', 'ws_davg_mean', 'temp_range_mean']

# ★ WGRADE 매핑
WGRADE_MAP = {'A': 0, 'B': 1, 'C': 2}
WGRADE_INV_MAP = {0: 'A', 1: 'B', 2: 'C'}

MAX_SAMPLES = 300_000
MAX_PREDICTION_LENGTH = 1
BATCH_SIZE = 64

print("=" * 70)
print("한우 멀티태스크 TFT 파이프라인 (분류 전용)")
print("분류 6개: INSFAT/YUKSAK/FATSAK/TISSUE/GROWTH + WGRADE(A/B/C)")
print("Kendall Uncertainty Weighting + 클래스 가중치 (역빈도)")
print("후처리: 육질등급 + WGRADE → LAST_GRADE")
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

# ★ WGRADE를 숫자 클래스로 변환 (A=0, B=1, C=2)
if 'WGRADE' in train_raw.columns:
    train_raw['WGRADE_CLASS'] = train_raw['WGRADE'].map(WGRADE_MAP)
    print(f"  WGRADE → WGRADE_CLASS 변환 완료")
else:
    raise ValueError("WGRADE 컬럼이 train 데이터에 없습니다")

# 타깃 결측 제거 (WGRADE_CLASS 포함)
FILTER_TARGETS = [
    'INSFAT_GRADE_CLASS', 'YUKSAK_GRADE_CLASS',
    'FATSAK_GRADE_CLASS', 'TISSUE_GRADE_CLASS',
    'GROWTH_STATUS_CLASS', 'WGRADE_CLASS'
]
before = len(train_raw)
train_raw = train_raw.dropna(subset=FILTER_TARGETS).reset_index(drop=True)
print(f"  타깃 결측 제거: {before} → {len(train_raw)}")

# 타깃 타입 변환
for col in CLASSIFICATION_TARGETS:
    train_raw[col] = pd.to_numeric(train_raw[col], errors='coerce').astype(int)

NUM_CLASSES = {}
for col in CLASSIFICATION_TARGETS:
    NUM_CLASSES[col] = int(train_raw[col].max()) + 1
    print(f"  {col}: {NUM_CLASSES[col]} classes, dist={dict(train_raw[col].value_counts().sort_index())}")

# 클래스 가중치 계산
CLASS_WEIGHTS = {}
print("\n  [클래스 가중치 계산]")
for col in CLASSIFICATION_TARGETS:
    if col in WEIGHTED_TARGETS:
        counts = train_raw[col].value_counts().sort_index()
        n_samples = len(train_raw)
        n_classes = NUM_CLASSES[col]
        w = torch.tensor(
            [n_samples / (n_classes * counts.get(i, 1)) for i in range(n_classes)],
            dtype=torch.float32
        )
        CLASS_WEIGHTS[col] = w
        print(f"    {col}: {[f'{x:.4f}' for x in w.tolist()]}")
    else:
        CLASS_WEIGHTS[col] = None
        print(f"    {col}: 없음 (균등 CrossEntropy)")

# =============================================================================
# 섹션 3: 피처 전처리
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

for col in static_real_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    median_val = train_df[col].median()
    if pd.isna(median_val):
        median_val = 0.0
    train_df[col] = train_df[col].fillna(median_val)
    if test_df is not None:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        test_df[col] = test_df[col].fillna(median_val)

print(f"  전처리 완료: train {train_df.shape}")

# =============================================================================
# 섹션 4: 날씨 데이터 주간 집계
# =============================================================================
print("\n[섹션 4] 날씨 데이터 로드 및 주간 집계...")

weather_raw = pd.read_csv(WEATHER_FILE)
print(f"  날씨 원본: {weather_raw.shape}")

weather_raw['date'] = pd.to_datetime(weather_raw['date'].astype(str), format='%Y%m%d', errors='coerce')
weather_raw = weather_raw.dropna(subset=['date'])
weather_raw['stn'] = weather_raw['stn'].astype(int)

weather_cols = ['ta_max', 'ta_min', 'rn_day', 'rhm_avg', 'ws_davg']
for col in weather_cols:
    weather_raw[col] = pd.to_numeric(weather_raw[col], errors='coerce')
weather_raw = weather_raw.sort_values(['stn', 'date'])
weather_raw[weather_cols] = weather_raw.groupby('stn')[weather_cols].transform(
    lambda x: x.interpolate(method='linear', limit_direction='both')
)
weather_raw['temp_range'] = weather_raw['ta_max'] - weather_raw['ta_min']

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

weather_dict = {}
for _, row in weather_weekly.iterrows():
    key = (int(row['stn']), row['year_week'])
    weather_dict[key] = {v: row[v] for v in WEATHER_VARS}

print(f"  주간 집계 완료: {len(weather_weekly)} rows, {len(weather_dict)} entries")

weather_global_median = {}
for v in WEATHER_VARS:
    weather_global_median[v] = weather_weekly[v].median()

# =============================================================================
# 섹션 5: 시퀀스 데이터 생성
# =============================================================================
print("\n[섹션 5] 시퀀스 데이터 생성...")

def build_sequences(df, weather_dict, weather_global_median, static_cols,
                    max_samples=None, is_test=False):
    valid = df.dropna(subset=['BIRTH_YMD', 'ABATT_DATE']).copy()
    valid['stn_int'] = pd.to_numeric(valid['stn'] if 'stn' in valid.columns else
                                      valid.get('stn_freq', np.nan), errors='coerce')
    if 'stn' in df.columns:
        valid['stn_int'] = pd.to_numeric(df.loc[valid.index, 'stn'], errors='coerce')
    valid = valid.dropna(subset=['stn_int'])
    valid['stn_int'] = valid['stn_int'].astype(int)
    
    if max_samples and len(valid) > max_samples:
        valid = valid.sample(n=max_samples, random_state=SEED).reset_index(drop=True)
    
    print(f"    시퀀스 생성 대상: {len(valid)} 개체")
    
    CHUNK_SIZE = 5000
    chunks = []
    skipped = 0
    total_rows = 0
    
    for chunk_start in range(0, len(valid), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(valid))
        chunk_rows = []
        
        for idx in range(chunk_start, chunk_end):
            row = valid.iloc[idx]
            birth = row['BIRTH_YMD']
            abatt = row['ABATT_DATE']
            stn = int(row['stn_int'])
            
            if pd.isna(birth) or pd.isna(abatt) or abatt <= birth:
                skipped += 1
                continue
            
            weeks = pd.date_range(start=birth, end=abatt, freq='W-MON')
            if len(weeks) < 3:
                skipped += 1
                continue
            
            static_vals = {}
            for col in static_cols:
                val = row.get(col, 0.0)
                static_vals[col] = float(val) if not pd.isna(val) else 0.0
            
            targets = {}
            if is_test:
                for t in ALL_TARGETS:
                    targets[t] = 0
            else:
                for t in ALL_TARGETS:
                    targets[t] = int(row[t])
            
            weight_val = float(row.get('WEIGHT', 0.0)) if not pd.isna(row.get('WEIGHT', np.nan)) else 0.0
            sex_val = row.get('JUDGE_SEX_orig', 'unknown')
            
            farm_id = row.get('FARM_UNIQUE_NO', f'farm_{idx}')
            if pd.isna(farm_id):
                farm_id = f'farm_{idx}'
            
            n_weeks = len(weeks)
            for t, week_date in enumerate(weeks):
                yr = week_date.isocalendar()[0]
                wk = week_date.isocalendar()[1]
                yw = f"{yr}-W{wk:02d}"
                
                seq_row = {
                    'cattle_idx': idx,
                    'time_idx': t,
                    'farm_id': farm_id,
                    'weight_val': weight_val,
                    'sex_val': sex_val,
                }
                seq_row.update(static_vals)
                
                wkey = (stn, yw)
                if wkey in weather_dict:
                    seq_row.update(weather_dict[wkey])
                else:
                    seq_row.update(weather_global_median)
                
                for tgt in ALL_TARGETS:
                    if t == n_weeks - 1:
                        seq_row[tgt] = targets[tgt]
                    else:
                        seq_row[tgt] = 0
                
                chunk_rows.append(seq_row)
        
        if chunk_rows:
            chunk_df = pd.DataFrame(chunk_rows)
            chunks.append(chunk_df)
            total_rows += len(chunk_rows)
            del chunk_rows
        
        processed = min(chunk_end, len(valid))
        print(f"      진행: {processed}/{len(valid)} 개체, 누적 {total_rows:,} rows")
    
    print(f"    청크 병합 중...")
    result = pd.concat(chunks, ignore_index=True)
    del chunks
    
    print(f"    생성 완료: {len(result):,} rows, 건너뜀: {skipped}")
    return result

seq_train = build_sequences(train_df, weather_dict, weather_global_median,
                            static_real_cols, max_samples=MAX_SAMPLES, is_test=False)

seq_lengths = seq_train.groupby('cattle_idx')['time_idx'].max() + 1
print(f"\n  시퀀스 길이: mean={seq_lengths.mean():.0f}, median={seq_lengths.median():.0f}, "
      f"min={seq_lengths.min()}, max={seq_lengths.max()}")

q95 = int(seq_lengths.quantile(0.95))
MAX_ENCODER_LENGTH = min(q95, 200)
print(f"  MAX_ENCODER_LENGTH = {MAX_ENCODER_LENGTH} (95th pct={q95})")
print(f"  MAX_PREDICTION_LENGTH = {MAX_PREDICTION_LENGTH}")

min_len = MAX_PREDICTION_LENGTH + 2
valid_cattle = seq_lengths[seq_lengths >= min_len].index
seq_train = seq_train[seq_train['cattle_idx'].isin(valid_cattle)].reset_index(drop=True)
n_entities = seq_train['cattle_idx'].nunique()
print(f"  유효 개체: {n_entities}, 총 rows: {len(seq_train)}")

# =============================================================================
# 섹션 6: Train/Val 분할 (8:2)
# =============================================================================
print("\n[섹션 6] Train/Val 분할 (8:2)...")

unique_cattle = seq_train['cattle_idx'].unique()
np.random.shuffle(unique_cattle)
split_idx = int(len(unique_cattle) * 0.8)
train_cattle = set(unique_cattle[:split_idx])
val_cattle = set(unique_cattle[split_idx:])

ts_train = seq_train[seq_train['cattle_idx'].isin(train_cattle)].reset_index(drop=True)
ts_val = seq_train[seq_train['cattle_idx'].isin(val_cattle)].reset_index(drop=True)

print(f"  Train: {len(train_cattle)} 개체, {len(ts_train)} rows")
print(f"  Val: {len(val_cattle)} 개체, {len(ts_val)} rows")

# =============================================================================
# 섹션 7: 커스텀 Dataset 정의
# =============================================================================
print("\n[섹션 7] 커스텀 Dataset 생성...")

class HanwooSequenceDataset(Dataset):
    def __init__(self, df, static_cols, weather_vars, max_encoder_length,
                 classification_targets):
        self.static_cols = static_cols
        self.weather_vars = weather_vars
        self.max_enc_len = max_encoder_length
        self.cls_targets = classification_targets
        
        self.groups = []
        for cattle_id, grp in df.groupby('cattle_idx'):
            grp = grp.sort_values('time_idx').reset_index(drop=True)
            self.groups.append(grp)
    
    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, idx):
        grp = self.groups[idx]
        seq_len = len(grp)
        
        if seq_len > self.max_enc_len:
            grp = grp.iloc[-self.max_enc_len:].reset_index(drop=True)
            seq_len = self.max_enc_len
        
        static_vals = torch.tensor(
            grp[self.static_cols].iloc[0].values.astype(np.float32),
            dtype=torch.float32
        )
        weather_seq = torch.tensor(
            grp[self.weather_vars].values.astype(np.float32),
            dtype=torch.float32
        )
        
        last_row = grp.iloc[-1]
        cls_vals = torch.tensor(
            [int(last_row[t]) for t in self.cls_targets],
            dtype=torch.long
        )
        
        weight_val = float(last_row.get('weight_val', 0.0))
        sex_val = str(last_row.get('sex_val', 'unknown'))
        
        return {
            'static': static_vals,
            'weather': weather_seq,
            'seq_len': seq_len,
            'cls_targets': cls_vals,
            'weight': weight_val,
            'sex': sex_val,
        }

def collate_fn(batch):
    max_len = max(b['seq_len'] for b in batch)
    bs = len(batch)
    n_static = batch[0]['static'].shape[0]
    n_weather = batch[0]['weather'].shape[1]
    
    static = torch.stack([b['static'] for b in batch])
    weather = torch.zeros(bs, max_len, n_weather)
    mask = torch.zeros(bs, max_len, dtype=torch.bool)
    cls_targets = torch.stack([b['cls_targets'] for b in batch])
    seq_lens = torch.tensor([b['seq_len'] for b in batch])
    weights = [b['weight'] for b in batch]
    sexes = [b['sex'] for b in batch]
    
    for i, b in enumerate(batch):
        slen = b['seq_len']
        weather[i, :slen, :] = b['weather']
        mask[i, :slen] = True
    
    return {
        'static': static,
        'weather': weather,
        'mask': mask,
        'seq_lens': seq_lens,
        'cls_targets': cls_targets,
        'weights': weights,
        'sexes': sexes,
    }

train_dataset = HanwooSequenceDataset(
    ts_train, static_real_cols, WEATHER_VARS, MAX_ENCODER_LENGTH,
    CLASSIFICATION_TARGETS
)
val_dataset = HanwooSequenceDataset(
    ts_val, static_real_cols, WEATHER_VARS, MAX_ENCODER_LENGTH,
    CLASSIFICATION_TARGETS
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# =============================================================================
# 섹션 8: Kendall Uncertainty Weighting
# =============================================================================
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.log_sigma_sq = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        total = 0.0
        for i, loss in enumerate(losses):
            sigma_sq = torch.exp(self.log_sigma_sq[i])
            total = total + 0.5 / sigma_sq * loss + 0.5 * torch.log(1 + sigma_sq)
        return total
    
    def get_weights(self):
        with torch.no_grad():
            sigma_sq = torch.exp(self.log_sigma_sq)
            weights = 0.5 / sigma_sq
        return weights.cpu().numpy(), sigma_sq.cpu().numpy()

# =============================================================================
# 섹션 9: TFT 기반 멀티태스크 모델 (분류 전용)
# =============================================================================
print("\n[섹션 9] 모델 정의...")

class TemporalBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,
                            num_layers=2, dropout=dropout, bidirectional=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
    
    def forward(self, x, mask=None):
        lstm_out, _ = self.lstm(x)
        if mask is not None:
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out,
                                key_padding_mask=key_padding_mask)
        x = self.norm1(lstm_out + attn_out)
        ff_out = self.ff(x)
        gate = self.gate(x)
        x = self.norm2(x + gate * ff_out)
        return x


class MultiTaskTFT(pl.LightningModule):
    def __init__(self, n_static, n_weather, hidden_size=128, num_heads=4,
                 dropout=0.1, num_classes_dict=None, learning_rate=1e-3,
                 class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.learning_rate = learning_rate
        self.num_classes_dict = num_classes_dict or {}
        
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.weather_proj = nn.Sequential(
            nn.Linear(n_weather, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.combine_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = TemporalBlock(
            hidden_size, hidden_size, num_heads, dropout
        )
        
        # ★ 분류 헤드만 (회귀 헤드 제거)
        self.cls_heads = nn.ModuleDict()
        for col, nc in self.num_classes_dict.items():
            self.cls_heads[col] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, nc),
            )
        
        # ★ 6개 태스크
        n_tasks = len(self.num_classes_dict)
        self.uncertainty_loss = UncertaintyWeightedLoss(n_tasks)
        
        self.ce_losses = nn.ModuleDict()
        for col in self.num_classes_dict:
            if class_weights and class_weights.get(col) is not None:
                self.ce_losses[col] = nn.CrossEntropyLoss(weight=class_weights[col])
            else:
                self.ce_losses[col] = nn.CrossEntropyLoss()
    
    def forward(self, static, weather, mask, seq_lens):
        bs = static.shape[0]
        
        static_h = self.static_proj(static)
        weather_h = self.weather_proj(weather)
        
        static_expanded = static_h.unsqueeze(1).expand(-1, weather_h.shape[1], -1)
        combined = torch.cat([static_expanded, weather_h], dim=-1)
        combined = self.combine_proj(combined)
        
        encoded = self.temporal_encoder(combined, mask)
        
        last_indices = (seq_lens - 1).long()
        last_hidden = encoded[torch.arange(bs, device=encoded.device), last_indices]
        
        cls_logits = {}
        for col in self.num_classes_dict:
            cls_logits[col] = self.cls_heads[col](last_hidden)
        
        return cls_logits
    
    def _compute_loss(self, batch):
        static = batch['static']
        weather = batch['weather']
        mask = batch['mask']
        seq_lens = batch['seq_lens']
        cls_targets = batch['cls_targets']
        
        cls_logits = self(static, weather, mask, seq_lens)
        
        cls_losses = []
        losses_dict = {}
        cls_cols = list(self.num_classes_dict.keys())
        for i, col in enumerate(cls_cols):
            logits = cls_logits[col]
            target = cls_targets[:, i].clamp(0, self.num_classes_dict[col] - 1)
            loss = self.ce_losses[col](logits, target)
            cls_losses.append(loss)
            losses_dict[col] = loss.detach()
        
        total_loss = self.uncertainty_loss(cls_losses)
        
        return total_loss, losses_dict, cls_logits
    
    def training_step(self, batch, batch_idx):
        total_loss, losses_dict, _ = self._compute_loss(batch)
        self.log('train_loss', total_loss, prog_bar=True)
        for col, loss in losses_dict.items():
            self.log(f'train_loss_{col}', loss)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        total_loss, losses_dict, _ = self._compute_loss(batch)
        self.log('val_loss', total_loss, prog_bar=True)
        for col, loss in losses_dict.items():
            self.log(f'val_loss_{col}', loss)
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

n_static = len(static_real_cols)
n_weather = len(WEATHER_VARS)

model = MultiTaskTFT(
    n_static=n_static,
    n_weather=n_weather,
    hidden_size=128,
    num_heads=4,
    dropout=0.1,
    num_classes_dict=NUM_CLASSES,
    learning_rate=1e-3,
    class_weights=CLASS_WEIGHTS,
)

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  총 파라미터: {total_params:,} ({trainable:,} trainable)")
print(f"  태스크: 분류 {len(NUM_CLASSES)}개")
for col, nc in NUM_CLASSES.items():
    w_info = "가중치 적용" if CLASS_WEIGHTS.get(col) is not None else "균등"
    print(f"    {col}: {nc} classes ({w_info})")

# =============================================================================
# 섹션 10: 학습
# =============================================================================
print("\n[섹션 10] 학습 시작...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=True),
    ModelCheckpoint(
        dirpath='checkpoints/',
        filename='tft_multitask_best',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True,
    ),
    LearningRateMonitor(logging_interval='epoch'),
]

trainer = pl.Trainer(
    max_epochs=30,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    gradient_clip_val=0.5,
    callbacks=callbacks,
    enable_progress_bar=True,
    log_every_n_steps=50,
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# =============================================================================
# 섹션 11: 등급 계산 함수
# =============================================================================

QUALITY_GRADE_MAP = {0: '1++', 1: '1+', 2: '1', 3: '2', 4: '3'}
QUALITY_DOWNGRADE = {'1++': '1+', '1+': '1', '1': '2', '2': '3', '3': '등외'}

def compute_quality_grade(insfat_cls, yuksak_cls, fatsak_cls, tissue_cls, growth_cls):
    worst = max(insfat_cls, yuksak_cls, fatsak_cls, tissue_cls)
    grade = QUALITY_GRADE_MAP.get(worst, '3')
    if growth_cls == 1 and worst < 4:
        grade = QUALITY_DOWNGRADE.get(grade, grade)
    return grade

def compute_last_grade(quality_grade, wgrade):
    return f"{quality_grade}{wgrade}"

# =============================================================================
# 섹션 12: 평가 (Validation)
# =============================================================================
print("\n[섹션 12] 검증 평가...")

best_path = callbacks[1].best_model_path
if best_path:
    model = MultiTaskTFT.load_from_checkpoint(
        best_path, class_weights=CLASS_WEIGHTS
    )
    print(f"  Best 모델 로드: {best_path}")
else:
    print("  체크포인트 없음 — 현재 모델 사용")

model.eval()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
device = next(model.parameters()).device

all_cls_pred = {col: [] for col in CLASSIFICATION_TARGETS}
all_cls_true = {col: [] for col in CLASSIFICATION_TARGETS}
all_weights, all_sexes = [], []
all_quality_pred, all_wgrade_pred, all_last_grade_pred = [], [], []
all_quality_true, all_wgrade_true, all_last_grade_true = [], [], []

with torch.no_grad():
    for batch in val_loader:
        static = batch['static'].to(device)
        weather = batch['weather'].to(device)
        mask = batch['mask'].to(device)
        seq_lens = batch['seq_lens'].to(device)
        cls_targets = batch['cls_targets']
        
        cls_logits = model(static, weather, mask, seq_lens)
        
        cls_cols = list(NUM_CLASSES.keys())
        for i, col in enumerate(cls_cols):
            preds = cls_logits[col].argmax(dim=-1).cpu().numpy()
            trues = cls_targets[:, i].numpy()
            all_cls_pred[col].extend(preds)
            all_cls_true[col].extend(trues)
        
        all_weights.extend(batch['weights'])
        all_sexes.extend(batch['sexes'])

# 분류 평가
print("\n" + "=" * 50)
print("분류 평가")
print("=" * 50)
TASK_CLASS_NAMES = {
    'INSFAT_GRADE_CLASS': ['1++', '1+', '1', '2', '3'],
    'YUKSAK_GRADE_CLASS': ['1++', '1+', '1', '2', '3'],
    'FATSAK_GRADE_CLASS': ['1++', '1+', '1', '2', '3'],
    'TISSUE_GRADE_CLASS': ['1++', '1+', '1', '2', '3'],
    'GROWTH_STATUS_CLASS': ['정상', '비정상'],
    'WGRADE_CLASS': ['A', 'B', 'C'],
}

for col in CLASSIFICATION_TARGETS:
    pred = np.array(all_cls_pred[col])
    true = np.array(all_cls_true[col])
    f1 = f1_score(true, pred, average='macro', zero_division=0)
    acc = accuracy_score(true, pred)
    names = TASK_CLASS_NAMES.get(col, None)
    w_info = "가중치 적용" if CLASS_WEIGHTS.get(col) is not None else "균등"
    print(f"\n  {col} ({w_info}): Macro F1={f1:.4f}, Accuracy={acc:.4f}")
    try:
        if names and len(names) == NUM_CLASSES[col]:
            print(classification_report(true, pred, target_names=names, zero_division=0))
        else:
            print(classification_report(true, pred, zero_division=0))
    except Exception as e:
        print(f"    classification_report 오류: {e}")

# Derived 평가 (LAST_GRADE)
print("\n" + "=" * 50)
print("Derived 평가 (육질등급, LAST_GRADE)")
print("=" * 50)

for i in range(len(all_weights)):
    # ★ WGRADE는 모델이 직접 예측한 클래스 사용
    wgrade_p = WGRADE_INV_MAP.get(int(all_cls_pred['WGRADE_CLASS'][i]), 'B')
    
    insfat_p = int(all_cls_pred['INSFAT_GRADE_CLASS'][i])
    yuksak_p = int(all_cls_pred['YUKSAK_GRADE_CLASS'][i])
    fatsak_p = int(all_cls_pred['FATSAK_GRADE_CLASS'][i])
    tissue_p = int(all_cls_pred['TISSUE_GRADE_CLASS'][i])
    growth_p = int(all_cls_pred['GROWTH_STATUS_CLASS'][i])
    
    quality_p = compute_quality_grade(insfat_p, yuksak_p, fatsak_p, tissue_p, growth_p)
    last_grade_p = compute_last_grade(quality_p, wgrade_p)
    
    all_wgrade_pred.append(wgrade_p)
    all_quality_pred.append(quality_p)
    all_last_grade_pred.append(last_grade_p)
    
    wgrade_t = WGRADE_INV_MAP.get(int(all_cls_true['WGRADE_CLASS'][i]), 'B')
    
    insfat_t = int(all_cls_true['INSFAT_GRADE_CLASS'][i])
    yuksak_t = int(all_cls_true['YUKSAK_GRADE_CLASS'][i])
    fatsak_t = int(all_cls_true['FATSAK_GRADE_CLASS'][i])
    tissue_t = int(all_cls_true['TISSUE_GRADE_CLASS'][i])
    growth_t = int(all_cls_true['GROWTH_STATUS_CLASS'][i])
    
    quality_t = compute_quality_grade(insfat_t, yuksak_t, fatsak_t, tissue_t, growth_t)
    last_grade_t = compute_last_grade(quality_t, wgrade_t)
    
    all_wgrade_true.append(wgrade_t)
    all_quality_true.append(quality_t)
    all_last_grade_true.append(last_grade_t)

quality_acc = accuracy_score(all_quality_true, all_quality_pred)
quality_f1 = f1_score(all_quality_true, all_quality_pred, average='macro', zero_division=0)
print(f"\n  육질등급: Accuracy={quality_acc:.4f}, Macro F1={quality_f1:.4f}")
try:
    print(classification_report(all_quality_true, all_quality_pred, zero_division=0))
except Exception as e:
    print(f"    {e}")

lg_acc = accuracy_score(all_last_grade_true, all_last_grade_pred)
lg_f1 = f1_score(all_last_grade_true, all_last_grade_pred, average='macro', zero_division=0)
print(f"\n  LAST_GRADE: Accuracy={lg_acc:.4f}, Macro F1={lg_f1:.4f}")
try:
    print(classification_report(all_last_grade_true, all_last_grade_pred, zero_division=0))
except Exception as e:
    print(f"    {e}")

# Kendall 가중치 출력
weights_k, sigma_sq = model.uncertainty_loss.get_weights()
task_names = list(NUM_CLASSES.keys())
print("\n" + "=" * 50)
print("Kendall Uncertainty Weights")
print("=" * 50)
for i, name in enumerate(task_names):
    print(f"  {name}: weight={weights_k[i]:.4f}, σ²={sigma_sq[i]:.4f}")

# =============================================================================
# 섹션 13: 테스트 데이터 추론
# =============================================================================
if has_test:
    print("\n[섹션 13] 테스트 데이터 추론...")
    
    # ★ 테스트 데이터에 WGRADE_CLASS 더미 추가
    if 'WGRADE_CLASS' not in test_df.columns:
        test_df['WGRADE_CLASS'] = 0
    
    seq_test = build_sequences(test_df, weather_dict, weather_global_median,
                               static_cols=static_real_cols, is_test=True)
    
    if len(seq_test) > 0:
        test_seq_lengths = seq_test.groupby('cattle_idx')['time_idx'].max() + 1
        valid_test_cattle = test_seq_lengths[test_seq_lengths >= min_len].index
        seq_test = seq_test[seq_test['cattle_idx'].isin(valid_test_cattle)].reset_index(drop=True)
        print(f"  테스트 유효 개체: {seq_test['cattle_idx'].nunique()}, 총 rows: {len(seq_test)}")
        
        test_dataset = HanwooSequenceDataset(
            seq_test, static_real_cols, WEATHER_VARS, MAX_ENCODER_LENGTH,
            CLASSIFICATION_TARGETS
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, collate_fn=collate_fn, pin_memory=True)
        
        test_results = []
        
        with torch.no_grad():
            for batch in test_loader:
                static = batch['static'].to(device)
                weather = batch['weather'].to(device)
                mask = batch['mask'].to(device)
                seq_lens = batch['seq_lens'].to(device)
                
                cls_logits = model(static, weather, mask, seq_lens)
                
                for j in range(static.shape[0]):
                    cls_preds = {}
                    for col in CLASSIFICATION_TARGETS:
                        cls_preds[col] = cls_logits[col][j].argmax().item()
                    
                    wgrade_p = WGRADE_INV_MAP.get(cls_preds['WGRADE_CLASS'], 'B')
                    quality_p = compute_quality_grade(
                        cls_preds['INSFAT_GRADE_CLASS'],
                        cls_preds['YUKSAK_GRADE_CLASS'],
                        cls_preds['FATSAK_GRADE_CLASS'],
                        cls_preds['TISSUE_GRADE_CLASS'],
                        cls_preds['GROWTH_STATUS_CLASS'],
                    )
                    last_grade_p = compute_last_grade(quality_p, wgrade_p)
                    
                    result = {
                        'WGRADE_pred': wgrade_p,
                        'quality_grade_pred': quality_p,
                        'LAST_GRADE_pred': last_grade_p,
                    }
                    result.update({f'{col}_pred': cls_preds[col] for col in CLASSIFICATION_TARGETS})
                    test_results.append(result)
        
        test_pred_df = pd.DataFrame(test_results)
        test_pred_df.to_csv('../data/test_predictions.csv', index=False)
        print(f"  테스트 예측 저장: ../data/test_predictions.csv ({len(test_pred_df)} rows)")
        print(f"  WGRADE 분포: {dict(test_pred_df['WGRADE_pred'].value_counts())}")
        print(f"  LAST_GRADE 분포: {dict(test_pred_df['LAST_GRADE_pred'].value_counts().head(10))}")
    else:
        print("  테스트 시퀀스 생성 실패")
else:
    print("\n[섹션 13] 테스트 파일 없음 — 건너뜀")

# =============================================================================
# 섹션 14: 모델 저장
# =============================================================================
print("\n[섹션 14] 모델 저장...")
save_path = 'checkpoints/tft_multitask_model.ckpt'
trainer.save_checkpoint(save_path)
print(f"  저장 완료: {save_path}")
print("\n" + "=" * 70)
print("파이프라인 완료!")
print("=" * 70)
