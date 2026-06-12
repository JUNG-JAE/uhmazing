import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    """
    ETTh1/ETTh2(시간 단위) 데이터셋 로더.

    채널 독립(channel independence) 처리가 핵심이다.
    __getitem__은 한 번에 '하나의 변수(채널)'에 대한 윈도우만 반환하며,
    __len__은 (윈도우 개수 x 변수 개수)가 된다. 즉, (윈도우, 채널) 쌍 하나가
    독립적인 단변량 샘플로 취급된다. 따라서 모델이 보는 입력은 항상 N=1 이다.

    train/val/test 분할은 ETT 공식 기준인 12/4/4 개월을 따른다.
    """

    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True, timeenc=0, freq='h', percent=100):
        # size = [입력길이, 예측길이]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent          # 학습 데이터 사용 비율(few-shot 실험용)
        self.features = features
        self.target = target
        self.scale = scale              # 표준화 여부
        self.timeenc = timeenc          # 0:달/일/요일/시 정수, 1:연속 시간 특성
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]                                   # 변수(채널) 개수
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1    # 채널당 윈도우 개수

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ETT 시간 단위 데이터의 12/4/4 개월 경계 (1개월 = 30*24 시간)
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # few-shot 실험: 학습 구간을 percent만큼만 사용
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # 사용할 변수 선택
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]      # 'date' 제외 전체 변수
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]     # 타깃 변수 하나만

        # 표준화: 학습 구간 통계로 fit한 뒤 전체에 적용 (정보 누수 방지)
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 시간 특성(time stamp) 인코딩
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # 전역 index를 (채널 id, 윈도우 시작 위치)로 분해 -> 채널 독립 처리
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len               # 입력 끝
        r_begin = s_end                              # 예측 구간 시작(입력 바로 다음)
        r_end = s_end + self.pred_len                # 예측 끝

        # 해당 채널(feat_id)의 슬라이스만 추출 -> 단변량 (길이, 1)
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # 전체 샘플 수 = 채널당 윈도우 수 x 채널 수
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_hanwoo(Dataset):
    """
    한 개체(행) = 한 샘플로 취급한다.
      - 시계열 입력: 개체가 겪은 날씨 (동일 stn, weather_mode/months로 구간 선택)
      - 프롬프트 입력: 개체/지역/농가/혈통/과거통계를 Time-LLM 스타일 자연어로 서술
      - y          : LAST_GRADE (추후 육량등급/육질등급으로 분리해 학습)
    """

    # 입력(프롬프트)으로 사용할 테이블 컬럼 (whitelist, 프롬프트 서술 순서와 동일)
    X_COLS = [
        # 1) 개체 기본
        'JUDGE_SEX', 'AGE', 'WEIGHT', 'BIRTH_YMD', 'BIRTH_SEASON', 'ABATT_DATE', 'ABATT_SEASON',
        # 2) 지역/환경
        'sido', 'stn',
        # 3) 농가 사육환경
        'FARM_AVG_HANWOO_COUNT', 'HANWOO_DENSITY', 'FARM_DEATH_AVG_COUNT',
        # 4) 혈통/번식 (count 뒤에 imputed)
        'FATHER_PAST_OFFSPRING_COUNT', 'FATHER_OFFSPRING_IMPUTED',
        'MOTHER_PAST_OFFSPRING_COUNT', 'MOTHER_OFFSPRING_IMPUTED',
        # 5) 과거 도체 통계
        'PAST_WEIGHT_MEAN', 'PAST_BACKFAT_MEAN', 'PAST_REA_MEAN', 'PAST_WINDEX_MEAN',
        # 6) 과거 등급 분포
        'PAST_WGRADE_A_RATIO', 'PAST_WGRADE_B_RATIO', 'PAST_WGRADE_C_RATIO',
        'PAST_QUALITY_GRADE_1PP_RATIO', 'PAST_QUALITY_GRADE_1P_RATIO',
        'PAST_QUALITY_GRADE_1_RATIO', 'PAST_QUALITY_GRADE_2_RATIO', 'PAST_QUALITY_GRADE_3_RATIO',
    ]

    # y 컬럼: LAST_GRADE만 (추후 1++A -> 육량 'A'(3클래스) + 육질 '1++'(5클래스)로 분리)
    Y_COLS = ['LAST_GRADE']

    # 시계열로 사용할 날씨 변수 (순서 고정, ws_davg는 시계열에서 제외하고 프롬프트 통계에만 사용)
    WEATHER_VARS = ['ta_max', 'ta_min', 'rn_day', 'rhm_avg', 'THI']

    # 프롬프트 통계용 원시 변수 (mean/variance 계산). 바람(ws_davg)은 사용 안 함.
    STAT_VARS = ['ta_max', 'ta_min', 'rn_day', 'rhm_avg']

    WEATHER_MISSING = -99      # 날씨 결측 코드 (통계 전 NaN으로 치환)

    # y(LAST_GRADE) 순서형 정수 매핑 (one-hot 금지)
    YIELD_MAP = {'A': 0, 'B': 1, 'C': 2}                                # 육량 3클래스
    QUALITY_MAP = {'1++': 0, '1+': 1, '1': 2, '2': 3, '3': 4}           # 육질 5클래스
    # LAST_GRADE 드롭 토큰: 등외(OUT)와 결측. (등외는 추후 별도 'is_out' head로 다룰 예정 — 미구현)
    DROP_TOKENS = {'등외', 'MISSING', '-99', 'nan', 'NaN', 'None', ''}
    # 입력 피처 결측 판별 토큰 (층화 split의 결측 보유 여부 계산용)
    MISSING_TOKENS = {'MISSING', '-99', '-99.0', 'nan', 'NaN', 'None', ''}

    # 시·도 한글 -> 영어 (프롬프트는 영어 서술)
    SIDO_EN = {
        '서울특별시': 'Seoul', '부산광역시': 'Busan', '대구광역시': 'Daegu', '인천광역시': 'Incheon',
        '광주광역시': 'Gwangju', '대전광역시': 'Daejeon', '울산광역시': 'Ulsan', '세종특별자치시': 'Sejong',
        '경기도': 'Gyeonggi-do', '강원특별자치도': 'Gangwon', '강원도': 'Gangwon-do',
        '충청북도': 'Chungcheongbuk-do', '충청남도': 'Chungcheongnam-do',
        '전북특별자치도': 'Jeonbuk', '전라북도': 'Jeollabuk-do', '전라남도': 'Jeollanam-do',
        '경상북도': 'Gyeongsangbuk-do', '경상남도': 'Gyeongsangnam-do', '제주특별자치도': 'Jeju',
    }
    SEASON_EN = {'봄': 'spring', '여름': 'summer', '가을': 'autumn', '겨울': 'winter'}
    SEX_EN = {'암': 'female', '거세': 'castrated (steer)', '수': 'male (bull)'}

    def __init__(self, root_path='./../data/', table_path='hanwoo_train_merged',
                 test_table_path='hanwoo_test_merged', weather_path='hanwoo_weather_imputed',
                 flag='train', file_type='csv', weather_interval='week',
                 weather_mode='abatt', weather_months=12, val_ratio=0.3, seed=2021):
        assert flag in ['train', 'test', 'val']
        assert weather_interval in ['day', 'week', 'month']
        assert weather_mode in ['abatt', 'birth', 'full']
        self.flag = flag
        self.root_path = root_path
        self.table_path = table_path               # train/val 테이블
        self.test_table_path = test_table_path     # test 테이블(라벨 없음, 순서 보존)
        self.weather_path = weather_path
        self.file_type = file_type                 # 'csv' 또는 'par'(parquet)
        self.weather_interval = weather_interval   # day/week/month
        self.weather_mode = weather_mode           # abatt/birth/full
        self.weather_months = weather_months       # 윈도우 길이(개월), full이면 무시
        self.val_ratio = val_ratio                 # train/val = (1-val_ratio)/val_ratio
        self.seed = seed
        self.__read_data__()

    # ----------------- 파일 읽기 (file_type으로 csv/parquet만 허용) -----------------
    def _read_any(self, name):
        if self.file_type == 'par':
            return pd.read_parquet(os.path.join(self.root_path, name + '.parquet'))
        elif self.file_type == 'csv':
            return pd.read_csv(os.path.join(self.root_path, name + '.csv'))
        raise ValueError("file_type은 'csv' 또는 'par'만 가능합니다: {}".format(self.file_type))

    def __read_data__(self):
        # 0) 날씨 로드 (train/val/test 공통)
        self._load_weather()

        # 1) test: 테이블 통째로(드롭/셔플/정렬 없이 순서 보존), 라벨 없음
        if self.flag == 'test':
            df = self._read_any(self.test_table_path)
            self.x_cols = [c for c in self.X_COLS if c in df.columns]
            self.y_cols = []
            self.prompt_cols = self.x_cols
            self.df = df[self.x_cols].reset_index(drop=True)
            n = len(self.df)
            self.y_yield = np.full(n, -1, dtype=np.int64)      # placeholder
            self.y_quality = np.full(n, -1, dtype=np.int64)
            self.split_idx = np.arange(n)                       # 원본 순서 그대로
            self.class_counts = None
            return

        # 2) train/val: 테이블 로드 + 라벨 파싱 + 등외/결측 드롭
        df = self._read_any(self.table_path)
        self.x_cols = [c for c in self.X_COLS if c in df.columns]
        self.y_cols = [c for c in self.Y_COLS if c in df.columns]
        self.prompt_cols = self.x_cols
        df = df[self.x_cols + self.y_cols].reset_index(drop=True)

        parsed = df['LAST_GRADE'].map(self._parse_last_grade)   # (y_yield, y_quality) 또는 None
        keep = parsed.notna().values
        self.df = df[keep].reset_index(drop=True)
        parsed = parsed[keep].reset_index(drop=True)
        self.y_yield = np.array([p[0] for p in parsed], dtype=np.int64)
        self.y_quality = np.array([p[1] for p in parsed], dtype=np.int64)

        # 3) 층화 train/val 분할: (육량,육질) + 입력 결측 보유 여부를 층화 키로
        has_missing = self._row_has_missing(self.df)
        strat = np.array(['{}_{}_{}'.format(y, q, int(m))
                          for y, q, m in zip(self.y_yield, self.y_quality, has_missing)])
        idx_all = np.arange(len(self.df))
        tr_idx, va_idx = self._stratified_split(idx_all, strat, self.val_ratio, self.seed)
        # self.split_idx = tr_idx if self.flag == 'train' else va_idx
        self.split_idx = tr_idx[:100] if self.flag == 'train' else va_idx[:100]
        
        self.split_sizes = {'train': len(tr_idx), 'val': len(va_idx)}

        # 클래스 빈도는 항상 train 부분 기준 (손실 가중치 산출용)
        self.class_counts = {
            'yield':   np.bincount(self.y_yield[tr_idx],   minlength=len(self.YIELD_MAP)),
            'quality': np.bincount(self.y_quality[tr_idx], minlength=len(self.QUALITY_MAP)),
        }

    # ----------------- 날씨 로드 + 결측 보간 (-99->NaN -> 스플라인) + stn 인덱싱 -----------------
    def _load_weather(self):
        w = self._read_any(self.weather_path)
        base = ['ta_max', 'rn_day', 'ta_min', 'rhm_avg', 'ws_davg']
        for c in base + ['THI']:
            if c in w.columns:
                w[c] = w[c].replace(self.WEATHER_MISSING, np.nan)
        if 'THI' not in w.columns:
            # THI = (1.8*T+32) - [(0.55 - 0.0055*H) * (1.8*T - 26.8)], T=ta_max, H=rhm_avg
            w['THI'] = (1.8 * w['ta_max'] + 32) - ((0.55 - 0.0055 * w['rhm_avg']) * (1.8 * w['ta_max'] - 26.8))
        w['date'] = pd.to_datetime(w['date'].astype(str), format='%Y%m%d')
        w = w.sort_values(['stn', 'date']).reset_index(drop=True)

        # 남은 결측(거의 없음)만 stn별 스플라인 보간 (없으면 통째로 skip)
        icols = [c for c in base + ['THI'] if c in w.columns]
        if w[icols].isna().any().any():
            w = w.groupby('stn', group_keys=False).apply(lambda g: self._interp_group(g, icols))

        self.weather_by_stn = {stn: g for stn, g in w.groupby('stn')}
        self._empty_weather = w.iloc[0:0]          # stn 미존재 시 반환할 빈 윈도우

    @staticmethod
    def _interp_group(g, cols):
        # 스플라인(3차) 보간, 실패 시 선형. 양끝 등 보간 불가분은 ffill/bfill로 마감.
        try:
            g[cols] = g[cols].interpolate(method='spline', order=3, limit_direction='both')
        except Exception:
            g[cols] = g[cols].interpolate(method='linear', limit_direction='both')
        g[cols] = g[cols].ffill().bfill()
        return g

    # ----------------- 라벨 파싱: LAST_GRADE -> (육량, 육질) -----------------
    @classmethod
    def _parse_last_grade(cls, v):
        # 접미 1글자=육량(A/B/C), 나머지 접두=육질(1++/1+/1/2/3). 등외/결측/파싱불가는 None(드롭).
        s = str(v).strip()
        if s in cls.DROP_TOKENS:
            return None
        yld = cls.YIELD_MAP.get(s[-1:])
        qul = cls.QUALITY_MAP.get(s[:-1])
        if yld is None or qul is None:
            return None
        return (yld, qul)

    # ----------------- 행별 입력 결측 보유 여부 (층화용) -----------------
    def _row_has_missing(self, df):
        miss = np.zeros(len(df), dtype=bool)
        for c in self.x_cols:
            tok = df[c].astype(str).str.strip().isin(self.MISSING_TOKENS).values
            miss |= (df[c].isna().values | tok)
        return miss

    @staticmethod
    def _stratified_split(idx, strat, val_ratio, seed):
        # 라벨 층화 분할. 희귀 조합으로 층화 실패 시 비층화로 fallback.
        try:
            tr, va = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=strat)
        except ValueError:
            tr, va = train_test_split(idx, test_size=val_ratio, random_state=seed, shuffle=True)
        return np.sort(tr), np.sort(va)

    # ----------------- 손실용 클래스 가중치 (train 분포 기준) -----------------
    def class_weights(self, head, scheme='balanced', clip=10.0):
        # scheme='balanced': 역빈도 N/(K*n_c). clip으로 초희귀 클래스 과대가중 방지.
        # (effective-number 방식은 추후 추가 — 미구현)
        counts = self.class_counts[head].astype(float)
        K = len(counts)
        w = counts.sum() / (K * np.maximum(counts, 1.0))
        if clip:
            w = np.minimum(w, clip)
        return w

    # ----------------- 날씨 윈도우 선택 (weather_mode/months, 동일 stn) -----------------
    def _weather_window(self, row):
        stn = row['stn']
        if stn not in self.weather_by_stn:
            return self._empty_weather
        g = self.weather_by_stn[stn]
        if self.weather_mode == 'abatt':                   # 도축일 기준 직전 n개월
            end = pd.Timestamp(row['ABATT_DATE'])
            start = end - pd.DateOffset(months=self.weather_months)
        elif self.weather_mode == 'birth':                 # 출생일 기준 직후 n개월
            start = pd.Timestamp(row['BIRTH_YMD'])
            end = start + pd.DateOffset(months=self.weather_months)
        else:                                              # full: 출생~도축 전체
            start = pd.Timestamp(row['BIRTH_YMD'])
            end = pd.Timestamp(row['ABATT_DATE'])
        return g[(g['date'] >= start) & (g['date'] <= end)].sort_values('date')

    # ----------------- 날씨 시계열 집계 (day/week/month) -----------------
    def _aggregate_weather(self, win):
        # day: 일 단위 그대로 / week: 달력 주(W) 평균 / month: 달력 월(MS) 평균
        if len(win) == 0:
            return np.zeros((0, len(self.WEATHER_VARS)), dtype=float), []
        if self.weather_interval == 'day':
            ts = win[self.WEATHER_VARS].to_numpy(dtype=float)
            dates = win['date'].dt.strftime('%Y-%m-%d').tolist()
            return ts, dates
        rule = 'W' if self.weather_interval == 'week' else 'MS'
        g = win.set_index('date')[self.WEATHER_VARS].resample(rule).mean().dropna(how='all')
        return g.to_numpy(dtype=float), g.index.strftime('%Y-%m-%d').tolist()

    # ----------------- 프롬프트용 날씨 통계 (집계 전 원본 윈도우 기준) -----------------
    def _weather_stats(self, win):
        stats = {}
        for v in self.STAT_VARS:
            s = win[v].dropna() if (v in win.columns and len(win)) else None
            if s is not None and len(s):
                stats[v] = (round(float(s.mean()), 2), round(float(s.var()), 2))   # (mean, variance)
            else:
                stats[v] = (None, None)
        stats['heat_level'] = self._dominant_thi_band(win)
        stats['period_days'] = int((win['date'].max() - win['date'].min()).days + 1) if len(win) else 0
        return stats

    # THI를 4구간으로 나눠 가장 많이 등장한 구간 라벨을 반환 (comfortable/mild/moderate/severe)
    def _dominant_thi_band(self, win):
        if 'THI' not in win.columns or len(win) == 0:
            return 'MISSING'
        thi = win['THI'].dropna()
        if len(thi) == 0:
            return 'MISSING'
        counts = {
            'comfortable': int((thi < 72).sum()),
            'mild':        int(((thi >= 72) & (thi < 79)).sum()),
            'moderate':    int(((thi >= 79) & (thi < 89)).sum()),
            'severe':      int((thi >= 89).sum()),
        }
        return max(counts, key=counts.get)

    # ----------------- 포맷 헬퍼 -----------------
    @staticmethod
    def _num(v, nd=2):
        # 숫자면 소수 nd자리, 결측/문자면 'MISSING'
        try:
            f = float(v)
            return 'MISSING' if pd.isna(f) else '{:.{nd}f}'.format(f, nd=nd)
        except (TypeError, ValueError):
            return 'MISSING'

    @staticmethod
    def _pct(v):
        # 비율(0~1) -> 정수 % (토큰 절약). 결측이면 'MISSING'
        try:
            f = float(v)
            return 'MISSING' if pd.isna(f) else '{:.0f}%'.format(f * 100)
        except (TypeError, ValueError):
            return 'MISSING'

    @staticmethod
    def _pct_group(values):
        # 비율 묶음 -> "22%/56%/22%". 단, 묶음 전체가 결측이면 단일 'MISSING'으로 축약.
        pcts = [Dataset_hanwoo._pct(v) for v in values]
        if all(p == 'MISSING' for p in pcts):
            return 'MISSING'
        return '/'.join(pcts)

    # ----------------- Time-LLM 스타일 프롬프트 생성 -----------------
    def build_prompt(self, row, stats):
        N = self._num
        def sfmt(x):
            return 'MISSING' if (x is None or pd.isna(x)) else '{:.2f}'.format(x)
        gv = lambda c: row[c] if c in row.index else 'MISSING'   # 컬럼 자체가 없으면(test 등) MISSING

        sido_en = self.SIDO_EN.get(str(gv('sido')), str(gv('sido')))
        sex = self.SEX_EN.get(str(gv('JUDGE_SEX')), str(gv('JUDGE_SEX')))
        bseason = self.SEASON_EN.get(str(gv('BIRTH_SEASON')), str(gv('BIRTH_SEASON')))
        aseason = self.SEASON_EN.get(str(gv('ABATT_SEASON')), str(gv('ABATT_SEASON')))

        # 시계열 범위 옵션(weather_mode/interval)에 따라 문장이 바뀜
        unit = {'day': 'days', 'week': 'weeks', 'month': 'months'}[self.weather_interval]
        n = stats['n_steps']
        if self.weather_mode == 'abatt':
            period_sentence = "Weather series spans {} {} preceding slaughter.".format(n, unit)
        elif self.weather_mode == 'full':
            period_sentence = "Historical weather sequence covering {} {} from birth to slaughter.".format(n, unit)
        else:  # birth
            period_sentence = "Weather series during the first {} {} after birth.".format(n, unit)

        tmax, tmin, rn, rh = (stats['ta_max'], stats['ta_min'], stats['rn_day'], stats['rhm_avg'])

        prompt = (
            "[BEGIN DATA]\n"
            "[Domain] This is a Korean cattle (Hanwoo) carcass grading task. Weather—especially "
            "heat stress during the fattening and pre-slaughter period—affects carcass weight and "
            "marbling, which determine the beef yield grade and meat quality grade.\n"
            "[Instruction] Given the animal profile, farm/region context, historical priors, and the "
            "weather time series the animal experienced, predict (1) the yield grade (A/B/C) and "
            "(2) the meat quality grade (1++/1+/1/2/3). MISSING indicates missing or unavailable data.\n"
            "[Statistics]\n"
            "- Animal: sex {sex}, slaughter age {age} months, carcass weight {weight} kg; "
            "born {birth} ({bseason}), slaughtered {abatt} ({aseason}).\n"
            "- Region & weather: province {sido}. {period} "
            "max temperature mean {tmax_m} (var {tmax_v}); "
            "min temperature mean {tmin_m} (var {tmin_v}); "
            "precipitation mean {rn_m} (var {rn_v}) mm; "
            "humidity mean {rh_m}% (var {rh_v}). "
            "Most frequent heat-stress level: {heat}.\n"
            "- Farm: avg herd size {herd}, stocking density {density}, avg annual deaths {deaths}.\n"
            "- Lineage: sire past offspring {fcnt} (imputed {fimp}), "
            "dam past offspring {mcnt} (imputed {mimp}).\n"
            "- Historical carcass priors of similar cattle: weight {pw}, backfat {pb}, "
            "ribeye area {pr}, yield index {pwx}.\n"
            "- Historical grade distribution of similar cattle: yield A/B/C = {yield_dist}; "
            "quality 1++/1+/1/2/3 = {quality_dist}.\n"
            "[END DATA]\n"
        ).format(
            sex=sex, age=gv('AGE'), weight=gv('WEIGHT'),
            birth=gv('BIRTH_YMD'), bseason=bseason, abatt=gv('ABATT_DATE'), aseason=aseason,
            sido=sido_en, period=period_sentence,
            tmax_m=sfmt(tmax[0]), tmax_v=sfmt(tmax[1]),
            tmin_m=sfmt(tmin[0]), tmin_v=sfmt(tmin[1]),
            rn_m=sfmt(rn[0]), rn_v=sfmt(rn[1]),
            rh_m=sfmt(rh[0]), rh_v=sfmt(rh[1]),
            heat=stats['heat_level'],
            herd=N(gv('FARM_AVG_HANWOO_COUNT')), density=N(gv('HANWOO_DENSITY')),
            deaths=N(gv('FARM_DEATH_AVG_COUNT')),
            fcnt=gv('FATHER_PAST_OFFSPRING_COUNT'), fimp=gv('FATHER_OFFSPRING_IMPUTED'),
            mcnt=gv('MOTHER_PAST_OFFSPRING_COUNT'), mimp=gv('MOTHER_OFFSPRING_IMPUTED'),
            pw=N(gv('PAST_WEIGHT_MEAN')), pb=N(gv('PAST_BACKFAT_MEAN')),
            pr=N(gv('PAST_REA_MEAN')), pwx=N(gv('PAST_WINDEX_MEAN')),
            yield_dist=self._pct_group([gv('PAST_WGRADE_A_RATIO'), gv('PAST_WGRADE_B_RATIO'),
                                        gv('PAST_WGRADE_C_RATIO')]),
            quality_dist=self._pct_group([gv('PAST_QUALITY_GRADE_1PP_RATIO'), gv('PAST_QUALITY_GRADE_1P_RATIO'),
                                          gv('PAST_QUALITY_GRADE_1_RATIO'), gv('PAST_QUALITY_GRADE_2_RATIO'),
                                          gv('PAST_QUALITY_GRADE_3_RATIO')]),
        )
        return prompt

    # ----------------- 인덱스 -> 샘플(시계열 + 프롬프트 + y) -----------------
    def build_sample(self, idx):
        row = self.df.iloc[idx]
        win = self._weather_window(row)
        ts, dates = self._aggregate_weather(win)            # (T, 5) — weather_interval 적용
        if len(ts) == 0:
            print('[경고] 빈 날씨 윈도우 (idx={}, stn={})'.format(idx, row['stn']))
        stats = self._weather_stats(win)
        stats['n_steps'] = len(dates)                       # 집계 후 스텝 수(프롬프트 문장에 사용)
        prompt = self.build_prompt(row, stats)
        return {
            'index': int(idx),
            'stn': int(row['stn']),
            'birth': str(row['BIRTH_YMD']),
            'abatt': str(row['ABATT_DATE']),
            'weather_mode': self.weather_mode,
            'weather_months': self.weather_months,
            'weather_interval': self.weather_interval,
            'time_series': ts,                              # 모델 입력 시계열
            'time_series_dates': dates,
            'prompt': prompt,                               # Time-LLM 스타일 프롬프트 문자열
            'y': {c: row[c] for c in self.y_cols},          # {'LAST_GRADE': ...}
        }

    def __len__(self):
        return len(self.split_idx)

    def __getitem__(self, i):
        row_idx = int(self.split_idx[i])
        s = self.build_sample(row_idx)
        ts = torch.from_numpy(np.asarray(s['time_series'], dtype=np.float32))
        return {
            'time_series': ts,                              # FloatTensor (T, 5)
            'length': int(ts.shape[0]),
            'prompt': s['prompt'],                          # str (토크나이즈는 모델 내부)
            'y_yield': int(self.y_yield[row_idx]),          # 0~2 (test는 -1)
            'y_quality': int(self.y_quality[row_idx]),      # 0~4 (test는 -1)
            'index': row_idx,                               # 원본(현재 df) 행 위치
        }


def hanwoo_collate(batch):
    """가변 길이 시계열을 zero-padding + mask로 배치화. 프롬프트는 문자열 리스트 그대로."""
    lengths = [b['length'] for b in batch]
    Tmax = max(lengths) if lengths else 0
    C = batch[0]['time_series'].shape[1] if (batch and batch[0]['time_series'].ndim == 2) \
        else len(Dataset_hanwoo.WEATHER_VARS)
    B = len(batch)
    ts = torch.zeros(B, Tmax, C, dtype=torch.float32)
    mask = torch.zeros(B, Tmax, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = b['length']
        if L > 0:
            ts[i, :L] = b['time_series']
            mask[i, :L] = True
    return {
        'time_series': ts,                                  # (B, Tmax, 5)
        'mask': mask,                                       # (B, Tmax) 유효 스텝 True
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'prompt': [b['prompt'] for b in batch],             # list[str] (len B)
        'y_yield': torch.tensor([b['y_yield'] for b in batch], dtype=torch.long),
        'y_quality': torch.tensor([b['y_quality'] for b in batch], dtype=torch.long),
        'index': torch.tensor([b['index'] for b in batch], dtype=torch.long),
    }
