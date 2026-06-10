import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
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
