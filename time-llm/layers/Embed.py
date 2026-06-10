"""시계열 임베딩 관련 레이어 모음. Time-LLM에서는 PatchEmbedding이 핵심으로 사용된다."""

import torch
import torch.nn as nn
from torch import Tensor
import math


class PositionalEmbedding(nn.Module):
    """사인/코사인 기반 고정 위치 임베딩."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """1D 컨볼루션으로 값을 d_model 차원으로 임베딩. (패치 임베딩의 핵심 연산)"""

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # circular 패딩 + 커널3 Conv1d로 지역 패턴을 d_model 차원으로 인코딩
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # (B, L, c_in) -> (B, c_in, L) -> Conv1d -> (B, L, d_model)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TimeFeatureEmbedding(nn.Module):
    """연속 시간 특성(timeF)을 선형층으로 d_model 차원에 매핑."""

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class ReplicationPad1d(nn.Module):
    """시퀀스 끝 값을 복제하여 오른쪽에 패딩. (마지막 패치를 만들기 위함)"""

    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        # 마지막 시점 값을 padding 길이만큼 복제하여 뒤에 이어 붙인다.
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output


class PatchEmbedding(nn.Module):
    """
    시계열을 겹치는 패치로 분할한 뒤 각 패치를 d_model 차원으로 임베딩한다.

    동작:
      1) 끝값 복제 패딩(ReplicationPad1d)으로 마지막 패치 확보
      2) unfold로 길이 patch_len, 간격 stride의 패치들로 분할
      3) (배치*채널, 패치수, patch_len) 형태로 reshape
      4) TokenEmbedding(Conv1d)으로 각 패치를 d_model 차원으로 임베딩
    """

    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]                       # 채널 수 (Time-LLM에서는 보통 1)
        x = self.padding_patch_layer(x)           # 끝값 복제 패딩
        # 길이 patch_len, 간격 stride 패치로 분할: (..., 패치수, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # (B*채널, 패치수, patch_len)로 정리
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)               # 각 패치 -> d_model 임베딩
        return self.dropout(x), n_vars
