"""RevIN (Reversible Instance Normalization) 구현."""

import torch
import torch.nn as nn


class Normalize(nn.Module):
    """
    가역 인스턴스 정규화 (RevIN).

    시계열의 분포 이동(distribution shift)을 완화하기 위해
    입력을 인스턴스(샘플)별로 정규화('norm')하고, 예측 후 다시 원래 스케일로
    되돌린다('denorm'). 정규화에 사용한 평균/표준편차를 저장해 두었다가 역변환에 재사용한다.
    """

    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: 변수(채널) 개수
        :param eps: 수치 안정성을 위한 작은 값
        :param affine: True면 학습 가능한 affine 파라미터 사용
        :param subtract_last: True면 평균 대신 마지막 시점 값을 빼서 정규화
        :param non_norm: True면 정규화를 수행하지 않음(통과)
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)   # 통계 계산 후 정규화
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)  # 저장된 통계로 역정규화
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        # 시간 축에 대해 평균/표준편차 계산 (배치/채널은 유지)
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev   # 표준편차 복원
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean  # 평균 복원
        return x
