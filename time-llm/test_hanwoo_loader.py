"""
Dataset_hanwoo 학습 파이프라인 검증 스크립트.

  ① 한 배치(collate 후) shape/mask/라벨 출력
  ② train/val 분할 크기 + head별 class_counts / class_weights 출력
  ③ time_series에 NaN 없음 assert
  ④ test flag 로드 시 행 순서/길이 보존 확인
실행: (time-llm 디렉토리에서)  python3 test_hanwoo_loader.py
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_loader import Dataset_hanwoo, hanwoo_collate

FILE_TYPE = 'par'
WEATHER_MODE = 'abatt'
WEATHER_MONTHS = 12
WEATHER_INTERVAL = 'day'


def main():
    print("=" * 72)
    print("  Dataset_hanwoo 파이프라인 검증 (mode=%s, months=%d, interval=%s)" % (WEATHER_MODE, WEATHER_MONTHS, WEATHER_INTERVAL))
    print("=" * 72)

    common = dict(root_path='./../data/', file_type=FILE_TYPE, weather_mode=WEATHER_MODE, weather_months=WEATHER_MONTHS, weather_interval=WEATHER_INTERVAL)

    train_set = Dataset_hanwoo(flag='train', **common)

    # ② 분할 크기 + 클래스 분포/가중치
    print("\n[1. 분할/클래스 분포]")
    print("  split sizes:", train_set.split_sizes, "| len(train_set):", len(train_set))
    print("  class_counts yield  :", train_set.class_counts['yield'].tolist())
    print("  class_counts quality:", train_set.class_counts['quality'].tolist())
    print("  class_weights yield  :", np.round(train_set.class_weights('yield'), 3).tolist())
    print("  class_weights quality:", np.round(train_set.class_weights('quality'), 3).tolist())

    # ① 한 배치 collate
    print("\n[2.배치 collate]")
    batch = hanwoo_collate([train_set[i] for i in range(4)])
    print("  time_series:", tuple(batch['time_series'].shape),
          "| mask:", tuple(batch['mask'].shape),
          "| lengths:", batch['lengths'].tolist())
    print("  y_yield:", batch['y_yield'].tolist(), "| y_quality:", batch['y_quality'].tolist())
    # print("  prompt[0] 첫 줄:", batch['prompt'][0].splitlines()[1])
    print("  prompt:", batch['prompt'][0])

    # ③ NaN 없음 assert
    assert not torch.isnan(batch['time_series']).any(), "time_series에 NaN 존재!"
    assert batch['y_yield'].min() >= 0 and batch['y_yield'].max() <= 2
    assert batch['y_quality'].min() >= 0 and batch['y_quality'].max() <= 4
    print("\n[③ assert] time_series NaN 없음 + 라벨 범위 정상 (yield 0-2, quality 0-4) ✓")

    # ④ test flag: 순서/길이 보존
    print("\n[④ test flag]")
    test_set = Dataset_hanwoo(flag='test', **common)
    print("  len(test_set):", len(test_set))
    first5 = [test_set[i]['index'] for i in range(5)]
    print("  앞 5개 index (원본 순서여야 함):", first5)
    assert first5 == [0, 1, 2, 3, 4], "test 행 순서가 보존되지 않음!"
    tb = hanwoo_collate([test_set[i] for i in range(4)])
    print("  test 배치 time_series:", tuple(tb['time_series'].shape),
          "| y_yield(placeholder):", tb['y_yield'].tolist())
    assert (tb['y_yield'] == -1).all(), "test 라벨은 -1 placeholder여야 함"
    print("  [PASS] test 순서 보존 + 라벨 placeholder(-1) ✓")

    print("\n" + "=" * 72)
    print("  ALL CHECKS PASSED")
    print("=" * 72)


if __name__ == '__main__':
    main()
