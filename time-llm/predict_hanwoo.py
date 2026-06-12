"""
한우 등급 추론/제출 스크립트 (분류 2-head).

  - test셋(hanwoo_test_merged, 라벨 없음)을 원본 순서 그대로 로드
  - 학습된 best 체크포인트로 (육량, 육질) 예측 -> LAST_GRADE 문자열 복원
  - 원본 행 순서대로 CSV 저장

실행 예시:
    python3 predict_hanwoo.py --checkpoint ./checkpoints/<setting>/checkpoint \
        --root_path ./../data/ --file_type par --weather_mode abatt \
        --weather_months 12 --weather_interval week --out submission.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import TimeLLM
from data_provider.data_loader import Dataset_hanwoo, hanwoo_collate
from utils.tools import combine_last_grade


def get_args():
    p = argparse.ArgumentParser(description='Hanwoo 추론')
    # --- 체크포인트/IO ---
    p.add_argument('--checkpoint', type=str, required=True, help='학습된 checkpoint 파일 경로')
    p.add_argument('--out', type=str, default='submission.csv', help='예측 결과 CSV 경로')
    # --- 데이터 ---
    p.add_argument('--root_path', type=str, default='./../data/')
    p.add_argument('--file_type', type=str, default='par', choices=['csv', 'par'])
    p.add_argument('--weather_mode', type=str, default='abatt', choices=['abatt', 'birth', 'full'])
    p.add_argument('--weather_months', type=int, default=12)
    p.add_argument('--weather_interval', type=str, default='week', choices=['day', 'week', 'month'])
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    # --- 모델 (학습 때와 동일하게 지정) ---
    p.add_argument('--task_name', type=str, default='long_term_forecast')
    p.add_argument('--task_type', type=str, default='classification')
    p.add_argument('--enc_in', type=int, default=5)
    p.add_argument('--d_model', type=int, default=32)
    p.add_argument('--d_ff', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--patch_len', type=int, default=16)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--cls_hidden', type=int, default=256)
    p.add_argument('--num_class_yield', type=int, default=3)
    p.add_argument('--num_class_quality', type=int, default=5)
    p.add_argument('--llm_model', type=str, default='GPT2')
    p.add_argument('--gpt2_size', type=str, default='small')
    p.add_argument('--llm_dim', type=int, default=4096)
    p.add_argument('--llm_layers', type=int, default=6)
    p.add_argument('--llm_dtype', type=str, default='fp32')
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test셋: 순서 보존, 라벨 없음
    test_set = Dataset_hanwoo(
        root_path=args.root_path, flag='test', file_type=args.file_type,
        weather_mode=args.weather_mode, weather_months=args.weather_months,
        weather_interval=args.weather_interval)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=args.num_workers, collate_fn=hanwoo_collate)
    print('test 행 수:', len(test_set))

    model = TimeLLM.Model(args)
    if args.llm_dtype == 'fp32':
        model = model.float()
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state)
    model.to(device).eval()

    idxs, yld, qul = [], [], []
    with torch.no_grad():
        for batch in loader:
            ts = batch['time_series'].float().to(device)
            mask = batch['mask'].to(device)
            logit_y, logit_q = model(ts, batch['prompt'], mask)
            yld.append(logit_y.argmax(-1).cpu().numpy())
            qul.append(logit_q.argmax(-1).cpu().numpy())
            idxs.append(batch['index'].numpy())

    idxs = np.concatenate(idxs)
    yld = np.concatenate(yld)
    qul = np.concatenate(qul)
    last_grade = combine_last_grade(yld, qul)   # 예: '1++A'  (등외는 드롭했으므로 예측에 안 나옴)

    # 원본 행 순서대로 정렬해 저장
    df = pd.DataFrame({'index': idxs, 'LAST_GRADE': last_grade}).sort_values('index').reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print('저장 완료:', args.out, '| 행 수:', len(df))


if __name__ == '__main__':
    main()
