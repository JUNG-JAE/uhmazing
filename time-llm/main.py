"""
Time-LLM 학습/평가 메인 스크립트.

accelerate를 사용해 단일/다중 GPU 학습을 모두 지원한다.
ETTh1 등 장기 예측(long-term forecast) 태스크를 대상으로 한다.

실행 예시(단일 GPU):
    accelerate launch --num_processes 1 main.py \
        --task_name long_term_forecast \
        --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 \
        --features M --seq_len 512 --pred_len 96 \
        --enc_in 7 --dec_in 7 --c_out 7 --d_model 32 --d_ff 128 \
        --batch_size 24 --learning_rate 0.01 --llm_layers 32 \
        --llm_dtype bf16 --gradient_checkpointing \
        --train_epochs 10
"""

import argparse
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import nn, optim
from tqdm import tqdm

from models import TimeLLM
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from utils.tools import vali, load_content, build_cls_criterion, vali_cls, TrainingLogger, build_lr_scheduler

parser = argparse.ArgumentParser(description='Time-LLM')

# 재현성을 위한 시드 고정
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# ---------------- 기본 설정 ----------------
parser.add_argument('--task_name', type=str, default='rollback', help='태스크 이름: [long_term_forecast, short_term_forecast]')
parser.add_argument('--task_type', type=str, default='classification', choices=['forecasting', 'classification'], help='forecasting: 기존 회귀 / classification: 한우 2-head 분류')
parser.add_argument('--des', type=str, default='Exp', help='실험 설명(체크포인트 폴더명에 사용)')
parser.add_argument('--seed', type=int, default=2021, help='랜덤 시드')

# ---------------- 데이터 로더 ----------------
parser.add_argument('--data', type=str, default='hanwoo', help='데이터셋 종류(Dataset 클래스 선택에 사용)')
parser.add_argument('--root_path', type=str, default='./../data/', help='데이터 루트 경로')
parser.add_argument('--data_path', type=str, default='hanwoo_train_merged', help='데이터 파일명')
parser.add_argument('--features', type=str, default='M', help='예측 형태: [M:다변량->다변량, S:단변량->단변량, MS:다변량->단변량]')
parser.add_argument('--target', type=str, default='OT', help='S/MS 태스크의 타깃 변수')
parser.add_argument('--freq', type=str, default='d', help='시간 특성 인코딩 주기: [s,t,h,d,b,w,m] (h=시간 단위)')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='체크포인트 저장 위치')
parser.add_argument('--file_type', type=str, default='par', help='csv: csv 파일로드, par: parquet 파일로드')
parser.add_argument('--weather_interval', type=str, default='day', help='day 하루단위, week 주단위 month 월 단위')
parser.add_argument('--weather_mode', type=str, default='abatt', choices=['abatt', 'birth', 'full'], help='시계열 윈도우 기준: abatt 도축일 기준 n개월, birth 출생일 기준 n개월, full 출생~도축 전체')
parser.add_argument('--weather_months', type=int, default=12, help='시계열 윈도우 길이(개월). weather_mode=full이면 무시')

# ---------------- 예측 태스크 ----------------
parser.add_argument('--seq_len', type=int, default=512, help='입력 시퀀스 길이') # 모델이 과거 몇스텝을 보고 예측할지 결정
parser.add_argument('--pred_len', type=int, default=96, help='예측 시퀀스 길이') # 모델이 몇스탭을 예측할지 결정, forecasting

# ---------------- 모델 정의 ----------------
parser.add_argument('--enc_in', type=int, default=5, help='입력 변수 개수(ETTh1=7)')
parser.add_argument('--dec_in', type=int, default=7, help='디코더 입력 크기') #
parser.add_argument('--c_out', type=int, default=7, help='출력 크기') #
parser.add_argument('--d_model', type=int, default=32, help='패치 임베딩 차원')
parser.add_argument('--n_heads', type=int, default=8, help='재프로그래밍 어텐션 헤드 수')
parser.add_argument('--d_ff', type=int, default=128, help='LLM 출력에서 사용할 특징 차원') # 
parser.add_argument('--moving_avg', type=int, default=12, help='이동평균 윈도우')
parser.add_argument('--factor', type=int, default=3, help='어텐션 factor')
parser.add_argument('--dropout', type=float, default=0.1, help='드롭아웃 비율')
parser.add_argument('--embed', type=str, default='timeF', help='시간 특성 인코딩 방식')
parser.add_argument('--activation', type=str, default='gelu', help='활성화 함수')
parser.add_argument('--output_attention', action='store_true', help='어텐션 출력 여부')
parser.add_argument('--patch_len', type=int, default=16, help='패치 길이')
parser.add_argument('--stride', type=int, default=8, help='패치 stride')
parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM 백본: [LLAMA, GPT2]')
parser.add_argument('--gpt2_size', type=str, default='large', choices=['small', 'medium', 'large', 'xl'], help='GPT2 백본 크기(llm_model=GPT2일 때): small(768)/medium(1024)/large(1280)/xl(1600)')
parser.add_argument('--llm_dim', type=int, default=4096, help='LLM hidden 차원. GPT2는 변형에 맞춰 자동 도출되므로 무시됨(LLaMA-7B:4096)')
parser.add_argument('--llm_layers', type=int, default=6, help='사용할 LLM 트랜스포머 레이어 수')
parser.add_argument('--llm_dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'], help='LLM 백본 로드 dtype. 전체 32-layer LLaMA를 단일 GPU에 올릴 땐 bf16 권장')
parser.add_argument('--gradient_checkpointing', action='store_true', default=False, help='LLM gradient checkpointing 사용(메모리 절약, 속도 다소 저하)')

# ---------------- 분류(classification) 전용 ----------------
parser.add_argument('--loss_type', type=str, default='balanced_ce', choices=['ce', 'balanced_ce', 'focal'], help='분류 손실: ce / balanced_ce(클래스가중) / focal')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal loss gamma')
parser.add_argument('--cls_hidden', type=int, default=256, help='분류 layer1 출력 차원')
parser.add_argument('--num_class_yield', type=int, default=3, help='육량 클래스 수(A/B/C)')
parser.add_argument('--num_class_quality', type=int, default=5, help='육질 클래스 수(1++/1+/1/2/3)')

# ---------------- 최적화 ----------------
parser.add_argument('--num_workers', type=int, default=4, help='데이터 로더 워커 수')
parser.add_argument('--itr', type=int, default=1, help='실험 반복 횟수')
parser.add_argument('--train_epochs', type=int, default=300, help='학습 에폭 수')
parser.add_argument('--batch_size', type=int, default=32, help='학습 배치 크기')
parser.add_argument('--eval_batch_size', type=int, default=8, help='평가 배치 크기')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='학습률')
parser.add_argument('--loss', type=str, default='MSE', help='손실 함수')
parser.add_argument('--lradj', type=str, default='constant', choices=['constant', 'linear_with_warmup', 'cosine_with_warmup'], help='학습률 스케줄 방식: constant / linear_with_warmup / cosine_with_warmup')
parser.add_argument('--warmup_ratio', type=float, default=0.1, help='전체 학습 step 중 warmup 비율(linear/cosine_with_warmup에서 사용)')
parser.add_argument('--use_amp', action='store_true', default=False, help='AMP 사용 여부')
parser.add_argument('--percent', type=int, default=100, help='학습 데이터 사용 비율(few-shot용)')
parser.add_argument('--grad_clip', type=float, default=0.0, help='gradient clipping max_norm (0이면 비활성). 전체 LLaMA-7B 학습 안정화에 권장(예: 1.0)')

args = parser.parse_args()

# 백본을 bf16/fp16으로 로드하면 가중치가 이미 저정밀이므로 accelerate autocast는 끈다('no').
# fp32 백본일 때만 bf16 autocast로 연산 효율을 높인다.
mixed_precision = 'bf16' if args.llm_dtype == 'fp32' else 'no'

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=mixed_precision)


for ii in range(args.itr):
    # 실험 설정 문자열(체크포인트 폴더명에 사용)
    # 테스크 종류, 데이터셋, LLM 백본, 변량, 입력길이, 예측길이, 패치 임베딩 차원, reprogramming 어텐션 수, LLM 출력 차원, 어텐션 factor, 실험설명, 실험 반복 수
    # setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_nh{}_df{}_fc{}_eb{}_{}_{}'.format(
    #     args.task_name, args.data, args.llm_model, args.features,
    #     args.seq_len, args.pred_len, args.d_model, args.n_heads,
    #     args.d_ff, args.factor, args.embed, args.des, ii)
    
    if args.llm_model == 'GPT2':
        setting = f'{args.data}_{args.llm_model}_{args.gpt2_size}_e{args.train_epochs}_{args.task_name}'
    elif args.llm_model == 'LLAMA':
        setting = f'{args.data}_{args.llm_model}_{args.llm_layers}_e{args.train_epochs}_{args.task_name}'

    # ============================ 분류(classification) 경로 ============================
    if args.task_type == 'classification':
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        accelerator.wait_for_everyone()
        logger = TrainingLogger(accelerator=accelerator, log_path=os.path.join(path, 'train.log'), name='hanwoo_training')
        
        # train/val 로더 (한우: dict 배치 + hanwoo_collate). test는 라벨 없어 main에선 미사용(추론은 predict_hanwoo.py).
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')

        model = TimeLLM.Model(args)
        if args.llm_dtype == 'fp32':
            model = model.float()

        # 학습 스케줄러
        trained_parameters = [p for p in model.parameters() if p.requires_grad]
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
        train_steps = len(train_loader)
        scheduler = build_lr_scheduler(model_optim, train_steps, args)

        # 클래스 가중치(train 분포 기준) -> head별 손실
        wy = torch.tensor(train_data.class_weights('yield'), dtype=torch.float)
        wq = torch.tensor(train_data.class_weights('quality'), dtype=torch.float)
        # 주의: hanwoo 배치는 dict + list[str]이므로 loader는 accelerate.prepare에서 제외(텐서는 수동 .to)
        model, model_optim, scheduler = accelerator.prepare(model, model_optim, scheduler)
        wy, wq = wy.to(accelerator.device), wq.to(accelerator.device)
        crit_y = build_cls_criterion(args.loss_type, weight=wy, gamma=args.focal_gamma)
        crit_q = build_cls_criterion(args.loss_type, weight=wq, gamma=args.focal_gamma)

        best_final_macro = -1.0
        for epoch in range(args.train_epochs):
            model.train(); train_loss = []; epoch_time = time.time()
            for batch in tqdm(train_loader):
                model_optim.zero_grad()
                ts = batch['time_series'].float().to(accelerator.device)
                mask = batch['mask'].to(accelerator.device)
                yq = batch['y_yield'].to(accelerator.device)
                qq = batch['y_quality'].to(accelerator.device)
                logit_y, logit_q = model(ts, batch['prompt'], mask)   # (B,3),(B,5)
                loss = crit_y(logit_y, yq) + crit_q(logit_q, qq)
                train_loss.append(loss.item())
                accelerator.backward(loss)
                if args.grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                model_optim.step()
                scheduler.step()

            train_loss = np.average(train_loss)
            res = vali_cls(accelerator, model, vali_loader, crit_y, crit_q)

            logger.log(f"Epoch {epoch + 1} ({time.time() - epoch_time:.1f}s) train {train_loss:.4f} | val {res['loss']:.4f}")
            logger.log(f"yield F1mi {res['yield'][0]:.2f} F1ma {res['yield'][1]:.2f} | quality F1mi {res['quality'][0]:.2f} F1ma {res['quality'][1]:.2f} QWK {res['quality'][2]:.2f} | last grade F1mi {res['final'][0]:.2f} F1ma {res['final'][1]:.2f}")
            logger.log(f"    yield class F1 | A: {res['yield_per_class']['A']:.2f} | B: {res['yield_per_class']['B']:.2f} | C: {res['yield_per_class']['C']:.2f}")
            logger.log(f"    quality class F1 | 1++: {res['quality_per_class']['1++']:.2f} | 1+: {res['quality_per_class']['1+']:.2f} | 1: {res['quality_per_class']['1']:.2f} | 2: {res['quality_per_class']['2']:.2f} | 3: {res['quality_per_class']['3']:.2f}")
            
            
            # best 체크포인트 기준 = 최종등급 macro-F1
            if res['final'][1] > best_final_macro:
                best_final_macro = res['final'][1]
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(path, 'checkpoint'))
                    logger.log("-> 체크포인트 저장 (final macro-F1={:.4f})".format(best_final_macro))
            logger.log(f" ")
                
        logger.log("분류 학습 완료. best final macro-F1 = {:.4f} | ckpt: {}".format(best_final_macro, path))
        continue

    # ============================ 예측(forecasting) 경로 (기존) ============================
    # 데이터 로더 준비 (train/val/test)
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # 모델 생성.
    # 주의: bf16/fp16 백본을 쓸 때 model.float()를 호출하면 동결 LLM까지 fp32로 되돌아가 메모리가 2배가 되어 OOM이 난다. 따라서 fp32 백본일 때만 .float()를 적용한다.
    model = TimeLLM.Model(args)
    if args.llm_dtype == 'fp32':
        model = model.float()

    path = os.path.join(args.checkpoints, setting)
    args.content = load_content(args)  # 프롬프트 도메인 설명 로드(항상 사용)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    train_steps = len(train_loader)
    best_vali_loss = np.inf

    # 학습 대상 파라미터만 옵티마이저에 등록 (동결된 LLM 파라미터 제외)
    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    # 학습률 스케줄러
    scheduler = build_lr_scheduler(model_optim, train_steps, args)

    criterion = nn.MSELoss()    # 학습 손실
    mae_metric = nn.L1Loss()    # 평가 지표(MAE)

    # accelerate로 분산/장치 배치 준비
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    for epoch in range(args.train_epochs):
        train_loss = []
        model.train()
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # 디코더 입력(모델은 인코더 전용이라 실제로 사용하지 않지만, 4-인자 시그니처 유지를 위해 0으로 채워 전달)
            dec_inp = torch.zeros_like(batch_y).float().to(accelerator.device)

            # 순방향
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            # 역전파 + (선택적) gradient clipping + 파라미터 갱신
            accelerator.backward(loss)
            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            model_optim.step()
            scheduler.step()
        
        train_loss = np.average(train_loss)

        # 검증셋만 매 epoch 평가한다 (best 체크포인트 선택 기준).
        # 테스트셋은 학습률/체크포인트 결정에 쓰이지 않으므로 학습 종료 후 1회만 평가한다.
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        accelerator.print("Epoch: {0} cost time: {1:.2f}s | Train Loss(MSE): {2:.4f} Vali Loss(MSE): {3:.4f} Vali MAE: {4:.4f}".format(epoch + 1, time.time() - epoch_time, train_loss, vali_loss, vali_mae_loss))

        # 검증 손실이 개선되면 체크포인트 저장 (조기 종료는 사용하지 않고 train_epochs까지 학습)
        if vali_loss < best_vali_loss:
            best_vali_loss = vali_loss
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(path, 'checkpoint'))
                accelerator.print("  -> 체크포인트 저장 (vali_loss={:.4f})".format(vali_loss))

    accelerator.wait_for_everyone()
    best_ckpt = os.path.join(path, 'checkpoint')
    if os.path.exists(best_ckpt):
        accelerator.unwrap_model(model).load_state_dict(
            torch.load(best_ckpt, map_location=accelerator.device))
    test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
    accelerator.print("테스트 (best 체크포인트) | Test Loss(MSE): {0:.4f} Test MAE: {1:.4f}".format(test_loss, test_mae_loss))

accelerator.wait_for_everyone()
accelerator.print('학습 완료. 최적 체크포인트는 다음 경로에 저장됨: {}'.format(path))
