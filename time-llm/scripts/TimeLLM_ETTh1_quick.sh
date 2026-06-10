#!/bin/bash
# =====================================================================
# Time-LLM ETTh1 - 빠른 동작 확인용 설정
#
# LLaMA-7B의 앞쪽 6개 레이어만 사용(논문 Table 6의 축소 변형과 유사).
# fp32 로드 + accelerate bf16 autocast 조합으로 batch_size=24 기준 약 13.6GB 사용.
# gradient checkpointing을 쓰지 않으므로 빠르다(에폭당 ~20분, RTX A5500 기준).
#
# 단 3 에폭만 돌려도 ETTh1(pred_len=96) 테스트 MSE가 ~0.39까지 수렴하여
# 구현이 올바르게 동작함을 확인할 수 있다. (논문 32-layer/100epoch: MSE 0.362)
# =====================================================================

train_epochs=3
learning_rate=0.01
llama_layers=6

master_port=00097
num_process=1
batch_size=24
d_model=32
d_ff=128

accelerate launch --num_processes $num_process --main_process_port $master_port main.py \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model LLAMA \
  --llm_dim 4096 \
  --llm_layers $llama_layers \
  --llm_dtype fp32 \
  --train_epochs $train_epochs
