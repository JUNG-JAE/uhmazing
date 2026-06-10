#!/bin/bash
# =====================================================================
# Time-LLM ETTh1 장기 예측 - 전체 LLaMA-7B(32 레이어) 설정
#
# 논문 기본 백본인 LLaMA-7B 전체 32개 레이어를 사용한다.
# 단일 24GB GPU(예: RTX A5500)에서 동작하도록 다음 두 옵션을 사용:
#   --llm_dtype bf16             : 백본 가중치를 bf16으로 로드 (메모리 절반: ~14GB)
#   --gradient_checkpointing     : 역전파 시 활성값 재계산 (활성 메모리 대폭 절감)
# 위 설정에서 batch_size=24 기준 peak 메모리는 약 15.7GB.
#
# 주의: gradient checkpointing은 활성값을 재계산하므로 학습 속도가 느리다.
#       빠른 동작 확인은 scripts/TimeLLM_ETTh1_quick.sh 를 사용할 것.
# =====================================================================

# 전체 32-layer LLaMA는 후반 레이어의 큰 활성 outlier로 인해 높은 lr에서 발산하기 쉽다.
# 따라서 낮은 학습률 + gradient clipping으로 안정화한다.
train_epochs=10
learning_rate=0.0005
llama_layers=32
grad_clip=1.0

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
  --llm_dtype bf16 \
  --gradient_checkpointing \
  --grad_clip $grad_clip \
  --train_epochs $train_epochs
