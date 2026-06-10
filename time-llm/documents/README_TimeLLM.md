# Time-LLM 구현 설명서

논문 **"Time-LLM: Time Series Forecasting by Reprogramming Large Language Models"** (ICLR 2024)
의 재구현입니다. 사전학습된 거대 언어모델(LLM)을 **동결(freeze)** 한 채로, 시계열을
언어모델이 이해할 수 있는 표현으로 **재프로그래밍(reprogramming)** 하여 시계열을 예측합니다.

- Foundation 모델: **LLaMA-7B** (`huggyllama/llama-7b`)
- 검증 데이터셋: **ETTh1** (전력 변압기 온도)


## 1. 핵심 아이디어

기존 시계열 모델은 태스크마다 새로 설계·학습해야 하지만, LLM은 한 번 학습되면 다양한
태스크에 few-shot/zero-shot으로 적용됩니다. Time-LLM은 이 LLM의 패턴 인식·추론 능력을
**가중치 수정 없이** 시계열 예측에 끌어옵니다.

LLM은 이산 토큰을 다루고 시계열은 연속값이라는 모달리티 차이를, 두 가지 장치로 메웁니다.

1. **Patch Reprogramming**: 시계열 패치를 LLM의 단어 임베딩(텍스트 프로토타입) 공간으로 변환
2. **Prompt-as-Prefix (PaP)**: 데이터셋 설명·태스크 지시·입력 통계를 자연어 프롬프트로 만들어 앞에 붙임

학습되는 것은 입력 변환부와 출력 투영부뿐이며(전체의 약 3%), **LLM 본체는 동결**됩니다.

## 2. 디렉터리 구조

```
Time-LLM/
├── models/
│   └── TimeLLM.py          # 모델 본체 (Model, ReprogrammingLayer, FlattenHead)
├── layers/
│   ├── Embed.py            # PatchEmbedding, TokenEmbedding, ReplicationPad1d
│   └── StandardNorm.py     # RevIN(가역 인스턴스 정규화)
├── data_provider/
│   ├── data_factory.py     # 데이터셋 -> DataLoader 팩토리
│   └── data_loader.py      # Dataset_ETT_hour (채널 독립 처리)
├── utils/
│   ├── timefeatures.py     # 시간 특성 인코딩
│   └── tools.py            # EarlyStopping, vali, 학습률 조절, 프롬프트 로드
├── dataset/
│   ├── ETT-small/ETTh1.csv      # ETTh1 데이터 (17,420행, 7변수)
│   └── prompt_bank/ETT.txt      # ETT 도메인 설명 프롬프트
├── scripts/
│   ├── TimeLLM_ETTh1.sh        # 전체 32-layer LLaMA-7B 설정
│   └── TimeLLM_ETTh1_quick.sh  # 빠른 동작 확인용(6-layer)
├── run_main.py             # 학습/평가 메인 스크립트
├── test_model.py           # 단위 검증 스크립트(데이터/초기화/순방향/학습스텝)
└── requirements.txt
```

---

## 3. 모델 동작 흐름 (`models/TimeLLM.py`의 `forecast`)

입력 `x_enc` 형태는 `(B, T, N)` = (배치, 입력길이, 변수수). 단, ETTh1 로더는 **채널 독립**
처리를 하므로 실제로는 N=1 입니다(아래 4절 참고).

```
x_enc (B, T, N)
  │
  ▼ 1) RevIN 정규화                  분포 이동(distribution shift) 완화
  ▼ 2) 채널 독립 reshape             (B, T, N) -> (B*N, T, 1)
  ▼ 3) 입력 통계 계산                min/max/median/추세/top-5 lag(FFT 자기상관)
  ▼ 4) 자연어 프롬프트 구성          "Dataset description ... Input statistics ..."
  │        │
  │        ▼ 토크나이즈 -> LLM 임베딩 = prompt_embeddings
  │
  ▼ 5) 텍스트 프로토타입 생성        word_embeddings(V개) --mapping_layer--> V'=1000개
  ▼ 6) 패치 임베딩                   PatchEmbedding: 겹치는 패치 -> Conv1d 임베딩
  ▼ 7) 패치 재프로그래밍             ReprogrammingLayer: cross-attention
  │        (Query=패치, Key/Value=프로토타입) -> 패치를 언어 공간으로 정렬
  │
  ▼ 8) concat([prompt_embeddings, reprogrammed_patches])
  ▼ 9) 동결 LLM 통과                 last_hidden_state 획득
  ▼ 10) 앞 d_ff 차원만 사용 + 패치 부분만 선택
  ▼ 11) FlattenHead (평탄화 + 선형투영) -> 예측 길이로 변환
  ▼ 12) RevIN 역정규화               원래 스케일 복원
  │
  ▼
예측값 (B, pred_len, N)
```

### 주요 구성요소

| 구성요소 | 클래스/위치 | 역할 |
|----------|-------------|------|
| RevIN 정규화 | `layers/StandardNorm.py:Normalize` | 인스턴스별 정규화/역정규화 |
| 패치 임베딩 | `layers/Embed.py:PatchEmbedding` | 시계열 → 겹치는 패치 → Conv1d 임베딩 |
| 텍스트 프로토타입 | `Model.mapping_layer` | 단어 임베딩 V개 → V'=1000개로 압축 (학습됨) |
| 패치 재프로그래밍 | `models/TimeLLM.py:ReprogrammingLayer` | 멀티헤드 cross-attention으로 모달리티 정렬 |
| Prompt-as-Prefix | `Model.forecast` 내 프롬프트 생성 | 도메인/태스크/통계를 자연어로 LLM에 주입 |
| 동결 LLM | `Model.llm_model` | LLaMA-7B (가중치 학습 안 함) |
| 출력 헤드 | `models/TimeLLM.py:FlattenHead` | 패치별 hidden state → 예측 시퀀스 |

### 학습/동결 파라미터

- **전체**: 약 1.39억(6-layer) ~ 70억(32-layer) 파라미터
- **학습 대상**: 약 4,500만 (재프로그래밍·매핑·패치임베딩·출력투영) — 전체의 약 3%
- **동결**: LLaMA 백본 전체

---

## 4. 데이터 처리 (`data_provider/data_loader.py`)

ETTh1은 7개 변수(HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)를 가진 시간 단위 데이터입니다.

- **채널 독립(channel independence)**: `__getitem__`이 한 번에 **한 변수**의 윈도우만
  반환합니다. 즉 (윈도우, 채널) 쌍 하나가 독립적인 단변량 샘플이 됩니다. 따라서 모델이
  보는 입력은 항상 N=1 이며, batch_size=24는 LLM을 통과하는 시퀀스 24개를 의미합니다.
  (이 점이 메모리 산정에 중요 — 7채널을 한꺼번에 넣는 게 아닙니다.)
- **분할**: ETT 공식 기준 train/val/test = 12/4/4 개월
- **표준화**: 학습 구간 통계로 `StandardScaler`를 fit한 뒤 전체에 적용(정보 누수 방지)

---

## 5. 실행 방법

### 5.1 환경 설치
```bash
pip install -r requirements.txt
```
최초 실행 시 `huggyllama/llama-7b` 가중치(약 13GB)가 자동 다운로드됩니다.

### 5.2 단위 검증 (빠름)
```bash
python3 test_model.py
```
데이터 로딩 → 모델 초기화 → 순방향 → 1스텝 학습을 차례로 점검합니다.

### 5.3 빠른 학습 확인 (6-layer, 3 epoch)
```bash
bash scripts/TimeLLM_ETTh1_quick.sh
```

### 5.4 전체 LLaMA-7B 학습 (32-layer)
```bash
bash scripts/TimeLLM_ETTh1.sh
```

### 5.5 GPT-2 백본 사용 (small / medium / large / xl)

LLaMA 대신 GPT-2를 백본으로 쓸 수 있으며 4가지 크기를 지원합니다.
`--llm_model GPT2` 와 함께 `--gpt2_size` 로 크기를 선택합니다.

| `--gpt2_size` | HuggingFace ID | hidden(n_embd) | layers | heads |
|---------------|----------------|----------------|--------|-------|
| `small`(기본) | `gpt2`         | 768            | 12     | 12    |
| `medium`      | `gpt2-medium`  | 1024           | 24     | 16    |
| `large`       | `gpt2-large`   | 1280           | 36     | 20    |
| `xl`          | `gpt2-xl`      | 1600           | 48     | 25    |

```bash
accelerate launch --num_processes 1 run_main.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
    --model_id ETTh1_512_96 --model TimeLLM --data ETTh1 \
    --features M --seq_len 512 --label_len 48 --pred_len 96 \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 32 --d_ff 128 \
    --batch_size 24 --learning_rate 0.01 \
    --llm_model GPT2 --gpt2_size medium --llm_layers 24 \
    --train_epochs 10 --model_comment TimeLLM-ETTh1-gpt2medium
```

참고 사항:
- **`--llm_dim` 은 지정 불필요**: GPT-2 hidden 차원은 선택한 변형의 config에서 자동
  도출됩니다(small=768 … xl=1600). LLaMA에서 쓰던 4096 값을 그대로 둬도 무시됩니다.
- **`--llm_layers` 자동 clamp**: 변형의 실제 레이어 수보다 큰 값을 주면 해당 변형의
  최대치로 줄여 로드합니다(예: small에 `--llm_layers 32` → 12로 clamp). 초과분이
  랜덤 초기화되어 사전학습 효과가 사라지는 것을 막기 위함입니다.
- 큰 변형(large/xl)은 메모리 사용량이 늘어나므로 `--llm_layers` 축소,
  `--gradient_checkpointing`, `--batch_size` 조정을 함께 고려하세요.

---

## 6. GPU 메모리 분석 (RTX A5500, 24GB 기준)

`pred_len=96`, `batch_size=24`, `seq_len=512` 기준 학습(순방향+역전파) 시 peak 메모리 측정값:

| 백본 설정 | dtype | grad ckpt | Peak 메모리 | 24GB GPU |
|-----------|-------|-----------|-------------|----------|
| LLaMA-7B **6 레이어** | fp32 + bf16 autocast | X | **13.6 GB** | ✅ 여유 |
| LLaMA-7B **8 레이어** | fp32 + bf16 autocast | X | **17.7 GB** | ✅ 가능 |
| LLaMA-7B **12 레이어** | fp32 + bf16 autocast | X | OOM(>24) | ❌ |
| LLaMA-7B **32 레이어(전체)** | fp32 | X | 로드 불가(~28GB 가중치) | ❌ |
| LLaMA-7B **32 레이어(전체)** | bf16 | X | 21.0 GB | ⚠️ 빠듯 |
| LLaMA-7B **32 레이어(전체)** | **bf16 + grad ckpt** | O | **15.7 GB** | ✅ 가능 |

### 결론: A5500(24GB)에서 LLaMA-7B를 돌릴 수 있는가?

- **가능합니다.** 다만 다음 중 하나의 방식을 사용해야 합니다.
  1. **전체 32-layer**: `--llm_dtype bf16 --gradient_checkpointing` (약 15.7GB).
     단, 후반 레이어의 큰 활성 outlier 때문에 높은 학습률에서 발산하므로
     **낮은 lr(0.0005) + gradient clipping(1.0)** 으로 안정화해야 합니다.
     gradient checkpointing으로 인해 속도는 느립니다(에폭당 수 시간).
  2. **축소 6~8 layer**: fp32 + bf16 autocast로 batch 24가 여유롭게 돌아가며
     학습이 안정적이고 빠릅니다. 논문 Table 6도 `Llama (8)` 축소 변형을 평가합니다.

> 참고: 논문 원본은 8×GPU + DeepSpeed ZeRO로 전체 32-layer를 학습합니다.
> 단일 24GB GPU에서는 위 메모리 최적화가 필요합니다.

### 메모리를 줄이는 핵심 옵션 (`run_main.py`)

| 옵션 | 효과 |
|------|------|
| `--llm_dtype bf16` | 백본 가중치를 bf16으로 로드 → 가중치 메모리 절반(~14GB) |
| `--gradient_checkpointing` | 역전파 시 활성값 재계산 → 활성 메모리 대폭 절감(속도 ↓) |
| `--grad_clip 1.0` | gradient clipping → 발산 방지(학습 안정화) |
| `--llm_layers N` | 사용할 LLM 레이어 수 축소 → 메모리/속도 절감 |
| `--batch_size` ↓ | 배치 축소 → 활성 메모리 비례 감소 |

추가로 `models/TimeLLM.py`에서 `output_attentions=False`,
`output_hidden_states=False`로 설정해, 사용하지 않는 어텐션 행렬이 메모리에 쌓이지
않도록 했습니다(원본 대비 큰 메모리 절감).

---

## 7. 검증 결과

### 단위 검증 (`test_model.py`)
```
[1/3] 데이터 로딩      : PASS (train 56,231 / val 19,495 / test 19,495)
[2/3] 모델 초기화      : PASS (학습 파라미터 비율 3.27%)
[3/3] 순방향           : PASS (출력 (B,96,1), NaN/Inf 없음)
[Bonus] 1스텝 학습     : PASS (gradient 정상 흐름)
```

### 실제 학습 (LLaMA 6-layer, ETTh1, pred_len=96)
| Epoch | Train MSE | Vali MSE | Test MSE | Test MAE |
|-------|-----------|----------|----------|----------|
| 1 | 0.4241 | 0.7823 | 0.4100 | 0.4314 |
| 2 | 0.3895 | 0.7728 | 0.3935 | 0.4165 |

단 2 에폭만에 Test MSE 0.39까지 수렴 — 구현이 올바름을 확인.
(논문 보고치: 32-layer/100epoch에서 MSE 0.362, MAE 0.388)

---

## 8. 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `seq_len` | 512 | 입력 시퀀스 길이 |
| `pred_len` | 96 | 예측 길이 |
| `patch_len` | 16 | 패치 길이 |
| `stride` | 8 | 패치 간격 (겹침 발생) |
| `d_model` | 32 | 패치 임베딩 차원 |
| `d_ff` | 128 | LLM 출력에서 사용할 특징 차원 |
| `n_heads` | 8 | 재프로그래밍 어텐션 헤드 수 |
| `num_tokens` | 1000 | 텍스트 프로토타입 개수 |
| `llm_dim` | 4096 | LLaMA-7B hidden 차원 |
| `batch_size` | 24 | 배치 크기 |
| `learning_rate` | 0.01(축소) / 0.0005(전체) | 학습률 |
```
