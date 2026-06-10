"""
Time-LLM 모델 구현 (논문: "Time-LLM: Time Series Forecasting by Reprogramming
Large Language Models", ICLR 2024).

전체 흐름
----------------
1. RevIN 정규화        : 각 채널을 평균0/분산1로 정규화 (분포 이동 완화)
2. 패치 분할 + 임베딩  : 시계열을 겹치는 패치로 자르고 Conv1d로 임베딩
3. 패치 재프로그래밍   : 패치 임베딩을 LLM 단어 임베딩 공간으로 cross-attention 매핑
4. Prompt-as-Prefix    : 데이터셋 설명/태스크/입력통계를 자연어 프롬프트로 만들어 앞에 붙임
5. 동결된 LLM 통과     : [프롬프트 임베딩 ; 재프로그래밍된 패치]를 LLM에 입력
6. 출력 투영 + 역정규화: LLM 마지막 hidden state를 평탄화·선형투영하여 예측값 생성
"""

from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize


transformers.logging.set_verbosity_error() # transformers의 로그를 줄임


# GPT-2 백본 변형(size) -> HuggingFace 모델 ID 매핑.
#   small  : n_embd=768,  n_layer=12, n_head=12  (논문 기본, 24GB GPU에 여유)
#   medium : n_embd=1024, n_layer=24, n_head=16
#   large  : n_embd=1280, n_layer=36, n_head=20
#   xl     : n_embd=1600, n_layer=48, n_head=25
# hidden 차원(n_embd)은 모델마다 다르므로 d_llm은 로드된 config에서 자동으로 도출한다.
GPT2_VARIANTS = {
    'small':  'openai-community/gpt2',
    'medium': 'openai-community/gpt2-medium',
    'large':  'openai-community/gpt2-large',
    'xl':     'openai-community/gpt2-xl',
}


class FlattenHead(nn.Module):
    """LLM 출력(패치별 hidden state)을 평탄화한 뒤 선형층으로 예측 길이만큼 투영하는 출력 헤드."""

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)   # 마지막 두 축(패치수 x 특징)을 하나로 펼침
        self.linear = nn.Linear(nf, target_window)  # 펼친 차원 -> 예측 길이로 투영
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ReprogrammingLayer(nn.Module):
    """
    패치 재프로그래밍 레이어 (논문 식 (1)).

    멀티헤드 cross-attention으로 동작한다.
      - Query  : 시계열 패치 임베딩 (target_embedding)
      - Key/Value : LLM 단어 임베딩에서 추출한 "텍스트 프로토타입" (source/value_embedding)
    즉, 각 시계열 패치를 텍스트 프로토타입들의 가중합으로 다시 표현하여
    시계열 모달리티를 언어 모달리티 공간에 정렬시킨다.
    """

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # 패치 -> Query
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)      # 프로토타입 -> Key
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)    # 프로토타입 -> Value
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)      # 결과 -> LLM 차원으로 복원
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape   # B:배치, L:패치 개수
        S, _ = source_embedding.shape      # S:프로토타입 개수
        H = self.n_heads

        # 각 입력을 (헤드 수 H)로 분할
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)        # 헤드 결합
        return self.out_projection(out)    # LLM 차원(d_llm)으로 투영

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)  # 스케일드 닷-프로덕트 어텐션의 스케일 계수

        # 패치(Query)와 프로토타입(Key) 사이의 어텐션 점수 계산
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        # softmax로 정규화한 어텐션 가중치
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # 가중치로 Value(프로토타입)를 가중합 -> 재프로그래밍된 패치 표현
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class Model(nn.Module):
    """Time-LLM 백본."""

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len     # 예측 길이 (예: 96)
        self.seq_len = configs.seq_len       # 입력 길이 (예: 512)
        self.d_ff = configs.d_ff             # LLM 출력에서 잘라 쓸 특징 차원
        self.top_k = 5                       # 입력 통계 중 자기상관 lag 상위 k개
        self.d_llm = configs.llm_dim         # 초기값. 백본 로드 후 config.hidden_size로 자동 갱신됨
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # 동결 백본 로드 dtype.
        #   - 'fp32' : 기본. 작은 백본(예: GPT-2, 소수 레이어 LLaMA)을 accelerate bf16 autocast와 함께 사용.
        #   - 'bf16' : 백본 가중치를 bf16으로 로드(메모리 절반). 전체 32-layer LLaMA-7B를 단일 GPU에 올릴 때 사용.
        #   - 'fp16' : bf16 미지원 환경용 대안.
        dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
        self.llm_dtype = dtype_map.get(getattr(configs, 'llm_dtype', 'fp32'), torch.float32)

        # ---------------------- 사전학습 LLM 백본 로드 ----------------------
        if configs.llm_model == 'LLAMA':
            self.llm_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llm_config.num_hidden_layers = configs.llm_layers  # 사용할 트랜스포머 레이어 수
            # forecast()는 last_hidden_state만 사용하므로 attention/hidden_states 출력을 끈다.
            # (켜면 모든 레이어의 어텐션 행렬이 메모리에 쌓여 단일 GPU에서 OOM 발생)
            self.llm_config.output_attentions = False
            self.llm_config.output_hidden_states = False
            self.llm_model = self._load_pretrained(
                LlamaModel, 'huggyllama/llama-7b', self.llm_config)
            self.tokenizer = self._load_tokenizer(LlamaTokenizer, 'huggyllama/llama-7b')

        elif configs.llm_model == 'GPT2':
            # 논문에서 사용한 더 작은 백본(GPT-2). small/medium/large/xl 4종 지원.
            gpt2_size = getattr(configs, 'gpt2_size', 'small')
            if gpt2_size not in GPT2_VARIANTS:
                raise Exception('정의되지 않은 GPT-2 변형입니다: {} (가능: {})'.format(
                    gpt2_size, list(GPT2_VARIANTS.keys())))
            gpt2_name = GPT2_VARIANTS[gpt2_size]

            self.llm_config = GPT2Config.from_pretrained(gpt2_name)
            # 요청한 레이어 수가 해당 변형의 실제 레이어 수를 넘으면 clamp한다.
            # (넘으면 초과분이 랜덤 초기화되어 사전학습 효과가 사라짐)
            max_layers = self.llm_config.num_hidden_layers  # 변형의 원래 레이어 수
            self.llm_config.num_hidden_layers = min(configs.llm_layers, max_layers)
            self.llm_config.output_attentions = False
            self.llm_config.output_hidden_states = False
            self.llm_model = self._load_pretrained(GPT2Model, gpt2_name, self.llm_config)
            self.tokenizer = self._load_tokenizer(GPT2Tokenizer, gpt2_name)

        else:
            raise Exception('정의되지 않은 LLM 백본입니다: {}'.format(configs.llm_model))

        # hidden 차원(d_llm)은 로드된 백본 config에서 자동 도출한다.
        # GPT-2 변형마다 n_embd가 다르고(768/1024/1280/1600), 이렇게 하면
        # 사용자가 --llm_dim을 변형에 맞춰 수동으로 지정하지 않아도 항상 일치한다.
        # (GPT2Config.hidden_size는 n_embd, LlamaConfig.hidden_size는 4096으로 매핑됨)
        self.d_llm = self.llm_config.hidden_size

        # 패딩 토큰 설정 (없으면 eos 토큰 또는 새 [PAD] 토큰 사용)
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # 핵심: LLM 백본의 모든 파라미터를 동결 (학습하지 않음)
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # Gradient checkpointing: 역전파 시 중간 활성값을 저장하지 않고 재계산하여 메모리를 크게 절약.
        # (속도는 다소 느려지지만, 전체 32-layer LLaMA-7B를 단일 24GB GPU에 올릴 수 있게 해준다.)
        # 입력 임베딩(재프로그래밍 결과)이 grad를 요구하므로 동결 백본이어도 정상 동작한다.
        if getattr(configs, 'gradient_checkpointing', False):
            self.llm_model.gradient_checkpointing_enable()

        # 프롬프트에 들어갈 데이터셋 설명 (도메인 지식). 항상 포함한다.
        # configs.content(= prompt_bank에서 로드한 도메인 설명)가 있으면 사용하고,
        # 없으면 ETT 기본 설명으로 대체한다.
        self.description = getattr(configs, 'content', None) or (
            'The Electricity Transformer Temperature (ETT) is a crucial '
            'indicator in the electric power long-term deployment.')

        self.dropout = nn.Dropout(configs.dropout)

        # ---------------------- 학습되는 모듈들 ----------------------
        # 1) 패치 임베딩: 시계열 패치 -> d_model 차원 임베딩
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        # 2) 텍스트 프로토타입: LLM의 거대한 단어 임베딩(vocab_size개)을 num_tokens개로 압축
        #    word_embeddings 자체는 동결된 LLM의 것이지만, mapping_layer는 학습된다.
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000  # 프로토타입 개수 V' (V' << V)
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # 3) 패치 재프로그래밍 레이어
        self.reprogramming_layer = ReprogrammingLayer(
            configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # 패치 개수 계산: (입력길이 - 패치길이)/stride + 2  (+1은 마지막 패딩 패치)
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums  # 출력 헤드 입력 차원

        # 4) 출력 투영 헤드
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(
                configs.enc_in, self.head_nf, self.pred_len, head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        # 5) RevIN 정규화 레이어 (affine=False: 학습 파라미터 없는 단순 정규화)
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    # ----------------- LLM/토크나이저 로드 헬퍼 -----------------
    def _load_pretrained(self, model_cls, name, config):
        """로컬 캐시 우선, 없으면 다운로드. 동결 백본이므로 fp16으로 로드해 메모리 절약."""
        kwargs = dict(trust_remote_code=True, config=config, torch_dtype=self.llm_dtype)
        try:
            return model_cls.from_pretrained(name, local_files_only=True, **kwargs)
        except (EnvironmentError, OSError):
            print("로컬 모델 파일을 찾지 못했습니다. 다운로드를 시도합니다...")
            return model_cls.from_pretrained(name, local_files_only=False, **kwargs)

    def _load_tokenizer(self, tok_cls, name):
        try:
            return tok_cls.from_pretrained(name, trust_remote_code=True, local_files_only=True)
        except (EnvironmentError, OSError):
            print("로컬 토크나이저 파일을 찾지 못했습니다. 다운로드를 시도합니다...")
            return tok_cls.from_pretrained(name, trust_remote_code=True, local_files_only=False)

    # ----------------- 순방향 -----------------
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # 마지막 pred_len 스텝만 반환
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # ---- 1) RevIN 정규화 ----
        x_enc = self.normalize_layers(x_enc, 'norm')

        # ---- 채널 독립 처리: (B, T, N) -> (B*N, T, 1) ----
        # 각 변수(채널)를 독립적인 단변량 시계열로 취급한다.
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # ---- 2) 입력 통계 계산 (Prompt-as-Prefix 용) ----
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)            # 자기상관 기반 상위 lag
        trends = x_enc.diff(dim=1).sum(dim=1)      # 전체 추세 방향(증가/감소)

        # ---- 3) 자연어 프롬프트 구성 (채널마다 하나씩) ----
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous "
                f"{str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        # 통계 계산이 끝났으니 원래 형태로 복원
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # ---- 4) 프롬프트 토크나이즈 -> LLM 임베딩 ----
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True,
                                truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        # ---- 5) 텍스트 프로토타입 생성 ----
        # word_embeddings(동결, fp16) -> float로 올린 뒤 mapping_layer로 압축
        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0).float()).permute(1, 0)

        # ---- 6) 패치 임베딩 + 재프로그래밍 ----
        x_enc = x_enc.permute(0, 2, 1).contiguous()           # (B*N, 1, T)
        enc_out, n_vars = self.patch_embedding(x_enc)         # (B*N, 패치수, d_model)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # ---- 7) [프롬프트 임베딩 ; 재프로그래밍된 패치]를 이어붙여 LLM 통과 ----
        # 두 텐서의 dtype을 백본 임베딩과 동일하게 맞춰 concat 오류를 방지한다.
        # (autocast 환경에서 한쪽은 bf16, 다른 쪽은 fp32가 되는 문제를 막아준다.)
        cat_dtype = self.llm_model.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(cat_dtype)
        enc_out = enc_out.to(cat_dtype)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff].float()  # 앞 d_ff 차원만 사용, 이후 연산은 fp32

        # ---- 8) 채널 축 복원 후 출력 투영 ----
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        # 패치 부분(뒤쪽 patch_nums개)만 골라 예측으로 투영 (프롬프트 부분은 버림)
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # ---- 9) RevIN 역정규화 ----
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out

    def calcute_lags(self, x_enc):
        """FFT 기반 자기상관(autocorrelation)으로 상위 top_k lag를 구한다."""
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)           # 파워 스펙트럼
        corr = torch.fft.irfft(res, dim=-1)       # 역FFT -> 자기상관
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
