"""학습 보조 유틸리티: 학습률 조절, 조기 종료, 검증 루프, 프롬프트 로드 등."""

import numpy as np
import torch
import shutil
from tqdm import tqdm
import logging
from transformers import get_scheduler

def build_lr_scheduler(optimizer, train_steps, args):
    total_steps = train_steps * args.train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    if args.lradj == 'constant':
        return get_scheduler(name='constant', optimizer=optimizer)

    scheduler_name = {'linear_with_warmup': 'linear', 'cosine_with_warmup': 'cosine'}[args.lradj]
    
    return get_scheduler(name=scheduler_name, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

class TrainingLogger:
      def __init__(self, accelerator, log_path, name='hanwoo_training'):
          self.accelerator = accelerator
          self.logger = self.create_logger(log_path, name)
    
      def create_logger(self, log_path, name):
          if not self.accelerator.is_main_process:
              return None

          logger = logging.getLogger(name)
          logger.setLevel(logging.INFO)
          logger.propagate = False

          # 같은 프로세스에서 재생성할 때 handler 중복 방지
          logger.handlers.clear()

          formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

          file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
          file_handler.setFormatter(formatter)
          logger.addHandler(file_handler)

          return logger

      def log(self, message):
          self.accelerator.print(message)

          if self.logger is not None:
              self.logger.info(message)


class EarlyStopping:
    """검증 손실이 일정 횟수(patience) 동안 개선되지 않으면 학습을 조기 종료하고, 최적 모델을 저장한다."""

    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            # 첫 평가: 무조건 저장
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # 개선되지 않음 -> 카운터 증가
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 개선됨 -> 저장 후 카운터 리셋
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            msg = f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            if self.accelerator is not None:
                self.accelerator.print(msg)
            else:
                print(msg)
        # 학습 대상 파라미터만 저장 (동결 LLM 제외 권장이나 여기서는 전체 state_dict 저장)
        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """점(.) 표기법으로 접근 가능한 dict (예: args.seq_len)."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def del_files(dir_path):
    """디렉터리 전체 삭제."""
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    """검증/테스트 손실(MSE)과 MAE를 계산한다. (그래디언트 계산 없음)"""
    total_loss = []
    total_mae_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # 디코더 입력(모델은 인코더 전용이라 사용하지 않지만 시그니처 유지를 위해 0으로 전달)
            dec_inp = torch.zeros_like(batch_y).float().to(accelerator.device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # 분산 환경에서 모든 프로세스 결과를 모음
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pred = outputs.detach()
            true = batch_y.detach()

            total_loss.append(criterion(pred, true).item())
            total_mae_loss.append(mae_metric(pred, true).item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    model.train()
    return total_loss, total_mae_loss


def load_content(args):
    """데이터셋에 맞는 프롬프트 도메인 설명 텍스트를 dataset/prompt_bank에서 읽어온다."""
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content


# ============================ 한우 분류(2-head) 보조 ============================
import torch.nn as nn
import torch.nn.functional as F

# 육량/육질 정수 인덱스 -> 등급 문자열 (LAST_GRADE 복원용). Dataset_hanwoo.*_MAP의 역.
YIELD_INV = {0: 'A', 1: 'B', 2: 'C'}
QUALITY_INV = {0: '1++', 1: '1+', 2: '1', 3: '2', 4: '3'}


class FocalLoss(nn.Module):
    """다중분류 focal loss (선택적 클래스 가중치 weight 결합)."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction='none')
        p = torch.exp(-ce)                       # 정답 클래스 확률
        return ((1 - p) ** self.gamma * ce).mean()


def build_cls_criterion(loss_type, weight=None, gamma=2.0):
    """분류 손실 생성: ce / balanced_ce(weight) / focal."""
    if loss_type == 'focal':
        return FocalLoss(gamma=gamma, weight=weight)
    if loss_type == 'balanced_ce':
        return nn.CrossEntropyLoss(weight=weight)
    return nn.CrossEntropyLoss()


def combine_last_grade(yld_idx, qual_idx):
    """(육량idx, 육질idx) -> LAST_GRADE 문자열 리스트 (예: 0,0 -> '1++A')."""
    return [QUALITY_INV[int(q)] + YIELD_INV[int(y)] for y, q in zip(yld_idx, qual_idx)]


@torch.no_grad()
def vali_cls(accelerator, model, vali_loader, criterion_y, criterion_q):
    """분류 검증: 손실 + yield/quality/final 지표(micro/macro F1, QWK) 산출."""
    from sklearn.metrics import f1_score, cohen_kappa_score
    model.eval()
    losses, ys, ps_y, qs, ps_q = [], [], [], [], []
    for batch in tqdm(vali_loader):
        ts = batch['time_series'].float().to(accelerator.device)
        mask = batch['mask'].to(accelerator.device)
        yq = batch['y_yield'].to(accelerator.device)
        qq = batch['y_quality'].to(accelerator.device)
        logit_y, logit_q = model(ts, batch['prompt'], mask)
        loss = criterion_y(logit_y, yq) + criterion_q(logit_q, qq)
        losses.append(loss.item())
        ys.append(yq.cpu()); ps_y.append(logit_y.argmax(-1).cpu())
        qs.append(qq.cpu()); ps_q.append(logit_q.argmax(-1).cpu())
    model.train()

    ys = torch.cat(ys).numpy(); ps_y = torch.cat(ps_y).numpy()
    qs = torch.cat(qs).numpy(); ps_q = torch.cat(ps_q).numpy()

    def f1pair(t, p):
        return (f1_score(t, p, average='micro'), f1_score(t, p, average='macro'))

    true_final = combine_last_grade(ys, qs)
    pred_final = combine_last_grade(ps_y, ps_q)
    
    yield_f1_per_class = f1_score(ys, ps_y, labels=[0, 1, 2], average=None, zero_division=0)
    quality_f1_per_class = f1_score(qs, ps_q, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    
    return {
        'loss': float(np.mean(losses)),
        'yield': f1pair(ys, ps_y) + (cohen_kappa_score(ys, ps_y, weights='quadratic'),),
        'quality': f1pair(qs, ps_q) + (cohen_kappa_score(qs, ps_q, weights='quadratic'),),
        'final': f1pair(true_final, pred_final),
        'yield_per_class': {'A': yield_f1_per_class[0], 'B': yield_f1_per_class[1], 'C': yield_f1_per_class[2]},
      'quality_per_class': {'1++': quality_f1_per_class[0], '1+': quality_f1_per_class[1], '1': quality_f1_per_class[2], '2': quality_f1_per_class[3], '3': quality_f1_per_class[4]},
    }
