"""학습 보조 유틸리티: 학습률 조절, 조기 종료, 검증 루프, 프롬프트 로드 등."""

import numpy as np
import torch
import shutil
from tqdm import tqdm


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    """에폭에 따라 학습률을 조절한다. lradj 방식별로 스케줄이 다르다."""
    if args.lradj == 'type1':
        # 매 에폭마다 절반으로 감소
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'COS':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


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
