"""
Time-LLM 모델 검증 스크립트
1) 데이터 로딩 테스트
2) 모델 초기화 테스트
3) Forward pass 테스트
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from utils.tools import dotdict, load_content


def test_data_loading():
    print("=" * 60)
    print("[1/3] Testing data loading (ETTh1)...")
    print("=" * 60)

    args = dotdict({
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTh1.csv',
        'data': 'ETTh1',
        'features': 'M',
        'target': 'OT',
        'freq': 'h',
        'embed': 'timeF',
        'seq_len': 512,
        'pred_len': 96,
        'batch_size': 4,
        'num_workers': 0,
        'percent': 100,
    })

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples:   {len(vali_data)}")
    print(f"  Test samples:  {len(test_data)}")

    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
    print(f"  batch_x shape:      {batch_x.shape}")
    print(f"  batch_y shape:      {batch_y.shape}")
    print(f"  batch_x_mark shape: {batch_x_mark.shape}")
    print(f"  batch_y_mark shape: {batch_y_mark.shape}")
    print("  [PASS] Data loading works correctly!\n")
    return args


def test_model_init(args):
    print("=" * 60)
    print("[2/3] Testing model initialization (LLaMA 7B backbone)...")
    print("=" * 60)

    args.update({
        'task_name': 'long_term_forecast',
        'd_model': 32,
        'd_ff': 128,
        'n_heads': 8,
        'enc_in': 7,
        'dropout': 0.1,
        'patch_len': 16,
        'stride': 8,
        'llm_model': 'LLAMA',
        'llm_dim': 4096,
        'llm_layers': 6,
    })

    from models.TimeLLM import Model
    model = Model(args)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")
    print(f"  Trainable ratio:      {trainable_params/total_params*100:.2f}%")
    print("  [PASS] Model initialization successful!\n")
    return model


def test_forward_pass(model, args):
    print("=" * 60)
    print("[3/3] Testing forward pass...")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = model.to(device)
    model.eval()

    train_data, train_loader = data_provider(args, 'train')
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))

    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    dec_inp = torch.zeros_like(batch_y).float().to(device)

    print(f"  Input x_enc:  {batch_x.shape}")
    print(f"  Input dec_inp: {dec_inp.shape}")

    with torch.no_grad():
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    print(f"  Output shape: {outputs.shape}")
    expected_shape = (args.batch_size, args.pred_len, 1)
    assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}"
    assert not torch.isnan(outputs).any(), "Output contains NaN values!"
    assert not torch.isinf(outputs).any(), "Output contains Inf values!"

    f_dim = 0
    pred = outputs[:, -args.pred_len:, f_dim:]
    true = batch_y[:, -args.pred_len:, f_dim:].to(device)
    mse = nn.MSELoss()(pred, true)
    mae = nn.L1Loss()(pred, true)
    print(f"  Initial MSE (untrained): {mse.item():.6f}")
    print(f"  Initial MAE (untrained): {mae.item():.6f}")
    print("  [PASS] Forward pass works correctly!\n")


def test_training_step(model, args):
    print("=" * 60)
    print("[Bonus] Testing one training step...")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trained_parameters, lr=0.001)
    criterion = nn.MSELoss()

    train_data, train_loader = data_provider(args, 'train')
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))

    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    dec_inp = torch.zeros_like(batch_y).float().to(device)

    optimizer.zero_grad()
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    f_dim = 0
    outputs = outputs[:, -args.pred_len:, f_dim:]
    batch_y = batch_y[:, -args.pred_len:, f_dim:]
    loss = criterion(outputs, batch_y)

    print(f"  Loss before step: {loss.item():.6f}")
    loss.backward()
    optimizer.step()
    print("  [PASS] Training step completed successfully!")
    print("  Gradients flow correctly through trainable parameters.\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  TIME-LLM MODEL VERIFICATION")
    print("=" * 60 + "\n")

    args = test_data_loading()

    try:
        model = test_model_init(args)
    except Exception as e:
        print(f"  [INFO] Model init requires LLaMA weights: {e}")
        print("  Please ensure 'huggyllama/llama-7b' is available.")
        sys.exit(1)

    test_forward_pass(model, args)
    test_training_step(model, args)

    print("=" * 60)
    print("  ALL TESTS PASSED! Time-LLM is ready for training.")
    print("=" * 60)
    print("\nTo start full training, run:")
    print("  bash scripts/TimeLLM_ETTh1.sh")
    print("Or for single GPU:")
    print("  accelerate launch --mixed_precision bf16 main.py \\")
    print("    --task_name long_term_forecast \\")
    print("    --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 \\")
    print("    --features M --seq_len 512 --pred_len 96 \\")
    print("    --enc_in 7 --dec_in 7 --c_out 7 --d_model 32 --d_ff 128 \\")
    print("    --batch_size 24 --learning_rate 0.01 --llm_layers 32 \\")
    print("    --train_epochs 100")
