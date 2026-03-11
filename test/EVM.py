import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("/home/ubuntu/zq_mae/ViT")

#from Transformer_train import CSIDataset, CSITransformer
# 如果你要测别的模型，就改成对应导入：
# from LSTM_train import CSIDataset, OptimizedLSTM
# from RNN_train import CSIDataset, OptimizedRNN
from ViT_pro_timespace_attention import SpaceTimeViT, CSIDataset
from ViT_Original import ViTBaseline, ViTBlock


def evaluate_evm(model, dataloader, device, eps=1e-12):
    model.eval()

    total_error_power = 0.0
    total_ref_power = 0.0

    evm_per_sample = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)   # (B, T, F)
            y = y.to(device)   # (B, 2, 256, 128)

            pred = model(x)    # (B, 2, 256, 128)

            # 拉平后按样本计算
            pred_flat = pred.reshape(pred.size(0), -1)
            y_flat = y.reshape(y.size(0), -1)

            # 每个样本的误差功率和参考信号功率
            error_power = torch.sum((pred_flat - y_flat) ** 2, dim=1)   # (B,)
            ref_power = torch.sum(y_flat ** 2, dim=1).clamp_min(eps)    # (B,)

            # 累积整体 EVM
            total_error_power += error_power.sum().item()
            total_ref_power += ref_power.sum().item()

            # 每个样本单独的 EVM
            evm_batch = torch.sqrt(error_power / ref_power)              # (B,)
            evm_per_sample.extend(evm_batch.cpu().numpy().tolist())

    if total_ref_power < eps:
        raise ValueError("参考信号总功率过小，无法计算 EVM。")

    # 整体 EVM
    evm = np.sqrt(total_error_power / total_ref_power)
    evm_percent = evm * 100.0
    evm_db = 20.0 * np.log10(max(evm, eps))

    # 每样本平均 EVM
    evm_sample_mean = np.mean(evm_per_sample) if len(evm_per_sample) > 0 else None
    evm_sample_mean_percent = evm_sample_mean * 100.0 if evm_sample_mean is not None else None
    evm_sample_mean_db = 20.0 * np.log10(max(evm_sample_mean, eps)) if evm_sample_mean is not None else None

    return evm, evm_percent, evm_db, evm_sample_mean, evm_sample_mean_percent, evm_sample_mean_db


def main():
    # =========================
    # 1. 路径与参数
    # =========================
    npy_dir = "/home/ubuntu/zq_mae/ViT/ViT_data_timespace_10"
    # model_path = "/home/ubuntu/zq_mae/ViT/EVM_R2score/ViT_timespace_512_4_16_model_pth/checkpoints/best_model.pth"    # 训练好的模型权重
    # model_path = "/home/ubuntu/zq_mae/ViT/EVM_R2score/LSTM_256_model_pth/checkpoints/best_model.pth"
    # model_path = "/home/ubuntu/zq_mae/ViT/EVM_R2score/RNN_512_8_model_pth/checkpoints/best_model.pth"
    # model_path = "/home/ubuntu/zq_mae/ViT/EVM_R2score/Transformer_192_6_3_model_pth/checkpoints/best_model.pth"
    model_path = "/home/ubuntu/zq_mae/ViT/EVM_R2score/ViT_512_4_16_model_pth/checkpoints/best_model.pth"

    mean = 0.0
    std = 1.0

    scene_list = [
        "10kmh_1000",
        "19kmh_1000",
        "29kmh_1000",
        "43kmh_1000",
        "53kmh_1000",
        "62kmh_1000",
        "72kmh_1000",
        "81kmh_1000",
        "91kmh_1000",
        "100kmh_1000"
    ]

    window_size = 4
    batch_size = 128
    num_workers = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================
    # 2. 构建测试集
    # =========================
    test_dataset = CSIDataset(
        npy_dir=npy_dir,
        scene_list=scene_list,
        mean=mean,
        std=std,
        window_size=window_size,
        mode='test',
        split_ratio=0.8
    )

    print(f"Test samples: {len(test_dataset)}")
    if len(test_dataset) == 0:
        raise ValueError("测试集样本数为 0，请检查 scene_list、npy_dir、window_size 和 split_ratio。")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # =========================
    # 3. 构建模型
    # =========================
    #model = CSITransformer(window_size=4).to(device)

    # 如果你测别的模型，用对应版本替换
    # model = OptimizedLSTM().to(device)
    # model = OptimizedRNN().to(device)
    # model = SpaceTimeViT(
    #     img_size=(256, 128),
    #     patch_size=(16, 16),
    #     in_chans=2,
    #     T=4,
    #     embed_dim=512,
    #     depth=4,
    #     num_heads=16,
    #     drop_path_rate=0.1
    # ).to(device)
    # Original ViT
    model = ViTBaseline(
        img_size=(256, 128),
        patch_size=(16, 16),
        in_chans=2,
        T=4,
        embed_dim=512,
        depth=4,
        num_heads=16,
        drop_path_rate=0.1
    ).to(device)

    # =========================
    # 4. 加载权重
    # =========================
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("Model loaded successfully.")

    # =========================
    # 5. 计算 EVM
    # =========================
    evm, evm_percent, evm_db, evm_sample_mean, evm_sample_mean_percent, evm_sample_mean_db = evaluate_evm(
        model=model,
        dataloader=test_loader,
        device=device
    )

    print("\n===== Test EVM Results =====")
    print(f"EVM:                     {evm:.6f}")
    print(f"EVM (%):                 {evm_percent:.6f}")
    print(f"EVM (dB):                {evm_db:.6f}")

    if evm_sample_mean is not None:
        print(f"Sample-wise Mean EVM:    {evm_sample_mean:.6f}")
        print(f"Sample-wise Mean EVM(%): {evm_sample_mean_percent:.6f}")
        print(f"Sample-wise Mean EVM(dB):{evm_sample_mean_db:.6f}")
    else:
        print("Sample-wise Mean EVM:    None")


if __name__ == "__main__":
    main()