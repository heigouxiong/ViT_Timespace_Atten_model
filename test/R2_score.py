import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

import sys
sys.path.append("/home/ubuntu/zq_mae/ViT")
from ViT_pro_timespace_attention import SpaceTimeViT, SpaceTimeBlock, CSIDataset
from LSTM_train import  OptimizedLSTM
from RNN_train import  OptimizedRNN
from Transformer_train import CSITransformer
from ViT_Original import ViTBaseline, ViTBlock


def evaluate_r2(
    model,
    dataloader,
    device
):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)   # (B, T, F)
            y = y.to(device)   # (B, 2, 256, 128)

            pred = model(x)    # (B, 2, 256, 128)

            # 拉平成二维: (B, 2*256*128)
            pred_flat = pred.reshape(pred.size(0), -1).cpu().numpy()
            y_flat = y.reshape(y.size(0), -1).cpu().numpy()

            all_preds.append(pred_flat)
            all_targets.append(y_flat)

    all_preds = np.concatenate(all_preds, axis=0)      # (N, D)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, D)

    # 1. 整体平均 R2（推荐）
    r2_mean = r2_score(all_targets, all_preds, multioutput='uniform_average')

    # 2. 按输出维度加权的 R2
    r2_var_weighted = r2_score(all_targets, all_preds, multioutput='variance_weighted')

    # 3. 每个样本单独算 R2，再求平均（可选）
    r2_per_sample = []
    for i in range(all_targets.shape[0]):
        try:
            r2_i = r2_score(all_targets[i], all_preds[i])
            r2_per_sample.append(r2_i)
        except:
            pass
    r2_sample_mean = np.mean(r2_per_sample) if len(r2_per_sample) > 0 else None

    return r2_mean, r2_var_weighted, r2_sample_mean


def main():
    # =========================
    # 1. 路径与参数
    # =========================
    npy_dir = "/home/ubuntu/zq_mae/ViT/ViT_data_timespace_10"                 # 你的 .npy 数据目录
    # model_path = "/home/ubuntu/zq_mae/ViT/EVM_R2score/ViT_timespace_512_4_16_model_pth/checkpoints/best_model.pth"    # 训练好的模型权重
    # model_path = "/home/ubuntu/zq_mae/ViT/EVM_R2score/LSTM_256_model_pth/checkpoints/best_model.pth"
    # model_path = "/home/ubuntu/zq_mae/ViT/EVM_R2score/RNN_512_8_model_pth/checkpoints/best_model.pth"
    # model_path = "/home/ubuntu/zq_mae/ViT/EVM_R2score/Transformer_192_6_3_model_pth/checkpoints/best_model.pth"
    model_path = "/home/ubuntu/zq_mae/ViT/EVM_R2score/ViT_512_4_16_model_pth/checkpoints/best_model.pth"
    
    mean = 0.0                         # 如果不用标准化，这里随便写即可
    std = 1.0                          # 如果不用标准化，这里随便写即可

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Test samples: {len(test_dataset)}")

    # =========================
    # 3. 构建模型
    # =========================
    # model = SpaceTimeViT(
    #     img_size=(256, 128),
    #     patch_size=(16, 16),
    #     in_chans=2,
    #     T=4,                 # 你的输入历史帧数
    #     embed_dim=512,
    #     depth=4,
    #     num_heads=16,
    #     drop_path_rate=0.1
    # ).to(device)
    
    # LSTM_model
    # model = OptimizedLSTM().to(device)

    # RNN_model
    # model = OptimizedRNN().to(device)

    # Transformer model
    # model = CSITransformer(window_size=4).to(device)

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

    # 兼容两种保存方式：
    # 1) torch.save(model.state_dict(), path)
    # 2) torch.save({'model_state_dict': model.state_dict()}, path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("Model loaded successfully.")

    # =========================
    # 5. 计算 R2-score
    # =========================
    r2_mean, r2_var_weighted, r2_sample_mean = evaluate_r2(
        model=model,
        dataloader=test_loader,
        device=device
    )

    print("\n===== Test R2 Results =====")
    print(f"R2-score (uniform_average):   {r2_mean:.6f}")
    print(f"R2-score (variance_weighted): {r2_var_weighted:.6f}")
    if r2_sample_mean is not None:
        print(f"R2-score (sample-wise mean):  {r2_sample_mean:.6f}")
    else:
        print("R2-score (sample-wise mean):  None")


if __name__ == "__main__":
    main()