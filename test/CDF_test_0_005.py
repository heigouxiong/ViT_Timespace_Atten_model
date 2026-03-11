import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/ubuntu/zq_mae/ViT")

from ViT_pro_timespace_attention import SpaceTimeViT, SpaceTimeBlock, CSIDataset
from LSTM_train import OptimizedLSTM
from RNN_train import OptimizedRNN
from Transformer_train import CSITransformer


@torch.no_grad()
def compute_samplewise_nmse(model, dataloader, device, eps=1e-12):
    """
    返回测试集中每个样本的 NMSE（线性值）
    """
    model.eval()
    nmse_list = []

    for x, y in dataloader:
        x = x.to(device)   # (B, T, F)
        y = y.to(device)   # (B, 2, 256, 128)

        pred = model(x)    # (B, 2, 256, 128)

        B = pred.shape[0]
        pred = pred.reshape(B, -1)
        y = y.reshape(B, -1)

        num = torch.sum((pred - y) ** 2, dim=1)               # (B,)
        den = torch.sum(y ** 2, dim=1).clamp_min(eps)         # (B,)
        nmse = num / den                                      # (B,)

        nmse_list.extend(nmse.cpu().numpy().tolist())

    return np.array(nmse_list, dtype=np.float64)


def empirical_cdf(values):
    x = np.sort(values)
    n = len(x)
    y = np.arange(1, n + 1) / n
    return x, y


def load_checkpoint(model, model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def main():
    # =========================
    # 1. 数据配置
    # =========================
    npy_dir = "/home/ubuntu/zq_mae/ViT/ViT_data_timespace_10"
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
    # 2. 测试集
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

    if len(test_dataset) == 0:
        raise ValueError("测试集样本数为 0，请检查 npy_dir、scene_list、window_size 和 split_ratio。")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Test samples: {len(test_dataset)}")

    # =========================
    # 3. 四个模型及权重
    # =========================
    model_dict = {
        "ViT": {
            "model": SpaceTimeViT(
                img_size=(256, 128),
                patch_size=(16, 16),
                in_chans=2,
                T=4,
                embed_dim=512,
                depth=4,
                num_heads=16,
                drop_path_rate=0.1
            ).to(device),
            "ckpt": "/home/ubuntu/zq_mae/ViT/EVM_R2score/ViT_timespace_512_4_16_model_pth/checkpoints/best_model.pth"
        },
        "LSTM": {
            "model": OptimizedLSTM().to(device),
            "ckpt": "/home/ubuntu/zq_mae/ViT/EVM_R2score/LSTM_256_model_pth/checkpoints/best_model.pth"
        },
        "RNN": {
            "model": OptimizedRNN().to(device),
            "ckpt": "/home/ubuntu/zq_mae/ViT/EVM_R2score/RNN_512_8_model_pth/checkpoints/best_model.pth"
        },
        "Transformer": {
            "model": CSITransformer(window_size=4).to(device),
            "ckpt": "/home/ubuntu/zq_mae/ViT/EVM_R2score/Transformer_192_6_3_model_pth/checkpoints/best_model.pth"
        }
    }

    # =========================
    # 4. 计算每个模型的测试集 NMSE（线性值）
    # =========================
    results = {}

    for model_name, info in model_dict.items():
        print(f"\nLoading {model_name} ...")
        model = info["model"]
        ckpt_path = info["ckpt"]

        model = load_checkpoint(model, ckpt_path, device)
        print(f"{model_name} loaded successfully.")

        nmse_values = compute_samplewise_nmse(model, test_loader, device)
        results[model_name] = nmse_values

        print(f"{model_name}:")
        print(f"  Samples      = {len(nmse_values)}")
        print(f"  Mean NMSE    = {nmse_values.mean():.6f}")
        print(f"  Median NMSE  = {np.median(nmse_values):.6f}")
        print(f"  Min NMSE     = {nmse_values.min():.6f}")
        print(f"  Max NMSE     = {nmse_values.max():.6f}")

    # =========================
    # 5. 只画 x ∈ [0, 0.005] 的CDF图
    # =========================
    plt.figure(figsize=(8, 6))

    for model_name, nmse_values in results.items():
        x, y = empirical_cdf(nmse_values)
        plt.plot(x, y, linewidth=2, label=model_name)

    plt.xlabel("NMSE")
    plt.ylabel("CDF")
    plt.title("CDF of NMSE on Test Set")
    plt.xlim(0, 0.005)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_dir = "/home/ubuntu/zq_mae/ViT/EVM_R2score/cdf_results"
    os.makedirs(save_dir, exist_ok=True)

    fig_path = os.path.join(save_dir, "nmse_cdf_4models_0_0.005.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nCDF figure saved to: {fig_path}")

    plt.show()


if __name__ == "__main__":
    main()