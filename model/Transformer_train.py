import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1) NMSE Loss + NMSE(dB)
# ==========================================
class NMSELoss(nn.Module):
    """
    样本级 NMSE:
        NMSE = ||pred-target||^2 / (||target||^2 + eps)
    返回 batch mean（线性值）
    """
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        B = pred.shape[0]
        pred = pred.reshape(B, -1)
        target = target.reshape(B, -1)
        num = torch.sum((pred - target) ** 2, dim=1)
        den = torch.sum(target ** 2, dim=1).clamp_min(self.eps)
        nmse = num / den
        return nmse.mean()

@torch.no_grad()
def nmse_db(pred, target, eps=1e-12):
    """
    NMSE(dB) = 10 * log10(mean(NMSE_sample))
    """
    B = pred.shape[0]
    pred = pred.reshape(B, -1)
    target = target.reshape(B, -1)
    num = torch.sum((pred - target) ** 2, dim=1)
    den = torch.sum(target ** 2, dim=1).clamp_min(eps)
    nmse = num / den
    nmse_mean = nmse.mean().clamp_min(eps)
    return 10.0 * torch.log10(nmse_mean)

# ==========================================
# 2) CSIDataset：按时间顺序 8:2（场景内前80%训练，后20%测试）
# ==========================================
class CSIDataset(Dataset):
    def __init__(self, npy_dir, scene_list, mean, std,
                 window_size=4, mode='train', split_ratio=0.8):
        self.window_size = window_size
        self.mean = mean
        self.std = std
        self.samples = []

        for scene in scene_list:
            file_path = os.path.join(npy_dir, f"{scene}.npy")
            if not os.path.exists(file_path):
                continue

            data = np.load(file_path)  # 期望: (num_frames, 2, 96, 128)
            # data = (data - self.mean) / (self.std + 1e-12)

            num_frames = data.shape[0]
            split_idx = int(num_frames * split_ratio)

            # ✅ 时间切分
            scene_data = data[:split_idx] if mode == 'train' else data[split_idx:]

            # 滑窗：window_size 帧预测下一帧
            for i in range(scene_data.shape[0] - window_size):
                x = scene_data[i:i + window_size]      # (W, 2, 96, 128)
                y = scene_data[i + window_size]        # (2, 96, 128)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = x.reshape(self.window_size, -1)  # (W, 2*96*128)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# ==========================================
# 3) 精简版 Transformer（保持你的逻辑不变）
# ==========================================
class CSITransformer(nn.Module):
    def __init__(self, input_dim=2*256*128, embed_dim=192, nhead=6, num_layers=3, window_size=4, dropout=0.1):
        super(CSITransformer, self).__init__()

        # 输入投影：F(65536) -> 512 -> embed_dim(192)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 位置编码：随 window_size 自适配
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,   # 4x 更常见更稳
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影：embed_dim -> 512 -> F(65536)
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, input_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        x = self.input_projection(x)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        out = x[:, -1, :]               # 取最后一个历史时刻特征
        out = self.output_projection(out)
        return out.reshape(B, 2, 256, 128)

# ==========================================
# 4) 日志函数
# ==========================================
def log_write(file_path, message):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as f:
        f.write(message + '\n')
    print(message)

# ==========================================
# 5) 主训练流程（epochs=300 + NMSE + dB）
# ==========================================
def train():
    X_TTI = 4  # 你原来这里写“唯一修改点”，但现在保持 4 不动，想改直接改这里即可

    TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    epochs = 300

    RUN_DIR = f'/home/ubuntu/zq_mae/ViT/result_all_model/run_Transformer_{epochs}ep_TimeSplit_NMSE_{TIMESTAMP}_{X_TTI}TTI'
    SAVE_DIR = os.path.join(RUN_DIR, 'checkpoints')
    LOG_FILE = os.path.join(RUN_DIR, 'training_log.txt')
    os.makedirs(SAVE_DIR, exist_ok=True)

    NPY_DIR = '/home/ubuntu/zq_mae/ViT/ViT_data_mimo'
    G_MEAN, G_STD = -0.000000, 0.000025

    all_scenes = sorted([f.replace('.npy', '') for f in os.listdir(NPY_DIR)
                         if f.endswith('.npy') and 'global' not in f])

    train_loader = DataLoader(
        CSIDataset(NPY_DIR, all_scenes, G_MEAN, G_STD, window_size=X_TTI, mode='train', split_ratio=0.8),
        batch_size=128, shuffle=True, pin_memory=True, num_workers=4
    )
    test_loader = DataLoader(
        CSIDataset(NPY_DIR, all_scenes, G_MEAN, G_STD, window_size=X_TTI, mode='test', split_ratio=0.8),
        batch_size=128, shuffle=False, pin_memory=True, num_workers=4
    )

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    model = CSITransformer(window_size=X_TTI).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_write(LOG_FILE, f"{X_TTI} TTI Transformer | 参数量: {total_params:,} | 设备: {device}")
    log_write(LOG_FILE, f"训练集样本数: {len(train_loader.dataset)} | 测试集样本数: {len(test_loader.dataset)}")

    # ✅ Loss 换成 NMSE
    criterion = NMSELoss(eps=1e-12).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # 余弦退火 T_max 与 epochs 对齐
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_test_nmse = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_nmse_sum = 0.0
        train_db_sum = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, y)  # NMSE (linear)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_nmse_sum += loss.item()
            train_db_sum += nmse_db(outputs, y).item()

        scheduler.step()

        avg_train_nmse = train_nmse_sum / max(1, len(train_loader))
        avg_train_db = train_db_sum / max(1, len(train_loader))

        model.eval()
        test_nmse_sum = 0.0
        test_db_sum = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                outputs = model(x)

                test_nmse_sum += criterion(outputs, y).item()
                test_db_sum += nmse_db(outputs, y).item()

        avg_test_nmse = test_nmse_sum / max(1, len(test_loader))
        avg_test_db = test_db_sum / max(1, len(test_loader))

        log_write(
            LOG_FILE,
            f"Epoch [{epoch}/{epochs}] - LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Train NMSE: {avg_train_nmse:.8f} ({avg_train_db:.2f} dB) | "
            f"Test NMSE: {avg_test_nmse:.8f} ({avg_test_db:.2f} dB)"
        )

        if avg_test_nmse < best_test_nmse:
            best_test_nmse = avg_test_nmse
            torch.save({'model_state_dict': model.state_dict(), 'nmse': best_test_nmse, 'epoch': epoch},
                       os.path.join(SAVE_DIR, 'best_model.pth'))
            log_write(LOG_FILE, f"      >>> [New Best] NMSE: {best_test_nmse:.8f} ({avg_test_db:.2f} dB)")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'ckpt_epoch_{epoch}.pth'))

if __name__ == "__main__":
    train()