import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. NMSE Loss + NMSE(dB)
# ==========================================
class NMSELoss(nn.Module):
    """
    样本级 NMSE:
        NMSE = ||pred-target||^2 / (||target||^2 + eps)
    返回 batch mean
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
    dB = 10*log10(mean(NMSE))，先按样本算 NMSE，再 batch 平均，再转 dB
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
# 2. 数据集定义：按时间顺序 8:2（场景内前80%训练，后20%测试）
# ==========================================
class CSIDataset(Dataset):
    def __init__(self, npy_dir, scene_list, mean, std,
                 window_size=4, mode='train', split_ratio=0.8):
        self.window_size = window_size
        self.samples = []

        # mean/std 可以是 float 或 .npy 路径
        self.mean = np.load(mean) if isinstance(mean, str) else mean
        self.std  = np.load(std)  if isinstance(std, str)  else std

        for scene in scene_list:
            file_path = os.path.join(npy_dir, f"{scene}.npy")
            if not os.path.exists(file_path):
                continue

            data = np.load(file_path)  # 期望形状: (num_frames, 2, 256, 128)

            # 如果你确实不想标准化，可把下一行注释掉
            # 建议保留：NMSE训练/数值稳定通常更好
#            data = (data - self.mean) / (self.std + 1e-12)

            num_frames = data.shape[0]
            split_idx = int(num_frames * split_ratio)

            # 按时间切分（核心要求）
            scene_data = data[:split_idx] if mode == 'train' else data[split_idx:]

            # 滑窗：用过去 window_size 帧预测下一帧
            for i in range(scene_data.shape[0] - window_size):
                x = scene_data[i:i + window_size]      # (W, 2, 256, 128)
                y = scene_data[i + window_size]        # (2, 256, 128)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = x.reshape(self.window_size, -1)  # (W, 2*256*128)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# ==========================================
# 3. 增强型 LSTM 模型（保持不变）
# ==========================================
class OptimizedLSTM(nn.Module):
    def __init__(self, input_dim=2*256*128, compress_dim=256, hidden_dim=256, num_layers=4):
        super(OptimizedLSTM, self).__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, compress_dim),
            nn.LayerNorm(compress_dim),
            nn.GELU()
        )
        self.lstm = nn.LSTM(compress_dim, hidden_dim, num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x):
        B, T, F = x.shape
        x = self.input_projection(x)
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        out = self.output_projection(out)
        return out.reshape(B, 2, 256, 128)

# ==========================================
# 4. 日志函数
# ==========================================
def log_write(file_path, message):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as f:
        f.write(message + '\n')
    print(message)

# ==========================================
# 5. 主训练流程（300 epochs + NMSE + dB）
# ==========================================
def train():
    TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

    epochs = 300
    RUN_DIR = f'/home/ubuntu/zq_mae/ViT/result_all_model/run_LSTM_{epochs}ep_TimeSplit_NMSE_{TIMESTAMP}'
    SAVE_DIR = os.path.join(RUN_DIR, 'checkpoints')
    os.makedirs(SAVE_DIR, exist_ok=True)
    LOG_FILE = os.path.join(RUN_DIR, 'training_log.txt')

    NPY_DIR = '/home/ubuntu/zq_mae/ViT/ViT_data_mimo'
    # 如果你有全局 zscore 文件，也可以换成路径：
    # G_MEAN = '/home/ubuntu/zq_mae/ViT/ViT_data_mimo/zscore_mean_all.npy'
    # G_STD  = '/home/ubuntu/zq_mae/ViT/ViT_data_mimo/zscore_std_all.npy'
    G_MEAN, G_STD = -0.000000, 0.000025

    all_scenes = sorted([f.replace('.npy', '') for f in os.listdir(NPY_DIR)
                         if f.endswith('.npy') and 'global' not in f])

    train_loader = DataLoader(
        CSIDataset(NPY_DIR, all_scenes, G_MEAN, G_STD, mode='train', split_ratio=0.8, window_size=4),
        batch_size=128, shuffle=True, pin_memory=True, num_workers=4
    )
    test_loader = DataLoader(
        CSIDataset(NPY_DIR, all_scenes, G_MEAN, G_STD, mode='test', split_ratio=0.8, window_size=4),
        batch_size=128, shuffle=False, pin_memory=True, num_workers=4
    )

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = OptimizedLSTM().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_write(LOG_FILE, f"{'='*30}\nLSTM 参数量: {total_params:,}\n{'='*30}")
    log_write(LOG_FILE, f"训练集样本数: {len(train_loader.dataset)} | 测试集样本数: {len(test_loader.dataset)}")

    criterion = NMSELoss(eps=1e-12).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

    warmup_epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(epochs - warmup_epochs), eta_min=1e-7
    )

    best_test_nmse = float('inf')

    for epoch in range(1, epochs + 1):
        # Warmup
        if epoch <= warmup_epochs:
            curr_lr = 5e-4 * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

        # -------- Train --------
        model.train()
        train_nmse_sum = 0.0
        train_db_sum = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(x)

            loss = criterion(pred, y)  # NMSE (linear)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_nmse_sum += loss.item()
            train_db_sum += nmse_db(pred, y).item()

        if epoch > warmup_epochs:
            scheduler.step()

        avg_train_nmse = train_nmse_sum / max(1, len(train_loader))
        avg_train_db = train_db_sum / max(1, len(train_loader))

        # -------- Test --------
        model.eval()
        test_nmse_sum = 0.0
        test_db_sum = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x)

                test_nmse_sum += criterion(pred, y).item()
                test_db_sum += nmse_db(pred, y).item()

        avg_test_nmse = test_nmse_sum / max(1, len(test_loader))
        avg_test_db = test_db_sum / max(1, len(test_loader))

        log_write(
            LOG_FILE,
            f"Epoch [{epoch}/{epochs}] - LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Train NMSE: {avg_train_nmse:.8f} ({avg_train_db:.2f} dB) | "
            f"Test NMSE: {avg_test_nmse:.8f} ({avg_test_db:.2f} dB)"
        )

        # Save best by Test NMSE (linear)
        if avg_test_nmse < best_test_nmse:
            best_test_nmse = avg_test_nmse
            torch.save(
                {'model_state_dict': model.state_dict(), 'nmse': best_test_nmse, 'epoch': epoch},
                os.path.join(SAVE_DIR, 'best_model.pth')
            )
            log_write(LOG_FILE, f"      >>> [New Best] Test NMSE: {best_test_nmse:.8f} ({avg_test_db:.2f} dB)")

        # Regular checkpoint
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'ckpt_epoch_{epoch}.pth'))

    log_write(LOG_FILE, f"\nDone! Best Test NMSE: {best_test_nmse:.8f}")

if __name__ == "__main__":
    train()