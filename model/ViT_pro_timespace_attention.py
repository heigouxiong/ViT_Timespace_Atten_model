import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from timm.models.vision_transformer import Attention, Mlp
from timm.models.layers import DropPath

# ==========================================
# 1. 辅助工具 & 配置
# ==========================================
def log_write(file_path, message):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as f:
        f.write(message + '\n')
    print(message)

# ==========================================
# 2. NMSE Loss + NMSE(dB) 计算
# ==========================================
class NMSELoss(nn.Module):
    """
    样本级 NMSE:
        NMSE = ||e||^2 / (||y||^2 + eps)
    最后对 batch 求 mean，返回标量 loss
    """
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred/target: (B, 2, 256, 128)
        B = pred.shape[0]
        pred = pred.reshape(B, -1)
        target = target.reshape(B, -1)
        num = torch.sum((pred - target) ** 2, dim=1)                      # (B,)
        den = torch.sum(target ** 2, dim=1).clamp_min(self.eps)           # (B,)
        nmse = num / den                                                  # (B,)
        return nmse.mean()

@torch.no_grad()
def nmse_db(pred, target, eps=1e-12):
    """
    返回一个 batch 的 NMSE(dB)，先按样本算 NMSE，再对 batch 求均值，最后转 dB
        dB = 10 * log10(mean(NMSE))
    """
    B = pred.shape[0]
    pred = pred.reshape(B, -1)
    target = target.reshape(B, -1)
    num = torch.sum((pred - target) ** 2, dim=1)
    den = torch.sum(target ** 2, dim=1).clamp_min(eps)
    nmse = num / den  # (B,)
    nmse_mean = nmse.mean().clamp_min(eps)
    return 10.0 * torch.log10(nmse_mean)

# ==========================================
# 3. 数据集定义：按时间划分 8:2（场景内前80%训练，后20%测试）
# ==========================================
class CSIDataset(Dataset):
    """
    逻辑保持：用过去 window_size(=4) 帧预测未来 1 帧
    要求：按时间划分 8:2（不shuffle索引，不随机划分）
    同时支持 mean/std 为 float 或 .npy 路径
    """
    def __init__(self, npy_dir, scene_list, mean, std,
                 window_size=4, mode='train', split_ratio=0.8):
        self.window_size = window_size
        self.samples = []

        # mean/std 可以是数值，也可以是 .npy 路径
        self.mean = np.load(mean) if isinstance(mean, str) else mean
        self.std  = np.load(std)  if isinstance(std, str)  else std

        for scene in scene_list:
            file_path = os.path.join(npy_dir, f"{scene}.npy")
            if not os.path.exists(file_path):
                continue

            data = np.load(file_path)  # 期望形状: (T_all, 2, 256, 128)

            # 标准化（如果你确实想不用标准化，可把下一行注释掉）
#            # data = (data - self.mean) / (self.std + 1e-12)

            num_frames = data.shape[0]
            split_idx = int(num_frames * split_ratio)

            # 按时间切分
            scene_data = data[:split_idx] if mode == 'train' else data[split_idx:]

            # 滑动窗口：x=[i:i+W], y=[i+W]
            for i in range(scene_data.shape[0] - window_size):
                x = scene_data[i:i + window_size]        # (W, 2, 256, 128)
                y = scene_data[i + window_size]          # (2, 256, 128)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # 保持你原来的“LSTM式”输入接口： (W, -1)
        x = x.reshape(self.window_size, -1)              # (W, 2*256*128)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# ==========================================
# 4. 核心组件：时空分离 Attention Block
# ==========================================
class SpaceTimeBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop_path=0.):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, B, T, N):
        # x: (B*T, N, D)

        # Temporal Attention: (B*T,N,D)->(B,N,T,D)->(B*N,T,D)
        xt = x.view(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        res_t = self.temporal_attn(self.temporal_norm(xt))
        res_t = res_t.view(B, N, T, -1).permute(0, 2, 1, 3).reshape(B * T, N, -1)
        x = x + self.drop_path(res_t)

        # Spatial Attention: (B*T, N, D)
        x = x + self.drop_path(self.spatial_attn(self.spatial_norm(x)))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))
        return x

# ==========================================
# 5. 模型架构：SpaceTimeViT（逻辑不变，只适配你的输入是 (B,W,F)）
# ==========================================
class SpaceTimeViT(nn.Module):
    def __init__(self, img_size=(256, 128), patch_size=(16, 16), in_chans=2,
                 T=1, embed_dim=512, depth=4, num_heads=16, drop_path_rate=0.1):
        super().__init__()
        self.T = T
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_chans = in_chans

        self.grid_H = img_size[0] // patch_size[0]
        self.grid_W = img_size[1] // patch_size[1]
        self.num_patches = self.grid_H * self.grid_W

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, self.T, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(dim=embed_dim, num_heads=num_heads, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 16*16*2 = 512
        self.head = nn.Linear(embed_dim, patch_size[0] * patch_size[1] * 2)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.time_embed, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        你的 Dataset 输出 x: (B, T, F), 其中 F = 2*256*128
        这里不改变逻辑：仍然 reshape 回 (B, T, 2, 256, 128) 然后走原来的 ViT
        """
        B = x.shape[0]
        # x: (B, T, 2*256*128) -> (B, T, 2, 256, 128) -> (B*T, 2, 256, 128)
        x = x.view(B, self.T, 2, 256, 128).reshape(B * self.T, 2, 256, 128)

        # Patch embedding: (B*T, 2, 256, 128) -> (B*T, N, D)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Pos embed
        x = x + self.pos_embed

        # Time embed
        t_embed = self.time_embed.unsqueeze(2)  # (1, T, 1, D)
        x = x.view(B, self.T, self.num_patches, -1) + t_embed
        x = x.view(B * self.T, self.num_patches, -1)

        # Blocks
        for block in self.blocks:
            x = block(x, B, self.T, self.num_patches)

        x = self.norm(x)

        # last time step features
        x = x.view(B, self.T, self.num_patches, -1)
        x_last = x[:, -1, :, :]  # (B, N, D)

        # predict patches
        x = self.head(x_last)    # (B, N, 16*16*2)

        # reshape back to image
        x = x.reshape(B, self.grid_H, self.grid_W, self.patch_size[0], self.patch_size[1], 2)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, 2, 256, 128)
        return x

# ==========================================
# 6. 主训练流程：loss=NMSE，同时输出 train/test 的 dB
# ==========================================
def train():
    TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    RUN_DIR = f'/home/ubuntu/zq_mae/ViT/result_all_model/run_ViT_10_{TIMESTAMP}_SpaceTime_300ep_NMSE'
    SAVE_DIR = os.path.join(RUN_DIR, 'checkpoints')
    LOG_FILE = os.path.join(RUN_DIR, 'training_log.txt')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 数据配置 ---
    X_TTI = 4
    NPY_DIR = '/home/ubuntu/zq_mae/ViT/ViT_data_timespace_10'
    G_MEAN = '/home/ubuntu/zq_mae/ViT/ViT_data_mimo/zscore_mean_all.npy'
    G_STD  = '/home/ubuntu/zq_mae/ViT/ViT_data_mimo/zscore_std_all.npy'

    all_scenes = sorted([f.replace('.npy', '') for f in os.listdir(NPY_DIR)
                         if f.endswith('.npy') and 'global' not in f])

    # 按时间 8:2 划分（在 Dataset 内实现）
    train_set = CSIDataset(NPY_DIR, all_scenes, G_MEAN, G_STD,
                           window_size=X_TTI, mode='train', split_ratio=0.8)
    test_set  = CSIDataset(NPY_DIR, all_scenes, G_MEAN, G_STD,
                           window_size=X_TTI, mode='test', split_ratio=0.8)

    train_loader = DataLoader(train_set, batch_size=100, shuffle=True, pin_memory=True, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=100, shuffle=False, pin_memory=True, num_workers=4)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = SpaceTimeViT(img_size=(256, 128), patch_size=(16, 16), in_chans=2, T=X_TTI).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_write(LOG_FILE, f"SpaceTimeViT (300 Epochs, NMSE) | 参数量: {total_params:,} | 设备: {device}")
    log_write(LOG_FILE, f"训练集: {len(train_set)} | 测试集: {len(test_set)}")

    # --- Loss / Optim ---
    criterion = NMSELoss(eps=1e-12).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-2)

    epochs = 300
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_test_nmse = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_nmse_db_sum = 0.0

        # Warmup 10 epochs（保持你原来的逻辑）
        if epoch <= 10:
            lr_scale = epoch / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-4 * lr_scale

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)   # (B, T, F)
            targets = targets.to(device, non_blocking=True) # (B, 2, 256, 128)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_nmse_db_sum += nmse_db(outputs, targets).item()

        if epoch > 10:
            scheduler.step()

        # --- Eval ---
        model.eval()
        test_loss_sum = 0.0
        test_nmse_db_sum = 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)

                test_loss_sum += criterion(outputs, targets).item()
                test_nmse_db_sum += nmse_db(outputs, targets).item()

        avg_train_loss = train_loss_sum / max(1, len(train_loader))   # NMSE (linear)
        avg_test_loss  = test_loss_sum  / max(1, len(test_loader))    # NMSE (linear)

        avg_train_db = train_nmse_db_sum / max(1, len(train_loader))  # NMSE(dB)
        avg_test_db  = test_nmse_db_sum  / max(1, len(test_loader))   # NMSE(dB)

        curr_lr = optimizer.param_groups[0]['lr']
        info = (f"Epoch [{epoch}/{epochs}] - LR: {curr_lr:.6f} | "
                f"Train NMSE: {avg_train_loss:.8f} ({avg_train_db:.2f} dB) | "
                f"Test NMSE: {avg_test_loss:.8f} ({avg_test_db:.2f} dB)")
        log_write(LOG_FILE, info)

        # 保存最优模型（按 Test NMSE 线性值判断）
        if avg_test_loss < best_test_nmse:
            best_test_nmse = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'nmse': best_test_nmse,
            }, os.path.join(SAVE_DIR, 'best_model.pth'))
            log_write(LOG_FILE, f"      >>> [New Best] Test NMSE: {best_test_nmse:.8f} ({avg_test_db:.2f} dB)")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'ckpt_epoch_{epoch}.pth'))

    log_write(LOG_FILE, f"\nDone! Best Test NMSE: {best_test_nmse:.8f}")

if __name__ == "__main__":
    train()