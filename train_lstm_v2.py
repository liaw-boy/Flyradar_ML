#!/usr/bin/env python3
"""
LSTM 飛機位置預測訓練腳本 v2（高準確度版）

改進：
  1. 更大模型（hidden=256, layers=3）
  2. Haversine Loss（直接最小化公里誤差）
  3. 加入 Attention 機制（聚焦關鍵時間步）
  4. 評估時輸出實際公里誤差

使用方式：
  python3 train_lstm_v2.py
  python3 train_lstm_v2.py --epochs 50 --hidden_size 256
"""

import numpy as np
import argparse, os, json, time, math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── 地球半徑（公里）──────────────────────────────────────────
EARTH_R = 6371.0


# ── Haversine Loss ────────────────────────────────────────────
class HaversineLoss(nn.Module):
    """
    直接對 lat/lng 計算球面距離損失（公里）
    讓模型優化的是實際地理誤差，而非抽象的 MSE 值
    """
    def __init__(self, col_min, col_max, alt_weight=0.1):
        super().__init__()
        self.lat_min = col_min[0]; self.lat_max = col_max[0]
        self.lng_min = col_min[1]; self.lng_max = col_max[1]
        self.alt_min = col_min[2]; self.alt_max = col_max[2]
        self.alt_weight = alt_weight

    def denorm_lat(self, x):
        return x * (self.lat_max - self.lat_min) + self.lat_min

    def denorm_lng(self, x):
        return x * (self.lng_max - self.lng_min) + self.lng_min

    def forward(self, pred, target):
        # 反正規化到真實緯經度
        lat1 = torch.deg2rad(self.denorm_lat(target[:, 0]))
        lat2 = torch.deg2rad(self.denorm_lat(pred[:, 0]))
        dlat = lat2 - lat1
        dlng = torch.deg2rad(
            self.denorm_lng(pred[:, 1]) - self.denorm_lng(target[:, 1])
        )
        a    = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlng/2)**2
        dist = 2 * EARTH_R * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))  # km

        # 高度誤差（換算成 km 量級）
        alt_err = (pred[:, 2] - target[:, 2]).abs() * (self.alt_max - self.alt_min) * 0.000304878  # 呎→km

        return (dist + self.alt_weight * alt_err).mean()


# ── Attention + LSTM 模型 ─────────────────────────────────────
class Attention(nn.Module):
    """時間維度注意力：讓模型自動聚焦最重要的時間步"""
    def __init__(self, hidden_size):
        super().__init__()
        self.w = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (batch, window, hidden)
        scores  = self.w(lstm_out).squeeze(-1)     # (batch, window)
        weights = torch.softmax(scores, dim=1)     # (batch, window)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        return context


class AircraftLSTMv2(nn.Module):
    def __init__(self, n_features=8, hidden_size=256, n_layers=3, dropout=0.2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            dropout     = dropout,
            batch_first = True,
        )
        self.attention = Attention(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, window, hidden)
        ctx    = self.attention(out)   # (batch, hidden)  attention 加權
        ctx    = self.dropout(ctx)
        return self.fc(ctx)


# ── Dataset ───────────────────────────────────────────────────
class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── 訓練/驗證 ─────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            if train: optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        default="/home/lbw/project_aerostrat/training_data/prepared_v2/training.npz")
    p.add_argument("--output",      default="/home/lbw/project_aerostrat/training_data/model_v2")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=4096)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int,   default=256)
    p.add_argument("--n_layers",    type=int,   default=3)
    p.add_argument("--dropout",     type=float, default=0.2)
    return p.parse_args()


def main():
    args   = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置：{device}")

    print(f"載入資料：{args.data}")
    data    = np.load(args.data, allow_pickle=True)
    X_train = data["X_train"]; y_train = data["y_train"]
    X_val   = data["X_val"];   y_val   = data["y_val"]
    col_min = data["col_min"]; col_max = data["col_max"]
    print(f"訓練：{len(X_train):,}  驗證：{len(X_val):,}")

    train_dl = DataLoader(FlightDataset(X_train, y_train),
                          batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(FlightDataset(X_val,   y_val),
                          batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    n_features = X_train.shape[2]
    model      = AircraftLSTMv2(
        n_features  = n_features,
        hidden_size = args.hidden_size,
        n_layers    = args.n_layers,
        dropout     = args.dropout,
    ).to(device)
    print(f"模型參數：{sum(p.numel() for p in model.parameters()):,}")

    criterion = HaversineLoss(col_min, col_max).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val  = float("inf")
    history   = []
    print("\n開始訓練（Haversine Loss，單位：公里）\n")

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = run_epoch(model, train_dl, optimizer, criterion, device, train=True)
        val_loss   = run_epoch(model, val_dl,   optimizer, criterion, device, train=False)
        scheduler.step()
        elapsed    = time.time() - t0
        marker     = " ← best" if val_loss < best_val else ""

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}km  val={val_loss:.4f}km  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"{elapsed:.1f}s{marker}")

        history.append({"epoch": epoch, "train_km": train_loss, "val_km": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_km":      val_loss,
                "config": {
                    "n_features":  n_features,
                    "hidden_size": args.hidden_size,
                    "n_layers":    args.n_layers,
                    "dropout":     args.dropout,
                    "window_size": X_train.shape[1],
                },
                "normalization": {
                    "col_min": col_min.tolist(),
                    "col_max": col_max.tolist(),
                }
            }, os.path.join(args.output, "best_model.pt"))

    with open(os.path.join(args.output, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ 完成！最佳驗證誤差：{best_val:.4f} 公里（{best_val*1000:.1f} 公尺）")
    print(f"  模型：{args.output}/best_model.pt")


if __name__ == "__main__":
    main()
