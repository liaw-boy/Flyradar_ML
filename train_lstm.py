#!/usr/bin/env python3
"""
LSTM 飛機位置預測模型訓練腳本

架構：Sequence-to-One LSTM
  輸入：過去 10 筆位置（lat, lng, alt, velocity, heading, vertical_rate）
  輸出：下一筆位置（lat, lng, alt）

使用方式：
  python3 train_lstm.py
  python3 train_lstm.py --data ./training_data/prepared/training.npz --epochs 50
"""

import numpy as np
import argparse
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ── 模型定義 ─────────────────────────────────────────────────

class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AircraftLSTM(nn.Module):
    """
    雙層 LSTM + Dropout + 全連接輸出層
    輸入：(batch, window, n_features)
    輸出：(batch, 3)  → lat, lng, altitude
    """
    def __init__(self, n_features=6, hidden_size=128, n_layers=2, dropout=0.2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            dropout     = dropout if n_layers > 1 else 0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)      # out: (batch, window, hidden)
        out = out[:, -1, :]        # 取最後一個時間步
        out = self.dropout(out)
        return self.fc(out)


# ── 訓練流程 ─────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        default="/home/lbw/project_aerostrat/training_data/prepared/training.npz")
    p.add_argument("--output",      default="/home/lbw/project_aerostrat/training_data/model")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=2048)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--n_layers",    type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # 裝置選擇
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # 載入資料
    print(f"載入資料：{args.data}")
    data = np.load(args.data, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    col_min, col_max = data["col_min"],  data["col_max"]

    print(f"訓練集：{len(X_train):,} 筆，驗證集：{len(X_val):,} 筆")
    print(f"輸入維度：{X_train.shape[1:]}，輸出維度：{y_train.shape[1:]}")

    train_ds = FlightDataset(X_train, y_train)
    val_ds   = FlightDataset(X_val,   y_val)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 建立模型
    n_features = X_train.shape[2]
    model = AircraftLSTM(
        n_features  = n_features,
        hidden_size = args.hidden_size,
        n_layers    = args.n_layers,
        dropout     = args.dropout,
    ).to(device)
    print(f"\n模型參數量：{sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    # 訓練
    best_val_loss = float("inf")
    history = []
    print("\n開始訓練...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        val_loss   = val_epoch(model, val_dl, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        marker = " ← best" if val_loss < best_val_loss else ""

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"{elapsed:.1f}s{marker}")

        # 儲存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_loss":    val_loss,
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

    # 儲存訓練歷史
    with open(os.path.join(args.output, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ 訓練完成！最佳驗證損失：{best_val_loss:.6f}")
    print(f"  模型儲存於：{args.output}/best_model.pt")
    print("\n下一步：執行 predict.py 測試推論")


if __name__ == "__main__":
    main()
