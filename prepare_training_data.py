#!/usr/bin/env python3
"""
從 aerostrat.db 匯出 LSTM 訓練資料
輸出：training.npz（features + labels）

使用方式：
  python3 prepare_training_data.py
  python3 prepare_training_data.py --db /path/to/aerostrat.db --output ./training_data --window 10
"""

import sqlite3
import numpy as np
import argparse
import os
from pathlib import Path

# ── 設定 ─────────────────────────────────────────────────────
WINDOW_SIZE = 10        # 用過去幾筆來預測
MAX_TIME_GAP = 30       # 兩筆之間最大允許間隔（秒），超過則截斷
MIN_SESSION_LEN = 15    # session 至少要有幾筆才用

# 過濾條件
FILTER = """
    on_ground = 0
    AND velocity > 10 AND velocity < 700
    AND altitude > 500 AND altitude < 45000
    AND lat IS NOT NULL AND lng IS NOT NULL
    AND heading IS NOT NULL
    AND vertical_rate IS NOT NULL
"""

# 輸入特徵欄位（順序固定，模型會用到）
# delta_t = 距上一筆的秒數，讓模型知道時間間隔
FEATURE_COLS = ["lat", "lng", "altitude", "velocity", "heading", "vertical_rate", "delta_t"]
# 預測目標
LABEL_COLS   = ["lat", "lng", "altitude"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db",     default="/home/lbw/project_aerostrat/backend/data/aerostrat.db")
    p.add_argument("--output", default="/home/lbw/project_aerostrat/training_data/prepared")
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument("--limit",  type=int, default=0, help="限制 session 數（0=全部，測試用可設 1000）")
    return p.parse_args()


def normalize(arr, col_min, col_max):
    """Min-Max 正規化到 [0, 1]"""
    return (arr - col_min) / (col_max - col_min + 1e-8)


def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"連接資料庫：{args.db}")
    conn = sqlite3.connect(args.db)
    cur  = conn.cursor()

    # 取得所有有效 session（按時間排序）
    limit_sql = f"LIMIT {args.limit}" if args.limit > 0 else ""
    cur.execute(f"""
        SELECT session_id FROM (
            SELECT session_id, COUNT(*) as cnt
            FROM track_points
            WHERE {FILTER}
            GROUP BY session_id
            HAVING cnt >= {MIN_SESSION_LEN}
        ) {limit_sql}
    """)
    session_ids = [r[0] for r in cur.fetchall()]
    print(f"有效 session 數：{len(session_ids):,}")

    all_X, all_y = [], []
    skipped = 0

    for i, sid in enumerate(session_ids):
        if i % 5000 == 0:
            print(f"  處理中 {i:,}/{len(session_ids):,}，已收集 {len(all_X):,} 筆樣本")

        cur.execute(f"""
            SELECT ts, lat, lng, altitude, velocity, heading, vertical_rate
            FROM track_points
            WHERE session_id = ? AND {FILTER}
            ORDER BY ts ASC
        """, (sid,))
        rows = cur.fetchall()

        if len(rows) < args.window + 1:
            continue

        # 切斷時間跳躍（超過 MAX_TIME_GAP 秒就重新開始）
        segments = []
        seg = [rows[0]]
        for r in rows[1:]:
            if r[0] - seg[-1][0] <= MAX_TIME_GAP:
                seg.append(r)
            else:
                if len(seg) >= args.window + 1:
                    segments.append(seg)
                seg = [r]
        if len(seg) >= args.window + 1:
            segments.append(seg)

        # 滑動視窗產生訓練樣本
        for seg in segments:
            # 加入 delta_t（距上一筆的秒數，第一筆設為 0）
            arr = []
            for k, r in enumerate(seg):
                dt = float(r[0] - seg[k-1][0]) if k > 0 else 0.0
                arr.append([r[1], r[2], r[3], r[4], r[5], r[6], dt])
            arr = np.array(arr, dtype=np.float32)   # (len, 7)

            for j in range(len(arr) - args.window):
                X = arr[j : j + args.window]         # (window, 7)
                y = arr[j + args.window][:3]          # (3,) lat, lng, alt
                all_X.append(X)
                all_y.append(y)

    conn.close()

    X = np.array(all_X, dtype=np.float32)   # (N, window, 6)
    y = np.array(all_y, dtype=np.float32)   # (N, 3)
    print(f"\n總樣本數：{len(X):,}")
    print(f"X shape：{X.shape}，y shape：{y.shape}")

    # 計算正規化參數（存起來，推論時要用）
    X_flat = X.reshape(-1, X.shape[-1])
    col_min = X_flat.min(axis=0)
    col_max = X_flat.max(axis=0)

    # 正規化
    X_norm = normalize(X_flat, col_min, col_max).reshape(X.shape)
    y_norm = normalize(y, col_min[:3], col_max[:3])

    # 切分訓練/驗證集（90/10）
    split = int(len(X_norm) * 0.9)
    idx   = np.random.permutation(len(X_norm))
    train_idx, val_idx = idx[:split], idx[split:]

    out_path = os.path.join(args.output, "training.npz")
    np.savez_compressed(
        out_path,
        X_train    = X_norm[train_idx],
        y_train    = y_norm[train_idx],
        X_val      = X_norm[val_idx],
        y_val      = y_norm[val_idx],
        col_min    = col_min,
        col_max    = col_max,
        feature_cols = np.array(FEATURE_COLS),
        label_cols   = np.array(LABEL_COLS),
    )
    print(f"\n✓ 已儲存：{out_path}")
    print(f"  訓練集：{len(train_idx):,} 筆")
    print(f"  驗證集：{len(val_idx):,} 筆")
    print(f"  特徵欄位：{FEATURE_COLS}")
    print(f"  預測目標：{LABEL_COLS}")
    print("\n下一步：執行 train_lstm.py 開始訓練")


if __name__ == "__main__":
    main()
