#!/usr/bin/env python3
"""
LSTM 訓練資料前處理 v2（高準確度版）

改進：
  1. 航向改用 sin/cos 編碼（解決環形數值問題）
  2. 切出獨立測試集（8:1:1）
  3. 更嚴格的資料過濾

特徵欄位（共 8 個）：
  lat, lng, altitude, velocity, sin_hdg, cos_hdg, vertical_rate, delta_t

使用方式：
  python3 prepare_training_data_v2.py
"""

import sqlite3
import numpy as np
import argparse
import os
from pathlib import Path

WINDOW_SIZE   = 10
MAX_TIME_GAP  = 30
MIN_SEG_LEN   = 15

FILTER = """
    on_ground = 0
    AND velocity > 10 AND velocity < 700
    AND altitude > 500 AND altitude < 45000
    AND lat IS NOT NULL AND lng IS NOT NULL
    AND heading IS NOT NULL AND heading >= 0 AND heading <= 360
    AND vertical_rate IS NOT NULL AND ABS(vertical_rate) < 100
"""

# sin/cos 編碼後的特徵順序
FEATURE_COLS = ["lat", "lng", "altitude", "velocity", "sin_hdg", "cos_hdg", "vertical_rate", "delta_t"]
LABEL_COLS   = ["lat", "lng", "altitude"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db",     default="/home/lbw/project_aerostrat/backend/data/aerostrat.db")
    p.add_argument("--output", default="/home/lbw/project_aerostrat/training_data/prepared_v2")
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument("--limit",  type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"連接資料庫：{args.db}")
    conn = sqlite3.connect(args.db)
    cur  = conn.cursor()

    limit_sql = f"LIMIT {args.limit}" if args.limit > 0 else ""
    cur.execute(f"""
        SELECT session_id FROM (
            SELECT session_id, COUNT(*) as cnt
            FROM track_points
            WHERE {FILTER}
            GROUP BY session_id
            HAVING cnt >= {MIN_SEG_LEN}
        ) {limit_sql}
    """)
    session_ids = [r[0] for r in cur.fetchall()]
    print(f"有效 session 數：{len(session_ids):,}")

    all_X, all_y = [], []

    for i, sid in enumerate(session_ids):
        if i % 5000 == 0:
            print(f"  [{i:,}/{len(session_ids):,}] 已收集 {len(all_X):,} 筆")

        cur.execute(f"""
            SELECT ts, lat, lng, altitude, velocity, heading, vertical_rate
            FROM track_points
            WHERE session_id = ? AND {FILTER}
            ORDER BY ts ASC
        """, (sid,))
        rows = cur.fetchall()
        if len(rows) < args.window + 1:
            continue

        # 切斷時間跳躍
        segments, seg = [], [rows[0]]
        for r in rows[1:]:
            if r[0] - seg[-1][0] <= MAX_TIME_GAP:
                seg.append(r)
            else:
                if len(seg) >= args.window + 1:
                    segments.append(seg)
                seg = [r]
        if len(seg) >= args.window + 1:
            segments.append(seg)

        for seg in segments:
            arr = []
            for k, r in enumerate(seg):
                dt      = float(r[0] - seg[k-1][0]) if k > 0 else 0.0
                hdg_rad = np.deg2rad(r[5])
                arr.append([
                    r[1],               # lat
                    r[2],               # lng
                    r[3],               # altitude
                    r[4],               # velocity
                    np.sin(hdg_rad),    # sin(heading)  ← 解決環形問題
                    np.cos(hdg_rad),    # cos(heading)
                    r[6],               # vertical_rate
                    dt,                 # delta_t
                ])
            arr = np.array(arr, dtype=np.float32)

            for j in range(len(arr) - args.window):
                all_X.append(arr[j : j + args.window])
                all_y.append(arr[j + args.window][:3])  # lat, lng, alt

    conn.close()

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    print(f"\n總樣本數：{len(X):,}")
    print(f"X shape：{X.shape}，y shape：{y.shape}")

    # 正規化（sin/cos 已在 -1~1，不需要額外處理）
    X_flat  = X.reshape(-1, X.shape[-1])
    col_min = X_flat.min(axis=0)
    col_max = X_flat.max(axis=0)

    # sin/cos 欄位固定為 -1~1，不做 min-max
    col_min[4] = -1.0; col_max[4] = 1.0  # sin_hdg
    col_min[5] = -1.0; col_max[5] = 1.0  # cos_hdg

    def normalize(a, mn, mx):
        return (a - mn) / (mx - mn + 1e-8)

    X_norm = normalize(X_flat, col_min, col_max).reshape(X.shape)
    y_norm = normalize(y, col_min[:3], col_max[:3])

    # 切分 訓練 80% / 驗證 10% / 測試 10%
    idx   = np.random.permutation(len(X_norm))
    n     = len(idx)
    train_idx = idx[:int(n * 0.8)]
    val_idx   = idx[int(n * 0.8):int(n * 0.9)]
    test_idx  = idx[int(n * 0.9):]

    out_path = os.path.join(args.output, "training.npz")
    np.savez_compressed(
        out_path,
        X_train  = X_norm[train_idx],
        y_train  = y_norm[train_idx],
        X_val    = X_norm[val_idx],
        y_val    = y_norm[val_idx],
        X_test   = X_norm[test_idx],
        y_test   = y_norm[test_idx],
        col_min  = col_min,
        col_max  = col_max,
        feature_cols = np.array(FEATURE_COLS),
        label_cols   = np.array(LABEL_COLS),
    )
    print(f"\n✓ 已儲存：{out_path}")
    print(f"  訓練集：{len(train_idx):,} 筆")
    print(f"  驗證集：{len(val_idx):,} 筆")
    print(f"  測試集：{len(test_idx):,} 筆")
    print("\n下一步：python3 train_lstm_v2.py")


if __name__ == "__main__":
    main()
