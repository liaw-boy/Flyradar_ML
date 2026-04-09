#!/usr/bin/env python3
"""
LSTM 飛機位置推算 — 推論腳本

輸入：過去 10 筆位置資料
輸出：預測的下一個位置（lat, lng, altitude）

使用方式：
  python3 predict.py --model ./training_data/model/best_model.pt

整合到後端時，呼叫 predict_next_position() 函式即可。
"""

import numpy as np
import torch
import json
import argparse
from train_lstm import AircraftLSTM  # 共用模型定義


def load_model(model_path: str):
    """載入訓練好的模型"""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    model = AircraftLSTM(
        n_features  = cfg["n_features"],
        hidden_size = cfg["hidden_size"],
        n_layers    = cfg["n_layers"],
        dropout     = cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    norm = ckpt["normalization"]
    col_min = np.array(norm["col_min"])
    col_max = np.array(norm["col_max"])

    return model, col_min, col_max, cfg


def predict_next_position(model, col_min, col_max, recent_points: list[dict]) -> dict:
    """
    根據最近 N 筆位置推算下一個位置

    recent_points: list of dict，每筆包含
      { lat, lng, altitude, velocity, heading, vertical_rate }
      按時間由舊到新排列

    回傳: { lat, lng, altitude }
    """
    # 轉成 numpy
    arr = np.array([
        [p["lat"], p["lng"], p["altitude"], p["velocity"], p["heading"], p["vertical_rate"], p.get("delta_t", 15.0)]
        for p in recent_points
    ], dtype=np.float32)

    # 正規化
    arr_norm = (arr - col_min) / (col_max - col_min + 1e-8)

    # 推論
    X = torch.tensor(arr_norm[np.newaxis, :, :], dtype=torch.float32)  # (1, window, 6)
    with torch.no_grad():
        pred_norm = model(X).numpy()[0]  # (3,)

    # 反正規化
    pred = pred_norm * (col_max[:3] - col_min[:3] + 1e-8) + col_min[:3]

    return {
        "lat":      float(pred[0]),
        "lng":      float(pred[1]),
        "altitude": float(pred[2]),
    }


def demo(model_path: str):
    """簡單示範：用假資料測試推論"""
    print(f"載入模型：{model_path}")
    model, col_min, col_max, cfg = load_model(model_path)
    print(f"視窗大小：{cfg['window_size']}，特徵數：{cfg['n_features']}")

    # 模擬一架飛往東方的飛機（10 筆歷史資料）
    recent = []
    for i in range(cfg["window_size"]):
        recent.append({
            "lat":           25.0 + i * 0.01,
            "lng":           121.0 + i * 0.02,
            "altitude":      35000.0,
            "velocity":      480.0,
            "heading":       90.0,
            "vertical_rate": 0.0,
            "delta_t":       15.0,   # 每 15 秒更新一次
        })

    result = predict_next_position(model, col_min, col_max, recent)
    print(f"\n最後已知位置：lat={recent[-1]['lat']:.4f}, lng={recent[-1]['lng']:.4f}")
    print(f"預測下一位置：lat={result['lat']:.4f}, lng={result['lng']:.4f}, alt={result['altitude']:.0f}呎")
    print(f"位移：Δlat={result['lat']-recent[-1]['lat']:.4f}, Δlng={result['lng']-recent[-1]['lng']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/lbw/project_aerostrat/training_data/model/best_model.pt")
    args = p.parse_args()
    demo(args.model)
