#!/usr/bin/env python3
"""
OpenSky Network 台灣區域飛行資料收集腳本
用於收集 LSTM 位置預測模型的訓練資料

使用方式：
  python3 collect_training_data.py
  python3 collect_training_data.py --days 7 --output ./data
  python3 collect_training_data.py --username your_email --password your_pass
"""

import requests
import time
import csv
import json
import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── 台灣區域範圍 ──────────────────────────────────────────────
TAIWAN_BBOX = {
    "lamin": 21.5,   # 南端緯度
    "lamax": 25.5,   # 北端緯度
    "lomin": 119.5,  # 西端經度
    "lomax": 122.5,  # 東端經度
}

# ── API 設定 ──────────────────────────────────────────────────
OPENSKY_BASE = "https://opensky-network.org/api"
REQUEST_INTERVAL = 6      # 每次請求間隔（秒），避免觸發限制
CHUNK_MINUTES = 30        # 每次查詢時間區段（OpenSky 限制最大 30 分鐘）
MIN_POINTS = 10           # 軌跡點數少於此值則跳過（資料太短沒意義）

# ── CSV 欄位定義 ──────────────────────────────────────────────
FIELDNAMES = [
    "icao24",        # 飛機唯一識別碼
    "callsign",      # 航班號
    "timestamp",     # Unix 時間戳
    "datetime_utc",  # 可讀時間
    "latitude",      # 緯度
    "longitude",     # 經度
    "altitude_m",    # 高度（公尺，氣壓）
    "velocity_ms",   # 速度（公尺/秒）
    "heading",       # 航向（度，0=北）
    "vertical_rate", # 垂直速率（公尺/秒，正=爬升）
    "on_ground",     # 是否在地面
]


def parse_args():
    parser = argparse.ArgumentParser(description="收集 OpenSky 台灣區域飛行資料")
    parser.add_argument("--days", type=int, default=3,
                        help="收集過去幾天的資料（預設 3 天，免費帳號建議不超過 7 天）")
    parser.add_argument("--output", type=str, default="./training_data",
                        help="輸出目錄（預設 ./training_data）")
    parser.add_argument("--username", type=str, default="",
                        help="OpenSky 帳號（有帳號額度從 400 升到 4000）")
    parser.add_argument("--password", type=str, default="",
                        help="OpenSky 密碼")
    parser.add_argument("--resume", action="store_true",
                        help="從上次中斷的地方繼續")
    return parser.parse_args()


def make_session(username, password):
    session = requests.Session()
    if username and password:
        session.auth = (username, password)
        print(f"✓ 使用帳號登入：{username}")
    else:
        print("⚠ 未提供帳號，使用匿名模式（每日 400 次額度）")
    return session


def fetch_state_vectors(session, begin_ts, end_ts):
    """
    取得指定時間區間內台灣區域的所有飛機狀態
    回傳：list of state vectors
    """
    url = f"{OPENSKY_BASE}/states/all"
    params = {
        "time": end_ts,
        **TAIWAN_BBOX,
    }

    try:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            print("  ⚠ 達到額度限制，等待 60 秒...")
            time.sleep(60)
            return []
        if resp.status_code == 401:
            print("  ✗ 帳號或密碼錯誤")
            return []
        if resp.status_code != 200:
            print(f"  ✗ 請求失敗：HTTP {resp.status_code}")
            return []

        data = resp.json()
        states = data.get("states", []) or []
        return states

    except requests.exceptions.Timeout:
        print("  ⚠ 請求逾時，跳過此時間點")
        return []
    except Exception as e:
        print(f"  ✗ 錯誤：{e}")
        return []


def state_to_row(state, timestamp):
    """將 OpenSky state vector 轉換為 CSV 列"""
    # state 格式: [icao24, callsign, origin_country, time_position,
    #              last_contact, longitude, latitude, baro_altitude,
    #              on_ground, velocity, true_track, vertical_rate, ...]
    try:
        icao24      = state[0] or ""
        callsign    = (state[1] or "").strip()
        longitude   = state[5]
        latitude    = state[6]
        altitude    = state[7]   # 氣壓高度（公尺）
        on_ground   = state[8]
        velocity    = state[9]   # 公尺/秒
        heading     = state[10]  # 真實航向（度）
        vert_rate   = state[11]  # 公尺/秒

        # 過濾無效資料
        if latitude is None or longitude is None:
            return None
        if on_ground:
            return None  # 排除地面上的飛機

        return {
            "icao24":        icao24,
            "callsign":      callsign,
            "timestamp":     timestamp,
            "datetime_utc":  datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "latitude":      round(latitude, 6),
            "longitude":     round(longitude, 6),
            "altitude_m":    round(altitude, 1) if altitude else "",
            "velocity_ms":   round(velocity, 2) if velocity else "",
            "heading":       round(heading, 1) if heading else "",
            "vertical_rate": round(vert_rate, 2) if vert_rate else "",
            "on_ground":     on_ground,
        }
    except (IndexError, TypeError):
        return None


def load_progress(progress_file):
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {}


def save_progress(progress_file, progress):
    with open(progress_file, "w") as f:
        json.dump(progress, f)


def collect_day(session, day_start_ts, day_end_ts, output_dir, date_str):
    """收集單日資料，每 30 分鐘一個時間窗口"""
    output_file = output_dir / f"flights_{date_str}.csv"

    # 如果檔案已存在且非空，跳過
    if output_file.exists() and output_file.stat().st_size > 1000:
        print(f"  ✓ {date_str} 已有資料，跳過")
        return 0

    all_rows = {}  # icao24 -> list of rows（同一架飛機的軌跡）
    current_ts = day_start_ts
    chunk_seconds = CHUNK_MINUTES * 60
    total_chunks = int((day_end_ts - day_start_ts) / chunk_seconds)
    chunk_count = 0

    while current_ts < day_end_ts:
        chunk_count += 1
        dt = datetime.fromtimestamp(current_ts, tz=timezone.utc)
        print(f"  [{chunk_count}/{total_chunks}] 查詢 {dt.strftime('%H:%M')} UTC...", end=" ")

        states = fetch_state_vectors(session, current_ts, min(current_ts + chunk_seconds, day_end_ts))

        count = 0
        for state in states:
            row = state_to_row(state, current_ts)
            if row is None:
                continue
            icao = row["icao24"]
            if icao not in all_rows:
                all_rows[icao] = []
            all_rows[icao].append(row)
            count += 1

        print(f"取得 {count} 筆")
        current_ts += chunk_seconds
        time.sleep(REQUEST_INTERVAL)

    # 過濾軌跡點太少的飛機，並攤平寫入 CSV
    valid_flights = {k: v for k, v in all_rows.items() if len(v) >= MIN_POINTS}
    total_rows = sum(len(v) for v in valid_flights.values())

    if total_rows == 0:
        print(f"  ⚠ {date_str} 沒有有效資料")
        return 0

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for icao, rows in valid_flights.items():
            # 按時間排序
            rows.sort(key=lambda r: r["timestamp"])
            writer.writerows(rows)

    print(f"  ✓ {date_str} 完成：{len(valid_flights)} 架飛機，{total_rows} 筆資料 → {output_file.name}")
    return total_rows


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = output_dir / "progress.json"

    print("=" * 55)
    print("  OpenSky 台灣區域飛行資料收集器")
    print("=" * 55)
    print(f"  收集天數：{args.days} 天")
    print(f"  輸出目錄：{output_dir.resolve()}")
    print(f"  台灣範圍：{TAIWAN_BBOX}")
    print("=" * 55)

    session = make_session(args.username, args.password)
    progress = load_progress(progress_file) if args.resume else {}

    now = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    total_rows = 0

    for i in range(args.days, 0, -1):
        day = now - timedelta(days=i)
        date_str = day.strftime("%Y-%m-%d")

        if args.resume and progress.get(date_str) == "done":
            print(f"[{date_str}] 已完成，跳過")
            continue

        print(f"\n[{date_str}] 開始收集...")
        day_start = int(day.timestamp())
        day_end   = day_start + 86400  # 24 小時

        rows = collect_day(session, day_start, day_end, output_dir, date_str)
        total_rows += rows

        progress[date_str] = "done"
        save_progress(progress_file, progress)

    print("\n" + "=" * 55)
    print(f"  收集完成！總計 {total_rows} 筆資料")
    print(f"  資料位於：{output_dir.resolve()}")
    print("=" * 55)
    print("\n下一步：執行 preprocess.py 合併資料準備訓練")


if __name__ == "__main__":
    main()
