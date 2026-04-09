# Flyradar ML

基於 ADS-B 資料的 LSTM 飛機位置預測模型，為 [AEROSTRAT 雷達系統](https://github.com/liaw-boy/project_aerostrat) 的 Dead Reckoning 模組提供資料支援。

---

## 模型架構

**Sequence-to-One LSTM**
- 輸入：過去 10 筆位置序列（lat, lng, altitude, velocity, heading, vertical_rate）
- 輸出：預測下一筆位置（lat, lng, altitude）
- v2 改進：hidden=256, layers=3 + Attention + Haversine Loss + 航向 sin/cos 編碼

---

## 腳本說明

| 腳本 | 說明 |
|------|------|
| `collect_training_data.py` | 從 OpenSky Network 收集台灣區域 ADS-B 歷史資料 |
| `prepare_training_data.py` | 從 aerostrat.db 匯出並前處理成 LSTM 訓練格式（v1） |
| `prepare_training_data_v2.py` | 前處理 v2：sin/cos 航向編碼、嚴格過濾、8:1:1 切分 |
| `train_lstm.py` | LSTM 訓練腳本 v1 |
| `train_lstm_v2.py` | LSTM 訓練腳本 v2（Attention + Haversine Loss） |
| `predict.py` | 推論腳本，載入訓練好的模型預測下一個位置 |

---

## 使用流程

### 1. 收集資料

```bash
python3 collect_training_data.py --days 7 --output ./training_data
# 可選：加上 OpenSky 帳號提高 API 頻率限制
python3 collect_training_data.py --username your_email --password your_pass
```

### 2. 前處理

從 aerostrat.db 匯出（推薦 v2）：

```bash
python3 prepare_training_data_v2.py --db /path/to/aerostrat.db --output ./training_data
```

### 3. 訓練

```bash
python3 train_lstm_v2.py --data ./training_data/prepared_v2/training.npz --epochs 50
```

訓練完成後模型儲存至 `training_data/model/best_model.pt`。

### 4. 推論

```bash
python3 predict.py --model ./training_data/model/best_model.pt
```

或在程式中呼叫：

```python
from predict import predict_next_position

next_pos = predict_next_position(last_10_points, model_path="./model/best_model.pt")
# 回傳 {"lat": 25.123, "lng": 121.456, "altitude": 10000}
```

---

## 資料格式

每筆輸入點包含 6 個欄位：

| 欄位 | 說明 |
|------|------|
| lat | 緯度（WGS84） |
| lng | 經度（WGS84） |
| altitude | 氣壓高度（公尺） |
| velocity | 地速（m/s） |
| heading | 航向（0-360°） |
| vertical_rate | 垂直速率（m/s） |

---

## 環境需求

```bash
pip install numpy torch requests
```

- Python 3.9+
- PyTorch 2.0+
- 訓練資料來源：[OpenSky Network](https://opensky-network.org) / [aerostrat.db](../project_aerostrat)

---

## 資料集

訓練資料（`training_data/`）不包含在此 repo，需自行執行 `collect_training_data.py` 或從 aerostrat.db 匯出。
