# 空氣品質 AQI 預測地圖

這個專案使用環境部即時空氣品質 API 與歷史空氣品質 CSV 資料，訓練 LSTM 模型預測各測站未來 30 天 AQI，並以 Folium 產生互動式地圖。

## 執行前必要準備

1. 到「環境部環境資料開放平臺」申請 API 金鑰：<https://data.moenv.gov.tw/>
2. 下載過去空污測站記錄的空氣污染歷史數值 CSV 檔(最近12個月的csv檔)。<https://data.moenv.gov.tw/dataset/detail/AQX_P_488>
3. 將所有歷史 CSV 檔放在同一個資料夾中。
4. 打開 `final pius2.py`，替換下列設定：

```python
API_KEY = "您的api金鑰"
csv_folder = r"您的訓練csv檔資料夾路徑"
```

例如：

```python
API_KEY = "貼上您申請到的金鑰"
csv_folder = r"貼上存放所有歷史 CSV 檔的資料夾路徑"
```

> 請不要將自己的 API 金鑰或本機完整路徑提交到公開 GitHub repository。

## 執行方式

安裝必要套件後執行：

```bash
python "final pius2.py"
```

程式會讀取指定資料夾中的 CSV 檔，產生每個測站的預測圖，並輸出 `future_air_quality_map.html` 互動式地圖。
