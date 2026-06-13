import os
import glob
import pandas as pd
import numpy as np
import folium
import webbrowser
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import requests
import certifi
from datetime import datetime, timedelta
import base64
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.dates as mdates

# ----------------------------
# 0️⃣ 執行前設定
# ----------------------------
# 請先到環境部環境資料開放平臺申請 API 金鑰：
# https://data.moenv.gov.tw/
# 並下載歷史空氣品質監測 CSV 檔，將所有 CSV 放在同一個資料夾。
# 執行前請將下列兩個值替換成您自己的 API 金鑰與 CSV 資料夾路徑。
API_KEY = "您的api金鑰"
csv_folder = r"您的訓練csv檔資料夾路徑"

if API_KEY == "您的api金鑰":
    raise ValueError("請先將 API_KEY 替換成您在環境部環境資料開放平臺申請的 API 金鑰。")

if csv_folder == r"您的訓練csv檔資料夾路徑":
    raise ValueError("請先將 csv_folder 替換成存放歷史空氣品質 CSV 檔的資料夾路徑。")

# ----------------------------
# 1️⃣ 即時 AQI
# ----------------------------
API_URL = f"https://data.moenv.gov.tw/api/v2/aqx_p_432?offset=0&limit=200&api_key={API_KEY}"
try:
    response = requests.get(API_URL, verify=False, timeout=10)
    response.raise_for_status()
    data = response.json()
except:
    data = []

records = data['records'] if isinstance(data, dict) and 'records' in data else data
realtime_df = pd.DataFrame(records)
realtime_df.columns = [c.lower() for c in realtime_df.columns]

# 保留即時 API 裡面常用的空污欄位
realtime_cols = [
    'sitename', 'county',
    'aqi', 'pollutant', 'status',
    'pm2.5', 'pm10',
    'so2', 'co', 'o3', 'o3_8hr',
    'no2', 'nox', 'no',
    'windspeed', 'winddirec',
    'publishtime',
    'latitude', 'longitude'
]

# 只取實際存在的欄位，避免某些欄位不存在時報錯
realtime_cols = [col for col in realtime_cols if col in realtime_df.columns]
realtime_df = realtime_df[realtime_cols]

# 需要轉成數字的欄位
numeric_cols = [
    'aqi', 'latitude', 'longitude',
    'pm2.5', 'pm10',
    'so2', 'co', 'o3', 'o3_8hr',
    'no2', 'nox', 'no',
    'windspeed', 'winddirec'
]

for col in numeric_cols:
    if col in realtime_df.columns:
        realtime_df[col] = pd.to_numeric(realtime_df[col], errors='coerce')

realtime_df.dropna(subset=['latitude', 'longitude', 'aqi'], inplace=True)

# 這裡不要只保留 latitude、longitude、aqi，要把 PM2.5、PM10、風向等一起放進去
site_coords = realtime_df.set_index('sitename').to_dict('index')

# ----------------------------
# 2️⃣ 讀取歷史資料並清理
# ----------------------------
all_files = glob.glob(os.path.join(csv_folder, "*.csv"))
if not all_files:
    raise FileNotFoundError(f"在 {csv_folder} 找不到任何 CSV 檔，請確認歷史空氣品質 CSV 檔已下載並放在同一個資料夾。")

df_list = []
for f in all_files:
    try:
        df_list.append(pd.read_csv(f, encoding='utf-8'))
    except:
        df_list.append(pd.read_csv(f, encoding='cp950'))

df = pd.concat(df_list, ignore_index=True)
df.columns = [c.lower() for c in df.columns]
date_col_candidates = ['date','datacreationdate','data_creation_date']
for col in date_col_candidates:
    if col in df.columns:
        df['date'] = pd.to_datetime(df[col])
        break
df.sort_values(['sitename','date'], inplace=True)

# ----------------------------
# 3️⃣ 合併分割站點
# ----------------------------

rename_sites = {
    "屏東(枋山)": "屏東枋山",
    "屏東（枋山）": "屏東枋山",
    "新北(樹林)": "新北樹林",
    "新北（樹林）": "新北樹林"
}
df['sitename'] = df['sitename'].replace(rename_sites)
 # 刪除資料缺失的站點
exclude_sites = ["Annan","Banqiao","Cailiao","Changhua","Chaozhou","Chiayi",
                 "Chiayi（Performing Arts Center）","Dacheng","Dali","Daliao",
                 "Datong","Dayuan","Dongshan","Douliu","Erlin","Fengshan",
                 "Fengyuan","FugueiCape","Fuxing","Guanshan","Guanyin",
                 "Guting","Hengchun","Hsinchu","Hualien","Hukou","Keelung",
                 "Kinmen","Linkou","Linsen","Linyuan","Longtan","Lunbei",
                 "Magong","Mailiao","Matsu","Meinong","Miaoli","Nantou",
                 "Nantou（Lugu）","Nanzi","New Taipei（Shulin）","Pingtung",
                 "Pingtung（Liuqiu）","Pingtung（fangshan）","Pingzhen","Puli",
                 "Puzi","Qianjin","Qianzhen","Qiaotou","Renwu","Sanchong",
                 "Sanyi","Shalu","Shanhua","Shilin","Songshan","Tainan",
                 "Taitung","Taiwan Avenue","Taixi","Tamsui","Taoyuan","Toufen",
                 "Tucheng","Wanhua","Xianxi","Xiaogang","Xindian","Xingang",
                 "Xinying","Xinzhuang","Xitun","Xizhi","Yangming","Yilan",
                 "Yonghe","Yuanlin","Zhongli","Zhongming","Zhongshan","Zhudong",
                 "Zhushan","Zuoying","台中市（和平區消防隊）","台中市（和平區）",
                 "台中（和平）","嘉義（建國路）","嘉義（東區）","宜蘭（三星）","林森",
                 "臺南（南化）","臺灣大道","高雄（湖內）"
]
df = df[~df['sitename'].isin(exclude_sites)]

 # ----------------------------
 # 2️⃣ LSTM 預測 + 趨勢圖生成
 # ----------------------------
# 多變量輸入特徵：使用過去 AQI、PM2.5、PM10 等污染物資料預測未來 AQI
feature_candidates = ['aqi', 'pm2.5', 'pm10', 'so2', 'co', 'o3', 'no2', 'nox', 'no']
target_feature = 'aqi'
sequence_length = 90
predictions = []

plots_folder = os.path.join(csv_folder, "plots")
os.makedirs(plots_folder, exist_ok=True)

site_name_en = {
    "三義": "Sanyi",
    "三重": "Sanchong",
    "中壢": "Zhongli",
    "中山": "Zhongshan",
    "二林": "Erlin",
    "仁武": "Renwu",
    "冬山": "Dongshan",
    "前金": "Qianjin",
    "前鎮": "Qianzhen",
    "南投": "Nantou",
    "南投（鹿谷）": "Nantou Lugu",
    "古亭": "Guting",
    "員林": "Yuanlin",
    "善化": "Shanhua",
    "嘉義": "Chiayi",
    "土城": "Tucheng",
    "埔里": "Puli",
    "基隆": "Keelung",
    "士林": "Shilin",
    "大同": "Datong",
    "大園": "Dayuan",
    "大城": "Dacheng",
    "大寮": "Daliao",
    "大里": "Dali",
    "安南": "Annan",
    "宜蘭": "Yilan",
    "富貴角": "Fuguei Cape",
    "小港": "Xiaogang",
    "屏東": "Pingtung",
    "屏東枋山": "Pingtung Fangshan",
    "屏東（琉球）": "Pingtung Liuqiu",
    "崙背": "Lunbei",
    "左營": "Zuoying",
    "平鎮": "Pingzhen",
    "彰化": "Changhua",
    "復興": "Fuxing",
    "忠明": "Zhongming",
    "恆春": "Hengchun",
    "斗六": "Douliu",
    "新北樹林": "New Taipei Shulin",
    "新店": "Xindian",
    "新港": "Xingang",
    "新營": "Xinying",
    "新竹": "Hsinchu",
    "新莊": "Xinzhuang",
    "朴子": "Puzi",
    "松山": "Songshan",
    "板橋": "Banqiao",
    "林口": "Linkou",
    "林園": "Linyuan",
    "桃園": "Taoyuan",
    "楠梓": "Nanzi",
    "橋頭": "Qiaotou",
    "永和": "Yonghe",
    "汐止": "Xizhi",
    "沙鹿": "Shalu",
    "淡水": "Tamsui",
    "湖口": "Hukou",
    "潮州": "Chaozhou",
    "竹山": "Zhushan",
    "竹東": "Zhudong",
    "線西": "Xianxi",
    "美濃": "Meinong",
    "臺南": "Tainan",
    "臺東": "Taitung",
    "臺西": "Taixi",
    "花蓮": "Hualien",
    "苗栗": "Miaoli",
    "菜寮": "Cailiao",
    "萬華": "Wanhua",
    "西屯": "Xitun",
    "觀音": "Guanyin",
    "豐原": "Fengyuan",
    "金門": "Kinmen",
    "關山": "Guanshan",
    "陽明": "Yangming",
    "頭份": "Toufen",
    "馬公": "Magong",
    "馬祖": "Matsu",
    "鳳山": "Fengshan",
    "麥寮": "Mailiao",
    "龍潭": "Longtan"
}

# ✅ 清空舊圖，放在迴圈前
for f in glob.glob(os.path.join(plots_folder, "*.png")):
     os.remove(f)

def calibrate_future_volatility(future_preds, test_preds, history_values):
    """
    依照每個測站自己的歷史波動程度，調整未來 30 天預測的震盪幅度。
    避免某些測站太平滑，某些測站太震盪。
    """

    future_preds = np.array(future_preds, dtype=float)
    test_preds = np.array(test_preds, dtype=float)
    history_values = np.array(history_values, dtype=float)

    # 取測試區間最後 30 筆預測，以及歷史最後 30 筆真實 AQI
    recent_test_preds = test_preds[-30:] if len(test_preds) >= 30 else test_preds
    recent_history = history_values[-30:] if len(history_values) >= 30 else history_values

    # 目標波動幅度：一部分參考模型在測試區間的表現，一部分參考最近真實 AQI
    test_std = np.nanstd(recent_test_preds)
    history_std = np.nanstd(recent_history)
    target_std = 0.7 * test_std + 0.3 * history_std

    future_std = np.nanstd(future_preds)

    # 如果標準差太小，代表幾乎是直線，避免除以 0
    if future_std < 1e-6 or target_std < 1e-6:
        return future_preds

    # scale > 1：放大未來線起伏
    # scale < 1：壓低未來線起伏
    scale = target_std / future_std

    # 避免調整太激烈
    scale = np.clip(scale, 0.45, 2.5)

    future_mean = np.nanmean(future_preds)
    adjusted = future_mean + (future_preds - future_mean) * scale

    # 用歷史 AQI 的單日變化限制，避免像東山那樣出現太誇張尖峰
    recent_diff = np.abs(np.diff(recent_history))
    recent_diff = recent_diff[~np.isnan(recent_diff)]

    if len(recent_diff) > 0:
        max_daily_change = np.nanpercentile(recent_diff, 90) * 1.5

        for i in range(1, len(adjusted)):
            diff = adjusted[i] - adjusted[i - 1]

            if abs(diff) > max_daily_change:
                adjusted[i] = adjusted[i - 1] + np.sign(diff) * max_daily_change

    # AQI 不應小於 0
    adjusted = np.clip(adjusted, 0, None)

    return adjusted

for site in df['sitename'].unique():
    site_df = df[df['sitename']==site].copy()

# 將 AQI、PM2.5、PM10 等多個污染物欄位轉成數字，作為 many-to-one 模型輸入
    input_features = [feat for feat in feature_candidates if feat in site_df.columns]
    if target_feature not in input_features:
        print(f"跳過 {site}，缺少 AQI 欄位")
        continue

    for feat in input_features:
        site_df[feat] = pd.to_numeric(site_df[feat], errors='coerce')


    # 移除模型輸入特徵缺失資料
    site_df.dropna(subset=input_features, inplace=True)
    # 將逐時資料整理成每日平均，讓 sequence_length=90 代表過去 90 天
    site_df['date_only'] = site_df['date'].dt.date

    site_df = site_df.groupby('date_only', as_index=False)[input_features].mean()
    site_df['date'] = pd.to_datetime(site_df['date_only'])
    site_df = site_df[['date'] + input_features]

     # 資料不足 30 筆就跳過
    if len(site_df) < sequence_length:
        print(f"跳過 {site}，資料量不足")
        continue

    # 建立 30 列，對應未來 30 天預測
    site_pred = pd.DataFrame({'sitename': [site] * 30})



    values = site_df[input_features].values
    target_values = site_df[[target_feature]].values

 # 時間序列資料不能 random split，要按照時間前後切割

    train_size = int(len(values) * 0.8)


    if train_size <= sequence_length:
        print(f"跳過 {site}，訓練資料不足")
        continue

 # scaler 只用訓練集 fit，避免資料洩漏 data leakage

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(values[:train_size])

    target_scaler = MinMaxScaler()
    target_scaler.fit(target_values[:train_size])


    scaled = feature_scaler.transform(values)
    scaled_target = target_scaler.transform(target_values)

# ----------------------------
# 建立訓練集 train set
# many-to-one：過去 90 天多個污染物特徵 -> 下一天 AQI
# ----------------------------

    X_train, y_train = [], []
    for i in range(sequence_length, train_size):
        X_train.append(scaled[i-sequence_length:i, :])
        y_train.append(scaled_target[i, 0])


    X_train, y_train = np.array(X_train), np.array(y_train)

 # ----------------------------
 # 建立測試集 test set
 # 測試集用來畫「過去時間的模型預測線」
 # ----------------------------

    X_test = []
    test_dates = []

    for i in range(train_size, len(scaled)):
        X_test.append(scaled[i-sequence_length:i, :])
        test_dates.append(site_df['date'].iloc[i])

    X_test = np.array(X_test)

    model = Sequential()
    model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(64, activation='tanh'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=8,
        verbose=1,
        callbacks=[early_stop],
        validation_split=0.2,
        shuffle=False
    )

    # ----------------------------
    # 測試集預測：讓紅色預測線涵蓋過去時間
    # ----------------------------
    test_preds_scaled = model.predict(X_test, verbose=0)
    test_preds = target_scaler.inverse_transform(test_preds_scaled).flatten()

    # ----------------------------
    # 未來 30 天預測
    # ----------------------------
    last_seq = scaled[-sequence_length:]
    future_preds_scaled = []
    seq_input = last_seq.copy()
    target_idx = input_features.index(target_feature)

    # 取最近 30 天的多污染物變化模式，讓未來 PM2.5、PM10、O3、NO2 等欄位不要固定不動
    recent_pattern = scaled[-30:].copy()

    # alpha 越大，未來線越接近模型原始輸出；越小越平滑
    alpha = 0.9
    prev_pred = None

    for step in range(30):
        raw_pred = model.predict(
            seq_input.reshape((1, sequence_length, len(input_features))),
            verbose=0
        )[0, 0]

        # 避免模型輸出超出 MinMaxScaler 的合理範圍
        raw_pred = np.clip(raw_pred, 0, 1)

        # 保留一點平滑，但不要壓得太平
        if prev_pred is not None:
            pred = alpha * raw_pred + (1 - alpha) * prev_pred
        else:
            pred = raw_pred

        prev_pred = pred
        future_preds_scaled.append(pred)

        # 不要用最後一天完全複製；改用最近 30 天的污染物型態
        next_row = recent_pattern[step % len(recent_pattern)].copy()

        # AQI 欄位仍然放入模型預測值
        next_row[target_idx] = pred

        seq_input = np.vstack([seq_input[1:], next_row])

# 這裡才把 scaled 預測值轉回 AQI
    future_preds = target_scaler.inverse_transform(
        np.array(future_preds_scaled).reshape(-1, 1)
    ).flatten()

    future_preds = calibrate_future_volatility(
        future_preds=future_preds,
        test_preds=test_preds,
        history_values=site_df[target_feature].values
    )

    # 未來 30 天日期
    last_date = site_df['date'].max()

    if pd.isna(last_date):
        print(f"跳過 {site}，日期欄位無效，無法產生未來日期")
        continue

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=30,
        freq='D'
    )

    # 儲存未來 30 天預測結果
    site_pred[f"{target_feature}_pred"] = future_preds
    site_pred['date'] = future_dates

    # ----------------------------
    # 將測試集預測 + 未來預測接成同一條紅線
    # ----------------------------
    prediction_dates = list(test_dates) + list(future_dates)
    prediction_values = np.concatenate([test_preds, future_preds])

    # ----------------------------
    # 生成趨勢圖 PNG
    # ----------------------------
    fig, ax = plt.subplots(figsize=(16, 5))

    # 藍線：歷史真實 AQI
    ax.plot(
        site_df['date'],
        site_df[target_feature],
        label='Observed AQI',
        color='blue',
        linewidth=1.2
    )

# 紅線：測試集預測 + 未來 30 天預測
    ax.plot(
        prediction_dates,
        prediction_values,
        label='Model Prediction / Forecast',
        color='red',
        linewidth=1.2
    )

# 灰色虛線：訓練集 / 測試集分界點
    ax.axvline(
        site_df['date'].iloc[train_size],
        linestyle='--',
        color='gray',
        label='Train/Test Split'
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")

    site_title = site_name_en.get(site, site)
    ax.set_title(f"AQI Forecast - {site_title}")

# 讓 X 軸每個月顯示一次，比較不會擠在一起
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 讓 X 軸多留一點空間
    ax.set_xlim(
        site_df['date'].min(),
        future_dates.max() + pd.Timedelta(days=5)
    )

    plt.xticks(rotation=30)
    ax.legend()
    fig.tight_layout()

    fig.savefig(os.path.join(plots_folder, f"{site}_aqi.png"), dpi=150)
    plt.close(fig)

    predictions.append(site_pred)

pred_df = pd.concat(predictions, ignore_index=True)

 #點顏色

def get_aqi_color(aqi_val):
     if aqi_val <= 50:
         return 'green'
     elif aqi_val <= 100:
         return 'gold'
     elif aqi_val <= 150:
         return 'darkorange'
     elif aqi_val <= 200:
         return 'red'
     else:
         return 'darkred'


 #在 Folium 地圖前加入圖片轉 base64 函式
def image_to_base64(img_path):
    if not os.path.exists(img_path):
         return None

    with open(img_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")

    return encoded

def fmt_value(value, unit=""):
    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except:
        pass

    if isinstance(value, float):
        return f"{value:.2f}{unit}"

    return f"{value}{unit}"


def make_realtime_info_html(info):
    rows = []

    display_items = [
        ('publishtime', '發布時間', ''),
        ('status', '狀態', ''),
        ('pollutant', '主要污染物', ''),
        ('pm2.5', 'PM2.5', ' μg/m³'),
        ('pm10', 'PM10', ' μg/m³'),
        ('so2', 'SO₂', ' ppb'),
        ('co', 'CO', ' ppm'),
        ('o3', 'O₃', ' ppb'),
        ('o3_8hr', 'O₃ 8hr', ' ppb'),
        ('no2', 'NO₂', ' ppb'),
        ('nox', 'NOx', ' ppb'),
        ('no', 'NO', ' ppb'),
        ('windspeed', '風速', ' m/s'),
        ('winddirec', '風向', '°')
    ]

    for key, label, unit in display_items:
        if key in info:
            rows.append(
                f"""
                <tr>
                    <td style="padding:3px 8px; border-bottom:1px solid #ddd;">{label}</td>
                    <td style="padding:3px 8px; border-bottom:1px solid #ddd;">{fmt_value(info.get(key), unit)}</td>
                </tr>
                """
            )

    return f"""
    <table style="border-collapse:collapse; font-size:13px; margin-top:8px;">
        {''.join(rows)}
    </table>
    """

# ----------------------------
# 3️⃣ Folium 地圖
# ----------------------------
m = folium.Map(location=[23.6,120.9], zoom_start=8)

for sitename, info in site_coords.items():
    lat, lon, aqi_now = info['latitude'], info['longitude'], info['aqi']
    img_path = os.path.join(plots_folder, f"{sitename}_aqi.png")
    img_b64 = image_to_base64(img_path)

    realtime_info_html = make_realtime_info_html(info)

    if img_b64 is not None:
        html = f"""
        <div style="width:780px;">
            <h3 style="margin:4px 0;">{sitename}</h3>
            <div style="font-size:16px; margin-bottom:6px;">
                <b>Realtime AQI:</b> {aqi_now}
            </div>

            <img src="data:image/png;base64,{img_b64}" width="760">

            <h4 style="margin:10px 0 4px 0;">即時空氣污染資訊</h4>
            {realtime_info_html}
        </div>
        """
    else:
        html = f"""
        <div style="width:420px;">
            <h3 style="margin:4px 0;">{sitename}</h3>
            <div style="font-size:16px; margin-bottom:6px;">
                <b>Realtime AQI:</b> {aqi_now}
            </div>

            <p>Prediction chart not found.</p>

            <h4 style="margin:10px 0 4px 0;">即時空氣污染資訊</h4>
            {realtime_info_html}
        </div>
        """

    popup = folium.Popup(
        folium.IFrame(html=html, width=820, height=650),
        max_width=820
    )

    folium.CircleMarker(
        location=[lat, lon],
        radius=np.sqrt(aqi_now),
        color=get_aqi_color(aqi_now),
        fill=True,
        fill_color=get_aqi_color(aqi_now),
        fill_opacity=0.6,
        popup=popup,
        tooltip=folium.Tooltip(html, sticky=True)
    ).add_to(m)

map_file = os.path.join(os.getcwd(), "future_air_quality_map.html")
m.save(map_file)
webbrowser.open(f"file:///{map_file}")
