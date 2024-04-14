#%%
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

#%%
# prophetに向けた日付カラムの処理
def ds_column_cleaning(data):
    data = data.rename(columns = {
        "datetime": "ds" 
    })
    data['ds'] = pd.to_datetime(data['ds'])
    return data

# 日本の祝日の日付を取得する関数
def is_japanese_holiday(date):
    import jpholiday
    return int(jpholiday.is_holiday(date))

# porphet用にholiday dataframeを作成する
def make_holiday_dataframe_for_prophet(data):
    data['holiday'] = data['ds'].apply(lambda x: is_japanese_holiday(x))
    data = data[data["holiday"] == 1]
    data = data[["holiday","ds"]]
    data["holiday"] = "jp_holiday"
    return data

def run_prophet(data, data_holiday, START_TEST_DATE, SELECT_COLUMNS):

    # trainとtestに分割
    data_train = data[data["ds"] < START_TEST_DATE]

    # 前処理
    data_train = data_train[["ds","y"] + SELECT_COLUMNS]
    data = data[["ds"] + SELECT_COLUMNS]

    # 学習実装
    model = Prophet(holidays = data_holiday)
    for external_variable in SELECT_COLUMNS:
        model.add_regressor(external_variable)

    model.fit(data_train)

    # 予測
    data_forecast = model.predict(data)

    return model, data_forecast

def forecast_close_day(data):
    data_forecast_close_day = data[data["close"] == 1]
    data_forecast_close_day["y"] = 0
    data_forecast_close_day = data_forecast_close_day[["ds","y"]]
    data_forecast_close_day = data_forecast_close_day.rename(columns = {
            "y": "yhat" 
        })
    return data_forecast_close_day

def make_forecast_dataflame(data_forecast, data_forecast_close_day):
    data_forecast = pd.concat([data_forecast, data_forecast_close_day], ignore_index=True)
    data_forecast = data_forecast.sort_values(by = "ds")
    return data_forecast

def chk_concat_forecast_n_closing_forecast(data_forecast, data):
    assert len(data_forecast) == len(data), "結合がうまくいっていません"


# %%
data_train = pd.read_csv("../data/train.csv")
data_test = pd.read_csv("../data/test.csv")
data = pd.concat([data_train, data_test], ignore_index=True)

# データの前処理
data = ds_column_cleaning(data)
# 古いdsで料金区分カラムの欠損が見られるため、欠損のないdsに絞る。データポイント的にも問題なし
data = data[data["ds"] >= "2012-04-01"]

# prophet用にholiday dataframe作成する
data_holiday = make_holiday_dataframe_for_prophet(data)

# 休業日は引越し数が必ず0になるため
data_for_model = data[data["close"] == 0]

#%%
# prophet実装 × 予測
START_TEST_DATE = "2016-04-01"
SELECT_COLUMNS = []
model, data_forecast = run_prophet(data_for_model, data_holiday, START_TEST_DATE, SELECT_COLUMNS)
data_forecast_close_day = forecast_close_day(data)
data_forecast = make_forecast_dataflame(data_forecast, data_forecast_close_day)
chk_concat_forecast_n_closing_forecast(data_forecast, data)


#%%
fig1 = model.plot(data_forecast, figsize=(20, 12)) # 結果のプロット#1

#%%
fig2 = model.plot_components(data_forecast) # 結果のプロット#2



# %%
data_for_model[data_for_model["price_am"] == -1]

# %%
