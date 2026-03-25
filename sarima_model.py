import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# LOAD DATA

df = pd.read_csv("daily_sales_data.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
df.set_index("date", inplace=True)

# Ensure daily frequency
df = df.asfreq("D")

# TRAIN TEST SPLIT

train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

# BUILD SARIMA MODEL

model = SARIMAX(
    train["sales"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit()

print(results.summary())

# TEST FORECAST

forecast = results.get_forecast(steps=len(test))
forecast_mean = forecast.predicted_mean

mae = mean_absolute_error(test["sales"], forecast_mean)
print("\nSARIMA Mean Absolute Error:", mae)

# PLOT TRAIN VS TEST FORECAST

plt.figure(figsize=(12,6))
plt.plot(train.index, train["sales"], label="Train")
plt.plot(test.index, test["sales"], label="Actual")
plt.plot(test.index, forecast_mean, label="SARIMA Forecast", color="green")
plt.title("SARIMA Forecast vs Actual")
plt.legend()
plt.show()

# FUTURE 30 DAY FORECAST

future_steps = 30

future_forecast = results.get_forecast(steps=future_steps)

forecast_mean_future = future_forecast.predicted_mean
forecast_ci = future_forecast.conf_int()

# Generate future dates
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=future_steps + 1, freq="D")[1:]

# PLOT FUTURE FORECAST

plt.figure(figsize=(12,6))

plt.plot(df.index, df["sales"], label="Historical Data")
plt.plot(future_dates, forecast_mean_future, color="green", label="30 Day Forecast")

plt.fill_between(
    future_dates,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="green",
    alpha=0.2,
    label="Confidence Interval"
)

plt.title("30 Day Sales Forecast (SARIMA)")
plt.legend()
plt.show()

# EXPORT TEST FORECAST TO CSV

test_forecast_df = pd.DataFrame({
    "date": test.index,
    "actual_sales": test["sales"].values,
    "predicted_sales": forecast_mean.values
})

test_forecast_df.to_csv("test_forecast_results.csv", index=False)

print("✅ Test forecast exported to test_forecast_results.csv")


# EXPORT FUTURE FORECAST TO CSV


future_forecast_df = pd.DataFrame({
    "date": future_dates,
    "forecast_sales": forecast_mean_future.values,
    "lower_bound": forecast_ci.iloc[:, 0].values,
    "upper_bound": forecast_ci.iloc[:, 1].values
})

future_forecast_df.to_csv("future_30_day_forecast.csv", index=False)

print("✅ Future 30 day forecast exported to future_30_day_forecast.csv")