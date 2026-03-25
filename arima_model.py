import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# LOAD DATA
df = pd.read_csv("daily_sales_data.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df.set_index("date", inplace=True)

# TRAIN TEST SPLIT
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# BUILD ARIMA MODEL
model = ARIMA(train["sales"], order=(1,1,1))
model_fit = model.fit()

print(model_fit.summary())

# MAKE PREDICTIONS
forecast = model_fit.forecast(steps=len(test))

# EVALUATION
mae = mean_absolute_error(test["sales"], forecast)
print("\nMean Absolute Error:", mae)

# PLOT RESULTS
plt.figure(figsize=(12,6))
plt.plot(train.index, train["sales"], label="Train")
plt.plot(test.index, test["sales"], label="Actual")
plt.plot(test.index, forecast, label="Forecast")
plt.legend()
plt.title("ARIMA Forecast vs Actual")
plt.show()