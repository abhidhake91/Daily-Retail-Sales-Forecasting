import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# LOAD DATA
df = pd.read_csv("daily_sales_data.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df.set_index("date", inplace=True)

print(df.head())

# BASIC PLOT
plt.figure(figsize=(12,5))
plt.plot(df["sales"])
plt.title("Daily Sales Over Time")
plt.show()

# ROLLING AVERAGE
rolling_mean = df["sales"].rolling(window=30).mean()

plt.figure(figsize=(12,5))
plt.plot(df["sales"], label="Original")
plt.plot(rolling_mean, label="30 Day Rolling Mean", color="red")
plt.legend()
plt.title("Rolling Mean Analysis")
plt.show()

# SEASONAL DECOMPOSITION
decomposition = seasonal_decompose(df["sales"], model="additive", period=7)

decomposition.plot()
plt.show()