import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# LOAD DATA
df = pd.read_csv("daily_sales_data.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df.set_index("date", inplace=True)

# ADF TEST FUNCTION
def adf_test(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")

# RUN TEST ON ORIGINAL SERIES
print("=== ADF Test on Original Sales Data ===")
adf_test(df["sales"])

# FIRST DIFFERENCING
df["sales_diff"] = df["sales"].diff()

plt.figure(figsize=(12,5))
plt.plot(df["sales_diff"])
plt.title("First Order Differenced Series")
plt.show()

print("\n=== ADF Test on Differenced Data ===")
adf_test(df["sales_diff"].dropna())