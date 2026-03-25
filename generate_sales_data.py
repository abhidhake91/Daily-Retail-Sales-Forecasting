import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SETTINGS
np.random.seed(42)
days = 730  # 2 years
dates = pd.date_range(start="2023-01-01", periods=days)

# COMPONENTS

# Trend (gradual growth)
trend = np.linspace(200, 400, days)

# Weekly seasonality (higher sales on weekends)
weekly_seasonality = 20 * np.sin(2 * np.pi * dates.dayofweek / 7)

# Random noise
noise = np.random.normal(0, 15, days)

# Occasional spikes (promotions)
spikes = np.random.choice([0, 100], size=days, p=[0.95, 0.05])

# Final sales
sales = trend + weekly_seasonality + noise + spikes
sales = np.round(sales)

# CREATE DATAFRAME
df = pd.DataFrame({
    "date": dates,
    "sales": sales
})

df.to_csv("daily_sales_data.csv", index=False)

print("✅ daily_sales_data.csv created successfully!")

# QUICK VISUAL CHECK
plt.figure(figsize=(12,5))
plt.plot(df["date"], df["sales"])
plt.title("Daily Retail Sales (Simulated)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()