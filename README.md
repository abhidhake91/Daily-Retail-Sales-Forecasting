# 📈 Daily Retail Sales Forecasting

## 📌 Overview
This project predicts future retail sales using time series forecasting models (ARIMA and SARIMA).

## ❗ Problem
Retail businesses struggle to predict future demand, leading to inventory and planning issues.

## 💡 Solution
A data-driven forecasting system that:
- Analyzes historical sales
- Detects trends and seasonality
- Predicts future sales
- Provides confidence intervals

## ⚙️ Workflow
1. Generate synthetic data → generate_sales_data.py  
2. Perform EDA → eda_analysis.py  
3. Check stationarity → stationarity_check.py  
4. Train ARIMA model → arima_model.py  
5. Train SARIMA model → sarima_model.py  
6. Forecast future sales  

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib
- Statsmodels
- Scikit-learn

## 📊 Models Used
- ARIMA (trend-based forecasting)
- SARIMA (trend + seasonality)

## 📈 Output
- Forecast vs Actual comparison
- 30-day future forecast
- Confidence intervals

## 🚀 How to Run

```bash
pip install -r requirements.txt
python generate_sales_data.py
python eda_analysis.py
python stationarity_check.py
python arima_model.py
python sarima_model.py
```

## 📷 Results

### Forecast vs Actual
![Forecast](forecast_vs_actual.png)

### 30-Day Forecast
![Future Forecast](future_forecast.png)

## 🔐 Note
Synthetic dataset used. No real business data involved.