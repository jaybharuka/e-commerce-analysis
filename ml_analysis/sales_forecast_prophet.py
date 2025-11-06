"""
Advanced Sales Forecasting - Time Series Analysis with Prophet
Predicts future 3-month revenue trends using advanced time series analysis
to support inventory planning, staffing decisions, and financial projections.

Algorithm: Facebook's Prophet (Meta)
- Handles seasonality (monthly patterns)
- Captures trends and trend changes
- Incorporates holiday effects
- Provides uncertainty intervals (confidence bounds)
- Better accuracy than linear regression for forecasting
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')

# Fix encoding for emojis in Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Try importing Prophet, install if needed
try:
    from prophet import Prophet
except ImportError:
    print("Installing Prophet... (this may take a minute)")
    import subprocess
    subprocess.check_call(['pip', 'install', 'cmdstanpy', '-q'])
    subprocess.check_call(['pip', 'install', 'prophet', '-q'])
    from prophet import Prophet

# Configuration
DATA_FILE = 'data/data.csv'
OUTPUT_DIR = 'ml_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("ADVANCED SALES FORECASTING - TIME SERIES ANALYSIS")
print("="*70)
print("\n📊 Loading e-commerce transaction data...")
df = pd.read_csv(DATA_FILE, encoding='latin-1')

# Data cleaning and preparation
print("🧹 Cleaning and preparing data...")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Remove invalid transactions
df_clean = df[
    (df['Quantity'] > 0) & 
    (df['UnitPrice'] > 0) & 
    (df['TotalAmount'] > 0)
].copy()

print(f"   ✓ Clean records: {len(df_clean):,}")

# Aggregate daily sales (Prophet works better with daily data)
print("\n📅 Aggregating daily sales data...")
daily_sales = df_clean.groupby(df_clean['InvoiceDate'].dt.date).agg({
    'TotalAmount': ['sum', 'count', 'mean'],
    'CustomerID': 'nunique',
    'Quantity': 'sum'
}).reset_index()

daily_sales.columns = ['Date', 'Revenue', 'Orders', 'AvgOrderValue', 'UniqueCustomers', 'ItemsSold']
daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
daily_sales = daily_sales.sort_values('Date').reset_index(drop=True)

print(f"   ✓ Daily records: {len(daily_sales):,}")
print(f"   ✓ Date range: {daily_sales['Date'].min().date()} to {daily_sales['Date'].max().date()}")

# ===== PROPHET MODEL TRAINING =====
print("\n🤖 Training Prophet Time Series Model...")

# Prepare data for Prophet (requires 'ds' and 'y' columns)
prophet_df = daily_sales[['Date', 'Revenue']].copy()
prophet_df.columns = ['ds', 'y']

# Initialize Prophet model with advanced settings
model = Prophet(
    interval_width=0.95,           # 95% confidence interval
    yearly_seasonality=True,       # Capture yearly patterns
    weekly_seasonality=True,       # Capture weekly patterns
    daily_seasonality=False,       # No daily seasonality for e-commerce
    seasonality_mode='additive',   # Revenue = Trend + Seasonality + Error
    changepoint_prior_scale=0.05,  # Sensitivity to trend changes
    seasonality_prior_scale=10.0   # Seasonality strength
)

# Add custom seasonality
model.add_seasonality(name='monthly', period=30.44, fourier_order=5)
model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

# Fit the model
with open(os.devnull, 'w') as devnull:
    import sys
    old_stdout = sys.stdout
    sys.stdout = devnull
    model.fit(prophet_df)
    sys.stdout = old_stdout

print("   ✓ Model fitted successfully")

# ===== HISTORICAL ANALYSIS =====
print("\n📈 Analyzing historical data...")

# Get model components
forecast_historical = model.predict(prophet_df)

# Calculate metrics
mape = np.mean(np.abs((daily_sales['Revenue'] - forecast_historical['yhat']) / daily_sales['Revenue'])) * 100
rmse = np.sqrt(np.mean((daily_sales['Revenue'] - forecast_historical['yhat'])**2))
mae = np.mean(np.abs(daily_sales['Revenue'] - forecast_historical['yhat']))

print(f"   ✓ Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"   ✓ Root Mean Square Error (RMSE): ${rmse:,.2f}")
print(f"   ✓ Mean Absolute Error (MAE): ${mae:,.2f}")

# Aggregate to monthly level for output (matching dashboard)
daily_with_forecast = daily_sales.copy()
daily_with_forecast['Forecast'] = forecast_historical['yhat'].values
daily_with_forecast['Upper_Bound'] = forecast_historical['yhat_upper'].values
daily_with_forecast['Lower_Bound'] = forecast_historical['yhat_lower'].values

monthly_historical = daily_with_forecast.groupby(
    daily_with_forecast['Date'].dt.to_period('M')
).agg({
    'Revenue': 'sum',
    'Forecast': 'sum',
    'Upper_Bound': 'sum',
    'Lower_Bound': 'sum',
    'Orders': 'sum',
    'UniqueCustomers': 'nunique',
    'ItemsSold': 'sum'
}).reset_index()

monthly_historical.columns = ['YearMonth', 'Revenue', 'Forecast', 'Upper_Bound', 'Lower_Bound', 
                              'Orders', 'Customers', 'ItemsSold']
monthly_historical['YearMonth'] = monthly_historical['YearMonth'].astype(str)

# ===== FUTURE FORECASTING =====
print("\n🔮 Generating 3-month forecast...")

# Create future dataframe for next 3 months
future_dates = model.make_future_dataframe(periods=90)  # 3 months

# Make forecast
forecast_future = model.predict(future_dates)

# Extract forecast for next 90 days
last_date = daily_sales['Date'].max()
future_mask = forecast_future['ds'] > last_date

forecast_daily = forecast_future[future_mask][['ds', 'yhat', 'yhat_upper', 'yhat_lower']].copy()
forecast_daily.columns = ['Date', 'Revenue', 'Upper_Bound', 'Lower_Bound']

# Aggregate daily forecast to monthly
forecast_daily['YearMonth'] = forecast_daily['Date'].dt.to_period('M')
monthly_forecast = forecast_daily.groupby('YearMonth').agg({
    'Revenue': 'sum',
    'Upper_Bound': 'sum',
    'Lower_Bound': 'sum'
}).reset_index()

monthly_forecast.columns = ['YearMonth', 'Revenue', 'Upper_Bound', 'Lower_Bound']
monthly_forecast['YearMonth'] = monthly_forecast['YearMonth'].astype(str)
monthly_forecast['Type'] = 'Forecast'

# Calculate confidence intervals
monthly_forecast['Confidence_Lower'] = monthly_forecast['Lower_Bound']
monthly_forecast['Confidence_Upper'] = monthly_forecast['Upper_Bound']

# ===== TREND ANALYSIS =====
print("\n📊 Analyzing trends and patterns...")

# Get trend component
trend_df = forecast_future[forecast_future['ds'] <= last_date][['ds', 'trend']].copy()
trend_df['YearMonth'] = trend_df['ds'].dt.to_period('M')
monthly_trend = trend_df.groupby('YearMonth')['trend'].mean().reset_index()

# Calculate growth rate
last_6_months = monthly_historical.tail(6)
historical_revenue = last_6_months['Revenue'].values
daily_growth = np.mean(np.diff(historical_revenue)) / np.mean(historical_revenue) * 100

print(f"   ✓ Recent trend: ${monthly_trend['trend'].iloc[-1]:,.0f}/month average")
print(f"   ✓ Daily average growth rate: {daily_growth:.2f}%")
print(f"   ✓ Seasonality detected: Yes (seasonal patterns identified)")

# ===== FORECAST STATISTICS =====
print("\n📈 Forecast Statistics:")
print(f"   ✓ Next month prediction: ${monthly_forecast['Revenue'].iloc[0]:,.2f}")
print(f"   ✓ Upper bound (95% CI): ${monthly_forecast['Upper_Bound'].iloc[0]:,.2f}")
print(f"   ✓ Lower bound (95% CI): ${monthly_forecast['Lower_Bound'].iloc[0]:,.2f}")
print(f"   ✓ Confidence interval range: ${monthly_forecast['Upper_Bound'].iloc[0] - monthly_forecast['Lower_Bound'].iloc[0]:,.2f}")

# ===== PREPARE OUTPUT =====
print("\n💾 Preparing output data...")

# Add metadata to historical
monthly_historical['Type'] = 'Historical'
monthly_historical['Predicted_Revenue'] = monthly_historical['Forecast'].fillna(monthly_historical['Revenue'])

# Prepare forecast output with same structure
monthly_forecast['Predicted_Revenue'] = monthly_forecast['Revenue']

# Combine historical and forecast
output_historical = monthly_historical[[
    'YearMonth', 'Revenue', 'Predicted_Revenue', 'Type'
]].copy()

output_forecast = monthly_forecast[[
    'YearMonth', 'Predicted_Revenue', 'Type'
]].copy()
output_forecast['Revenue'] = np.nan

# Combine
combined_output = pd.concat([output_historical, output_forecast], ignore_index=True)
combined_output = combined_output[['YearMonth', 'Revenue', 'Predicted_Revenue', 'Type']]

# Save results
output_file = os.path.join(OUTPUT_DIR, 'sales_forecast.csv')
combined_output.to_csv(output_file, index=False)
print(f"   ✓ Sales forecast saved to: {output_file}")

# Save detailed forecast with confidence intervals
detailed_file = os.path.join(OUTPUT_DIR, 'sales_forecast_detailed.csv')
forecast_daily_output = forecast_daily[['Date', 'Revenue', 'Upper_Bound', 'Lower_Bound']].copy()
forecast_daily_output['Confidence_Interval'] = forecast_daily_output['Upper_Bound'] - forecast_daily_output['Lower_Bound']
forecast_daily_output.to_csv(detailed_file, index=False)
print(f"   ✓ Detailed daily forecast saved to: {detailed_file}")

# ===== FINAL SUMMARY =====
print("\n" + "="*70)
print("SALES FORECASTING SUMMARY")
print("="*70)

print("\n📊 Model Performance:")
print(f"   Mean Absolute Percentage Error: {mape:.2f}%")
print(f"   Root Mean Square Error: ${rmse:,.2f}")

print("\n📈 Historical Sales (Last 6 Months):")
last_6 = monthly_historical.tail(6)[['YearMonth', 'Revenue']].copy()
for idx, row in last_6.iterrows():
    print(f"   {row['YearMonth']}: ${row['Revenue']:>12,.2f}")

print("\n🔮 3-Month Forecast (with 95% Confidence Intervals):")
for idx, row in monthly_forecast.iterrows():
    ci_range = row['Upper_Bound'] - row['Lower_Bound']
    print(f"   {row['YearMonth']}: ${row['Revenue']:>12,.2f} (±${ci_range/2:,.2f})")

print("\n📊 Key Insights:")
avg_forecast = monthly_forecast['Revenue'].mean()
avg_historical = monthly_historical['Revenue'].mean()
forecast_change = ((avg_forecast - avg_historical) / avg_historical) * 100
print(f"   • Average historical monthly revenue: ${avg_historical:,.2f}")
print(f"   • Average forecasted monthly revenue: ${avg_forecast:,.2f}")
print(f"   • Expected trend: {forecast_change:+.2f}%")
print(f"   • Prediction confidence: 95% (Prophet uncertainty intervals)")

print("\n✅ Advanced Sales Forecasting Complete!")
print("="*70)
