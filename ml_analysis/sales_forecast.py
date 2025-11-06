"""
Sales Forecasting - Time Series Analysis
Predicts future sales trends
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# Configuration
DATA_FILE = '../data/data.csv'
OUTPUT_DIR = '../ml_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading e-commerce data...")
df = pd.read_csv(DATA_FILE, encoding='latin-1')

# Data cleaning
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

df_clean = df[
    (df['Quantity'] > 0) & 
    (df['UnitPrice'] > 0) & 
    (df['TotalAmount'] > 0)
].copy()

print(f"Clean records: {len(df_clean)}")

# Aggregate monthly sales
print("\nAggregating monthly sales...")
df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')

monthly_sales = df_clean.groupby('YearMonth').agg({
    'TotalAmount': 'sum',
    'InvoiceNo': 'nunique',
    'CustomerID': 'nunique'
}).reset_index()

monthly_sales.columns = ['YearMonth', 'Revenue', 'Orders', 'Customers']
monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)

print(f"Months of data: {len(monthly_sales)}")

# Add lag features for forecasting
monthly_sales['Month_Index'] = range(len(monthly_sales))
monthly_sales['Revenue_Lag1'] = monthly_sales['Revenue'].shift(1)
monthly_sales['Revenue_Lag2'] = monthly_sales['Revenue'].shift(2)
monthly_sales['Revenue_Lag3'] = monthly_sales['Revenue'].shift(3)
monthly_sales['Moving_Avg_3M'] = monthly_sales['Revenue'].rolling(window=3).mean()

# Prepare training data (remove rows with NaN from lag features)
train_data = monthly_sales.dropna()

if len(train_data) >= 5:
    # Train linear regression model
    print("\nTraining forecasting model...")
    X = train_data[['Month_Index', 'Revenue_Lag1', 'Revenue_Lag2', 'Moving_Avg_3M']]
    y = train_data['Revenue']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict on existing data
    train_data['Predicted_Revenue'] = model.predict(X)
    train_data['Accuracy'] = 100 * (1 - abs(train_data['Revenue'] - train_data['Predicted_Revenue']) / train_data['Revenue'])
    
    # Generate 3-month forecast
    print("\nGenerating 3-month forecast...")
    forecasts = []
    
    # Get last month's data
    last_month_idx = monthly_sales['Month_Index'].max()
    last_revenue = monthly_sales['Revenue'].iloc[-1]
    last_3_revenues = monthly_sales['Revenue'].tail(3).values
    
    # Calculate trend from last 6 months
    recent_data = monthly_sales.tail(6)
    trend_coef = np.polyfit(recent_data['Month_Index'], recent_data['Revenue'], 1)[0]
    
    print(f"  Recent trend: ${trend_coef:,.0f} per month")
    
    # Generate forecast with trend
    for i in range(1, 4):
        month_idx = last_month_idx + i
        
        # Use actual forecast formula with trend
        if i == 1:
            lag1 = last_revenue
            lag2 = last_3_revenues[-2]
            moving_avg = np.mean(last_3_revenues)
        else:
            # Use previously predicted values as lags
            lag1 = forecasts[-1]['Predicted_Revenue']
            if i == 2:
                lag2 = last_revenue
            else:
                lag2 = forecasts[-2]['Predicted_Revenue']
            # Update moving average with predictions
            recent_vals = list(last_3_revenues[-(3-i):]) + [f['Predicted_Revenue'] for f in forecasts]
            moving_avg = np.mean(recent_vals[-3:])
        
        X_future = [[month_idx, lag1, lag2, moving_avg]]
        base_pred = model.predict(X_future)[0]
        
        # Apply trend to make forecast more realistic
        pred_with_trend = base_pred + (trend_coef * i * 0.7)  # Dampen trend slightly
        
        # Get the next period (month/year)
        last_period = pd.Period(monthly_sales['YearMonth'].iloc[-1], freq='M')
        future_period = last_period + i
        
        forecasts.append({
            'YearMonth': str(future_period),
            'Revenue': None,
            'Predicted_Revenue': max(0, pred_with_trend),  # Ensure non-negative
            'Type': 'Forecast'
        })
    
    # Combine historical and forecast
    historical = train_data[['YearMonth', 'Revenue', 'Predicted_Revenue']].copy()
    historical['Type'] = 'Historical'
    
    df_forecast = pd.DataFrame(forecasts)
    combined = pd.concat([historical, df_forecast], ignore_index=True)
    
    # Save results
    output_file = os.path.join(OUTPUT_DIR, 'sales_forecast.csv')
    combined.to_csv(output_file, index=False)
    print(f"\n✅ Sales forecast saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SALES FORECASTING SUMMARY")
    print("="*60)
    
    print("\nModel Performance:")
    print(f"  R² Score: {model.score(X, y):.4f}")
    print(f"  Mean Accuracy: {train_data['Accuracy'].mean():.2f}%")
    
    print("\nHistorical Sales (Last 6 Months):")
    print(historical.tail(6)[['YearMonth', 'Revenue', 'Predicted_Revenue']].to_string(index=False))
    
    print("\n3-Month Forecast:")
    print(df_forecast[['YearMonth', 'Predicted_Revenue']].to_string(index=False))
    
    print("\n✅ Sales Forecasting Complete!")
else:
    print("⚠️ Insufficient data for forecasting (need at least 5 months)")
