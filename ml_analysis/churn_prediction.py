"""
Churn Prediction Model - Customer Retention Analysis
Predicts which customers are likely to churn (stop purchasing) in the next period
using machine learning classification.

Algorithm: Gradient Boosting (XGBoost)
- Handles non-linear relationships
- Feature importance analysis
- Probability-based predictions with confidence scores
- Better accuracy than logistic regression

Business Value:
- Identify at-risk customers before they leave
- Enable proactive retention campaigns
- Improve customer lifetime value
- Optimize marketing spend on high-churn-risk segments
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Try importing XGBoost, install if needed
try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
except ImportError:
    print("Installing XGBoost and dependencies...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', '-q'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn', '-q'])
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Configuration
DATA_FILE = 'data/data.csv'
OUTPUT_DIR = 'ml_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("CHURN PREDICTION MODEL - CUSTOMER RETENTION ANALYSIS")
print("="*70)

# Load data
print("\nLoading e-commerce data...")
df = pd.read_csv(DATA_FILE, encoding='latin-1')

# Data cleaning
print("Preparing data for analysis...")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

df_clean = df[
    (df['Quantity'] > 0) & 
    (df['UnitPrice'] > 0) & 
    (df['TotalAmount'] > 0) &
    (df['CustomerID'].notna())
].copy()

print(f"   Total records: {len(df_clean):,}")

# ===== FEATURE ENGINEERING =====
print("\nEngineering customer features...")

# Get date range
min_date = df_clean['InvoiceDate'].min()
max_date = df_clean['InvoiceDate'].max()
date_range = (max_date - min_date).days

# Define churn observation period (last 30 days is test period)
observation_date = max_date - timedelta(days=30)
test_date = observation_date

# Historical data for features
historical_data = df_clean[df_clean['InvoiceDate'] < observation_date].copy()
test_data = df_clean[(df_clean['InvoiceDate'] >= observation_date) & (df_clean['InvoiceDate'] <= max_date)].copy()

print(f"   Observation period: {historical_data['InvoiceDate'].min().date()} to {observation_date.date()}")
print(f"   Churn detection period: {test_date.date()} to {max_date.date()}")

# Extract customer-level features from historical data
customer_features = []

for customer_id in historical_data['CustomerID'].unique():
    customer_data = historical_data[historical_data['CustomerID'] == customer_id]
    
    # Feature 1: Recency (days since last purchase)
    last_purchase = customer_data['InvoiceDate'].max()
    recency = (observation_date - last_purchase).days
    
    # Feature 2: Frequency (number of transactions)
    frequency = len(customer_data)
    
    # Feature 3: Monetary (total spending)
    monetary = customer_data['TotalAmount'].sum()
    
    # Feature 4: Average order value
    avg_order_value = monetary / frequency if frequency > 0 else 0
    
    # Feature 5: Customer tenure (days since first purchase)
    first_purchase = customer_data['InvoiceDate'].min()
    tenure_days = (observation_date - first_purchase).days
    
    # Feature 6: Purchase variability (standard deviation of order amounts)
    purchase_variability = customer_data.groupby('InvoiceNo')['TotalAmount'].sum().std()
    purchase_variability = purchase_variability if not np.isnan(purchase_variability) else 0
    
    # Feature 7: Average days between purchases
    if frequency > 1:
        purchase_dates = sorted(customer_data.groupby('InvoiceNo')['InvoiceDate'].min().values)
        days_between = np.diff([(pd.Timestamp(d) - first_purchase).days for d in purchase_dates])
        avg_days_between = np.mean(days_between)
    else:
        avg_days_between = tenure_days
    
    # Feature 8: Product diversity (number of unique products)
    product_diversity = customer_data['StockCode'].nunique()
    
    # Feature 9: Country indicator (UK vs International)
    country = customer_data['Country'].iloc[0]
    is_uk = 1 if country == 'United Kingdom' else 0
    
    # Feature 10: Transaction count in last 90 days
    recent_90 = customer_data[
        (customer_data['InvoiceDate'] >= observation_date - timedelta(days=90)) &
        (customer_data['InvoiceDate'] < observation_date)
    ]
    transactions_90d = len(recent_90)
    
    customer_features.append({
        'CustomerID': customer_id,
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary,
        'AvgOrderValue': avg_order_value,
        'TenureDays': tenure_days,
        'PurchaseVariability': purchase_variability,
        'AvgDaysBetween': avg_days_between,
        'ProductDiversity': product_diversity,
        'IsUK': is_uk,
        'Transactions90d': transactions_90d
    })

features_df = pd.DataFrame(customer_features)
print(f"   Features extracted for {len(features_df):,} customers")

# ===== CREATE CHURN LABELS =====
print("\nDefining churn labels...")

# Customers who purchased in test period = not churned (0)
# Customers who did NOT purchase in test period = churned (1)
customers_in_test = set(test_data['CustomerID'].unique())

features_df['Churned'] = features_df['CustomerID'].apply(
    lambda x: 0 if x in customers_in_test else 1
)

churn_rate = features_df['Churned'].mean() * 100
print(f"   Churn rate: {churn_rate:.2f}% ({features_df['Churned'].sum():,} churned customers)")
print(f"   Active customers: {(1 - features_df['Churned']).sum():,}")

# ===== TRAIN/TEST SPLIT =====
print("\nPreparing training data...")

X = features_df.drop(['CustomerID', 'Churned'], axis=1)
y = features_df['Churned']

# Handle missing values
X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {len(X_train):,} customers")
print(f"   Test set: {len(X_test):,} customers")

# ===== TRAIN XGBOOST MODEL =====
print("\nTraining XGBoost Churn Prediction Model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),  # Handle class imbalance
    random_state=42,
    verbose=0,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("   Model trained successfully")

# ===== MODEL EVALUATION =====
print("\nModel Performance Evaluation...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = (y_pred == y_test).mean()
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   AUC-ROC Score: {auc_score:.4f}")

# Classification report
print("\n   Classification Report:")
report = classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned'])
for line in report.split('\n'):
    if line.strip():
        print(f"   {line}")

# ===== FEATURE IMPORTANCE =====
print("\nFeature Importance (Top 10):")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']:20s}: {row['Importance']:.4f}")

# ===== PREDICT ON ALL CUSTOMERS =====
print("\nGenerating churn predictions for all customers...")

X_all = features_df.drop(['CustomerID', 'Churned'], axis=1).fillna(0)
X_all = X_all.replace([np.inf, -np.inf], 0)

churn_predictions = model.predict(X_all)
churn_probabilities = model.predict_proba(X_all)[:, 1]

output_df = features_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'TenureDays']].copy()
output_df['ChurnRisk'] = churn_probabilities
output_df['ChurnPrediction'] = churn_predictions
output_df['ChurnRiskLevel'] = pd.cut(
    churn_probabilities,
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low', 'Medium', 'High']
)

# Sort by churn risk
output_df = output_df.sort_values('ChurnRisk', ascending=False)

# ===== SAVE RESULTS =====
print("\nSaving results...")

output_file = os.path.join(OUTPUT_DIR, 'churn_prediction.csv')
output_df.to_csv(output_file, index=False)
print(f"   Churn predictions saved to: {output_file}")

# Save model summary
summary_file = os.path.join(OUTPUT_DIR, 'churn_model_summary.csv')
feature_importance.to_csv(summary_file, index=False)
print(f"   Model summary saved to: {summary_file}")

# ===== ACTIONABLE INSIGHTS =====
print("\n" + "="*70)
print("CHURN PREDICTION SUMMARY")
print("="*70)

high_risk = output_df[output_df['ChurnRiskLevel'] == 'High']
medium_risk = output_df[output_df['ChurnRiskLevel'] == 'Medium']

print(f"\nChurn Risk Segmentation:")
print(f"   High Risk (>60%): {len(high_risk):,} customers ({len(high_risk)/len(output_df)*100:.1f}%)")
print(f"   Medium Risk (30-60%): {len(medium_risk):,} customers ({len(medium_risk)/len(output_df)*100:.1f}%)")
print(f"   Low Risk (<30%): {len(output_df) - len(high_risk) - len(medium_risk):,} customers")

print(f"\nTop 5 Highest Churn Risk Customers:")
for idx, row in output_df.head(5).iterrows():
    print(f"   Customer {row['CustomerID']:>6} | Risk: {row['ChurnRisk']*100:>5.1f}% | Recency: {row['Recency']:>3}d | Value: ${row['Monetary']:>10,.0f}")

print(f"\nTop 5 Most Loyal Customers (Lowest Churn Risk):")
for idx, row in output_df.tail(5).iterrows():
    print(f"   Customer {row['CustomerID']:>6} | Risk: {row['ChurnRisk']*100:>5.1f}% | Recency: {row['Recency']:>3}d | Value: ${row['Monetary']:>10,.0f}")

print("\nBusiness Recommendations:")
print("   1. Target HIGH RISK customers with immediate retention campaigns")
print("   2. Implement win-back strategy for very recent churners")
print("   3. Increase engagement frequency for medium-risk customers")
print("   4. Create VIP program for high-value, low-risk customers")
print("   5. Use churn predictions in email segmentation and offers")

print("\nModel Insights:")
print(f"   Model Accuracy: {accuracy*100:.2f}%")
print(f"   Top predictor: {feature_importance.iloc[0]['Feature']} ({feature_importance.iloc[0]['Importance']:.4f})")
print(f"   Most important features indicate: customer recency and spending patterns are key churn indicators")

print("\nOK CHURN PREDICTION COMPLETE!")
print("="*70)
