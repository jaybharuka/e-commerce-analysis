"""
RFM Analysis - Direct from CSV
Generates customer value segments based on Recency, Frequency, Monetary
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configuration
DATA_FILE = '../data/data.csv'
OUTPUT_DIR = '../ml_results'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading e-commerce data...")
df = pd.read_csv(DATA_FILE, encoding='latin-1')

print(f"Total records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Data cleaning
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Filter valid transactions
df_clean = df[
    (df['Quantity'] > 0) & 
    (df['UnitPrice'] > 0) & 
    (df['CustomerID'].notna()) &
    (df['TotalAmount'] > 0)
].copy()

print(f"Clean records: {len(df_clean)}")
print(f"Unique customers: {df_clean['CustomerID'].nunique()}")

# Calculate RFM metrics
print("\nCalculating RFM metrics...")

# Reference date (latest transaction + 1 day)
reference_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Reference date: {reference_date}")

# Group by customer
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalAmount': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print(f"RFM metrics calculated for {len(rfm)} customers")

# Calculate RFM scores (1-5 scale)
print("\nAssigning RFM scores...")

def rfm_score(x, column, quintiles):
    """Assign score based on quintiles (1=worst, 5=best)"""
    if column == 'Recency':
        # Lower recency is better
        if x <= quintiles[0.2]:
            return 5
        elif x <= quintiles[0.4]:
            return 4
        elif x <= quintiles[0.6]:
            return 3
        elif x <= quintiles[0.8]:
            return 2
        else:
            return 1
    else:
        # Higher frequency/monetary is better
        if x >= quintiles[0.8]:
            return 5
        elif x >= quintiles[0.6]:
            return 4
        elif x >= quintiles[0.4]:
            return 3
        elif x >= quintiles[0.2]:
            return 2
        else:
            return 1

# Calculate quintiles
r_quintiles = rfm['Recency'].quantile([0.2, 0.4, 0.6, 0.8])
f_quintiles = rfm['Frequency'].quantile([0.2, 0.4, 0.6, 0.8])
m_quintiles = rfm['Monetary'].quantile([0.2, 0.4, 0.6, 0.8])

# Assign scores
rfm['R_Score'] = rfm['Recency'].apply(lambda x: rfm_score(x, 'Recency', r_quintiles))
rfm['F_Score'] = rfm['Frequency'].apply(lambda x: rfm_score(x, 'Frequency', f_quintiles))
rfm['M_Score'] = rfm['Monetary'].apply(lambda x: rfm_score(x, 'Monetary', m_quintiles))

# Calculate overall RFM score (weighted: R=30%, F=30%, M=40%)
rfm['RFM_Score'] = (rfm['R_Score'] * 0.3 + rfm['F_Score'] * 0.3 + rfm['M_Score'] * 0.4).round(2)

# Segment customers based on RFM score
def segment_customer(row):
    score = row['RFM_Score']
    if score >= 4.5:
        return 'Champions'
    elif score >= 4.0:
        return 'Loyal Customers'
    elif score >= 3.5:
        return 'Potential Loyalists'
    elif score >= 3.0:
        return 'Recent Customers'
    elif score >= 2.5:
        return 'Promising'
    elif score >= 2.0:
        return 'Needs Attention'
    elif score >= 1.5:
        return 'At Risk'
    else:
        return 'Lost'

rfm['RFM_Segment'] = rfm.apply(segment_customer, axis=1)

# Calculate churn risk based on Recency score
def churn_risk(r_score):
    if r_score >= 4:
        return 'Low'
    elif r_score >= 3:
        return 'Medium'
    else:
        return 'High'

rfm['Churn_Risk'] = rfm['R_Score'].apply(churn_risk)

# Save results
output_file = os.path.join(OUTPUT_DIR, 'rfm_analysis.csv')
rfm.to_csv(output_file, index=False)
print(f"\n✅ RFM analysis saved to: {output_file}")

# Print summary
print("\n" + "="*60)
print("RFM ANALYSIS SUMMARY")
print("="*60)

print("\nCustomer Segment Distribution:")
segment_dist = rfm['RFM_Segment'].value_counts().sort_values(ascending=False)
for segment, count in segment_dist.items():
    pct = (count / len(rfm) * 100)
    print(f"  {segment:25s}: {count:5d} ({pct:5.1f}%)")

print("\nChurn Risk Distribution:")
risk_dist = rfm['Churn_Risk'].value_counts()
for risk, count in risk_dist.items():
    pct = (count / len(rfm) * 100)
    print(f"  {risk:10s}: {count:5d} ({pct:5.1f}%)")

print("\nTop 10 Champions:")
champions = rfm[rfm['RFM_Segment'] == 'Champions'].nlargest(10, 'RFM_Score')
print(champions[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score']].to_string(index=False))

print("\nAverage Metrics by Segment:")
segment_metrics = rfm.groupby('RFM_Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'RFM_Score': 'mean'
}).round(2)
print(segment_metrics)

print("\n✅ RFM Analysis Complete!")
