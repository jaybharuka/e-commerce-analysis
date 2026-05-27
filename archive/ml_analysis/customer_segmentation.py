"""
Customer Segmentation - K-Means Clustering
Groups customers by purchasing behavior
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
    (df['CustomerID'].notna()) &
    (df['TotalAmount'] > 0)
].copy()

print(f"Clean records: {len(df_clean)}")

# Create customer features
print("\nCreating customer features...")
customer_features = df_clean.groupby('CustomerID').agg({
    'TotalAmount': 'sum',      # Total spending
    'InvoiceNo': 'nunique',    # Number of orders
    'Quantity': 'sum'          # Total items purchased
}).reset_index()

customer_features.columns = ['CustomerID', 'TotalSpending', 'TotalOrders', 'TotalItems']
customer_features['AvgOrderValue'] = customer_features['TotalSpending'] / customer_features['TotalOrders']

print(f"Customer features created for {len(customer_features)} customers")

# Standardize features for clustering
print("\nPerforming K-Means clustering...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(customer_features[['TotalSpending', 'TotalOrders']])

# K-Means with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
customer_features['Cluster'] = kmeans.fit_predict(features_scaled)

# Name segments based on characteristics
def name_segment(row):
    cluster = row['Cluster']
    spending = row['TotalSpending']
    orders = row['TotalOrders']
    
    # Get cluster means
    cluster_data = customer_features[customer_features['Cluster'] == cluster]
    avg_spending = cluster_data['TotalSpending'].mean()
    avg_orders = cluster_data['TotalOrders'].mean()
    
    if avg_spending > customer_features['TotalSpending'].median() * 2:
        if avg_orders > customer_features['TotalOrders'].median():
            return 'High Value Frequent'
        else:
            return 'High Value Occasional'
    elif avg_orders > customer_features['TotalOrders'].median():
        return 'Regular Shoppers'
    else:
        return 'Low Engagement'

customer_features['Segment'] = customer_features.apply(name_segment, axis=1)

# Save results
output_file = os.path.join(OUTPUT_DIR, 'customer_segments.csv')
customer_features.to_csv(output_file, index=False)
print(f"\n✅ Customer segmentation saved to: {output_file}")

# Print summary
print("\n" + "="*60)
print("CUSTOMER SEGMENTATION SUMMARY")
print("="*60)

print("\nSegment Distribution:")
segment_dist = customer_features['Segment'].value_counts()
for segment, count in segment_dist.items():
    pct = (count / len(customer_features) * 100)
    print(f"  {segment:25s}: {count:5d} ({pct:5.1f}%)")

print("\nAverage Metrics by Segment:")
segment_metrics = customer_features.groupby('Segment').agg({
    'TotalSpending': 'mean',
    'TotalOrders': 'mean',
    'TotalItems': 'mean',
    'AvgOrderValue': 'mean'
}).round(2)
print(segment_metrics)

print("\n✅ Customer Segmentation Complete!")
