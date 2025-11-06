"""
Product Recommendation Model - Collaborative Filtering
Recommends products to customers based on their purchase history and similar customers.

Algorithm: Content-Based & Collaborative Filtering with Matrix Factorization
- Item-based recommendations (what other customers with similar items bought)
- Customer similarity analysis
- Personalized product rankings
- Confidence scores for recommendations

Business Value:
- Increase average order value through targeted recommendations
- Improve customer experience with relevant products
- Optimize catalog visibility
- Drive cross-selling and upselling opportunities
- Reduce product discovery friction
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = 'data/data.csv'
OUTPUT_DIR = 'ml_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("PRODUCT RECOMMENDATION MODEL - COLLABORATIVE FILTERING")
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
    (df['CustomerID'].notna()) &
    (df['Description'].notna())
].copy()

print(f"   Total records: {len(df_clean):,}")
print(f"   Unique customers: {df_clean['CustomerID'].nunique():,}")
print(f"   Unique products: {df_clean['StockCode'].nunique():,}")

# ===== CREATE CUSTOMER-PRODUCT MATRIX =====
print("\nBuilding customer-product interaction matrix...")

# Create a customer-product purchase matrix
customer_product_matrix = pd.crosstab(
    df_clean['CustomerID'],
    df_clean['StockCode'],
    values=df_clean['TotalAmount'],
    aggfunc='sum'
)

# Fill NaN with 0 (no purchase)
customer_product_matrix = customer_product_matrix.fillna(0)

print(f"   Matrix shape: {customer_product_matrix.shape[0]:,} customers x {customer_product_matrix.shape[1]:,} products")

# Normalize the matrix (0-1 scale) for similarity calculation
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
matrix_normalized = pd.DataFrame(
    scaler.fit_transform(customer_product_matrix),
    index=customer_product_matrix.index,
    columns=customer_product_matrix.columns
)

# ===== CALCULATE PRODUCT SIMILARITY =====
print("\nCalculating product similarities...")

# Transpose to get product-customer matrix for similarity
product_customer_matrix = matrix_normalized.T

# Calculate cosine similarity between products
product_similarity = cosine_similarity(product_customer_matrix)
product_similarity_df = pd.DataFrame(
    product_similarity,
    index=product_customer_matrix.index,
    columns=product_customer_matrix.index
)

print(f"   Similarity matrix computed for {len(product_similarity_df):,} products")

# ===== CALCULATE CUSTOMER SIMILARITY =====
print("Calculating customer similarities...")

# Calculate cosine similarity between customers
customer_similarity = cosine_similarity(matrix_normalized)
customer_similarity_df = pd.DataFrame(
    customer_similarity,
    index=matrix_normalized.index,
    columns=matrix_normalized.index
)

print(f"   Customer similarity matrix computed for {len(customer_similarity_df):,} customers")

# ===== GENERATE PRODUCT RECOMMENDATIONS =====
print("\nGenerating personalized recommendations...")

recommendations_list = []

# For each customer, recommend top products from similar customers
customer_list = list(customer_product_matrix.index)[:100]  # Top 100 active customers for demo

for customer_id in customer_list:
    # Get products already purchased by this customer
    purchased_products = set(customer_product_matrix.columns[customer_product_matrix.loc[customer_id] > 0])
    
    if len(purchased_products) == 0:
        continue
    
    # Find similar customers
    similar_scores = customer_similarity_df.loc[customer_id]
    similar_customers = similar_scores.nlargest(6).iloc[1:]  # Top 5 similar (excluding self)
    
    # Find products bought by similar customers but not by this customer
    recommendations = {}
    
    for sim_customer in similar_customers.index:
        sim_products = set(customer_product_matrix.columns[customer_product_matrix.loc[sim_customer] > 0])
        new_products = sim_products - purchased_products
        
        for product in new_products:
            if product not in recommendations:
                recommendations[product] = 0
            # Weight by similarity and product value
            recommendations[product] += (
                similar_scores[sim_customer] * 
                customer_product_matrix.loc[sim_customer, product]
            )
    
    # Sort recommendations by score
    if recommendations:
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for rank, (product_id, score) in enumerate(top_recommendations, 1):
            # Get product info
            product_info = df_clean[df_clean['StockCode'] == product_id].iloc[0]
            
            recommendations_list.append({
                'CustomerID': customer_id,
                'ProductStockCode': product_id,
                'ProductDescription': product_info['Description'],
                'RecommendationScore': score,
                'Rank': rank,
                'AveragePrice': product_info['UnitPrice'],
                'ProductCategory': product_info.get('Description', '').split()[0] if pd.notna(product_info.get('Description')) else 'Unknown'
            })

recommendations_df = pd.DataFrame(recommendations_list)
print(f"   Generated {len(recommendations_df):,} recommendations for top customers")

# ===== PRODUCT AFFINITY ANALYSIS =====
print("\nAnalyzing product affinities...")

# For each product, find most frequently co-purchased products
product_cooccurrence = {}

# Sample invoices for faster processing (last 5000 invoices)
sample_invoices = df_clean['InvoiceNo'].unique()[-5000:]

for invoice in sample_invoices:
    invoice_products = df_clean[df_clean['InvoiceNo'] == invoice]['StockCode'].unique()
    
    for product1 in invoice_products:
        for product2 in invoice_products:
            if product1 != product2:
                key = tuple(sorted([product1, product2]))
                if key not in product_cooccurrence:
                    product_cooccurrence[key] = 0
                product_cooccurrence[key] += 1

# Convert to dataframe
affinity_list = []
for (product1, product2), count in sorted(product_cooccurrence.items(), key=lambda x: x[1], reverse=True)[:100]:
    p1_info = df_clean[df_clean['StockCode'] == product1]
    p2_info = df_clean[df_clean['StockCode'] == product2]
    
    if len(p1_info) > 0 and len(p2_info) > 0:
        affinity_list.append({
            'Product1': product1,
            'Product1Description': p1_info['Description'].iloc[0],
            'Product2': product2,
            'Product2Description': p2_info['Description'].iloc[0],
            'CooccurrenceCount': count,
            'BundleScore': count / len(sample_invoices)  # Normalize by sample invoices
        })

affinity_df = pd.DataFrame(affinity_list)

# ===== SAVE RESULTS =====
print("\nSaving results...")

# Save recommendations
rec_file = os.path.join(OUTPUT_DIR, 'product_recommendations.csv')
recommendations_df.to_csv(rec_file, index=False)
print(f"   Recommendations saved to: {rec_file}")

# Save product affinities
affinity_file = os.path.join(OUTPUT_DIR, 'product_affinity.csv')
affinity_df.to_csv(affinity_file, index=False)
print(f"   Product affinities saved to: {affinity_file}")

# ===== SUMMARY STATISTICS =====
print("\n" + "="*70)
print("PRODUCT RECOMMENDATION SUMMARY")
print("="*70)

print(f"\nRecommendation Statistics:")
print(f"   Total recommendations: {len(recommendations_df):,}")
print(f"   Unique customers targeted: {recommendations_df['CustomerID'].nunique():,}")
print(f"   Unique products recommended: {recommendations_df['ProductStockCode'].nunique():,}")
print(f"   Average recommendations per customer: {len(recommendations_df) / recommendations_df['CustomerID'].nunique():.1f}")

print(f"\nTop 10 Most Recommended Products:")
top_products = recommendations_df['ProductDescription'].value_counts().head(10)
for idx, (product, count) in enumerate(top_products.items(), 1):
    print(f"   {idx}. {product[:50]:50s} ({count:>3} recommendations)")

print(f"\nProduct Affinity Insights:")
print(f"   Total product pairs analyzed: {len(affinity_df):,}")
print(f"   Strongest bundle opportunities (Top 5):")
for idx, row in affinity_df.head(5).iterrows():
    print(f"   {idx+1}. {row['Product1Description'][:30]:30s} + {row['Product2Description'][:30]:30s}")
    print(f"      Co-occurrence: {row['CooccurrenceCount']:>3} times | Bundle Score: {row['BundleScore']:.4f}")

print(f"\nCross-Selling Opportunities:")
avg_price = df_clean['UnitPrice'].mean()
high_value_recs = recommendations_df[recommendations_df['AveragePrice'] > avg_price * 1.5]
print(f"   High-value product recommendations: {len(high_value_recs):,}")
print(f"   Potential revenue uplift opportunities: Identified for targeted campaigns")

print(f"\nRecommendation Quality Metrics:")
print(f"   Average recommendation score: {recommendations_df['RecommendationScore'].mean():.4f}")
print(f"   Max recommendation score: {recommendations_df['RecommendationScore'].max():.4f}")
print(f"   Recommendation coverage: {recommendations_df['CustomerID'].nunique() / df_clean['CustomerID'].nunique() * 100:.1f}% of active customers")

print("\nBusiness Impact:")
print("   1. Implement 'Frequently bought together' section on product pages")
print("   2. Personalize email campaigns with tailored product recommendations")
print("   3. Create intelligent product bundles based on affinity analysis")
print("   4. Optimize product placement in store based on co-purchase patterns")
print("   5. A/B test recommendations to measure uplift in AOV and conversion")

print("\nRECOMMENDATION MODEL COMPLETE!")
print("="*70)
