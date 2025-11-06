"""
Market Basket Analysis - Association Rules
Finds products frequently bought together
"""
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
import os

# Configuration
DATA_FILE = '../data/data.csv'
OUTPUT_DIR = '../ml_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading e-commerce data...")
df = pd.read_csv(DATA_FILE, encoding='latin-1')

# Data cleaning
df_clean = df[
    (df['Quantity'] > 0) & 
    (df['Description'].notna()) &
    (df['InvoiceNo'].notna())
].copy()

print(f"Clean records: {len(df_clean)}")

# Create shopping baskets
print("\nCreating shopping baskets...")
baskets = df_clean.groupby('InvoiceNo')['Description'].apply(list).reset_index()
baskets.columns = ['InvoiceNo', 'Products']

# Filter baskets with at least 2 items
baskets = baskets[baskets['Products'].apply(len) >= 2]
print(f"Baskets with 2+ items: {len(baskets)}")

# Find frequent item pairs
print("\nFinding frequent product pairs...")
all_pairs = []
for products in baskets['Products']:
    # Get all pairs in this basket
    pairs = list(combinations(sorted(set(products)), 2))
    all_pairs.extend(pairs)

# Count pair frequencies
pair_counts = Counter(all_pairs)
total_baskets = len(baskets)

# Calculate support (frequency)
frequent_pairs = []
for (prod1, prod2), count in pair_counts.most_common(100):
    support = count / total_baskets
    if support >= 0.01:  # 1% minimum support
        frequent_pairs.append({
            'Product_A': prod1,
            'Product_B': prod2,
            'Frequency': count,
            'Support': round(support * 100, 2)
        })

df_pairs = pd.DataFrame(frequent_pairs)

# Save results
output_file = os.path.join(OUTPUT_DIR, 'product_associations.csv')
df_pairs.to_csv(output_file, index=False)
print(f"\n✅ Product associations saved to: {output_file}")

# Print summary
print("\n" + "="*60)
print("MARKET BASKET ANALYSIS SUMMARY")
print("="*60)

print(f"\nTotal baskets analyzed: {total_baskets}")
print(f"Frequent product pairs found: {len(df_pairs)}")

print("\nTop 20 Product Associations:")
if len(df_pairs) > 0:
    top_20 = df_pairs.head(20)
    for idx, row in top_20.iterrows():
        print(f"{idx+1:2d}. {row['Product_A'][:40]:40s} + {row['Product_B'][:40]:40s}")
        print(f"    Bought together {row['Frequency']} times ({row['Support']}% of baskets)")
else:
    print("  No frequent pairs found")

print("\n✅ Market Basket Analysis Complete!")
