from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import col, collect_list, size, explode, struct

# Create SparkSession
spark = SparkSession.builder \
    .appName("E-Commerce ML - Market Basket Analysis") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
    .enableHiveSupport() \
    .getOrCreate()

# Use ecommerce database
spark.sql("USE ecommerce_db")

# Drop existing tables
spark.sql("DROP TABLE IF EXISTS product_associations")
spark.sql("DROP TABLE IF EXISTS frequent_itemsets")

print("Loading transaction data for basket analysis...")

# Load transactions - group by invoice to get baskets
transactions = spark.sql("""
    SELECT 
        InvoiceNo,
        Description,
        Country
    FROM ecommerce_transformed
    WHERE Description IS NOT NULL 
    AND InvoiceNo IS NOT NULL
    AND Quantity > 0
""")

print(f"Total transaction items: {transactions.count()}")

# Create baskets (list of products per invoice)
print("Creating shopping baskets...")
baskets = transactions.groupBy("InvoiceNo") \
    .agg(collect_list("Description").alias("items"))

# Filter baskets with at least 2 items
baskets = baskets.filter(size(col("items")) >= 2)

print(f"Total baskets with 2+ items: {baskets.count()}")

# Train FP-Growth model for frequent itemsets
print("Running FP-Growth algorithm...")
fpGrowth = FPGrowth(
    itemsCol="items",
    minSupport=0.01,  # 1% minimum support
    minConfidence=0.3  # 30% minimum confidence
)

model = fpGrowth.fit(baskets)

# Get frequent itemsets
print("Extracting frequent itemsets...")
frequent_itemsets = model.freqItemsets

# Get top frequent pairs
frequent_pairs = frequent_itemsets.filter(size(col("items")) == 2) \
    .orderBy(col("freq").desc()) \
    .limit(100)

print("\nTop 10 Frequent Product Pairs:")
frequent_pairs.select(
    col("items").getItem(0).alias("Product1"),
    col("items").getItem(1).alias("Product2"),
    col("freq").alias("Frequency")
).show(10, truncate=False)

# Save frequent itemsets
frequent_itemsets.write \
    .mode("overwrite") \
    .format("parquet") \
    .saveAsTable("frequent_itemsets")

# Get association rules
print("Generating association rules...")
association_rules = model.associationRules

# Filter for strong rules (confidence > 50%)
strong_rules = association_rules.filter(col("confidence") > 0.5) \
    .orderBy(col("confidence").desc(), col("lift").desc())

print(f"\nTotal strong association rules: {strong_rules.count()}")

print("\nTop 10 Association Rules:")
strong_rules.select(
    col("antecedent").alias("If_Bought"),
    col("consequent").alias("Then_Bought"),
    col("confidence").alias("Confidence"),
    col("lift").alias("Lift")
).show(10, truncate=False)

# Save association rules
strong_rules.write \
    .mode("overwrite") \
    .format("parquet") \
    .saveAsTable("product_associations")

print("\n" + "="*50)
print("Market Basket Analysis Complete")
print("="*50)
print(f"Frequent Itemsets: {frequent_itemsets.count()}")
print(f"Association Rules: {association_rules.count()}")
print(f"Strong Rules (>50% confidence): {strong_rules.count()}")
print("\nTables created:")
print("  - frequent_itemsets")
print("  - product_associations")

# Stop Spark session
spark.stop()
