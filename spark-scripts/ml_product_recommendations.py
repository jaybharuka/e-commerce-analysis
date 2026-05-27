from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, explode, array, lit

# Create SparkSession
spark = SparkSession.builder \
    .appName("E-Commerce ML - Product Recommendations") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
    .enableHiveSupport() \
    .getOrCreate()

# Use ecommerce database
spark.sql("USE ecommerce_db")

# Drop existing tables if exist
spark.sql("DROP TABLE IF EXISTS product_recommendations")

# Load transformed data
transactions = spark.sql("""
    SELECT 
        CAST(CustomerID AS INT) as CustomerID,
        StockCode,
        Quantity,
        TotalAmount
    FROM ecommerce_transformed
    WHERE CustomerID IS NOT NULL 
    AND StockCode IS NOT NULL
    AND Quantity > 0
""")

# Create ratings (using quantity as implicit feedback)
# Normalize quantity to rating scale (1-5)
from pyspark.sql.functions import when, greatest, least
from pyspark.ml.feature import StringIndexer

# Convert StockCode to numeric
indexer = StringIndexer(inputCol="StockCode", outputCol="StockCodeIndex")
indexer_model = indexer.fit(transactions)
transactions_indexed = indexer_model.transform(transactions)

ratings = transactions_indexed.groupBy("CustomerID", "StockCodeIndex") \
    .agg({"Quantity": "sum"}) \
    .withColumnRenamed("sum(Quantity)", "TotalQuantity")

# Convert to rating scale (1-5)
ratings = ratings.withColumn(
    "rating",
    when(col("TotalQuantity") >= 50, 5.0)
    .when(col("TotalQuantity") >= 20, 4.0)
    .when(col("TotalQuantity") >= 10, 3.0)
    .when(col("TotalQuantity") >= 5, 2.0)
    .otherwise(1.0)
)

# Train ALS model
als = ALS(
    userCol="CustomerID",
    itemCol="StockCodeIndex",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    implicitPrefs=False,
    rank=10,
    maxIter=10,
    regParam=0.1
)

model = als.fit(ratings)

# Generate top 10 product recommendations for each customer
user_recs = model.recommendForAllUsers(10)

# Flatten recommendations
user_recs_flat = user_recs.select(
    col("CustomerID"),
    explode(col("recommendations")).alias("rec")
).select(
    "CustomerID",
    # ALS emits the item column used at training time (StockCodeIndex), not the original StockCode
    col("rec.StockCodeIndex").alias("RecommendedProductIndex"),
    col("rec.rating").alias("RecommendationScore")
)

# Reverse-map StockCodeIndex back to StockCode using the fitted indexer
# Build an index→StockCode lookup from the indexer labels
index_to_stock = spark.createDataFrame(
    [(float(i), label) for i, label in enumerate(indexer_model.labels)],
    ["StockCodeIndex", "StockCode"]
)

user_recs_mapped = user_recs_flat.join(index_to_stock, on="StockCodeIndex", how="left")

# Add product descriptions
products = spark.sql("""
    SELECT DISTINCT StockCode, Description
    FROM ecommerce_transformed
    WHERE StockCode IS NOT NULL AND Description IS NOT NULL
""")

recommendations_with_desc = user_recs_mapped.join(
    products,
    on="StockCode",
    how="left"
)

# Save recommendations
recommendations_with_desc.select(
    "CustomerID",
    col("StockCode").alias("RecommendedProduct"),
    "Description",
    "RecommendationScore"
).write \
    .mode("overwrite") \
    .format("parquet") \
    .saveAsTable("product_recommendations")

# Show statistics
print("=" * 50)
print("Product Recommendation Model Results")
print("=" * 50)
total_customers = user_recs.count()
total_recs = user_recs_flat.count()
print(f"Total Customers: {total_customers}")
print(f"Total Recommendations: {total_recs}")
print(f"Average Recommendations per Customer: {total_recs / total_customers:.2f}")

print("\nSample Recommendations:")
recommendations_with_desc.limit(20).show(truncate=False)

# Stop Spark session
spark.stop()
