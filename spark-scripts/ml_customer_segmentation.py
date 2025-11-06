from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col, when, lit

# Create SparkSession
spark = SparkSession.builder \
    .appName("E-Commerce ML - Customer Segmentation") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
    .enableHiveSupport() \
    .getOrCreate()

# Use ecommerce database
spark.sql("USE ecommerce_db")

# Drop existing table if exists
spark.sql("DROP TABLE IF EXISTS customer_segments")

# Load customer metrics
customer_metrics_df = spark.sql("SELECT * FROM customer_metrics WHERE CustomerID IS NOT NULL")

# Prepare features for clustering
assembler = VectorAssembler(
    inputCols=["total_spent", "total_orders"],
    outputCol="features"
)

features_df = assembler.transform(customer_metrics_df)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(features_df)
scaled_df = scaler_model.transform(features_df)

# Train K-Means clustering model with 4 clusters
kmeans = KMeans(k=4, seed=1, featuresCol="scaledFeatures")
model = kmeans.fit(scaled_df)

# Make predictions
predictions = model.transform(scaled_df)

# Add segment labels based on cluster characteristics
# Analyze cluster centers to determine segment names
predictions_with_labels = predictions.withColumn(
    "SegmentName",
    when(col("prediction") == 0, "VIP Customers")
    .when(col("prediction") == 1, "Regular Customers")
    .when(col("prediction") == 2, "Occasional Buyers")
    .when(col("prediction") == 3, "New Customers")
    .otherwise("Unknown")
)

# Save results to new Hive table
predictions_with_labels.select(
    "CustomerID",
    "total_spent",
    "total_orders",
    col("prediction").alias("SegmentID"),
    "SegmentName"
).write \
    .mode("overwrite") \
    .format("parquet") \
    .saveAsTable("customer_segments")

# Show cluster centers and statistics
print("=" * 50)
print("Customer Segmentation Model Results")
print("=" * 50)

centers = model.clusterCenters()
for i, center in enumerate(centers):
    print(f"Cluster {i}: TotalPurchases={center[0]:.2f}, Transactions={center[1]:.2f}")

# Count customers per segment
segment_counts = predictions_with_labels.groupBy("SegmentName").count().orderBy("count", ascending=False)
print("\nCustomer Distribution by Segment:")
segment_counts.show()

# Stop Spark session
spark.stop()
