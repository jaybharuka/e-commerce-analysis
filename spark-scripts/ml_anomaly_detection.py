from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col, when, lit, abs as spark_abs
from pyspark.ml.stat import Correlation

# Create SparkSession
spark = SparkSession.builder \
    .appName("E-Commerce ML - Anomaly Detection") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
    .enableHiveSupport() \
    .getOrCreate()

# Use ecommerce database
spark.sql("USE ecommerce_db")

# Drop existing table if exists
spark.sql("DROP TABLE IF EXISTS transaction_anomalies")

# Load transformed data
transactions = spark.sql("""
    SELECT 
        InvoiceNo,
        CustomerID,
        StockCode,
        Description,
        Quantity,
        UnitPrice,
        TotalAmount,
        InvoiceDate,
        Country
    FROM ecommerce_transformed
    WHERE TotalAmount > 0
""")

# Calculate statistics for anomaly detection
from pyspark.sql.functions import mean, stddev, count

# Calculate mean and standard deviation for each feature
stats = transactions.agg(
    mean("Quantity").alias("mean_qty"),
    stddev("Quantity").alias("std_qty"),
    mean("UnitPrice").alias("mean_price"),
    stddev("UnitPrice").alias("std_price"),
    mean("TotalAmount").alias("mean_amount"),
    stddev("TotalAmount").alias("std_amount")
).collect()[0]

mean_qty = stats["mean_qty"]
std_qty = stats["std_qty"]
mean_price = stats["mean_price"]
std_price = stats["std_price"]
mean_amount = stats["mean_amount"]
std_amount = stats["std_amount"]

# Calculate z-scores for each transaction
transactions_with_scores = transactions.withColumn(
    "qty_zscore",
    spark_abs((col("Quantity") - lit(mean_qty)) / lit(std_qty))
).withColumn(
    "price_zscore",
    spark_abs((col("UnitPrice") - lit(mean_price)) / lit(std_price))
).withColumn(
    "amount_zscore",
    spark_abs((col("TotalAmount") - lit(mean_amount)) / lit(std_amount))
)

# Define anomaly threshold (z-score > 3 is typically considered an outlier)
anomaly_threshold = 3.0

# Flag anomalies
transactions_with_anomalies = transactions_with_scores.withColumn(
    "is_anomaly",
    when(
        (col("qty_zscore") > anomaly_threshold) |
        (col("price_zscore") > anomaly_threshold) |
        (col("amount_zscore") > anomaly_threshold),
        1
    ).otherwise(0)
).withColumn(
    "anomaly_type",
    when(col("qty_zscore") > anomaly_threshold, "High Quantity")
    .when(col("price_zscore") > anomaly_threshold, "High Price")
    .when(col("amount_zscore") > anomaly_threshold, "High Amount")
    .otherwise("Normal")
).withColumn(
    "anomaly_score",
    (col("qty_zscore") + col("price_zscore") + col("amount_zscore")) / 3.0
)

# Calculate severity level
transactions_with_severity = transactions_with_anomalies.withColumn(
    "severity",
    when(col("anomaly_score") >= 5.0, "Critical")
    .when(col("anomaly_score") >= 4.0, "High")
    .when(col("anomaly_score") >= 3.0, "Medium")
    .otherwise("Low")
)

# Save anomaly results
transactions_with_severity.select(
    "InvoiceNo",
    "CustomerID",
    "StockCode",
    "Description",
    "Quantity",
    "UnitPrice",
    "TotalAmount",
    "InvoiceDate",
    "Country",
    "is_anomaly",
    "anomaly_type",
    "anomaly_score",
    "severity",
    "qty_zscore",
    "price_zscore",
    "amount_zscore"
).write \
    .mode("overwrite") \
    .format("parquet") \
    .saveAsTable("transaction_anomalies")

# Show statistics
print("=" * 50)
print("Anomaly Detection Results")
print("=" * 50)

total_transactions = transactions_with_severity.count()
anomalies = transactions_with_severity.filter(col("is_anomaly") == 1).count()
anomaly_rate = (anomalies / total_transactions * 100) if total_transactions > 0 else 0.0

print(f"Total Transactions: {total_transactions}")
print(f"Anomalies Detected: {anomalies}")
print(f"Anomaly Rate: {anomaly_rate:.2f}%")

print("\nAnomaly Distribution by Type:")
transactions_with_severity.filter(col("is_anomaly") == 1) \
    .groupBy("anomaly_type").count() \
    .orderBy("count", ascending=False).show()

print("\nAnomaly Distribution by Severity:")
transactions_with_severity.filter(col("is_anomaly") == 1) \
    .groupBy("severity").count() \
    .orderBy("count", ascending=False).show()

print("\nTop 10 Critical Anomalies:")
transactions_with_severity.filter(col("severity") == "Critical") \
    .select("InvoiceNo", "CustomerID", "Description", "Quantity", "UnitPrice", "TotalAmount", "anomaly_score") \
    .orderBy(col("anomaly_score").desc()) \
    .limit(10).show(truncate=False)

print("\nStatistics:")
if mean_qty is not None and std_qty is not None:
    print(f"Mean Quantity: {mean_qty:.2f} (StdDev: {std_qty:.2f})")
    print(f"Mean Price: {mean_price:.2f} (StdDev: {std_price:.2f})")
    print(f"Mean Amount: {mean_amount:.2f} (StdDev: {std_amount:.2f})")
else:
    print("Statistics not available (empty dataset or null values)")

# Stop Spark session
spark.stop()
