from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, current_date, max as spark_max, count, sum as spark_sum, avg, when, lit
from pyspark.sql.types import IntegerType

# Create SparkSession
spark = SparkSession.builder \
    .appName("E-Commerce ML - RFM Analysis") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
    .enableHiveSupport() \
    .getOrCreate()

# Use ecommerce database
spark.sql("USE ecommerce_db")

# Drop existing table
spark.sql("DROP TABLE IF EXISTS customer_rfm_analysis")

print("Loading customer transaction data...")

# Load transformed data with dates
transactions = spark.sql("""
    SELECT 
        CustomerID,
        InvoiceDate,
        TotalAmount,
        InvoiceNo
    FROM ecommerce_transformed
    WHERE CustomerID IS NOT NULL 
    AND TotalAmount > 0
    AND InvoiceDate IS NOT NULL
""")

print(f"Total transactions: {transactions.count()}")

# Get the max date in the dataset as reference date
max_date = transactions.agg(spark_max("InvoiceDate")).collect()[0][0]
print(f"Reference date (latest transaction): {max_date}")

# Calculate RFM metrics
print("Calculating RFM metrics...")

# Convert max_date to lit() for PySpark operations
max_date_lit = lit(max_date)

rfm_data = transactions.groupBy("CustomerID").agg(
    # Recency: Days since last purchase
    datediff(max_date_lit, spark_max("InvoiceDate")).alias("Recency"),
    # Frequency: Number of transactions
    count("InvoiceNo").alias("Frequency"),
    # Monetary: Total amount spent
    spark_sum("TotalAmount").alias("Monetary")
)

print("Assigning RFM scores (1-5)...")

# Calculate quartiles for scoring (using percentile approximation)
# Score 5 = best, 1 = worst
rfm_with_scores = rfm_data.withColumn(
    "R_Score",
    when(col("Recency") <= 30, 5)
    .when(col("Recency") <= 60, 4)
    .when(col("Recency") <= 90, 3)
    .when(col("Recency") <= 180, 2)
    .otherwise(1)
).withColumn(
    "F_Score", 
    when(col("Frequency") >= 50, 5)
    .when(col("Frequency") >= 20, 4)
    .when(col("Frequency") >= 10, 3)
    .when(col("Frequency") >= 5, 2)
    .otherwise(1)
).withColumn(
    "M_Score",
    when(col("Monetary") >= 5000, 5)
    .when(col("Monetary") >= 2000, 4)
    .when(col("Monetary") >= 1000, 3)
    .when(col("Monetary") >= 500, 2)
    .otherwise(1)
)

# Calculate overall RFM score (weighted: R=30%, F=30%, M=40%)
rfm_final = rfm_with_scores.withColumn(
    "RFM_Score",
    (col("R_Score") * 0.3 + col("F_Score") * 0.3 + col("M_Score") * 0.4)
).withColumn(
    "RFM_Segment",
    when(col("RFM_Score") >= 4.5, "Champions")
    .when(col("RFM_Score") >= 4.0, "Loyal Customers")
    .when(col("RFM_Score") >= 3.5, "Potential Loyalists")
    .when(col("RFM_Score") >= 3.0, "Recent Customers")
    .when(col("RFM_Score") >= 2.5, "Promising")
    .when(col("RFM_Score") >= 2.0, "Needs Attention")
    .when(col("RFM_Score") >= 1.5, "At Risk")
    .otherwise("Lost")
).withColumn(
    "Churn_Risk",
    when(col("R_Score") <= 2, "High")
    .when(col("R_Score") == 3, "Medium")
    .otherwise("Low")
)

print("Saving RFM analysis to Hive...")

# Save to Hive table
rfm_final.write \
    .mode("overwrite") \
    .format("parquet") \
    .saveAsTable("customer_rfm_analysis")

print("\n" + "="*50)
print("RFM Analysis Results")
print("="*50)

# Show segment distribution
print("\nCustomer Segment Distribution:")
rfm_final.groupBy("RFM_Segment").count() \
    .orderBy("count", ascending=False).show()

print("\nChurn Risk Distribution:")
rfm_final.groupBy("Churn_Risk").count() \
    .orderBy("count", ascending=False).show()

print("\nTop 10 Champions:")
rfm_final.filter(col("RFM_Segment") == "Champions") \
    .orderBy(col("RFM_Score").desc()) \
    .select("CustomerID", "Recency", "Frequency", "Monetary", "RFM_Score") \
    .limit(10).show()

print("\nRFM analysis completed successfully!")
print("Table 'customer_rfm_analysis' created in Hive")

# Stop Spark session
spark.stop()
