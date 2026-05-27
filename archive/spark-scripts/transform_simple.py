from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, sum, count, round, month, year

# Create SparkSession
spark = SparkSession.builder \
    .appName("E-Commerce Transform - Simplified") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
    .enableHiveSupport() \
    .getOrCreate()

print("=" * 60)
print("STARTING TRANSFORM JOB - SIMPLIFIED VERSION")
print("=" * 60)

# Use database
spark.sql("USE ecommerce_db")

# Load raw data
print("\n1. Loading raw data...")
raw_df = spark.sql("SELECT * FROM ecommerce_raw")
print(f"   Loaded {raw_df.count()} rows")

# Transform data
print("\n2. Transforming data...")
transformed_df = (raw_df
    .withColumn("InvoiceDate", to_timestamp(col("InvoiceDate"), "M/d/yyyy H:mm"))
    .withColumn("Month", month(col("InvoiceDate")))
    .withColumn("Year", year(col("InvoiceDate")))
    .withColumn("TotalAmount", round(col("Quantity") * col("UnitPrice"), 2))
    .filter((col("Quantity") > 0) & (col("UnitPrice") > 0))
)

# Aggregations
print("\n3. Creating aggregations...")

sales_per_country_df = transformed_df.groupBy("Country").agg(
    sum("TotalAmount").alias("TotalSales")
).orderBy(col("TotalSales").desc())

monthly_sales_df = transformed_df.groupBy("Year", "Month").agg(
    sum("TotalAmount").alias("TotalSales")
).orderBy("Year", "Month")

customer_metrics_df = transformed_df.groupBy("CustomerID").agg(
    sum("TotalAmount").alias("TotalPurchases"),
    count("InvoiceNo").alias("NumberOfTransactions")
)

product_performance_df = transformed_df.groupBy("StockCode", "Description").agg(
    sum("Quantity").alias("TotalQuantitySold"),
    sum("TotalAmount").alias("TotalRevenue")
).orderBy(col("TotalRevenue").desc())

# Create tables using SQL - NO PARTITIONING
print("\n4. Creating Hive tables (no partitioning)...")

# Register temp views
transformed_df.createOrReplaceTempView("temp_transformed")
sales_per_country_df.createOrReplaceTempView("temp_sales_country")
monthly_sales_df.createOrReplaceTempView("temp_monthly")
customer_metrics_df.createOrReplaceTempView("temp_customers")
product_performance_df.createOrReplaceTempView("temp_products")

# Drop and create tables using SQL
tables = [
    ("ecommerce_transformed", "temp_transformed"),
    ("sales_per_country", "temp_sales_country"),
    ("monthly_sales", "temp_monthly"),
    ("customer_metrics", "temp_customers"),
    ("product_performance", "temp_products")
]

for table_name, view_name in tables:
    print(f"\n   Creating {table_name}...")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    spark.sql(f"CREATE TABLE {table_name} STORED AS PARQUET AS SELECT * FROM {view_name}")
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {table_name}").collect()[0]['cnt']
    print(f"   ✅ {table_name}: {count} rows")

print("\n" + "=" * 60)
print("✅ TRANSFORM COMPLETE - ALL TABLES CREATED")
print("=" * 60)

# Verify all tables
print("\nFinal table list:")
spark.sql("SHOW TABLES").show()

spark.stop()
