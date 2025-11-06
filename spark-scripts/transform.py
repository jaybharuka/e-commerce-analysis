from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, sum,  count, round, expr, month, year

# Create a SparkSession
spark = SparkSession.builder \
    .appName("E-Commerce Data Transformation") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .enableHiveSupport() \
    .getOrCreate()



# Select the database
spark.sql("USE ecommerce_db")

# Verifie tables
spark.sql("SHOW TABLES").show()

# Load the data from the Hive table
raw_df = spark.sql("SELECT * FROM ecommerce_raw")

# Transformations
# # 1. Convert InvoiceDate to Timestamp
# transformed_df = raw_df.withColumn("InvoiceDate", to_timestamp(col("InvoiceDate"), "M/d/yyyy H:mm"))

# # 2. Extract the month from InvoiceDate (enables month-based aggregation)
# transformed_df = raw_df.withColumn("Month",month(col("InvoiceDate")))

# # 3. Extract the year from InvoiceDate
# transformed_df = raw_df.withColumn("Year",year(col("InvoiceDate")))

# # 2. Calculate Total Amount (Quantity * UnitPrice)
# transformed_df = transformed_df.withColumn("TotalAmount", round(col("Quantity") * col("UnitPrice"), 2))

# # 3. Filter out invalid data (e.g., negative or zero Quantity/UnitPrice)
# transformed_df = transformed_df.filter((col("Quantity") > 0) & (col("UnitPrice") > 0))


transformed_df = (raw_df
    .withColumn("InvoiceDate", to_timestamp(col("InvoiceDate"), "M/d/yyyy H:mm"))
    .withColumn("Month", month(col("InvoiceDate")))
    .withColumn("Year", year(col("InvoiceDate")))
    .withColumn("TotalAmount", round(col("Quantity") * col("UnitPrice"), 2))
    .filter((col("Quantity") > 0) & (col("UnitPrice") > 0))
)

# 4. Aggregate Data: Total Sales per Country
sales_per_country_df = transformed_df.groupBy("Country").agg(
    sum("TotalAmount").alias("TotalSales")
).orderBy(col("TotalSales").desc())

# 5. Aggregate Data: Monthly Sales Trends
monthly_sales_df = transformed_df.groupBy("Year", "Month").agg(
    sum("TotalAmount").alias("MonthlySales")
).orderBy("Year", "Month")


# Customer Segmentation
customer_metrics_df = transformed_df.groupBy("CustomerID").agg(
    sum("TotalAmount").alias("TotalPurchases"),
    count("InvoiceNo").alias("NumberOfTransactions")
)

# Product Performance
product_performance_df = transformed_df.groupBy("StockCode", "Description").agg(
    sum("Quantity").alias("TotalQuantitySold"),
    sum("TotalAmount").alias("TotalRevenue")
).orderBy(col("TotalRevenue").desc())

# Drop existing tables if they exist to avoid conflicts
spark.sql("DROP TABLE IF EXISTS ecommerce_transformed")
spark.sql("DROP TABLE IF EXISTS sales_per_country")
spark.sql("DROP TABLE IF EXISTS monthly_sales")
spark.sql("DROP TABLE IF EXISTS customer_metrics")
spark.sql("DROP TABLE IF EXISTS product_performance")

# Write transformed data back to Hive in Parquet format
transformed_df.write \
    .mode("overwrite") \
    .format("parquet") \
    .saveAsTable("ecommerce_transformed")

# Write results back to Hive in Parquet format with partitioning
sales_per_country_df.write \
    .mode("overwrite") \
    .partitionBy("Country") \
    .format("parquet") \
    .saveAsTable("sales_per_country")

monthly_sales_df.write \
    .mode("overwrite") \
    .partitionBy("Year", "Month") \
    .format("parquet") \
    .saveAsTable("monthly_sales")

customer_metrics_df.write \
    .mode("overwrite") \
    .partitionBy("CustomerID") \
    .format("parquet") \
    .saveAsTable("customer_metrics")

product_performance_df.write \
    .mode("overwrite") \
    .partitionBy("Description") \
    .format("parquet") \
    .saveAsTable("product_performance")

# Display the results (for debugging or verification)
# sales_per_country_df.show(5, truncate=False)
# monthly_sales_df.show(5, truncate=False)
# customer_metrics_df.show(5, truncate=False)
# product_performance_df.show(5, truncate=False)

# Stop the Spark session
spark.stop()
