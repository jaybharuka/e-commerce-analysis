from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, sum, round, expr

# Create a SparkSession
spark = SparkSession.builder \
    .appName("E-Commerce Data Loading") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
    .enableHiveSupport() \
    .getOrCreate()

    
# Step 1: Create a database if it doesn't already exist
spark.sql("CREATE DATABASE IF NOT EXISTS ecommerce_db")

# Step 2: Switch to the newly created (or existing) database
spark.sql("USE ecommerce_db")

# Step 3: Create the `ecommerce_raw` table with appropriate schema
spark.sql("""
    CREATE TABLE IF NOT EXISTS ecommerce_raw (
        InvoiceNo STRING,
        StockCode STRING,
        Description STRING,
        Quantity INT,
        InvoiceDate STRING,
        UnitPrice FLOAT,
        CustomerID STRING,
        Country STRING
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    STORED AS TEXTFILE
    TBLPROPERTIES ("skip.header.line.count"="1")
""")

# Step 4: Load data from HDFS into the `ecommerce_raw` table
spark.sql("LOAD DATA INPATH '/user/root/ecommerce/data.csv' INTO TABLE ecommerce_raw")

# Stop the Spark session
spark.stop()