from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, month, year, unix_timestamp, avg, sum as spark_sum, lit

# Create SparkSession
spark = SparkSession.builder \
    .appName("E-Commerce ML - Sales Forecasting") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
    .enableHiveSupport() \
    .getOrCreate()

# Use ecommerce database
spark.sql("USE ecommerce_db")

# Drop existing table if exists
spark.sql("DROP TABLE IF EXISTS sales_forecast")

# Load monthly sales data
monthly_sales = spark.sql("SELECT * FROM monthly_sales ORDER BY Year, Month")

# Create time-based features
monthly_sales = monthly_sales.withColumn(
    "MonthIndex", 
    (col("Year") - 2010) * 12 + col("Month")
)

# Add lag features (previous months' sales)
from pyspark.sql.window import Window
windowSpec = Window.orderBy("Year", "Month").rowsBetween(-2, -1)

monthly_sales = monthly_sales.withColumn(
    "AvgPrevSales",
    avg("MonthlySales").over(windowSpec)
)

# Fill nulls for first few months
monthly_sales = monthly_sales.fillna({"AvgPrevSales": 0})

# Prepare features for regression
assembler = VectorAssembler(
    inputCols=["MonthIndex", "Month", "AvgPrevSales"],
    outputCol="features"
)

data = assembler.transform(monthly_sales)

# Split into training and test sets (80/20)
train_data = data.filter(col("MonthIndex") <= 10)
test_data = data.filter(col("MonthIndex") > 10)

# Train Linear Regression model
lr = LinearRegression(
    featuresCol="features",
    labelCol="MonthlySales",
    predictionCol="PredictedSales"
)

model = lr.fit(train_data)

# Make predictions on test data
predictions = model.transform(test_data)

# Create forecast for next 3 months
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

# Get the last month's data
last_month = monthly_sales.orderBy(col("MonthIndex").desc()).first()
last_index = last_month["MonthIndex"]
last_year = last_month["Year"]
last_month_num = last_month["Month"]
last_sales = last_month["MonthlySales"]

# Generate future months
future_data = []
for i in range(1, 4):  # Next 3 months
    future_index = last_index + i
    future_month = (last_month_num + i - 1) % 12 + 1
    future_year = last_year + ((last_month_num + i - 1) // 12)
    
    future_data.append((future_year, future_month, future_index, last_sales))

schema = StructType([
    StructField("Year", IntegerType(), True),
    StructField("Month", IntegerType(), True),
    StructField("MonthIndex", IntegerType(), True),
    StructField("AvgPrevSales", DoubleType(), True)
])

future_df = spark.createDataFrame(future_data, schema)
future_features = assembler.transform(future_df)

# Predict future sales
future_predictions = model.transform(future_features)

# Combine historical and forecast data
forecast_results = predictions.select(
    "Year", 
    "Month", 
    col("MonthlySales").alias("ActualSales"),
    col("PredictedSales"),
    lit("Historical").alias("Type")
).union(
    future_predictions.select(
        "Year",
        "Month",
        lit(None).cast(DoubleType()).alias("ActualSales"),
        "PredictedSales",
        lit("Forecast").alias("Type")
    )
)

# Save forecast results
forecast_results.write \
    .mode("overwrite") \
    .format("parquet") \
    .saveAsTable("sales_forecast")

# Show model statistics
print("=" * 50)
print("Sales Forecasting Model Results")
print("=" * 50)
print(f"RMSE: {model.summary.rootMeanSquaredError:.2f}")
print(f"R2 Score: {model.summary.r2:.4f}")
print(f"MAE: {model.summary.meanAbsoluteError:.2f}")

print("\nForecast for Next 3 Months:")
future_predictions.select("Year", "Month", "PredictedSales").show()

# Stop Spark session
spark.stop()
