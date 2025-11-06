from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, year, sum as spark_sum, avg, lag, lead, count
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

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

# Drop existing table
spark.sql("DROP TABLE IF EXISTS sales_forecast")

print("Loading monthly sales data...")

# Aggregate sales by month
monthly_sales_data = spark.sql("""
    SELECT 
        year,
        month,
        SUM(TotalAmount) as total_revenue,
        COUNT(DISTINCT CustomerID) as unique_customers,
        COUNT(DISTINCT InvoiceNo) as total_orders,
        AVG(TotalAmount) as avg_order_value,
        SUM(Quantity) as total_items_sold
    FROM ecommerce_transformed
    WHERE TotalAmount > 0
    GROUP BY year, month
    ORDER BY year, month
""")

print("Monthly sales aggregated")
monthly_sales_data.show()

# Create time-based features
print("Creating lag features for forecasting...")

# Define window for lag features
window_spec = Window.orderBy("year", "month")

# Add lag features (previous months' data)
sales_with_lags = monthly_sales_data.withColumn(
    "revenue_lag1", lag("total_revenue", 1).over(window_spec)
).withColumn(
    "revenue_lag2", lag("total_revenue", 2).over(window_spec)
).withColumn(
    "revenue_lag3", lag("total_revenue", 3).over(window_spec)
).withColumn(
    "orders_lag1", lag("total_orders", 1).over(window_spec)
).withColumn(
    "customers_lag1", lag("unique_customers", 1).over(window_spec)
).withColumn(
    "moving_avg_3m", 
    avg("total_revenue").over(
        Window.orderBy("year", "month").rowsBetween(-2, 0)
    )
)

# Remove rows with null lags (first few months)
sales_features = sales_with_lags.filter(
    col("revenue_lag1").isNotNull() & 
    col("revenue_lag2").isNotNull() &
    col("revenue_lag3").isNotNull()
)

print(f"Training data points: {sales_features.count()}")

if sales_features.count() > 3:
    # Prepare features for model
    feature_cols = [
        "month", "revenue_lag1", "revenue_lag2", "revenue_lag3",
        "orders_lag1", "customers_lag1", "moving_avg_3m"
    ]
    
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    
    training_data = assembler.transform(sales_features).select(
        "year", "month", "features", 
        col("total_revenue").alias("label")
    )
    
    # Split data into train/test (80/20)
    train_data, test_data = training_data.randomSplit([0.8, 0.2], seed=42)
    
    print(f"Training set: {train_data.count()} months")
    print(f"Test set: {test_data.count()} months")
    
    # Train Linear Regression model
    print("Training forecasting model...")
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.8
    )
    
    model = lr.fit(train_data)
    
    # Make predictions
    predictions = model.transform(training_data)
    
    # Calculate forecast accuracy
    evaluator = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    rmse = evaluator.evaluate(predictions)
    r2 = evaluator.setMetricName("r2").evaluate(predictions)
    
    print(f"\nModel Performance:")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R² Score: {r2:.4f}")
    
    # Create forecast with confidence intervals
    final_forecast = predictions.withColumn(
        "forecast_lower", col("prediction") * 0.9
    ).withColumn(
        "forecast_upper", col("prediction") * 1.1
    ).withColumn(
        "accuracy", 
        100 * (1 - abs(col("label") - col("prediction")) / col("label"))
    )
    
    # Save forecast results
    print("Saving forecasts to Hive...")
    final_forecast.select(
        "year",
        "month",
        col("label").alias("actual_revenue"),
        col("prediction").alias("forecasted_revenue"),
        "forecast_lower",
        "forecast_upper",
        "accuracy"
    ).write \
        .mode("overwrite") \
        .format("parquet") \
        .saveAsTable("sales_forecast")
    
    print("\n" + "="*50)
    print("Sales Forecasting Complete")
    print("="*50)
    
    print("\nForecast Summary:")
    final_forecast.select(
        "year", "month",
        col("label").alias("Actual"),
        col("prediction").alias("Forecast"),
        "accuracy"
    ).orderBy("year", "month").show(12)
    
    print("\nTable 'sales_forecast' created in Hive")

else:
    print("Insufficient data for forecasting (need at least 4 months)")
    # Create empty table
    empty_forecast = spark.createDataFrame(
        [],
        "year INT, month INT, actual_revenue DOUBLE, forecasted_revenue DOUBLE, forecast_lower DOUBLE, forecast_upper DOUBLE, accuracy DOUBLE"
    )
    empty_forecast.write.mode("overwrite").saveAsTable("sales_forecast")

# Stop Spark session
spark.stop()
