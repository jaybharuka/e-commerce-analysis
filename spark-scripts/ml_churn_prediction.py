from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col, datediff, max as spark_max, min as spark_min, when, lit
from pyspark.sql.window import Window

# Create SparkSession
spark = SparkSession.builder \
    .appName("E-Commerce ML - Churn Prediction") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
    .enableHiveSupport() \
    .getOrCreate()

# Use ecommerce database
spark.sql("USE ecommerce_db")

# Drop existing table if exists
spark.sql("DROP TABLE IF EXISTS customer_churn_prediction")

# Load transformed data with dates
transactions = spark.sql("""
    SELECT 
        CustomerID,
        InvoiceDate,
        TotalAmount,
        Quantity
    FROM ecommerce_transformed
    WHERE CustomerID IS NOT NULL
""")

# Calculate churn features per customer
from pyspark.sql.functions import count, avg, stddev, sum as spark_sum

# Get first and last purchase dates
customer_dates = transactions.groupBy("CustomerID").agg(
    spark_min("InvoiceDate").alias("FirstPurchase"),
    spark_max("InvoiceDate").alias("LastPurchase"),
    count("*").alias("TotalTransactions"),
    spark_sum("TotalAmount").alias("TotalSpent"),
    avg("TotalAmount").alias("AvgOrderValue"),
    stddev("TotalAmount").alias("StdOrderValue")
)

# Calculate recency (days since last purchase from the last date in dataset)
max_date = transactions.agg(spark_max("InvoiceDate")).collect()[0][0]

customer_features = customer_dates.withColumn(
    "Recency",
    datediff(lit(max_date), col("LastPurchase"))
).withColumn(
    "CustomerLifetime",
    datediff(col("LastPurchase"), col("FirstPurchase"))
)

# Fill nulls
customer_features = customer_features.fillna({"StdOrderValue": 0, "CustomerLifetime": 0})

# Define churn label: churned if no purchase in last 90 days
customer_features = customer_features.withColumn(
    "Churn",
    when(col("Recency") > 90, 1).otherwise(0)
)

# Prepare features
feature_cols = ["Recency", "TotalTransactions", "TotalSpent", "AvgOrderValue", "CustomerLifetime"]
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

data = assembler.transform(customer_features)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(data)
scaled_data = scaler_model.transform(data)

# Split data into training and test sets (80/20)
train_data, test_data = scaled_data.randomSplit([0.8, 0.2], seed=42)

# Train Random Forest Classifier
rf = RandomForestClassifier(
    featuresCol="scaledFeatures",
    labelCol="Churn",
    predictionCol="ChurnPrediction",
    probabilityCol="ChurnProbability",
    numTrees=50,
    maxDepth=10,
    seed=42
)

model = rf.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate model
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# AUC
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="Churn",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = evaluator_auc.evaluate(predictions)

# Accuracy
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="Churn",
    predictionCol="ChurnPrediction",
    metricName="accuracy"
)
accuracy = evaluator_acc.evaluate(predictions)

# Precision
evaluator_prec = MulticlassClassificationEvaluator(
    labelCol="Churn",
    predictionCol="ChurnPrediction",
    metricName="weightedPrecision"
)
precision = evaluator_prec.evaluate(predictions)

# Make predictions on all customers
all_predictions = model.transform(scaled_data)

# Extract probability of churning
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT

def get_prob_churn(probability):
    return float(probability[1])

get_prob_udf = udf(get_prob_churn, DoubleType())

final_predictions = all_predictions.withColumn(
    "ChurnRisk",
    get_prob_udf(col("ChurnProbability"))
).withColumn(
    "RiskCategory",
    when(col("ChurnRisk") >= 0.7, "High Risk")
    .when(col("ChurnRisk") >= 0.4, "Medium Risk")
    .otherwise("Low Risk")
)

# Save predictions
final_predictions.select(
    "CustomerID",
    "Recency",
    "TotalTransactions",
    "TotalSpent",
    "AvgOrderValue",
    "Churn",
    "ChurnPrediction",
    "ChurnRisk",
    "RiskCategory"
).write \
    .mode("overwrite") \
    .format("parquet") \
    .saveAsTable("customer_churn_prediction")

# Show model statistics
print("=" * 50)
print("Churn Prediction Model Results")
print("=" * 50)
print(f"AUC-ROC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")

# Show feature importance
feature_importance = model.featureImportances.toArray()
for i, importance in enumerate(feature_importance):
    print(f"{feature_cols[i]}: {importance:.4f}")

print("\nChurn Distribution:")
final_predictions.groupBy("RiskCategory").count().orderBy("count", ascending=False).show()

print("\nSample High-Risk Customers:")
final_predictions.filter(col("RiskCategory") == "High Risk") \
    .select("CustomerID", "Recency", "TotalSpent", "ChurnRisk") \
    .limit(10).show()

# Stop Spark session
spark.stop()
