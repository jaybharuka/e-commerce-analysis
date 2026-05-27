"""
ecommerce-data-pipeline.py
--------------------------
Airflow DAG for the e-commerce ELT pipeline.

Flow:
    wait_for_data_file
        → upload_data_to_hdfs
        → load_spark_job
        → transform_spark_job
        → validate_row_counts          ← data-quality gate
        → [ml_customer_segmentation,
           ml_sales_forecasting,
           ml_product_recommendations,
           ml_churn_prediction,
           ml_anomaly_detection]       ← parallel ML jobs
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

# ---------------------------------------------------------------------------
# Default arguments
# ---------------------------------------------------------------------------
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------------------------------
# Configuration — override via Airflow Variables or environment variables
# ---------------------------------------------------------------------------
DATA_FOLDER = os.getenv("PIPELINE_DATA_FOLDER", "/home/")
RAW_DATA_FILE = os.getenv("PIPELINE_RAW_DATA_FILE", "data.csv")
HDFS_DATA_PATH = os.getenv("PIPELINE_HDFS_PATH", "/input/ecommerce_data/")
SCRIPTS_PATH = os.getenv("PIPELINE_SCRIPTS_PATH", "/home/scripts")

# Minimum acceptable row counts after each stage (set to 0 to skip threshold)
MIN_ROWS_TRANSFORMED = int(os.getenv("PIPELINE_MIN_ROWS_TRANSFORMED", "1000"))
MIN_ROWS_CUSTOMER_METRICS = int(os.getenv("PIPELINE_MIN_ROWS_CUSTOMER_METRICS", "100"))

# ---------------------------------------------------------------------------
# Row-count validation helper
# ---------------------------------------------------------------------------
def validate_row_counts(**context) -> None:
    """
    Connect to Hive via pyhive and assert that key tables have at least the
    minimum expected number of rows.  Raises ValueError if any check fails,
    which will mark the task (and the DAG run) as failed.
    """
    from pyhive import hive  # installed in the Airflow environment

    checks = {
        "ecommerce_transformed": MIN_ROWS_TRANSFORMED,
        "customer_metrics": MIN_ROWS_CUSTOMER_METRICS,
        "monthly_sales": 1,
        "sales_per_country": 1,
        "product_performance": 1,
    }

    hive_host = os.getenv("HIVE_HOST", "hive-server")
    hive_port = int(os.getenv("HIVE_PORT", "10000"))

    conn = hive.Connection(host=hive_host, port=hive_port, database="ecommerce_db")
    cursor = conn.cursor()

    failures = []
    for table, min_count in checks.items():
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        actual = cursor.fetchone()[0]
        status = "✅" if actual >= min_count else "❌"
        print(f"  {status}  {table}: {actual:,} rows (min={min_count:,})")
        if actual < min_count:
            failures.append(
                f"{table}: expected >= {min_count} rows, got {actual}"
            )

    cursor.close()
    conn.close()

    if failures:
        raise ValueError(
            "Row-count validation FAILED:\n" + "\n".join(f"  • {f}" for f in failures)
        )

    print("Row-count validation passed for all tables.")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    "ecommerce_data_pipeline",
    default_args=default_args,
    description="E-Commerce ELT pipeline: ingest → transform → validate → ML",
    schedule_interval=None,
    catchup=False,
    tags=["ecommerce", "elt", "spark", "ml"],
) as dag:

    # 1. Wait for source data file
    wait_for_data_file = FileSensor(
        task_id="wait_for_data_file",
        filepath=os.path.join(DATA_FOLDER, RAW_DATA_FILE),
        poke_interval=30,
        timeout=60 * 60,
        mode="reschedule",
        soft_fail=False,
    )

    # 2. Upload raw CSV to HDFS
    upload_data_to_hdfs = BashOperator(
        task_id="upload_data_to_hdfs",
        bash_command=(
            f"hdfs dfs -test -d {HDFS_DATA_PATH} "
            f"|| hdfs dfs -mkdir -p {HDFS_DATA_PATH} && "
            f"hdfs dfs -put -f {os.path.join(DATA_FOLDER, RAW_DATA_FILE)} {HDFS_DATA_PATH}"
        ),
    )

    # 3. Load raw data into Hive (ecommerce_raw table)
    load_data = SparkSubmitOperator(
        task_id="load_spark_job",
        application=f"{SCRIPTS_PATH}/load.py",
        conn_id="spark_default",
        verbose=True,
    )

    # 4. Transform: clean, enrich, aggregate → ecommerce_transformed + summary tables
    transform_data = SparkSubmitOperator(
        task_id="transform_spark_job",
        application=f"{SCRIPTS_PATH}/transform.py",
        conn_id="spark_default",
        verbose=True,
    )

    # 5. Data-quality gate: assert row counts meet minimum thresholds
    validate_data = PythonOperator(
        task_id="validate_row_counts",
        python_callable=validate_row_counts,
        provide_context=True,
    )

    # 6a. ML — Customer Segmentation (K-Means)
    ml_customer_segmentation = SparkSubmitOperator(
        task_id="ml_customer_segmentation",
        application=f"{SCRIPTS_PATH}/ml_customer_segmentation.py",
        conn_id="spark_default",
        verbose=True,
    )

    # 6b. ML — Sales Forecasting (Linear Regression)
    ml_sales_forecasting = SparkSubmitOperator(
        task_id="ml_sales_forecasting",
        application=f"{SCRIPTS_PATH}/ml_sales_forecasting.py",
        conn_id="spark_default",
        verbose=True,
    )

    # 6c. ML — Product Recommendations (ALS collaborative filtering)
    ml_product_recommendations = SparkSubmitOperator(
        task_id="ml_product_recommendations",
        application=f"{SCRIPTS_PATH}/ml_product_recommendations.py",
        conn_id="spark_default",
        verbose=True,
    )

    # 6d. ML — Churn Prediction (Random Forest)
    ml_churn_prediction = SparkSubmitOperator(
        task_id="ml_churn_prediction",
        application=f"{SCRIPTS_PATH}/ml_churn_prediction.py",
        conn_id="spark_default",
        verbose=True,
    )

    # 6e. ML — Anomaly Detection (z-score)
    ml_anomaly_detection = SparkSubmitOperator(
        task_id="ml_anomaly_detection",
        application=f"{SCRIPTS_PATH}/ml_anomaly_detection.py",
        conn_id="spark_default",
        verbose=True,
    )

    # ---------------------------------------------------------------------------
    # Task dependencies
    # ---------------------------------------------------------------------------
    (
        wait_for_data_file
        >> upload_data_to_hdfs
        >> load_data
        >> transform_data
        >> validate_data
        >> [
            ml_customer_segmentation,
            ml_sales_forecasting,
            ml_product_recommendations,
            ml_churn_prediction,
            ml_anomaly_detection,
        ]
    )
