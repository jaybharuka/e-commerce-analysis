import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
import pandas as pd
import logging
from airflow.utils.dates import days_ago

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# Configuration
DATA_FOLDER = '/home/'
RAW_DATA_FILE = 'data.csv'
HDFS_DATA_PATH = '/input/ecommerce_data/'

# Create the DAG
with DAG(
    'ecommerce_data_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:
    
    # Sensor to wait for the data file
    wait_for_data_file = FileSensor(
        task_id='wait_for_data_file',
        filepath=os.path.join(DATA_FOLDER, RAW_DATA_FILE),
        poke_interval=30,  # Check every 30 seconds
        timeout=60 * 60,  # Timeout after 1 hour
        mode='reschedule',  # Allows Airflow to run other tasks while waiting
        soft_fail=False  # Fail the task if file not found
    )
    
    
    # Upload to HDFS
    upload_data_to_hdfs = BashOperator(
        task_id='upload_data_to_hdfs',
        bash_command=f"""
        hdfs dfs -test -d {HDFS_DATA_PATH} || hdfs dfs -mkdir -p {HDFS_DATA_PATH} && \
        hdfs dfs -put -f {os.path.join(DATA_FOLDER, RAW_DATA_FILE)} {HDFS_DATA_PATH}
        """,
    )
    
    
    # Load 
    load_data = SparkSubmitOperator(
        task_id='load_spark_job',
        application='/home/scripts/load.py',
        conn_id='spark_default',
        verbose=True,
        application_args=[],
        dag=dag,
    )
    
    
    # Transform
    transform_data = SparkSubmitOperator(
        task_id='transform_spark_job',
        application='/home/scripts/transform.py',
        conn_id='spark_default',
        verbose=True,
        application_args=[],
        dag=dag,
    )
    
    # ML Model 1: Customer Segmentation
    ml_customer_segmentation = SparkSubmitOperator(
        task_id='ml_customer_segmentation',
        application='/home/scripts/ml_customer_segmentation.py',
        conn_id='spark_default',
        verbose=True,
        application_args=[],
        dag=dag,
    )
    
    # ML Model 2: Sales Forecasting
    ml_sales_forecasting = SparkSubmitOperator(
        task_id='ml_sales_forecasting',
        application='/home/scripts/ml_sales_forecasting.py',
        conn_id='spark_default',
        verbose=True,
        application_args=[],
        dag=dag,
    )
    
    # ML Model 3: Product Recommendations
    ml_product_recommendations = SparkSubmitOperator(
        task_id='ml_product_recommendations',
        application='/home/scripts/ml_product_recommendations.py',
        conn_id='spark_default',
        verbose=True,
        application_args=[],
        dag=dag,
    )
    
    # ML Model 4: Churn Prediction
    ml_churn_prediction = SparkSubmitOperator(
        task_id='ml_churn_prediction',
        application='/home/scripts/ml_churn_prediction.py',
        conn_id='spark_default',
        verbose=True,
        application_args=[],
        dag=dag,
    )
    
    # ML Model 5: Anomaly Detection
    ml_anomaly_detection = SparkSubmitOperator(
        task_id='ml_anomaly_detection',
        application='/home/scripts/ml_anomaly_detection.py',
        conn_id='spark_default',
        verbose=True,
        application_args=[],
        dag=dag,
    )
    
    # Define task dependencies
    wait_for_data_file >> upload_data_to_hdfs >> load_data >> transform_data
    
    # ML models run in parallel after transform completes
    transform_data >> [ml_customer_segmentation, ml_sales_forecasting, ml_product_recommendations, ml_churn_prediction, ml_anomaly_detection]
