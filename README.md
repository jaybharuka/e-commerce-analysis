# 🛒 E-Commerce Data Pipeline with ML

## 📄 Description
This project implements an advanced ELT (Extract, Load, Transform) data pipeline for analyzing e-commerce transaction data from a UK-based online retailer. It uses a modern, containerized tech stack to enable efficient data ingestion, transformation, warehousing, and visualization. The pipeline now includes **5 machine learning models** that provide actionable insights for customer segmentation, sales forecasting, product recommendations, churn prediction, and anomaly detection.

---

## 🚀 Features
- **Data Extraction**: Ingest raw e-commerce transaction data from CSV files.
- **Data Transformation**: Clean, aggregate, and enrich data using Apache Spark.
- **Data Warehousing**: Store processed data in Hive tables for querying.
- **Workflow Orchestration**: Manage pipeline tasks using Apache Airflow.
- **Machine Learning**: 5 production-ready ML models for business insights
  - 👥 **Customer Segmentation** (K-Means clustering)
  - 📈 **Sales Forecasting** (Linear Regression)
  - 🎯 **Product Recommendations** (ALS collaborative filtering)
  - ⚠️ **Churn Prediction** (Random Forest classifier)
  - 🚨 **Anomaly Detection** (Statistical z-score analysis)
- **Interactive Dashboard**: Streamlit-powered analytics with ML insights
- **Scalability**: Leverage Hadoop for distributed data storage and processing.
- **Containerized Deployment**: Use Docker to run the entire stack seamlessly.

---
## 📝 Project Architecture
![Airflow Pipeline Graph](/output/ELT-Pipeline-Workflow.png)
*ELT pipeline workflow with integrated ML models*

---

## 🛠️ Tech Stack

| Technology | Purpose | Version |
|--- |--- | --- |
| Hadoop | Distributed storage (HDFS) | 3.2.1 |
| Python | Programming and scripting  | 3.9 |
| Spark | Distributed data processing | 3.2.2
| PySpark MLlib | Machine learning library | 3.2.2
| Airflow | Workflow orchestration | 2.3.3 |
| Zeppelin | Web-based notebook for exploration | 0.10.1 |
| Hive | Data warehousing solution | 2.3.2 |
| Postgres | Hive metastore backend | 15.1 |
| Streamlit | Interactive dashboard | Latest |

---

## 📂 Project Structure

```
.
├── dags/                        # Airflow DAGs for pipeline orchestration
├── configs/                     # Configuration files for Spark, Hive, and Hadoop
├── spark-scripts/               # Scripts for data ingestion, transformation, and ML models
│   ├── load.py                  # Data loading script
│   ├── transform.py             # Data transformation script
│   ├── ml_customer_segmentation.py      # K-Means clustering
│   ├── ml_sales_forecasting.py          # Sales prediction
│   ├── ml_product_recommendations.py    # ALS recommendations
│   ├── ml_churn_prediction.py           # Churn risk scoring
│   └── ml_anomaly_detection.py          # Anomaly detection
├── data/                        # Folder to store raw data
├── output/                      # Folder for results
├── streamlit/                   # Streamlit app for data visualization with ML insights
├── docker-compose.yml           # Docker Compose file to orchestrate services
└── ML_IMPLEMENTATION.md         # Comprehensive ML documentation

```

---

## 🚀 Quick Start Guide

### Prerequisites

- Docker
- Docker Compose
- Minimum 16GB RAM recommended
- Git

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AbderrahmaneOd/spark-hive-airflow-ecommerce-pipeline.git
   cd spark-hive-airflow-ecommerce-pipeline
   ```

2. **Launch Infrastructure**
   ```bash
   docker-compose up -d
   ```

3. **Access Services**

   | Service | URL | Credentials |
   |---------|-----|-------------|
   | HDFS Namenode | http://localhost:9870 | - |
   | YARN ResourceManager | http://localhost:8088 | - |
   | Spark Master | http://localhost:8080 | - |
   | Spark Worker | http://localhost:8081 | - |
   | Zeppelin | http://localhost:8082 | - |
   | Airflow | http://localhost:3000 | admin@gmail.com / admin |

4. **Configure Spark Connection**
   - Navigate to Airflow UI
   - Go to Admin > Connections
   - Edit "spark_default"
     * Host: spark://namenode
     * Port: 7077

---

## 🔍 Pipeline Workflow

The Airflow DAG demonstrates a complete data and ML workflow:
1. **Wait for Data File** - FileSensor monitors for new data
2. **Upload to HDFS** - BashOperator copies data to distributed storage
3. **Load Data** - Spark reads CSV and creates raw Hive table
4. **Transform Data** - Spark performs ETL and creates analytical tables
5. **ML Models (Run in Parallel)** - 5 ML models generate insights:
   - Customer Segmentation (K-Means)
   - Sales Forecasting (Linear Regression)
   - Product Recommendations (ALS)
   - Churn Prediction (Random Forest)
   - Anomaly Detection (Z-Score)

---

## 🤖 Machine Learning Models

### 1. Customer Segmentation 👥
- **Algorithm**: K-Means Clustering (k=4)
- **Features**: Total Purchases, Number of Transactions
- **Output**: 4 customer segments (High/Medium/Low Value, New Buyers)
- **Business Use**: Targeted marketing, personalized campaigns

### 2. Sales Forecasting 📈
- **Algorithm**: Linear Regression
- **Features**: Month Index, Lag features (1-3 months)
- **Output**: Next 3 months sales predictions
- **Business Use**: Inventory planning, revenue projections

### 3. Product Recommendations 🎯
- **Algorithm**: ALS Collaborative Filtering
- **Features**: Customer-Product-Rating matrix
- **Output**: Top 10 product recommendations per customer
- **Business Use**: Cross-selling, personalization, email marketing

### 4. Churn Prediction ⚠️
- **Algorithm**: Random Forest Classifier
- **Features**: RFM (Recency, Frequency, Monetary)
- **Output**: Churn probability and risk level (High/Medium/Low)
- **Business Use**: Retention campaigns, customer intervention

### 5. Anomaly Detection 🚨
- **Algorithm**: Statistical Z-Score Analysis
- **Features**: Quantity, Unit Price, Total Amount
- **Output**: Anomalous transactions with severity levels
- **Business Use**: Fraud detection, data quality monitoring

**📚 For detailed ML documentation, see [ML_IMPLEMENTATION.md](ML_IMPLEMENTATION.md)**

---
   
## 📊 Data Visualization
The project includes an enhanced Streamlit app with both business analytics and ML insights:

**Business Metrics**:
- Total Transactions, Quantity Sold, Revenue
- Sales by Country and Year
- Top Selling Products
- Sales Trend Over Time

**ML Insights**:
- Customer Segment Distribution (pie charts)
- Sales Forecast vs. Actual (line charts)
- Top Product Recommendations (tables)
- Churn Risk Analysis (bar charts, gauges)
- Transaction Anomalies (severity charts, alerts)

**Interactive Features**:
- Country-based filtering
- Date range selection
- Downloadable filtered data
- Real-time ML insights

---
 
## 🖼️ Project Outputs

### 📊 Streamlit Dashboard Screenshots

#### Overview & Business Metrics
![Dashboard Overview 1](dash-pictures/Screenshot\ 2025-10-30\ 231306.png)
*Main dashboard with business overview and key metrics*

![Dashboard Overview 2](dash-pictures/Screenshot\ 2025-10-30\ 231323.png)
*Sales trends and business performance indicators*

![Dashboard Overview 3](dash-pictures/Screenshot\ 2025-10-30\ 231341.png)
*Business summary with interactive controls*

#### RFM Analysis & Customer Segmentation
![RFM Analysis](dash-pictures/Screenshot\ 2025-10-30\ 231406.png)
*RFM (Recency, Frequency, Monetary) customer analysis*

![Customer Segmentation 1](dash-pictures/Screenshot\ 2025-10-30\ 231605.png)
*Customer segmentation visualization and insights*

![Customer Segmentation 2](dash-pictures/Screenshot\ 2025-10-30\ 231621.png)
*Detailed segmentation metrics and distribution*

#### Market Basket & Product Analysis
![Market Basket Analysis](dash-pictures/Screenshot\ 2025-10-30\ 231651.png)
*Product association rules and market basket insights*

![Product Analysis](dash-pictures/Screenshot\ 2025-10-30\ 231709.png)
*Top products and product performance metrics*

#### Advanced ML Models
![Sales Forecasting](dash-pictures/Screenshot\ 2025-10-30\ 231729.png)
*Sales forecast with Prophet time series prediction*

![Churn Prediction](dash-pictures/Screenshot\ 2025-10-30\ 231831.png)
*Customer churn risk prediction and analysis*

![Churn Details](dash-pictures/Screenshot\ 2025-10-30\ 231850.png)
*Detailed churn prediction metrics and risk distribution*

#### Product Recommendations
![Product Recommendations 1](dash-pictures/Screenshot\ 2025-10-30\ 231902.png)
*AI-powered product recommendations with rankings*

![Product Recommendations 2](dash-pictures/Screenshot\ 2025-10-30\ 231917.png)
*Bundle opportunities and cross-sell analysis*

#### Action Items & Summary
![Action Items](dash-pictures/Screenshot\ 2025-10-30\ 231943.png)
*Actionable recommendations and business insights*

---

### Architecture Diagrams

#### Airflow Pipeline Visualization
![Airflow Pipeline Graph](/output/ELT-Pipeline-Workflow.png)
*Airflow DAG (Directed Acyclic Graph) showing the pipeline workflow*

#### Airflow Task Dependencies
![Airflow Task Dependencies](/output/ecommerce_data_pipeline-Graph-Airflow.png)
*Detailed view of task dependencies in the data pipeline*

#### RFM Analysis Chart
![RFM Analysis Chart](/output/rfm.png)
*RFM segmentation visualization from ML pipeline*

````
