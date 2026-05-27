# E-Commerce ELT Analytics Pipeline

An end-to-end data engineering project that ingests raw e-commerce transaction data, transforms it through a Hadoop/Hive/Spark stack, runs five parallel ML models, and surfaces insights via a Streamlit dashboard — all orchestrated by Apache Airflow and deployed to AWS with Terraform through a Jenkins CI/CD pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Ingestion                                                              │
│  data.csv  ──Flume──►  HDFS (/input/ecommerce_data/)                   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ Airflow DAG: ecommerce_data_pipeline
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Load & Transform  (PySpark → Hive)                                     │
│                                                                         │
│  load.py           →  ecommerce_raw            (Hive / Parquet)         │
│  transform.py      →  ecommerce_transformed    (cleaned + enriched)     │
│                    →  sales_per_country                                 │
│                    →  monthly_sales                                     │
│                    →  customer_metrics                                  │
│                    →  product_performance                               │
│                                                                         │
│  validate_row_counts  ◄── data-quality gate (min row thresholds)        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ fan-out (parallel)
          ┌─────────────────────┼──────────────────────────┐
          ▼                     ▼                          ▼
┌─────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ Customer Seg.   │  │  Sales Forecasting   │  │ Product Recomm.      │
│ (K-Means)       │  │  (Linear Regression) │  │ (ALS collab. filter) │
└─────────────────┘  └──────────────────────┘  └──────────────────────┘
          ▼                     ▼
┌─────────────────┐  ┌──────────────────────┐
│ Churn Predict.  │  │  Anomaly Detection   │
│ (Random Forest) │  │  (z-score)           │
└─────────────────┘  └──────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Streamlit Dashboard  (port 8501)                                       │
│  Reads Hive tables → Plotly charts → interactive filters                │
└─────────────────────────────────────────────────────────────────────────┘

CI/CD (Jenkins)
  Checkout → Unit Tests → SonarQube → Docker Build → Push → Terraform Apply → EC2
```

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Orchestration | Apache Airflow | 2.x |
| Distributed storage | Apache Hadoop HDFS | 3.2 |
| Query engine | Apache Hive | 2.3 |
| Processing | Apache Spark / PySpark | 3.5 |
| Ingestion | Apache Flume | 1.x |
| Dashboard | Streamlit | 1.35 |
| Visualisation | Plotly | 5.22 |
| Containerisation | Docker / Docker Compose | latest |
| IaC | Terraform | 1.x |
| CI/CD | Jenkins | 2.x |
| Code quality | SonarQube | LTS community |
| Monitoring | Prometheus + Grafana | latest |
| Cloud | AWS EC2 (ap-south-1) | — |

---

## ML Models

| Task | Algorithm | Input table | Output table |
|------|-----------|-------------|--------------|
| Customer Segmentation | K-Means (k=4) | `customer_metrics` | `customer_segments` |
| Sales Forecasting | Linear Regression | `monthly_sales` | `sales_forecast` |
| Product Recommendations | ALS collaborative filtering | `ecommerce_transformed` | `product_recommendations` |
| Churn Prediction | Random Forest (50 trees) | `ecommerce_transformed` | `customer_churn_prediction` |
| Anomaly Detection | Z-score (3-sigma threshold) | `ecommerce_transformed` | `transaction_anomalies` |

---

## Three Run Modes

### Mode 1 — Local Docker Compose (development)

Spins up the full Hadoop / Hive / Spark / Airflow cluster on your machine.
Requires ~16 GB RAM.

```bash
# 1. Create the secret env files (gitignored)
cp sonarqube.env.example sonarqube.env
# Edit sonarqube.env and set a real password

# 2. Start all services
docker-compose up -d

# 3. Trigger the Airflow DAG manually
#    Open http://localhost:3001  → enable & trigger 'ecommerce_data_pipeline'

# 4. View the dashboard
open http://localhost:8501
```

### Mode 2 — Jenkins CI/CD → AWS EC2 (production)

Triggered automatically on every push to `main`.

```
Jenkins prerequisites
─────────────────────
Credential ID               Kind                     Value
dockerhub-creds             Username/password        DockerHub login
aws-credentials             Username/password        AWS_ACCESS_KEY_ID / SECRET
sonarqube-token             Secret text              SonarQube token
grafana-admin-password      Secret text              Strong Grafana admin password
```

```bash
# 1. Configure the credentials above in Jenkins
# 2. Create a Pipeline job pointing at this repo
# 3. Push to main — pipeline runs automatically

# Pipeline stages:
#   Checkout → Unit Tests → SonarQube → Docker Build
#   → Push DockerHub → Terraform Init/Validate/Plan/Apply → Deployed
```

### Mode 3 — Unit tests only (no cluster needed)

```bash
pip install -r requirements.txt
pytest tests/ -v

# Expected:
#   tests/test_transformations.py ......................  22 passed
```

---

## Project Structure

```
ecommerce-elt-pipeline-main/
├── dags/
│   └── ecommerce-data-pipeline.py   # Airflow DAG (canonical)
├── spark-scripts/
│   ├── load.py                      # Ingest CSV → Hive ecommerce_raw
│   ├── transform.py                 # Clean + aggregate → 5 Hive tables
│   ├── ml_customer_segmentation.py  # K-Means segmentation
│   ├── ml_sales_forecasting.py      # Linear Regression forecast
│   ├── ml_product_recommendations.py # ALS recommendations
│   ├── ml_churn_prediction.py       # Random Forest churn
│   ├── ml_anomaly_detection.py      # Z-score anomaly detection
│   ├── ml_rfm_analysis.py           # RFM scoring (standalone)
│   └── ml_market_basket.py          # Market basket (standalone)
├── streamlit/
│   ├── app.py                       # Dashboard (canonical)
│   └── dockerfile
├── tests/
│   └── test_transformations.py      # Unit tests for transform logic
├── terraform/
│   ├── main.tf                      # EC2 + SG + user_data bootstrap
│   ├── variables.tf                 # All input variables (no secrets)
│   ├── outputs.tf
│   ├── backend.tf
│   └── provider.tf
├── configs/                         # Hadoop/Hive/Flume config files
├── docker_image_conf/               # Dockerfiles for each cluster node
├── flume_config/
├── data/
│   └── data.csv                     # Sample dataset (UCI Online Retail)
├── archive/                         # Superseded scripts — do not import
├── docker-compose.yaml
├── Jenkinsfile
├── prometheus.yml
├── requirements.txt                 # Version-pinned Python deps
├── sonarqube.env.example            # Template — copy to sonarqube.env
└── .gitignore
```

---

## Data Pipeline — Key Hive Schema

### `ecommerce_transformed` (main fact table)

| Column | Type | Notes |
|--------|------|-------|
| InvoiceNo | STRING | |
| StockCode | STRING | |
| Description | STRING | |
| Quantity | INT | Filtered: > 0 |
| InvoiceDate | TIMESTAMP | Parsed from `M/d/yyyy H:mm` |
| UnitPrice | DOUBLE | Filtered: > 0 |
| CustomerID | STRING | |
| Country | STRING | |
| Month | INT | Extracted from InvoiceDate |
| Year | INT | Extracted from InvoiceDate |
| TotalAmount | DOUBLE | `round(Quantity * UnitPrice, 2)` |

### `customer_metrics` (input to Customer Segmentation ML)

| Column | Type |
|--------|------|
| CustomerID | STRING |
| TotalPurchases | DOUBLE |
| NumberOfTransactions | LONG |

> `ml_customer_segmentation.py` references **`TotalPurchases`** and **`NumberOfTransactions`** exactly.

---

## CI/CD Stages

```
Checkout → Unit Tests → SonarQube → Docker Build → Push DockerHub
  → TF Init → TF Validate → TF Plan → TF Apply → EC2 Online
                                                      │
                                            Streamlit  :8501
                                            Prometheus :9090
                                            Grafana    :3000
```

On **failure**, the post-build step runs `terraform destroy` automatically to avoid orphaned AWS resources.

---

## Environment Variables / Secrets

| Variable | Where set | Purpose |
|----------|-----------|---------|
| `PIPELINE_DATA_FOLDER` | Airflow env | Path to CSV on namenode |
| `PIPELINE_HDFS_PATH` | Airflow env | HDFS destination |
| `PIPELINE_MIN_ROWS_TRANSFORMED` | Airflow env | Row-count gate (default: 1000) |
| `PIPELINE_MIN_ROWS_CUSTOMER_METRICS` | Airflow env | Row-count gate (default: 100) |
| `HIVE_HOST` / `HIVE_PORT` | Airflow env | Hive connectivity for validator |
| `TF_VAR_grafana_admin_password` | Jenkins secret → env | Grafana admin password |
| `SONAR_DB_USER` / `SONAR_DB_PASSWORD` | `sonarqube.env` | SonarQube Postgres creds |
| `AWS_ACCESS_KEY_ID` | Jenkins secret | Terraform AWS auth |
| `AWS_SECRET_ACCESS_KEY` | Jenkins secret | Terraform AWS auth |

---

## Known Limitations

1. **Batch-only ingestion** — the FileSensor polls for a new CSV file. A Kafka → Spark Structured Streaming path would enable real-time processing.

2. **Single-node Spark** — Docker Compose runs Spark in standalone mode. Datasets significantly larger than the UCI sample (~541K rows) need a proper multi-node cluster or a managed service (EMR, Dataproc).

3. **ALS cold-start** — the recommendations model cannot produce scores for customers or products with no purchase history (`coldStartStrategy="drop"`). A hybrid content-based fallback would be needed for new users/items.

4. **3-month forecast horizon only** — `ml_sales_forecasting.py` uses a simple linear model. A time-series approach (Prophet, SARIMA) would improve seasonality capture.

5. **Full table rewrites on every run** — all Hive tables are recreated from scratch (`DROP TABLE IF EXISTS` + `overwrite`). Incremental `INSERT OVERWRITE PARTITION` logic is needed for production-scale runs.

6. **Grafana dashboards not version-controlled** — dashboards must be configured manually after first boot. Use Grafana provisioning YAML to export and commit them.

7. **Local Terraform state** — `terraform/backend.tf` uses a local state file. An S3 + DynamoDB remote backend is required for team use or pipeline-managed deploys.

---

## Dataset

[UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) — 541,909 transactions across 8 columns from a UK-based online retailer (2010–2011).
