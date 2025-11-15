# 🛒 E-Commerce Analysis Platform - Setup Guide

## Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Git
- VS Code (recommended)

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/jaybharuka/e-commerce-analysis.git
cd e-commerce-analysis
```

### 2. Create `.env` File
Create a `.env` file in the root directory to store API keys (if needed for future extensions):
```bash
# Example .env file
DEBUG=True
STREAMLIT_THEME=dark
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note**: Main dependencies include:
> - `streamlit` - Interactive dashboard
> - `pandas`, `numpy` - Data processing
> - `scikit-learn` - Machine learning
> - `prophet` - Time series forecasting
> - `xgboost` - Gradient boosting
> - `plotly` - Interactive visualizations

---

## Running the Application

### Option 1: Streamlit Dashboard (Recommended for Development)

**Terminal 1 - Start the Dashboard:**
```bash
python -m streamlit run streamlit/app_complete.py --server.port=8502
```

Dashboard will be available at: **http://localhost:8502**

### Option 2: ML Models (Batch Processing)

**Run all ML models:**
```bash
python ml_analysis/run_all_analyses.py
```

**Run individual models:**
```bash
# RFM Analysis
python ml_analysis/rfm_analysis.py

# Customer Segmentation
python ml_analysis/customer_segmentation.py

# Market Basket Analysis
python ml_analysis/market_basket.py

# Sales Forecasting (Prophet)
python ml_analysis/sales_forecast_prophet.py

# Churn Prediction
python ml_analysis/churn_prediction.py

# Product Recommendations
python ml_analysis/product_recommendation.py
```

### Option 3: Docker Deployment

**Start the entire stack:**
```bash
docker-compose up -d
```

**View logs:**
```bash
docker-compose logs -f streamlit
```

**Stop services:**
```bash
docker-compose down
```

---

## Project Structure

```
e-commerce-analysis/
├── streamlit/
│   ├── app_complete.py          # Main Streamlit dashboard
│   ├── app.py                    # Alternative dashboard
│   └── app_direct.py             # Direct data analysis app
├── ml_analysis/
│   ├── rfm_analysis.py           # RFM customer segmentation
│   ├── customer_segmentation.py  # K-Means clustering
│   ├── market_basket.py          # Apriori algorithm
│   ├── sales_forecast_prophet.py # Prophet time series
│   ├── churn_prediction.py       # XGBoost classification
│   ├── product_recommendation.py # Collaborative filtering
│   └── run_all_analyses.py       # Run all models
├── spark-scripts/
│   ├── load.py                   # Data loading
│   ├── transform.py              # Data transformation
│   └── ml_*.py                   # Spark ML models
├── data/
│   └── data.csv                  # Source e-commerce data
├── ml_results/
│   ├── rfm_analysis.csv
│   ├── customer_segments.csv
│   ├── product_associations.csv
│   ├── sales_forecast.csv
│   ├── churn_prediction.csv
│   └── product_recommendations.csv
├── configs/                      # Configuration files
├── dags/                         # Airflow DAGs
├── docker_image_conf/            # Docker configurations
├── dash-pictures/                # Dashboard screenshots
├── output/                       # Output files
├── docker-compose.yaml           # Docker Compose config
├── README.md                     # Project documentation
└── SETUP.md                      # This file
```

---

## Dashboard Features

### 📊 Available Sections

1. **Business Overview**
   - Total Transactions, Revenue, Quantity Sold
   - Sales by Country and Year
   - Interactive filters

2. **RFM Analysis**
   - Customer value scoring
   - Recency, Frequency, Monetary metrics
   - Segmentation insights

3. **Customer Segmentation**
   - K-Means clustering (4 segments)
   - Behavioral analysis
   - Segment distribution

4. **Market Basket Analysis**
   - Product association rules
   - Co-purchase patterns
   - Bundle opportunities

5. **Sales Forecasting**
   - Prophet time series predictions
   - 3-month forecast with confidence intervals
   - Trend analysis

6. **Churn Prediction**
   - XGBoost churn risk scoring
   - Risk distribution analysis
   - High-risk customer identification

7. **Product Recommendations**
   - Collaborative filtering
   - Personalized suggestions
   - Cross-sell opportunities

8. **Action Items**
   - Automated business insights
   - Strategic recommendations

---

## ML Models Summary

| Model | Algorithm | Output | Use Case |
|-------|-----------|--------|----------|
| RFM Analysis | Rule-based | Customer scores | Segmentation |
| Segmentation | K-Means | 4 clusters | Behavioral groups |
| Market Basket | Apriori | Association rules | Cross-selling |
| Sales Forecast | Prophet | 3-month forecast | Revenue planning |
| Churn Prediction | XGBoost | Risk scores | Retention |
| Recommendations | Collaborative Filtering | Top products | Personalization |

---

## Data Files

### Input Data
- **Location**: `data/data.csv`
- **Records**: 541,909 transactions
- **Customers**: 4,338
- **Products**: 3,665

### Output Data
All ML model results are saved in `ml_results/` as CSV files:
- `rfm_analysis.csv`
- `customer_segments.csv`
- `product_associations.csv`
- `sales_forecast.csv`
- `churn_prediction.csv`
- `product_recommendations.csv`

---

## Configuration

### Streamlit Settings
Located in `streamlit/app_complete.py`:
- **Theme**: Dark mode
- **Port**: 8502 (configurable)
- **Layout**: Wide mode

### ML Model Parameters
Each model file contains configurable parameters:
```python
# Example: sales_forecast_prophet.py
FORECAST_PERIODS = 90  # 3 months
CONFIDENCE_INTERVAL = 0.95

# Example: churn_prediction.py
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Example: customer_segmentation.py
N_CLUSTERS = 4
```

---

## Troubleshooting

### Issue: Port 8502 already in use
**Solution**: Change port in command
```bash
python -m streamlit run streamlit/app_complete.py --server.port=8503
```

### Issue: Module not found errors
**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

### Issue: Slow ML model training
**Solution**: Check system resources or reduce data size in model scripts

### Issue: Dashboard not loading images
**Solution**: Ensure `dash-pictures/` folder exists with screenshots

---

## Performance Metrics

### Model Accuracy
- **Churn Prediction**: 68.80% accuracy, 0.69 AUC-ROC
- **Sales Forecast**: MAPE 31.56%, RMSE $13,761.67
- **Customer Segmentation**: 4 distinct clusters identified

### Data Processing
- **Total Records**: 541,909 transactions
- **Processing Time**: ~2-5 seconds for all models
- **Project Size**: ~44 MB

---

## Next Steps

1. ✅ Clone the repository
2. ✅ Install requirements
3. ✅ Start the Streamlit dashboard
4. ✅ Explore the ML models
5. ✅ Run batch analyses
6. ✅ Deploy with Docker (optional)

---

## Support & Documentation

- **GitHub**: https://github.com/jaybharuka/e-commerce-analysis
- **Dashboard**: http://localhost:8502 (after starting)
- **Data Source**: UK Online Retail Dataset (541K+ transactions)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Last Updated**: November 15, 2025  
**Version**: 1.0
