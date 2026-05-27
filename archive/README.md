# Archive

This directory contains **legacy and duplicate scripts** that have been superseded by canonical versions in the main tree.
Do **not** import or run these directly; they are kept for reference only.

| File | Canonical Replacement | Reason archived |
|------|-----------------------|-----------------|
| `spark-scripts/ml_sales_forecast.py` | `spark-scripts/ml_sales_forecasting.py` | Duplicate forecasting script |
| `spark-scripts/transform_simple.py` | `spark-scripts/transform.py` | Duplicate transform; also has wrong `monthly_sales` column alias (`TotalSales` instead of `MonthlySales`) |
| `streamlit/app_direct.py` | `streamlit/app.py` | Second Streamlit variant reading from CSV instead of Hive |
| `ml_analysis/*` | `spark-scripts/ml_*.py` | Non-Spark prototype analyses; replaced by full PySpark pipeline |
