# Complete ML Pipeline Execution Script
# Runs all 5 ML models and verifies results

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "     ML PIPELINE EXECUTION - ALL 5 MODELS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Function to run a Spark job
function Run-SparkJob {
    param(
        [string]$JobName,
        [string]$Script,
        [string]$Icon
    )
    
    Write-Host "$Icon Running $JobName..." -ForegroundColor Yellow
    $result = docker exec namenode spark-submit --master spark://namenode:7077 $Script 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ $JobName COMPLETE" -ForegroundColor Green
        Write-Host ""
        return $true
    } else {
        Write-Host "  ❌ $JobName FAILED" -ForegroundColor Red
        Write-Host ""
        Write-Host "Error: $($result | Select-Object -Last 5)" -ForegroundColor Red
        Write-Host ""
        return $false
    }
}

# Check current tables
Write-Host "Current Tables:" -ForegroundColor Cyan
$tables = docker exec -it hive-server beeline -u "jdbc:hive2://localhost:10000/ecommerce_db" -e "SHOW TABLES;" --silent=true 2>$null
Write-Host $tables
Write-Host ""

# Run ML Models
$success_count = 0

Write-Host "Starting ML Model Execution" -ForegroundColor Cyan
Write-Host ""

# Model 1: Customer Segmentation
if (Run-SparkJob "Customer Segmentation" "/home/scripts/ml_customer_segmentation.py" "Model1") {
    $success_count++
}

# Model 2: Sales Forecasting
if (Run-SparkJob "Sales Forecasting" "/home/scripts/ml_sales_forecasting.py" "Model2") {
    $success_count++
}

# Model 3: Product Recommendations
if (Run-SparkJob "Product Recommendations" "/home/scripts/ml_product_recommendations.py" "Model3") {
    $success_count++
}

# Model 4: Churn Prediction
if (Run-SparkJob "Churn Prediction" "/home/scripts/ml_churn_prediction.py" "Model4") {
    $success_count++
}

# Model 5: Anomaly Detection
if (Run-SparkJob "Anomaly Detection" "/home/scripts/ml_anomaly_detection.py" "Model5") {
    $success_count++
}

# Final Results
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "     ML PIPELINE RESULTS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Completed Models: $success_count / 5" -ForegroundColor Green
Write-Host ""

# Verify all tables
Write-Host "Final Table List:" -ForegroundColor Cyan
docker exec -it hive-server beeline -u "jdbc:hive2://localhost:10000/ecommerce_db" -e "SHOW TABLES;" --silent=true 2>$null

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if ($success_count -eq 5) {
    Write-Host "SUCCESS! All ML models completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Yellow
    Write-Host "   1. Open Dashboard: http://localhost:8501" -ForegroundColor White
    Write-Host "   2. Scroll to Machine Learning Insights" -ForegroundColor White
    Write-Host "   3. Explore all 5 ML visualizations!" -ForegroundColor White
    Write-Host ""
    
    $response = Read-Host "Open dashboard now? (Y/N)"
    if ($response -eq "Y" -or $response -eq "y") {
        Start-Process "http://localhost:8501"
    }
} else {
    Write-Host "Some models failed. Check the errors above." -ForegroundColor Yellow
    Write-Host "   You can re-run individual models or check logs." -ForegroundColor White
    Write-Host ""
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
