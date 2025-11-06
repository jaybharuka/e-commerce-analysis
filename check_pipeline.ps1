# ML Pipeline Status Checker
# Run this script to check the current status of your ML pipeline

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  E-Commerce ML Pipeline Status Check" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. Check if containers are running
Write-Host "1. Checking Docker Containers..." -ForegroundColor Yellow
$containers = docker-compose ps --services --filter "status=running" 2>$null
if ($containers) {
    Write-Host "   ✅ Containers Running: " -ForegroundColor Green -NoNewline
    Write-Host "$($containers.Count) containers" -ForegroundColor White
} else {
    Write-Host "   ❌ No containers running!" -ForegroundColor Red
    Write-Host "   Run: docker-compose up -d" -ForegroundColor Yellow
    exit
}

# 2. Check Hive Tables
Write-Host "`n2. Checking Hive Tables..." -ForegroundColor Yellow
try {
    $tables = docker-compose exec -T hive-server beeline -u "jdbc:hive2://localhost:10000/ecommerce_db" -e "SHOW TABLES;" --silent=true 2>$null | Select-String "ecommerce|sales|customer|product|anomal|segment|churn|forecast|recommendation"
    
    $tableList = @()
    foreach ($line in $tables) {
        if ($line -match "\|\s*(\w+)\s*\|") {
            $tableList += $matches[1]
        }
    }
    
    Write-Host "   Tables Found: $($tableList.Count)" -ForegroundColor White
    
    # Base Tables
    Write-Host "`n   Base Tables:" -ForegroundColor Cyan
    $baseTables = @("ecommerce_raw", "ecommerce_transformed", "monthly_sales", "sales_per_country", "customer_metrics", "product_performance")
    foreach ($table in $baseTables) {
        if ($tableList -contains $table) {
            Write-Host "   ✅ $table" -ForegroundColor Green
        } else {
            Write-Host "   ⏳ $table (pending...)" -ForegroundColor Gray
        }
    }
    
    # ML Tables
    Write-Host "`n   ML Tables:" -ForegroundColor Cyan
    $mlTables = @("customer_segments", "sales_forecast", "product_recommendations", "customer_churn_prediction", "transaction_anomalies")
    foreach ($table in $mlTables) {
        if ($tableList -contains $table) {
            Write-Host "   ✅ $table" -ForegroundColor Green
        } else {
            Write-Host "   ⏳ $table (pending...)" -ForegroundColor Gray
        }
    }
    
} catch {
    Write-Host "   ⚠️  Could not connect to Hive" -ForegroundColor Yellow
}

# 3. Quick Links
Write-Host "`n3. Service URLs:" -ForegroundColor Yellow
Write-Host "   🌐 Airflow UI:   http://localhost:3000" -ForegroundColor White
Write-Host "      Login: admin@gmail.com / admin" -ForegroundColor Gray
Write-Host "   📊 Dashboard:    http://localhost:8501" -ForegroundColor White
Write-Host "   🗄️  HDFS:         http://localhost:9870" -ForegroundColor White
Write-Host "   ⚡ Spark:        http://localhost:8080" -ForegroundColor White

# 4. Instructions
Write-Host "`n4. What to Do Next:" -ForegroundColor Yellow
Write-Host "   • Open Airflow UI to monitor DAG progress" -ForegroundColor White
Write-Host "   • Wait for all tasks to turn GREEN ✅" -ForegroundColor White
Write-Host "   • Expected time: 15-20 minutes total" -ForegroundColor White
Write-Host "   • Once complete, open Dashboard to see ML insights" -ForegroundColor White

Write-Host "`n========================================`n" -ForegroundColor Cyan

# Option to open UIs
Write-Host "Would you like to open the UIs? (Y/N): " -ForegroundColor Yellow -NoNewline
$response = Read-Host

if ($response -eq "Y" -or $response -eq "y") {
    Write-Host "`nOpening Airflow UI and Dashboard...`n" -ForegroundColor Green
    Start-Process "http://localhost:3000"
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:8501"
} else {
    Write-Host "`nManually open:" -ForegroundColor Yellow
    Write-Host "  Airflow: http://localhost:3000" -ForegroundColor White
    Write-Host "  Dashboard: http://localhost:8501`n" -ForegroundColor White
}
