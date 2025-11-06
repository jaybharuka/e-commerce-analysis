#!/usr/bin/env python3
"""
Live ML Pipeline Monitor
Continuously checks and displays the status of your ML pipeline
"""

import subprocess
import time
import sys
from datetime import datetime

def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout
    except:
        return ""

def get_hive_tables():
    """Get list of tables in Hive"""
    cmd = 'docker-compose exec -T hive-server beeline -u "jdbc:hive2://localhost:10000/ecommerce_db" -e "SHOW TABLES;" --silent=true 2>nul'
    output = run_command(cmd)
    
    tables = []
    for line in output.split('\n'):
        if '|' in line and 'tab_name' not in line:
            parts = line.split('|')
            if len(parts) >= 2:
                table_name = parts[1].strip()
                if table_name and table_name != '':
                    tables.append(table_name)
    return tables

def clear_screen():
    """Clear the terminal screen"""
    subprocess.run('cls' if sys.platform == 'win32' else 'clear', shell=True)

def display_status():
    """Display the current pipeline status"""
    clear_screen()
    
    print("=" * 60)
    print("     🚀 E-COMMERCE ML PIPELINE - LIVE MONITOR")
    print("=" * 60)
    print(f"\n⏰ Last Update: {datetime.now().strftime('%H:%M:%S')}\n")
    
    # Get current tables
    tables = get_hive_tables()
    
    # Define expected tables
    base_tables = {
        'ecommerce_raw': '📦 Raw Data',
        'ecommerce_transformed': '🔄 Transformed Data',
        'monthly_sales': '📅 Monthly Sales',
        'sales_per_country': '🌍 Country Sales',
        'customer_metrics': '👤 Customer Metrics',
        'product_performance': '📊 Product Performance'
    }
    
    ml_tables = {
        'customer_segments': '👥 Customer Segmentation',
        'sales_forecast': '📈 Sales Forecast',
        'product_recommendations': '🎯 Product Recommendations',
        'customer_churn_prediction': '⚠️  Churn Prediction',
        'transaction_anomalies': '🚨 Anomaly Detection'
    }
    
    # Display Base Tables
    print("📦 BASE TABLES:")
    print("-" * 60)
    base_complete = 0
    for table, desc in base_tables.items():
        if table in tables:
            print(f"  ✅ {desc:<30} [{table}]")
            base_complete += 1
        else:
            print(f"  ⏳ {desc:<30} [pending...]")
    
    print(f"\n  Progress: {base_complete}/{len(base_tables)} complete")
    
    # Display ML Tables
    print("\n🤖 ML MODEL TABLES:")
    print("-" * 60)
    ml_complete = 0
    for table, desc in ml_tables.items():
        if table in tables:
            print(f"  ✅ {desc:<30} [{table}]")
            ml_complete += 1
        else:
            print(f"  ⏳ {desc:<30} [pending...]")
    
    print(f"\n  Progress: {ml_complete}/{len(ml_tables)} complete")
    
    # Overall Progress
    total_complete = base_complete + ml_complete
    total_tables = len(base_tables) + len(ml_tables)
    progress_pct = (total_complete / total_tables) * 100
    
    print("\n" + "=" * 60)
    print(f"📊 OVERALL PROGRESS: {total_complete}/{total_tables} tables ({progress_pct:.1f}%)")
    print("=" * 60)
    
    # Status message
    if total_complete == 0:
        print("\n🔄 Pipeline starting... Please wait...")
    elif base_complete < len(base_tables):
        print("\n🔄 Base pipeline running... Transform & analytics in progress")
    elif ml_complete == 0:
        print("\n🔄 Base complete! ML models starting...")
    elif ml_complete < len(ml_tables):
        print(f"\n🤖 ML models running... {ml_complete}/{len(ml_tables)} models complete")
    else:
        print("\n🎉 PIPELINE COMPLETE! All tables created successfully!")
        print("\n✨ Next Steps:")
        print("   1. Open Dashboard: http://localhost:8501")
        print("   2. Scroll to 'Machine Learning Insights'")
        print("   3. Explore the 5 ML visualizations!")
        return True  # Signal completion
    
    # Service Links
    print("\n🌐 SERVICE URLS:")
    print("-" * 60)
    print("   Airflow:  http://localhost:3000 (admin@gmail.com / admin)")
    print("   Dashboard: http://localhost:8501")
    print("   Spark:     http://localhost:8080")
    
    print("\n💡 Press Ctrl+C to stop monitoring\n")
    
    return False  # Not complete yet

def main():
    """Main monitoring loop"""
    print("\n🚀 Starting ML Pipeline Monitor...")
    print("📡 Will refresh every 15 seconds\n")
    time.sleep(2)
    
    try:
        while True:
            complete = display_status()
            if complete:
                print("\n✅ Monitoring complete! Pipeline finished successfully.\n")
                break
            time.sleep(15)  # Refresh every 15 seconds
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user.")
        print("💡 Run this script again anytime to check status.\n")

if __name__ == "__main__":
    main()
