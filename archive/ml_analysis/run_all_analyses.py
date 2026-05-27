"""
Run All ML Analyses
Executes all ML models and generates CSV results for the dashboard
"""
import subprocess
import os
import sys

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

print("="*70)
print("🚀 E-COMMERCE ML ANALYSIS PIPELINE")
print("="*70)

analyses = [
    ('rfm_analysis.py', '💎 RFM Analysis'),
    ('customer_segmentation.py', '👥 Customer Segmentation'),
    ('market_basket.py', '🛒 Market Basket Analysis'),
    ('sales_forecast_prophet.py', '📈 Advanced Sales Forecasting (Prophet)'),
    ('churn_prediction.py', '⚠️ Churn Prediction Model'),
    ('product_recommendation.py', '🎁 Product Recommendation Model')
]

results = []

for script, name in analyses:
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}\n")
    
    script_path = os.path.join(script_dir, script)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            results.append((name, '✅ SUCCESS'))
        else:
            results.append((name, '❌ FAILED'))
            print(f"ERROR: {result.stderr}")
    except subprocess.TimeoutExpired:
        results.append((name, '⏱️ TIMEOUT'))
        print(f"⏱️ Timeout after 5 minutes")
    except Exception as e:
        results.append((name, f'❌ ERROR: {str(e)}'))
        print(f"❌ Error: {str(e)}")

# Print summary
print("\n" + "="*70)
print("📊 ANALYSIS RESULTS SUMMARY")
print("="*70)

for name, status in results:
    print(f"  {name:40s}: {status}")

success_count = sum(1 for _, status in results if '✅' in status)
print(f"\n✅ Completed: {success_count}/{len(analyses)}")

print("\n" + "="*70)
print("🎉 ML Analysis Pipeline Complete!")
print("="*70)
print("\n📁 Results saved in: ml_results/")
print("🌐 View dashboard at: http://localhost:8501")
