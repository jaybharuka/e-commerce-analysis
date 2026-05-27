"""
run_local.py
------------
End-to-end local runner — no Docker, no Spark, no Hive.

Runs the pandas-based ML analyses directly against data/data.csv,
writes CSV results to ml_results/, then launches the Streamlit dashboard.

Usage:
    python run_local.py            # run all analyses + open dashboard
    python run_local.py --ml-only  # run analyses, skip streamlit
    python run_local.py --dash     # skip analyses, just open dashboard
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
ARCHIVE_DIR = ROOT / "archive"
SCRIPTS_DIR = ARCHIVE_DIR / "ml_analysis"
ML_RESULTS_DIR = ROOT / "ml_results"
DATA_FILE = ROOT / "data" / "data.csv"
DASHBOARD = ARCHIVE_DIR / "streamlit" / "app_direct.py"

# ---------------------------------------------------------------------------
# Analyses — ordered so fast scripts run first, Prophet last (slowest)
# ---------------------------------------------------------------------------
ANALYSES = [
    ("rfm_analysis.py",          "RFM Analysis"),
    ("customer_segmentation.py", "Customer Segmentation (K-Means)"),
    ("market_basket.py",         "Market Basket Analysis"),
    ("churn_prediction.py",      "Churn Prediction (XGBoost)"),
    ("product_recommendation.py","Product Recommendations"),
    ("sales_forecast_prophet.py","Sales Forecasting (Prophet)"),
]


def check_prerequisites() -> bool:
    """Fail fast if the data file is missing."""
    ok = True
    if not DATA_FILE.exists():
        print(f"[ERROR] Data file not found: {DATA_FILE}")
        print("        Download the UCI Online Retail dataset and place it at data/data.csv")
        ok = False
    if not SCRIPTS_DIR.exists():
        print(f"[ERROR] archive/ml_analysis/ not found: {SCRIPTS_DIR}")
        ok = False
    ML_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return ok


def run_analyses() -> dict:
    """
    Run each ML script as a subprocess.

    CWD is set to ARCHIVE_DIR so that the scripts' relative paths
    '../data/data.csv' and '../ml_results' resolve to the correct locations:
        ARCHIVE_DIR/../data/data.csv  →  ROOT/data/data.csv  ✓
        ARCHIVE_DIR/../ml_results     →  ROOT/ml_results/    ✓
    """
    results = {}
    total = len(ANALYSES)

    for i, (script_file, label) in enumerate(ANALYSES, 1):
        script_path = SCRIPTS_DIR / script_file
        if not script_path.exists():
            print(f"[{i}/{total}] SKIP  {label}  (file not found: {script_path})")
            results[label] = "SKIP"
            continue

        print(f"\n[{i}/{total}] Running: {label}")
        print("-" * 60)

        t0 = time.time()
        # Force UTF-8 stdout so emoji in script print() don't crash on
        # Windows cp1252 terminals.  PYTHONUTF8=1 is equivalent to -X utf8.
        child_env = os.environ.copy()
        child_env["PYTHONUTF8"] = "1"
        child_env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ARCHIVE_DIR),   # ← key: makes ../data/ and ../ml_results/ resolve correctly
            capture_output=False,   # stream output live so user can watch progress
            text=True,
            env=child_env,
        )
        elapsed = time.time() - t0

        if proc.returncode == 0:
            print(f"     OK  ({elapsed:.1f}s)")
            results[label] = "OK"
        else:
            print(f"     FAIL  (exit {proc.returncode}, {elapsed:.1f}s)")
            results[label] = "FAIL"

    return results


def print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for label, status in results.items():
        icon = {"OK": "[OK]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}.get(status, status)
        print(f"  {icon:<8} {label}")

    ok_count = sum(1 for s in results.values() if s == "OK")
    print(f"\n  {ok_count}/{len(results)} analyses completed successfully")

    if ML_RESULTS_DIR.exists():
        csv_files = list(ML_RESULTS_DIR.glob("*.csv"))
        print(f"\n  Output files in ml_results/:")
        for f in sorted(csv_files):
            size_kb = f.stat().st_size // 1024
            print(f"    {f.name:<45}  {size_kb:>5} KB")


def launch_dashboard() -> None:
    print("\n" + "=" * 60)
    print("Launching Streamlit dashboard...")
    print("Dashboard: http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Run streamlit from ROOT so ./ml_results and ./data paths resolve correctly
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(DASHBOARD),
         "--server.headless", "false"],
        cwd=str(ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="E-Commerce local ML runner")
    parser.add_argument("--ml-only", action="store_true", help="Run ML analyses only, skip dashboard")
    parser.add_argument("--dash",    action="store_true", help="Launch dashboard only, skip ML analyses")
    args = parser.parse_args()

    print("=" * 60)
    print("E-COMMERCE ML PIPELINE  (local / no-Spark mode)")
    print("=" * 60)
    print(f"  Data:      {DATA_FILE}")
    print(f"  Results:   {ML_RESULTS_DIR}")
    print(f"  Dashboard: {DASHBOARD}")
    print()

    if not check_prerequisites():
        sys.exit(1)

    if not args.dash:
        results = run_analyses()
        print_summary(results)

    if not args.ml_only:
        launch_dashboard()


if __name__ == "__main__":
    main()
