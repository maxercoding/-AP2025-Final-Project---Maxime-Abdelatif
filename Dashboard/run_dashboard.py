#!/usr/bin/env python3
"""
ML Market Regime Forecasting - Dashboard Launcher
=================================================
Launches the Streamlit dashboard with clean output.

Usage:
    python Dashboard/run_dashboard.py python &
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)
    
    # Suppress Streamlit's file watcher warning and other noise
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    
    print("=" * 50)
    print("  ML Regime Forecasting Dashboard")
    print("=" * 50)
    print(f"  Opening: http://localhost:8501")
    print(f"  Press Ctrl+C to stop")
    print("=" * 50)
    print()
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--theme.base=light",
        "--logger.level=error",  # Suppress info/warning logs
        "--client.showErrorDetails=false",
    ]
    
    try:
        # Run with stderr filtered to reduce noise
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n\nDashboard closed.")
    except subprocess.CalledProcessError:
        print("Dashboard stopped.")
    except FileNotFoundError:
        print("Error: Streamlit not installed.")
        print("Run: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()