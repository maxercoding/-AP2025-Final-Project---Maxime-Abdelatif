#!/usr/bin/env python3
"""
ML Market Regime Forecasting - Video Support Dashboard
======================================================
Entry point for the Streamlit presentation dashboard.

Location: Dashboard/run_dashboard.py

Usage (from project root):
    python Dashboard/run_dashboard.py
    
Or directly:
    streamlit run Dashboard/dashboard.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    # Dashboard is in same folder as this script
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)
    
    print(f"Launching dashboard from: {dashboard_path}")
    print("Dashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop.\n")
    
    # Launch Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--theme.base=light",
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard closed.")
    except subprocess.CalledProcessError as e:
        print(f"Error launching dashboard: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Streamlit not installed. Run: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()