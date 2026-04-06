"""
Entry point — pre-warms data and CV metrics, then launches the Streamlit app.
Run:  python main.py
Or directly:  streamlit run src/app.py
"""

import subprocess
import sys
from pathlib import Path


def preflight():
    """Download data and run CV if not already done."""
    print("=" * 60)
    print("  StockAI — Preflight checks")
    print("=" * 60)

    # ── Historical data ────────────────────────────────────────────
    print("\n[1/2] Fetching historical price data …")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from ingest import fetch_all
        datasets = fetch_all()
        for t, df in datasets.items():
            print(f"      {t}: {len(df):,} rows  ({df.index[0].date()} → {df.index[-1].date()})")
    except Exception as exc:
        print(f"      Warning: {exc}")

    # ── Cross-validation metrics ───────────────────────────────────
    metrics_path = Path(__file__).parent / "data" / "metrics.json"
    if not metrics_path.exists():
        print("\n[2/2] Running 10-fold cross-validation …")
        try:
            from filter import cross_validate_and_save
            m = cross_validate_and_save()
            print(f"      Accuracy: {m['accuracy']['mean']:.4f} ± {m['accuracy']['std']:.4f}")
        except Exception as exc:
            print(f"      Warning: {exc}")
    else:
        print("\n[2/2] CV metrics already exist — skipping.")

    print("\nPreflight complete.\n")


def launch():
    cmd = [sys.executable, "-m", "streamlit", "run", "src/app.py",
           "--server.headless", "false",
           "--server.port", "8501"]
    print("Launching Streamlit …")
    print("  Local URL:  http://localhost:8501")
    print("  Press Ctrl+C to stop.\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    preflight()
    launch()
