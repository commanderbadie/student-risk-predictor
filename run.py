"""
run.py — Entry point for the Student Academic Risk Predictor.

Usage:
    python run.py          # launches the Streamlit app
    python run.py --train  # re-trains models from scratch
    python run.py --data   # regenerates dataset only
"""
import sys
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def check_models():
    """Return True if trained models exist."""
    return (
        os.path.exists(os.path.join(PROJECT_ROOT, "models", "risk_classifier.pkl"))
        and os.path.exists(os.path.join(PROJECT_ROOT, "models", "gpa_regressor.pkl"))
    )


def run_script(script_path):
    result = subprocess.run([sys.executable, script_path], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[ERROR] Script failed: {script_path}")
        sys.exit(1)


def main():
    args = sys.argv[1:]

    if "--data" in args:
        print("=== Regenerating dataset ===")
        run_script(os.path.join(PROJECT_ROOT, "generate_dataset.py"))
        return

    if "--train" in args:
        print("=== Regenerating dataset ===")
        run_script(os.path.join(PROJECT_ROOT, "generate_dataset.py"))
        print("=== Training models ===")
        run_script(os.path.join(PROJECT_ROOT, "src", "train.py"))
        return

    # Default: launch the app, auto-train if models missing
    if not check_models():
        print("[INFO] No trained models found — running full pipeline first...")
        run_script(os.path.join(PROJECT_ROOT, "generate_dataset.py"))
        run_script(os.path.join(PROJECT_ROOT, "src", "train.py"))

    print("=== Launching Streamlit app ===")
    app_path = os.path.join(PROJECT_ROOT, "app", "app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless", "false"],
        cwd=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()
