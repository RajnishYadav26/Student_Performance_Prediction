from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "student-performance.arff"

MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"

SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"

REPORT_PATH = BASE_DIR / "reports"