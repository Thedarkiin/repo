
# src/utils.py

from pathlib import Path
import logging

def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)

# Dynamically compute project root and figures path
BASE_DIR     = Path(__file__).resolve().parents[1]
FIGURES_PATH = BASE_DIR / "reports" / "figures"
