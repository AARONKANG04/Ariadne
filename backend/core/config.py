"""Application configuration."""
from pathlib import Path

# backend/ directory (parent of core/)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
# MAG id (OpenAlex URL) -> title; lives at backend/mag_id_to_title.json
MAG_ID_TO_TITLE_PATH = BASE_DIR / "mag_id_to_title.json"
UPLOAD_DIR = BASE_DIR / "uploads"
