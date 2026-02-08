"""Application configuration."""
import os
from pathlib import Path

# backend/ directory (parent of core/)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
# MAG id (OpenAlex URL) -> title; lives at backend/mag_id_to_title.json
MAG_ID_TO_TITLE_PATH = BASE_DIR / "mag_id_to_title.json"
# MAG id -> node index; 1D array in same order as sorted MAG_ID_TO_TITLE_PATH keys
MAG_TO_NODE_IDX_PATH = BASE_DIR / "mag_to_node_idx.npy"
# Node index -> paper id (from same mapping as mag_to_node_idx); for for-you recommendations
NODE_TO_MAG_ID_PATH = BASE_DIR / "node_to_mag_id.npy"
# L2-normalized paper embeddings, shape (num_nodes, 256); row i = node i
PAPER_EMBEDDINGS_PATH = BASE_DIR / "paper_embeddings_256d.npy"
UPLOAD_DIR = BASE_DIR / "uploads"

# Auth0 (optional): for JWT validation and Management API user_metadata)
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "").rstrip("/")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "")  # API identifier for JWT validation
AUTH0_M2M_CLIENT_ID = os.getenv("AUTH0_M2M_CLIENT_ID", "")
AUTH0_M2M_CLIENT_SECRET = os.getenv("AUTH0_M2M_CLIENT_SECRET", "")
AUTH0_MANAGEMENT_AUDIENCE = f"https://{AUTH0_DOMAIN}/api/v2/" if AUTH0_DOMAIN else ""
