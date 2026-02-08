"""Paper-related endpoints: arXiv PDF URL by MAG id, For You feed."""
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.core.auth import get_sub_from_token, security
from backend.core.config import (
    MAG_TO_NODE_IDX_PATH,
    NODE_TO_MAG_ID_PATH,
    PAPER_EMBEDDINGS_PATH,
)
from backend.services import auth0_storage, paper_service

router = APIRouter(prefix="/api/papers", tags=["papers"])

# In-memory click history: node ids (for terminal logging only; Auth0 is source of truth)
_click_history: list[int] = []

# mag_to_node dict (paper id -> node idx), loaded from mag_to_node_idx.npy
_mag_to_node: dict | None = None
# node_to_mag dict (node idx -> paper id), loaded from node_to_mag_id.npy
_node_to_mag: dict | None = None
# L2-normalized embeddings, shape (num_nodes, 256)
_embeddings: np.ndarray | None = None


def _load_mag_to_node() -> dict:
    """Load mag_to_node dict from mag_to_node_idx.npy (saved as np.save(dict))."""
    global _mag_to_node
    if _mag_to_node is None:
        if not MAG_TO_NODE_IDX_PATH.exists():
            raise FileNotFoundError(f"Mapping not found: {MAG_TO_NODE_IDX_PATH}")
        arr = np.load(MAG_TO_NODE_IDX_PATH, allow_pickle=True)
        _mag_to_node = arr.item()
    return _mag_to_node


def _load_node_to_mag() -> dict:
    """Load node_to_mag dict from node_to_mag_id.npy (saved as np.save(dict))."""
    global _node_to_mag
    if _node_to_mag is None:
        if not NODE_TO_MAG_ID_PATH.exists():
            raise FileNotFoundError(f"Mapping not found: {NODE_TO_MAG_ID_PATH}")
        arr = np.load(NODE_TO_MAG_ID_PATH, allow_pickle=True)
        _node_to_mag = arr.item()
    return _node_to_mag


def _load_embeddings() -> np.ndarray:
    """Load and L2-normalize paper embeddings."""
    global _embeddings
    if _embeddings is None:
        if not PAPER_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"Embeddings not found: {PAPER_EMBEDDINGS_PATH}")
        emb = np.load(PAPER_EMBEDDINGS_PATH).astype(np.float32)
        _embeddings = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return _embeddings


def _mag_id_to_node_id(mag_id: str) -> int | None:
    """Resolve MAG id (URL or numeric) to node idx using mag_to_node_idx.npy."""
    mapping = _load_mag_to_node()
    numeric = paper_service._mag_id_to_numeric(mag_id)
    if not numeric.isdigit():
        return None
    key_int = int(numeric)
    if key_int in mapping:
        return int(mapping[key_int])
    if numeric in mapping:
        return int(mapping[numeric])
    return None


def _node_id_to_mag_id_url(node_id: int) -> str | None:
    """Convert node idx to OpenAlex URL using node_to_mag_id.npy."""
    mapping = _load_node_to_mag()
    node_id = int(node_id)
    if node_id not in mapping:
        return None
    paper_id = mapping[node_id]
    return f"https://openalex.org/W{paper_id}"


class ClickRequest(BaseModel):
    mag_id: str


@router.get("/paper-info")
def get_paper_info(mag_id: str = Query(..., description="MAG/OpenAlex id")):
    """Look up paper title, DOI URL, and abstract by MAG id."""
    result = paper_service.get_paper_info_by_mag_id(mag_id)
    if result["title"] is None and result["doi_url"] is None:
        raise HTTPException(status_code=404, detail=f"No paper found for MAG id: {mag_id}")
    return result


@router.post("/click")
def register_click(
    body: ClickRequest,
    credentials=Depends(security),
):
    """
    Register a paper click: convert mag ID to node ID, update Auth0 node_history (max 5).
    Requires Bearer token.
    """
    node_id = _mag_id_to_node_id(body.mag_id)
    if node_id is None:
        print(f"\n[Click] Unknown MAG id (not in mag_to_node_idx): {body.mag_id!r}\n")
        return {"ok": False, "error": "mag_id not in mapping"}
    sub = get_sub_from_token(credentials)
    history = auth0_storage.append_node_to_history(sub, str(node_id))
    _click_history.append(node_id)
    print("\n[Click] Current click history (node ids):")
    for i, nid in enumerate(history, 1):
        print(f"  {i}. {nid}")
    print()
    return {"ok": True, "history": history}


@router.get("/for-you")
def get_for_you_papers(
    n: int = Query(50, ge=1, le=500, description="Number of papers to return (default 50)"),
    credentials=Depends(security),
):
    """
    Return n papers for the For You page: get user's click history from Auth0 (max 5),
    average their embeddings, return n nearest nodes by cosine similarity.
    If no Bearer token or invalid token, treats as no history and returns 50 papers (fallback).
    """
    history: list[str] = []
    if credentials and credentials.credentials:
        try:
            sub = get_sub_from_token(credentials)
            try:
                history = auth0_storage.get_node_history(sub)
            except Exception:
                pass
        except HTTPException:
            pass
    embeddings = _load_embeddings()
    node_to_mag = _load_node_to_mag()
    num_nodes = embeddings.shape[0]

    # Build average vector from click history (or fallback to random if empty)
    if history:
        node_ids = []
        for x in history:
            try:
                ni = int(x)
                if 0 <= ni < num_nodes:
                    node_ids.append(ni)
            except (ValueError, TypeError):
                continue
        if node_ids:
            avg = embeddings[node_ids].mean(axis=0)
            avg = avg / np.linalg.norm(avg)
        else:
            avg = embeddings.mean(axis=0)
            avg = avg / np.linalg.norm(avg)
    else:
        avg = embeddings.mean(axis=0)
        avg = avg / np.linalg.norm(avg)

    # Cosine similarity (embeddings already normalized)
    scores = embeddings @ avg
    # Descending; exclude nodes we don't have mag_id for, and exclude user's history
    history_set = {int(x) for x in history if isinstance(x, str) and x.isdigit()}
    candidate_nodes = []
    for idx in np.argsort(-scores):
        idx_int = int(idx)
        if idx_int in history_set or idx_int not in node_to_mag:
            continue
        mag_url = _node_id_to_mag_id_url(idx_int)
        if mag_url and paper_service.get_title_by_mag_id(mag_url):
            candidate_nodes.append(idx_int)
        if len(candidate_nodes) >= n:
            break

    papers = []
    for node_id in candidate_nodes[:n]:
        mag_id = _node_id_to_mag_id_url(node_id)
        if not mag_id:
            continue
        title = paper_service.get_title_by_mag_id(mag_id) or "—"
        papers.append({"mag_id": mag_id, "title": title})

    # Fallback: if we got no papers (e.g. no history + sparse mapping), return random
    if not papers:
        papers = paper_service.get_random_papers(n=n)

    return {"papers": papers, "count": len(papers)}
