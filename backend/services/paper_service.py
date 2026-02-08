"""Paper lookup and PDF URL resolution via OpenAlex + arXiv fallback."""
import json
import random
import urllib.request
from pathlib import Path

import arxiv
from ..core.config import MAG_ID_TO_TITLE_PATH

# In-memory cache: OpenAlex ID -> title
_mag_id_to_title: dict[str, str] | None = None

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def _normalize_mag_id(mag_id: str) -> str:
    """Normalize MAG id to OpenAlex URL key format (for our JSON lookup)."""
    s = mag_id.strip()
    if s.isdigit():
        return f"https://openalex.org/W{s}"
    if not s.startswith("http"):
        return f"https://openalex.org/W{s}" if s.startswith("W") else f"https://openalex.org/{s}"
    return s


def _mag_id_to_numeric(mag_id: str) -> str:
    """Extract numeric MAG id for OpenAlex API filter (e.g. W1578902217 or full URL -> 1578902217)."""
    s = mag_id.strip()
    if s.isdigit():
        return s
    if s.startswith("https://openalex.org/W"):
        return s.replace("https://openalex.org/W", "")
    if s.startswith("W"):
        return s[1:]
    return s


def _load_mag_id_to_title() -> dict[str, str]:
    global _mag_id_to_title
    if _mag_id_to_title is None:
        if not MAG_ID_TO_TITLE_PATH.exists():
            raise FileNotFoundError(f"MAG title mapping not found: {MAG_ID_TO_TITLE_PATH}")
        with open(MAG_ID_TO_TITLE_PATH, encoding="utf-8") as f:
            _mag_id_to_title = json.load(f)
    return _mag_id_to_title


def get_title_by_mag_id(mag_id: str) -> str | None:
    """Return paper title for a given MAG/OpenAlex id from backend/mag_id_to_title.json, or None."""
    mapping = _load_mag_id_to_title()
    key = _normalize_mag_id(mag_id)
    return mapping.get(key)


def _abstract_inverted_index_to_text(inverted: dict | None) -> str | None:
    """Convert OpenAlex abstract_inverted_index to plain text. Returns None if invalid or empty."""
    if not inverted or not isinstance(inverted, dict):
        return None
    # Inverted index: word -> list of positions. Build list of (position, word) then sort and join.
    pairs: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for p in positions:
            if isinstance(p, int) and p >= 0:
                pairs.append((p, str(word)))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    return " ".join(w for (_, w) in pairs)


def _fetch_openalex_work_by_mag_id(mag_id: str) -> dict | None:
    """Query OpenAlex API by MAG id; return first work with doi and abstract_inverted_index selected."""
    numeric_id = _mag_id_to_numeric(mag_id)
    if not numeric_id:
        return None
    url = f"{OPENALEX_WORKS_URL}?filter=ids.mag:{numeric_id}&select=doi,abstract_inverted_index&per_page=1"
    req = urllib.request.Request(url, headers={"User-Agent": "AriadneBackend/1.0 (mailto:optional@example.com)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None
    results = data.get("results", [])
    return results[0] if results else None


def get_doi_for_mag_id(mag_id: str) -> str | None:
    """Query OpenAlex API by MAG id to get DOI URL for the work."""
    work = _fetch_openalex_work_by_mag_id(mag_id)
    return work.get("doi") if work else None


def get_abstract_for_mag_id(mag_id: str) -> str | None:
    """Query OpenAlex API by MAG id and return abstract as plain text, or None."""
    work = _fetch_openalex_work_by_mag_id(mag_id)
    if not work:
        return None
    inv = work.get("abstract_inverted_index")
    return _abstract_inverted_index_to_text(inv)


def get_paper_info_by_mag_id(mag_id: str) -> dict:
    """Return title (from backend/mag_id_to_title.json), DOI URL, and abstract from OpenAlex."""
    normalized = _normalize_mag_id(mag_id)
    title = get_title_by_mag_id(mag_id)
    work = _fetch_openalex_work_by_mag_id(mag_id)
    doi_url = work.get("doi") if work else None
    abstract = _abstract_inverted_index_to_text(work.get("abstract_inverted_index")) if work else None
    return {"mag_id": normalized, "title": title, "doi_url": doi_url, "abstract": abstract}


def get_random_papers(n: int = 50) -> list[dict]:
    """Return n random papers (mag_id, title) from the mapping. For 'For You' placeholder."""
    mapping = _load_mag_id_to_title()
    keys = list(mapping.keys())
    if n >= len(keys):
        chosen = keys
    else:
        chosen = random.sample(keys, n)
    return [{"mag_id": k, "title": mapping[k]} for k in chosen]