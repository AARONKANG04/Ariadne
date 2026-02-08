"""Update Auth0 user_metadata (e.g. node history) via Management API."""
from typing import List
from urllib.parse import quote

import httpx
from fastapi import HTTPException, status

from core.config import (
    AUTH0_DOMAIN,
    AUTH0_MANAGEMENT_AUDIENCE,
    AUTH0_M2M_CLIENT_ID,
    AUTH0_M2M_CLIENT_SECRET,
)

NODE_HISTORY_KEY = "node_history"
NODE_HISTORY_MAX_SIZE = 5


def _get_m2m_token() -> str:
    if not AUTH0_DOMAIN or not AUTH0_M2M_CLIENT_ID or not AUTH0_M2M_CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth0 Management API not configured (AUTH0_DOMAIN, AUTH0_M2M_CLIENT_ID, AUTH0_M2M_CLIENT_SECRET)",
        )
    url = f"https://{AUTH0_DOMAIN}/oauth/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": AUTH0_M2M_CLIENT_ID,
        "client_secret": AUTH0_M2M_CLIENT_SECRET,
        "audience": AUTH0_MANAGEMENT_AUDIENCE,
    }
    with httpx.Client() as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    access_token = data.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to obtain Auth0 Management API token",
        )
    return access_token


def _get_user_metadata(user_id: str, token: str) -> dict:
    """Get user record; return user_metadata dict (empty if missing)."""
    encoded_id = quote(user_id, safe="")
    url = f"https://{AUTH0_DOMAIN}/api/v2/users/{encoded_id}"
    with httpx.Client() as client:
        resp = client.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        resp.raise_for_status()
        data = resp.json()
    return data.get("user_metadata") or {}


def _patch_user_metadata(user_id: str, token: str, user_metadata: dict) -> None:
    encoded_id = quote(user_id, safe="")
    url = f"https://{AUTH0_DOMAIN}/api/v2/users/{encoded_id}"
    with httpx.Client() as client:
        resp = client.patch(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"user_metadata": user_metadata},
        )
        if resp.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        resp.raise_for_status()


def get_node_history(user_id: str) -> List[str]:
    """
    Return the user's node_history from Auth0 user_metadata (max 5, newest last).
    Returns [] if not set or Auth0 not configured.
    """
    try:
        token = _get_m2m_token()
        meta = _get_user_metadata(user_id, token)
        history = list(meta.get(NODE_HISTORY_KEY) or [])
        if not isinstance(history, list):
            return []
        return [str(x) for x in history if x is not None][-NODE_HISTORY_MAX_SIZE:]
    except HTTPException:
        raise
    except Exception:
        return []


def append_node_to_history(user_id: str, node_id: str) -> List[str]:
    """
    Append node_id to the user's node_history in Auth0 user_metadata.
    Keeps at most NODE_HISTORY_MAX_SIZE (5) entries, newest last.
    Returns the updated history list.
    """
    token = _get_m2m_token()
    meta = _get_user_metadata(user_id, token)
    history: List[str] = list(meta.get(NODE_HISTORY_KEY) or [])
    if not isinstance(history, list):
        history = []
    # Ensure string ids and remove duplicates (keep last occurrence)
    history = [str(x) for x in history if x is not None]
    if node_id in history:
        history.remove(node_id)
    history.append(str(node_id))
    history = history[-NODE_HISTORY_MAX_SIZE:]
    meta[NODE_HISTORY_KEY] = history
    _patch_user_metadata(user_id, token, meta)
    return history
