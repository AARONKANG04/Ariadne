"""User-related endpoints (e.g. Auth0 user_metadata history)."""
from fastapi import APIRouter, Depends

from ..core.auth import get_sub_from_token, security
from ..services import auth0_storage

router = APIRouter(prefix="/api/user", tags=["user"])


@router.post("/node-history/{node_id}")
def add_node_to_history(
    node_id: str,
    credentials=Depends(security),
):
    """
    Add a node id to the current user's history queue (stored in Auth0 user_metadata).
    Queue has max size 5; newest entries are kept. Requires Bearer token.
    """
    sub = get_sub_from_token(credentials)
    history = auth0_storage.append_node_to_history(sub, node_id)
    return {"node_id": node_id, "history": history}
