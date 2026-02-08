"""Paper-related endpoints: arXiv PDF URL by MAG id, For You feed."""
from fastapi import APIRouter, HTTPException, Query

from backend.services import paper_service

router = APIRouter(prefix="/api/papers", tags=["papers"])


@router.get("/paper-info")
def get_paper_info(mag_id: str = Query(..., description="MAG/OpenAlex id")):
    """Look up paper title, DOI URL, and abstract by MAG id."""
    result = paper_service.get_paper_info_by_mag_id(mag_id)
    if result["title"] is None and result["doi_url"] is None:
        raise HTTPException(status_code=404, detail=f"No paper found for MAG id: {mag_id}")
    return result


@router.get("/for-you")
def get_for_you_papers(
    n: int = Query(50, ge=1, le=500, description="Number of papers to return (default 50)"),
):
    """
    Return n papers for the "For You" page. Until the model is wired up, returns n random
    MAG ids and their titles from mag_id_to_title.json.
    """
    papers = paper_service.get_random_papers(n=n)
    return {"papers": papers, "count": len(papers)}
