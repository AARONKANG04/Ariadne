"""File upload endpoint."""
from pathlib import Path

import shutil
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.core.config import UPLOAD_DIR

router = APIRouter(prefix="/api", tags=["upload"])

UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_paper(file: UploadFile = File(...)):
    """Accept a PDF upload and save it to the backend uploads folder.

    Expects a multipart/form-data POST with field name `file`.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    safe_name = Path(file.filename).name
    dest_path = UPLOAD_DIR / safe_name

    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}")

    return {"filename": safe_name, "size": dest_path.stat().st_size, "message": "Upload successful"}
