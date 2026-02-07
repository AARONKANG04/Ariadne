from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/api/upload", status_code=status.HTTP_201_CREATED)
async def upload_paper(file: UploadFile = File(...)):
    """Accept a PDF upload and save it to the backend uploads folder.

    Expects a multipart/form-data POST with field name `file`.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Basic content-type check
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Sanitize filename (simple)
    safe_name = Path(file.filename).name
    dest_path = UPLOAD_DIR / safe_name

    try:
        # Stream write to disk
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}")

    return {"filename": safe_name, "size": dest_path.stat().st_size, "message": "Upload successful"}