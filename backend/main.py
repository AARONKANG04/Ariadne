from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import papers, upload, user
import numpy as np

app = FastAPI(
    title="Ariadne API",
    description="Backend for paper discovery and uploads",
)

# Load precomputed embeddings (no torch needed!)
embeddings = np.load("paper_embeddings_256d.npy").astype('float32')
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

print(f"✅ Loaded {embeddings.shape[0]} paper embeddings")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["ariadne-cxc-2026.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(papers.router)
app.include_router(user.router)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Keep this endpoint but use precomputed embeddings
from pydantic import BaseModel
from typing import List

class CitationListRequest(BaseModel):
    ids: List[int]

@app.post("/get_new_node_embedding")
def get_new_node_embedding(request: CitationListRequest):
    """Get embedding for a virtual user node based on clicked papers."""
    # Average the clicked papers' embeddings
    user_embedding = embeddings[request.ids].mean(axis=0)
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    return {"embedding": user_embedding.tolist()}