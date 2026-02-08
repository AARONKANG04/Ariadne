from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import papers, upload

app = FastAPI(
    title="Ariadne API",
    description="Backend for paper discovery and uploads",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(papers.router)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
