from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import papers, upload
from ml_pipeline.src.data_loader import load_ogbn_arxiv, unsafe_load_ogbn_arxiv
from pydantic import BaseModel
from typing import List

from torch_geometric.data import Data
import torch

app = FastAPI(
    title="Ariadne API",
    description="Backend for paper discovery and uploads",
)

dataset = unsafe_load_ogbn_arxiv()
graph_dict, labels = dataset[0]
graph = Data(
    x=torch.tensor(graph_dict['node_feat'], dtype=torch.float),
    edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long),
    num_nodes=graph_dict['num_nodes']
)
# Add to feature dimension to simulate the reembedded graph
# TODO: replace with actual reembedded graph
graph.x = torch.cat([graph.x, torch.zeros(graph.num_nodes, 256)], dim=1)


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

from ml_pipeline.src.gnn_embed_new import endpoint

class CitationListRequest(BaseModel):
    ids: List[int]

# sample usage of endpoint below
# Gets full embedding: probably impractical for frontend
@app.post("/get_new_node_embedding")
def get_new_node_embedding(request: CitationListRequest):
    return {"embedding": endpoint(graph, request.ids).tolist()}
