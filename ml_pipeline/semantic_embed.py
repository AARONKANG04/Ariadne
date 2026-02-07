import os
from dotenv import load_dotenv
import torch

load_dotenv()

OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

if OPEN_ROUTER_API_KEY is None:
    raise ValueError("Please set OPEN_ROUTER_API_KEY in the .env file")

import requests

def get_semantic_embed(text: str, dimensions: int=768) -> list[float]:

    response = requests.post(
    "https://openrouter.ai/api/v1/embeddings",
    headers={
        "Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "google/gemini-embedding-001",
        "input": text,
        "dimensions": dimensions
    }
    )

    data = response.json()
    embedding = torch.tensor(data["data"][0]["embedding"], dtype=torch.float32)
    
    return embedding

if __name__ == "__main__":
    print(get_semantic_embed("Some article abstract here.").shape, 256)

