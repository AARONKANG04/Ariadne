import json
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

_MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")
_MODEL_INSTANCE = None

def get_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
        
        print(f"Loading model into cache at: {_MODEL_CACHE_DIR}")
        
        _MODEL_INSTANCE = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B", 
            trust_remote_code=False,
            cache_folder=_MODEL_CACHE_DIR
        )
        
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        _MODEL_INSTANCE.to(device)
        
    return _MODEL_INSTANCE

def get_qwen_embedding(text: str, truncate_dim: int=256) -> list[float]:
    model = get_model()
    embedding = model.encode(text, 
                            show_progress_bar=True,
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                            truncate_dim=truncate_dim)[0]
    return embedding

def generate_qwen_embeddings(input_file, output_file, truncate_dim: int=256):
    print(f"Loading {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        data = json.load(f)

    texts = []
    for item in data:
        title = item.get('title', "")
        abstract = item.get('abstract', "")
        if not title and not abstract:
            text = "Paper content unavailable"
        else:
            text = f"{title}. {abstract}"
        texts.append(text)

    print(f"Loaded {len(texts)} papers. Loading model...")

    model = get_model()

    BATCH_SIZE = 128
    
    print("Generating embeddings...")
    embeddings = model.encode(
        texts, 
        batch_size=BATCH_SIZE, 
        show_progress_bar=True, 
        convert_to_tensor=True,
        normalize_embeddings=True,
        truncate_dim=truncate_dim
    )
    embeddings = embeddings[:, :truncate_dim]
    print(f"Original shape: {embeddings.shape}")
    print(f"Truncated shape: {embeddings.shape}")

    print(f"Saving to {output_file}...")
    torch.save(embeddings.cpu(), output_file)
    print("Done!")