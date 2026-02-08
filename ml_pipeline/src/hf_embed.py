from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import torch


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

load_dotenv()

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")

model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE_DIR)

def get_semantic_embed(text: str, truncate_dim: int=256) -> list[float]:
    return model.encode(text, truncate_dim=truncate_dim)

if __name__ == "__main__":
    abstract = "Humans invent new words when there is a rising demand for a new useful concept\ (e.g., doomscrolling). We explore and validate a similar idea in our communication with LLMs: introducing new words to better understand and control the models, expanding on the recently introduced neologism learning. This method introduces a new word by adding a new word embedding and training with examples that exhibit the concept with no other changes in model parameters. We show that adding a new word allows for control of concepts such as flattery, incorrect answers, text length, as well as more complex concepts in AxBench. We discover that neologisms can also further our understanding of the model via self-verbalization: models can describe what each new word means to them in natural language, like explaining that a word that represents a concept of incorrect answers means ``a lack of complete, coherent, or meaningful answers...'' To validate self-verbalizations, we introduce plug-in evaluation: we insert the verbalization into the context of a model and measure whether it controls the target concept. In some self-verbalizations, we find machine-only synonyms: words that seem unrelated to humans but cause similar behavior in machines. Finally, we show how neologism learning can jointly learn multiple concepts in multiple words."
    print(get_semantic_embed(abstract, truncate_dim=256).shape)