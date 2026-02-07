import os
import json
import torch
import time
from tqdm import tqdm
from google import genai
from google.genai import types

# 1. Setup Client
# Ensure GOOGLE_API_KEY is in your environment variables
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def generate_embeddings(input_file, output_file):
    # 2. Load Data
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract abstracts (Handle missing/empty ones to prevent API errors)
    abstracts = [item.get('abstract') if item.get('abstract') else "No abstract available" for item in data]
    print(f"Found {len(abstracts)} abstracts. Starting embedding generation...")
    
    embeddings_list = []
    BATCH_SIZE = 50  # Safe batch size
    
    # 3. Batch Process
    for i in tqdm(range(0, len(abstracts), BATCH_SIZE)):
        batch = abstracts[i : i + BATCH_SIZE]
        
        try:
            # 4. The New API Call
            # 'text-embedding-004' natively supports MRL
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=batch,
                config=types.EmbedContentConfig(
                    output_dimensionality=256,  # <--- Truncates to 256 dims here
                    task_type="RETRIEVAL_DOCUMENT" # Optimize for storage
                )
            )
            
            # The new SDK returns objects, we need to extract the vector values
            # response.embeddings is a list of Embedding objects
            batch_embeddings = [e.values for e in response.embeddings]
            embeddings_list.extend(batch_embeddings)
            
            # Rate limiting (adjust based on your tier)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            # Fallback: append zero vectors to keep alignment with IDs
            embeddings_list.extend([[0.0] * 256] * len(batch))

    # 5. Save as Tensor
    print("Converting to tensor...")
    # Verify shape before saving
    if not embeddings_list:
        print("No embeddings were generated. Check your API key or input data.")
        return

    embedding_tensor = torch.tensor(embeddings_list, dtype=torch.float32)
    
    print(f"Saving tensor of shape {embedding_tensor.shape} to {output_file}...")
    torch.save(embedding_tensor, output_file)
    print("Done!")

if __name__ == "__main__":
    generate_embeddings('arxiv_mag_metadata.json', 'arxiv_embeddings_256.pt')