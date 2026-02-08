import argparse
import asyncio
import aiohttp
import pandas as pd
import json
import os
import sys
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

# Allow importing from ml_pipeline.src when running from scripts/
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
from src.data_loader import load_ogbn_arxiv, DATA_DIR

def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    word_index = []
    for word, indices in inverted_index.items():
        for index in indices:
            word_index.append((index, word))
    return " ".join([word for _, word in sorted(word_index)])

async def fetch_batch(session, ids, pbar=None):
    ids_param = "|".join([f"mag:{i}" for i in ids])
    url = f"https://api.openalex.org/works?filter=ids.mag:{ids_param}&per-page=50&select=id,ids,title,abstract_inverted_index"
    
    retries = 3
    base_delay = 1

    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('results', [])
                    
                    processed_results = []
                    for item in results:
                        abstract_text = reconstruct_abstract(item.get('abstract_inverted_index'))
                        processed_results.append({
                            "id": item.get("id"),
                            "mag_id": item.get("ids", {}).get("mag"),
                            "title": item.get("title"),
                            "abstract": abstract_text
                        })
                    
                    if pbar:
                        pbar.update(len(ids))
                    return processed_results
                
                elif response.status == 429:
                    wait_time = base_delay * (2 ** attempt)
                    if pbar:
                        pbar.set_description(f"Rate limited. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Error {response.status}: {url}")
                    return []
        except Exception as e:
            print(f"Request failed: {e}")
            await asyncio.sleep(1)
            
    return [] # Failed after retries

async def process_data(input_file, output_file, batch_size, limit):
    print(f"Reading ID mapping from {input_file}...")
    
    try:
        if input_file.endswith('.gz'):
            df = pd.read_csv(input_file, compression='gzip')
        else:
            df = pd.read_csv(input_file)
            
        if 'paper id' in df.columns:
            df = df.rename(columns={'paper id': 'mag_id'})
        elif 'mag_id' not in df.columns:
            df.rename(columns={df.columns[0]: 'mag_id'}, inplace=True)
            
        all_ids = df['mag_id'].dropna().astype(str).tolist()
        
        if limit:
            all_ids = all_ids[:limit]
            print(f"Limiting to first {limit} IDs.")
            
        print(f"Found {len(all_ids)} MAG IDs to fetch.")
        
    except Exception as e:
        print(f"Failed to read input file: {e}")
        sys.exit(1)

    batches = [all_ids[i:i + batch_size] for i in range(0, len(all_ids), batch_size)]
    
    connector = aiohttp.TCPConnector(limit_per_host=5) 
    
    results = []
    
    print("Starting async fetch...")
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        pbar = tqdm_asyncio(total=len(all_ids), desc="Fetching Papers")
        
        for batch in batches:
            tasks.append(fetch_batch(session, batch, pbar))
        
        batch_results = await asyncio.gather(*tasks)
        
        for batch_res in batch_results:
            results.extend(batch_res)
            
        pbar.close()

    print(f"Successfully fetched {len(results)} papers.")
    print(f"Saving to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch paper metadata from OpenAlex based on MAG IDs.")
    default_mapping = DATA_DIR / "ogbn-arxiv" / "mapping" / "nodeidx2paperid.csv.gz"
    parser.add_argument("--input", type=str, default=str(default_mapping),
                        help="Path to the input CSV/GZ file containing MAG IDs.")
    parser.add_argument("--output", type=str, default="arxiv_mag_metadata.json",
                        help="Path to save the output JSON.")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Number of IDs to query per API call (Max 50 recommended for OpenAlex).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of IDs to process (for testing).")

    args = parser.parse_args()

    print("Loading ogbn-arxiv (downloads to cache if not present)...")
    load_ogbn_arxiv()
    print("Proceeding with metadata fetch...")

    asyncio.run(process_data(args.input, args.output, args.batch_size, args.limit))