import asyncio
import aiohttp
import pandas as pd
import json
import sys
from tqdm.asyncio import tqdm_asyncio

# --- CONFIGURATION ---
INPUT_FILE = "data/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz"
OUTPUT_FILE = "arxiv_mag_metadata.json"
EMAIL = "your_email@example.com"  # <--- TODO: Put your email here for faster speeds
MAX_CONCURRENT_REQUESTS = 10      # OpenAlex polite limit is ~10 req/s

def reconstruct_abstract(inverted_index):
    """
    OpenAlex stores abstracts as an inverted index (word -> [positions]).
    This function reconstructs the readable text for your UI.
    """
    if not inverted_index:
        return None
    
    # Create a list of (position, word) tuples
    word_list = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_list.append((pos, word))
            
    # Sort by position and join
    return " ".join(word for _, word in sorted(word_list))

async def fetch_batch(session, ids, semaphore):
    # Filter syntax: ids.mag:123|456|789
    ids_param = "|".join(ids)
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"ids.mag:{ids_param}",
        "per_page": 100,
        "select": "id,ids,title,abstract_inverted_index,primary_location",
        "mailto": EMAIL
    }

    async with semaphore:  # Wait here if too many requests are active
        while True:  # Retry loop
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])
                        
                        # Process abstracts immediately to save space/time
                        processed_results = []
                        for r in results:
                            # Extract ArXiv ID cleanly
                            arxiv_id = r.get('ids', {}).get('arxiv', '').replace("https://arxiv.org/abs/", "")
                            
                            processed_results.append({
                                "mag_id": str(r['ids'].get('mag')), # Ensure string for consistency
                                "arxiv_id": arxiv_id,
                                "title": r.get('title'),
                                "abstract": reconstruct_abstract(r.get('abstract_inverted_index')),
                                "pdf_url": r.get('primary_location', {}).get('pdf_url')
                            })
                        return processed_results
                    
                    elif response.status == 429:
                        # Rate limited: Wait and retry
                        await asyncio.sleep(2)
                        continue  # Try again
                    
                    elif response.status >= 500:
                        # Server error: Wait and retry
                        await asyncio.sleep(5)
                        continue
                        
                    else:
                        print(f"Error {response.status}: {await response.text()}")
                        return []  # Fatal error for this batch
                        
            except Exception as e:
                print(f"Request failed: {e}")
                await asyncio.sleep(1)

async def main():
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE, compression='gzip')
        # Handle inconsistent column names in OGB vs custom files
        col_name = 'paper id' if 'paper id' in df.columns else df.columns[1]
        mag_ids = df[col_name].astype(str).tolist()
        print(f"Loaded {len(mag_ids)} IDs.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. Prepare Batches
    batch_size = 50
    batches = [mag_ids[i:i + batch_size] for i in range(0, len(mag_ids), batch_size)]
    
    # 3. Setup Async
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=None) # We limit via semaphore instead
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_batch(session, batch, semaphore) for batch in batches]
        
        # tqdm_asyncio automatically handles the gather + progress bar
        results_list = await tqdm_asyncio.gather(*tasks, desc="Fetching from OpenAlex")

    # 4. Flatten Results
    # results_list is a list of lists: [[paper1, paper2], [paper3, paper4]...]
    flat_results = [item for sublist in results_list for item in sublist]
    
    # 5. Convert to Dictionary (MAG_ID -> Data) for O(1) Lookup
    final_db = {item['mag_id']: item for item in flat_results}

    print(f"Successfully fetched {len(final_db)} papers.")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_db, f)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())