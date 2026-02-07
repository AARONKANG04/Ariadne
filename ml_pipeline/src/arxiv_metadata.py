import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm
import json

try:
    df = pd.read_csv("data/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz")
    print(f"Loaded {len(df)} IDs.")
    
    # OGB mapping files usually have a column 'paper id' for the MAG ID
    # We rename it to 'mag_id' for clarity
    if 'paper id' in df.columns:
        df = df.rename(columns={'paper id': 'mag_id'})
    
    # Ensure they are strings for the URL
    mag_ids = df['mag_id'].astype(str).tolist()
    
except KeyError:
    print(f"Error: Columns found: {df.columns}. Please check the CSV header.")
    mag_ids = []

# 2. Async Fetcher
async def fetch_batch(session, ids):
    # Join IDs with pipe | for "OR" logic
    # Filter 'ids.mag' tells OpenAlex these are Microsoft Academic Graph IDs
    ids_param = "|".join(ids)
    url = f"https://api.openalex.org/works?filter=ids.mag:{ids_param}&per-page=100&select=id,ids,title,abstract_inverted_index"
    
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('results', [])
            elif response.status == 429:
                # Rate limited? Wait a sec and retry (basic handling)
                await asyncio.sleep(2)
                return []
            else:
                return []
    except Exception as e:
        print(f"Error: {e}")
        return []

async def main(ids_list):
    batch_size = 50 # 50 is safer for OpenAlex URL length limits than 100
    tasks = []
    results = []
    
    # Limit to 10 simultaneous connections to be polite
    connector = aiohttp.TCPConnector(limit_per_host=10)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create all tasks
        for i in range(0, len(ids_list), batch_size):
            batch = ids_list[i : i + batch_size]
            tasks.append(fetch_batch(session, batch))
        
        # Run them with a progress bar
        # gathered_results will be a list of lists
        gathered_results = await tqdm.gather(*tasks)
        
    # Flatten the list of lists
    for batch_result in gathered_results:
        results.extend(batch_result)
        
    return results

if __name__ == "__main__":
    if len(mag_ids) > 0:
        # CORRECT WAY for .py scripts:
        final_data = asyncio.run(main(mag_ids))
        
        print(f"Fetched metadata for {len(final_data)} papers.")
        
        # 4. Save to JSON
        with open('arxiv_mag_metadata.json', 'w') as f:
            json.dump(final_data, f)
        print("Saved to arxiv_mag_metadata.json")