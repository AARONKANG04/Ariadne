# Script for running inference on the GNN for the purpose of embedding new papers / nodes
# whose abstracts are unknown but whose citation list is known.
# Also includes a test run in __main__ for a random chunk of the train graph. (not sure if this works anymore)

import sys
from pathlib import Path
# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from contextlib import contextmanager
from src.model import EmbedderGNNv3
from src.data_loader import load_ogbn_arxiv, unsafe_load_ogbn_arxiv
from src.hf_embed import get_semantic_embed

INPUT_DIM = 384   # 128 (Original) + 256 (Qwen)
HIDDEN_DIM = 256
OUTPUT_DIM = 256

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

# Load model
model = EmbedderGNNv3(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, num_layers=4)
model.load_state_dict(torch.load("ml_pipeline/models/gnn_contrastive_v2.pth", map_location=device))
model.eval()

def get_cluster(data, center=None, num_neighbors=[10, 10, 5]):
    """Extract k-hop neighborhood with sampling. num_neighbors: list of ints per hop."""
    if center is None:
        center = np.random.randint(0, data.num_nodes)
    
    # Sample neighbors per hop (similar to NeighborLoader)
    all_nodes = {center}
    current_layer = {center}
    
    for n_sample in num_neighbors:
        next_layer = set()
        for node in current_layer:  # Only sample from current layer, not all nodes
            neighbors = data.edge_index[1, data.edge_index[0] == node].numpy()
            if len(neighbors) > n_sample:
                neighbors = np.random.choice(neighbors, n_sample, replace=False)
            next_layer.update(neighbors)
        all_nodes.update(next_layer)
        current_layer = next_layer  # Move to next layer
    
    subset = torch.tensor(sorted(all_nodes), dtype=torch.long)
    _, sub_edge_index, mapping, _ = k_hop_subgraph(subset, 0, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    mapping = (subset == center).nonzero(as_tuple=True)[0].item()
    return Data(x=data.x[subset], edge_index=sub_edge_index, num_nodes=len(subset)), mapping, subset


def build_query_graph(base_graph, cited_node_ids, num_neighbors=[10, 10, 5]):
    """
    This is what "adds" the new paper to the graph in preparation for inference.

    Given our entire graph, and a new paper's citation list,
    builds a new subgraph of the new paper's neighbourhood with center at index 0.
    
    Args:
        base_graph: Full ogbn-arxiv graph (Data object)
        cited_node_ids: List of node IDs the new paper cites
        num_neighbors: Sampling strategy per hop
        
    Returns:
        subgraph_data (with center at index 0), original_node_ids
    """
    # 1. Create extended graph with new node
    num_original_nodes = base_graph.num_nodes
    new_node_id = num_original_nodes  # New node ID
    
    # 2. Extend node features (use zeros placeholder - will be masked anyway)
    placeholder_embedding = torch.zeros(1, base_graph.x.shape[1])
    extended_x = torch.cat([base_graph.x, placeholder_embedding], dim=0)
    
    # 3. Create edges: new_node → cited_nodes
    new_edges = torch.tensor([
        [new_node_id] * len(cited_node_ids),
        cited_node_ids
    ], dtype=torch.long)
    
    # 4. Extend edge_index and create extended graph
    extended_edge_index = torch.cat([base_graph.edge_index, new_edges], dim=1)
    extended_graph = Data(
        x=extended_x,
        edge_index=extended_edge_index,
        num_nodes=num_original_nodes + 1
    )
    
    # 5. Use get_cluster to sample neighborhood
    subgraph, center_idx, orig_ids = get_cluster(
        extended_graph,
        center=new_node_id,
        num_neighbors=num_neighbors
    )
    
    # 6. Reorder to put center at index 0
    other_indices = [i for i in range(subgraph.num_nodes) if i != center_idx]
    reorder = torch.tensor([center_idx] + other_indices)
    
    # Reorder features and original IDs
    reordered_x = subgraph.x[reorder]
    reordered_orig_ids = orig_ids[reorder]
    
    # Remap edge indices
    old_to_new = torch.zeros(subgraph.num_nodes, dtype=torch.long)
    old_to_new[reorder] = torch.arange(subgraph.num_nodes)
    reordered_edge_index = old_to_new[subgraph.edge_index]
    
    # Create final subgraph
    final_subgraph = Data(
        x=reordered_x,
        edge_index=reordered_edge_index,
        num_nodes=subgraph.num_nodes
    )
    
    return final_subgraph, reordered_orig_ids


def endpoint(graph, citation_ids): 
    # Build the neighbourhood subgraph (center at index 0, will be masked)
    neighbourhood_graph, orig_ids = build_query_graph(graph, citation_ids)
    # Forward pass
    with torch.no_grad():
        embeddings = model(neighbourhood_graph.x, neighbourhood_graph.edge_index)
    # Return the embedding of the new paper (always at index 0)
    return embeddings[0]

if __name__ == "__main__":
    # Load data
    dataset = unsafe_load_ogbn_arxiv()
    graph_dict, labels = dataset[0]
    data = Data(
        x=torch.tensor(graph_dict['node_feat'], dtype=torch.float),
        edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long),
        num_nodes=graph_dict['num_nodes']
    )
    # Add to feature dimension to simulate the reembedded graph
    data.x = torch.cat([data.x, torch.zeros(data.num_nodes, 256)], dim=1)
    
    # Get cluster and run inference
    print(endpoint(data, [6767, 6, 7, 67]))
