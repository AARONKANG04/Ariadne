import torch
import torch_geometric
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from src.model import EmbedderGNNv2
from tqdm import tqdm
from src.data_loader import load_ogbn_arxiv

from contextlib import contextmanager


WITH_WANDB = False
BATCH_SIZE = 1024
NUM_NEIGHBORS = [16, 8]
LEARNING_RATE = 0.01
NUM_EPOCHS = 100
HIDDEN_DIM = 128

if WITH_WANDB: 
    import wandb
    wandb.init(
        project="embedder-gnn-at-cxc",
        config={
            "learning_rate": LEARNING_RATE,
        }
    )


@contextmanager
def allow_unsafe_torch_load():
    # Temporarily allow torch.load with weights_only=False
    # Maybe move to data_loader.py?
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load

with allow_unsafe_torch_load():
    dataset = load_ogbn_arxiv()

graph_dict, labels = dataset[0]
print(graph_dict["edge_index"].shape) # (2, 1166243) (2, E)
print(graph_dict["node_year"].shape) # (169343, 1) (N, 1)
print(graph_dict["num_nodes"]) # Scal.
print(graph_dict["node_feat"].shape) # (169343, 128) (N, F)

EMBED_SIZE = graph_dict["node_feat"].shape[-1]

data = Data(
    x=torch.tensor(graph_dict['node_feat'], dtype=torch.float),
    edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long),
    y=torch.tensor(labels.squeeze(), dtype=torch.long),
    num_nodes=graph_dict['num_nodes']
)

split_idx = dataset.get_idx_split()
train_loader = NeighborLoader(data, num_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE, input_nodes=split_idx["train"], shuffle=True)
valid_loader = NeighborLoader(data, num_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE, input_nodes=split_idx["valid"], shuffle=False)

loss_fn = torch.nn.MSELoss()

model = EmbedderGNNv2(EMBED_SIZE, HIDDEN_DIM, EMBED_SIZE, dropout=0.2)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
    total_train_loss = 0
    total_valid_loss = 0

    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        x, edge_index = batch.x, batch.edge_index
        y = batch.y
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = loss_fn(out, y)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    model.eval()
    for batch in tqdm(valid_loader, desc=f"Validating Epoch {epoch}"):
        x, edge_index = batch.x, batch.edge_index
        y = batch.y
        out = model(x, edge_index)
        loss = loss_fn(out, y)
        total_valid_loss += loss.item()
    
    wandb.log({"train_loss": total_train_loss / len(train_loader), "valid_loss": total_valid_loss / len(valid_loader)})

if WITH_WANDB:
    wandb.finish()