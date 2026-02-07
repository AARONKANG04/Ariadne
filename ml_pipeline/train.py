import torch
import torch_geometric
from model import EmbedderGNN
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader



WITH_WANDB = True
BATCH_SIZE = 1024
NUM_NEIGHBORS = 128
LEARNING_RATE = 0.01
NUM_EPOCHS = 100

if WITH_WANDB: 
    import wandb
    wandb.init(
        project="embedder-gnn-at-cxc"
        config={
            "learning_rate": LEARNING_RATE,
        }
    )


loss_fn = torch.nn.MSELoss()
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