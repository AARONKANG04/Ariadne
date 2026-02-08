import torch
import torch_geometric
from torch_geometric.nn import SAGEConv, BatchNorm
import torch.nn.functional as F

class EmbedderGNNv1(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = SAGEConv(dim, dim, aggr='mean')
        self.conv2 = SAGEConv(dim, dim, aggr='mean')
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)

    # x: node features (B, N, C)
    # edge_index: source node indices, target node indices (2, E)
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = h + x

        h2 = self.conv2(h, edge_index)
        h2 = self.norm2(h2)
        return h2 + h


class EmbedderGNNv2(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr='mean'))
        self.bns.append(BatchNorm(hidden_dim))

        self.mask_embed = torch.nn.Parameter(torch.randn(in_dim), requires_grad=True)
        self.register_parameter("mask_embed", self.mask_embed)

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            self.bns.append(BatchNorm(hidden_dim))

        self.convs.append(SAGEConv(hidden_dim, out_dim, aggr='mean'))
        self.bns.append(BatchNorm(out_dim))

class EmbedderGNNv3(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Layer 1: Input -> Hidden
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr='mean'))
        self.bns.append(BatchNorm(hidden_dim))

        # Learnable Mask Token (Replacing Node 0's features)
        self.mask_embed = torch.nn.Parameter(torch.randn(in_dim), requires_grad=True)
        # self.register_parameter("mask_embed", self.mask_embed) # Not strictly needed if assigned to self

        # Hidden Layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            self.bns.append(BatchNorm(hidden_dim))

        # Output Layer: Hidden -> Out
        self.convs.append(SAGEConv(hidden_dim, out_dim, aggr='mean'))
        self.bns.append(BatchNorm(out_dim))

    def forward(self, x, edge_index):
        # 1. Masking Strategy
        # We clone x so we don't modify the original dataset in memory
        # 'detach()' stops gradients flowing back to raw features (which are fixed anyway)
        x = x.clone().detach()
        
        # Replace the first node in the batch with the learnable mask token
        # Note: In LinkNeighborLoader, node 0 is usually a target node for the batch
        x[0] = self.mask_embed

        # 2. Message Passing Loop
        for i in range(self.num_layers):
            h = self.convs[i](x, edge_index)
            h = self.bns[i](h)
        
            # Activation & Dropout (except for last layer)
            if i != self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            # Residual Connection
            # Only apply if shapes match (Input Dim != Hidden Dim on layer 0)
            if x.shape == h.shape:
                h = h + x
            
            x = h

        return x

    # Input format (probably): 
    # x: (n, F)
    # edge_index: (2, E)
    def forward(self, x, edge_index):
        x = x.clone().detach()
        x[0] = self.mask_embed

        for i in range(self.num_layers):
            h = self.convs[i](x, edge_index)
            
            h = self.bns[i](h)
        
            if i != self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            if x.shape == h.shape:
                h = h + x
            
            x = h

        return x