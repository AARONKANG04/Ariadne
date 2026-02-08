import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm, JumpingKnowledge
import torch.nn.functional as F

class EmbedderGNNv1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = SAGEConv(dim, dim, aggr='mean')
        self.conv2 = SAGEConv(dim, dim, aggr='mean')
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

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


class EmbedderGNNv2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr='mean'))
        self.bns.append(BatchNorm(hidden_dim))

        self.mask_embed = nn.Parameter(torch.randn(in_dim), requires_grad=True)
        self.register_parameter("mask_embed", self.mask_embed)

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            self.bns.append(BatchNorm(hidden_dim))

        self.convs.append(SAGEConv(hidden_dim, out_dim, aggr='mean'))
        self.bns.append(BatchNorm(out_dim))

class EmbedderGNNv3(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Layer 1: Input -> Hidden
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr='mean'))
        self.bns.append(BatchNorm(hidden_dim))

        # Learnable Mask Token (Replacing Node 0's features)
        self.mask_embed = nn.Parameter(torch.randn(in_dim), requires_grad=True)
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


class EmbedderGNNv4(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.mask_token = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.input_projection = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            self.norms.append(BatchNorm(hidden_dim))

        self.output_projection = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        mask_vector = self.mask_token
        x_masked = torch.cat([mask_vector.unsqueeze(0), x[1:].detach()], dim=0)

        x = self.input_projection(x_masked) 
        x = self.input_norm(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            x = x + h

        out = self.output_projection(x)

        return out




class GatedBlock(nn.Module):
    def __init__(self, dim, dropout=0.5):
        super().__init__()
        self.conv = SAGEConv(dim, dim, aggr='mean')
        self.norm = BatchNorm(dim)
        self.dropout = dropout

        self.linear_gate = nn.Linear(2 * dim, dim)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        gate = torch.sigmoid(self.linear_gate(torch.cat([x, h], dim=1)))
        out = gate * x + (1 - gate) * h
        return out


class EmbedderGNNv5(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.input_projection = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.mask_token = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(GatedBlock(hidden_dim, dropout))

        self.jk = JumpingKnowledge(mode='cat', channels=hidden_dim, num_layers=num_layers)

        self.output_projection = nn.Linear(hidden_dim * num_layers, out_dim)

    def forward(self, x, edge_index):
        mask_vector = self.mask_token
        x_masked = torch.cat([mask_vector.unsqueeze(0), x[1:].detach()], dim=0)

        x = self.input_proj(x_masked)
        x = self.input_norm(x)
        x = F.gelu(x)

        layer_outputs = []

        for layer in self.layers:
            x = layer(x, edge_index)
            layer_outputs.append(x)

        x = self.jk(layer_outputs)

        x = self.out_proj(x)
        return x