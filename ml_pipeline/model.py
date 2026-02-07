import torch
import torch_geometric
from torch_geometric.nn import SAGEConv
from torch.nn.functional import relu

class EmbedderGNN(torch.nn.Module):
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
        h = relu(h)
        h = h + x

        h2 = self.conv2(h, edge_index)
        h2 = self.norm2(h2)
        return h2 + h
