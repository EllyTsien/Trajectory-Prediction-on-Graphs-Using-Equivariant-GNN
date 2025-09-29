import torch
import torch_geometric
from torch_geometric.nn.conv import GATv2Conv
import torch.nn.functional as F

class GATNetwork(torch.nn.Module):
    def __init__(self, node_features, gat_dims, fc_dims, n_heads, gat_activation=F.relu, fc_activation=F.relu):
        super().__init__()
        self.gat_dims = gat_dims
        self.fc_dims = fc_dims
        self.n_heads = n_heads
        self.gat_activation = gat_activation
        self.fc_activation = fc_activation
        self.gat_layers = []

        last_layer_dim = node_features
        for d in gat_dims:
            self.gat_layers.append(GATv2Conv(last_layer_dim, d // n_heads, n_heads))
            last_layer_dim = (d // n_heads) * n_heads
        
        self.linear_layers = []
        for d in fc_dims:
            self.linear_layers.append(torch.nn.Linear(last_layer_dim, d))
            last_layer_dim = d
        
        # prediction layer
        self.linear_layers.append(torch.nn.Linear(last_layer_dim, 1))

        self.gat_layers = torch.nn.ModuleList(self.gat_layers)
        self.linear_layers = torch.nn.ModuleList(self.linear_layers)

    def forward(self, x, edge_index):
        for layer in self.gat_layers:
            x = layer(x, edge_index)
            x = self.gat_activation(x)
        
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if i < len(self.linear_layers)-1:
                x = self.fc_activation(x)
                
        return x