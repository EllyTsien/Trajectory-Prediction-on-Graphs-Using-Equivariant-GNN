import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, activation=F.relu):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.layers = []

        last_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(torch.nn.Linear(last_dim, dim))
            last_dim = dim
        self.layers.append(torch.nn.Linear(last_dim, 1)) #output dimension is set to 1

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:  # Apply activation except for the last layer
                x = self.activation(x)
        return x
