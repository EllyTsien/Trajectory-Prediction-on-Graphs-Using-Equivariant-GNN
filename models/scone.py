import torch

class SCoNeLayer(torch.nn.Module):
    # see https://arxiv.org/pdf/2102.10058: Algorithm 1 and Algorithm S-2
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W0 = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.W1 = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.W2 = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.W0)
        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.xavier_uniform_(self.W2)

    def forward(self, x, B1, B2):
        # d2 = B2 @ B2^T @ x @ W2^l
        d2 = torch.sparse.mm(B2, torch.sparse.mm(B2.t(), torch.matmul(x, self.W2)))
        
        # d1 = x @ W1^l
        d1 = torch.matmul(x, self.W1)
        
        # d0 = B1^T @ B1 @ x @ W0^l
        d0 = torch.sparse.mm(B1.t(), torch.sparse.mm(B1, torch.matmul(x, self.W0)))
        
        return torch.relu(d2 + d1 + d0)

class SCoNe(torch.nn.Module):
    def __init__(self, in_features, hidden_features, num_layers):
        super().__init__()
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList([
            SCoNeLayer(in_features if i == 0 else hidden_features, hidden_features)
            for i in range(num_layers)
        ])
        self.W0_L = torch.nn.Parameter(torch.Tensor(hidden_features, 1))
        torch.nn.init.xavier_uniform_(self.W0_L)

    def forward(self, x, B1, B2):
        for layer in self.layers:
            x = layer(x, B1, B2)

        # x = B1 @ x @ W0^L
        x = torch.sparse.mm(B1, torch.matmul(x, self.W0_L))
        return x