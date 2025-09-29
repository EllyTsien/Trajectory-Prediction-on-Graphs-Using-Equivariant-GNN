import torch
import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn.functional as F



class SimpleGNN(torch.nn.Module):
	def __init__(self, node_features, gcn_dims, fc_dims, gcn_activation=F.relu, fc_activation=F.relu):
		super().__init__()
		self.gcn_dims = gcn_dims
		self.fc_dims = fc_dims
		self.gcn_activation = gcn_activation
		self.fc_activation = fc_activation
		self.gcn_layers = []

		last_layer_dim = node_features
		for d in gcn_dims:
			self.gcn_layers.append(GCNConv(last_layer_dim, d))
			last_layer_dim = d


		self.linear_layers = []
		for d in fc_dims:
			self.linear_layers.append(torch.nn.Linear(last_layer_dim, d))
			last_layer_dim = d
		# append prediction layer
		self.linear_layers.append(torch.nn.Linear(last_layer_dim, 1))

		self.gcn_layers = torch.nn.ModuleList(self.gcn_layers)
		self.linear_layers = torch.nn.ModuleList(self.linear_layers)

	def forward(self, x, edge_index):

		for layer in self.gcn_layers:
			x = layer(x, edge_index)
			x = self.gcn_activation(x)
		
		for i, layer in enumerate(self.linear_layers):
			x = layer(x)
			if i < len(self.linear_layers)-1:
				x = self.fc_activation(x)

		return x


class SimpleGNN_2Layer(torch.nn.Module):
	def __init__(self, node_features):
		super().__init__()
		self.conv1 = GCNConv(node_features+1, 32)
		self.conv2 = GCNConv(32, 16)
		self.linear = torch.nn.Linear(16, 1)

	def forward(self, x, edge_index):

		x = self.conv1(x, edge_index)
		x = F.relu(x)
		next_node = self.conv2(x, edge_index)
		next_node = F.relu(next_node)
		next_node = self.linear(next_node)

		return next_node