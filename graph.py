import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_undirected

class Graph():
    def __init__(self, node_features, node_coordinates, edges):
        """
        Initialize the class with node features, node coordinates, and edges.
        """
        self.nodes = node_features
        self.node_coordinates = node_coordinates
        self.edges = edges
        self.edge_index = edges.T.contiguous()
        self.calculate_neighbors()

        self.nx_graph = nx.from_edgelist(self.edges.cpu().numpy())
        
        # required for SCoNe
        self.B1, self.B2 = self.graph_to_simplicial_complex()
    
    def calculate_neighbors(self):
        self.neighbor = {}
        for e in self.edges:
            if e[0].item() not in self.neighbor:
                self.neighbor[e[0].item()] = set()
            if e[1].item() not in self.neighbor:
                self.neighbor[e[1].item()] = set()
            
            self.neighbor[e[0].item()].add(e[1].item())
            self.neighbor[e[1].item()].add(e[0].item())
    
    def get(self, node_representation='coordinate', edge_representation='list'):
        """
        Retrieves the nodes and edges of the graph.
        """
        nodes = None
        edges = None

        if node_representation == 'coordinate':
            nodes = self.nodes
        else:
            raise NotImplementedError(f"Not defined nodes representation '{node_representation}'")

        if edge_representation == 'list':
            edges = self.edges
        else:
            raise NotImplementedError(f"Not defined edge representation '{edge_representation}'")
        
        return nodes, edges
    
    def get_plot_data(self):
        pos = {k: x for k, x in enumerate(self.node_coordinates.cpu().numpy())}

        return self.nx_graph, pos
    
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        _, pos = self.get_plot_data()
        nx.draw_networkx(self.nx_graph, with_labels=False, pos=pos, ax=ax)
    
    def get_neighbors(self, node):
        return self.neighbor[node]
    
    def get_all_shortest_path(self):
        nx_graph = nx.from_edgelist(np.array(self.edges))
        self.all_shortest_path = dict(nx.all_pairs_shortest_path(nx_graph))
    
    def graph_to_simplicial_complex(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.nodes.shape[0]))
        directed_edges = to_undirected(self.edges.T).t()
        G.add_edges_from(directed_edges.tolist())

        edges = list(G.edges())
        self.edge_to_idx = {tuple(sorted(e)): i for i, e in enumerate(edges)}
        triangles = list(nx.find_cliques(self.nx_graph))
        triangles = [t for t in triangles if len(t) == 3]

        # B1 (signed incidence matrix)
        B1_data, B1_row, B1_col = [], [], []
        for i, (u, v) in enumerate(self.edges):
            B1_data.extend([1, -1])
            B1_row.extend([u, v])
            B1_col.extend([i, i])

        # B2
        B2_data, B2_row, B2_col = [], [], []
        for i, (u, v, w) in enumerate(triangles):
            for e in [(u, v), (v, w), (w, u)]:
                edge_idx = self.edge_to_idx[tuple(sorted(e))]
                B2_data.append(1 if e[0] < e[1] else -1)
                B2_row.append(edge_idx)
                B2_col.append(i)

        # Convert to sparse tensors
        B1 = torch.sparse_coo_tensor(torch.tensor([B1_row, B1_col]), torch.tensor(B1_data, dtype=torch.float32), size=(self.nodes.shape[0], len(edges)))
        B2 = torch.sparse_coo_tensor(torch.tensor([B2_row, B2_col]), torch.tensor(B2_data, dtype=torch.float32), size=(len(edges), len(triangles)))

        return B1, B2
    
class Trajectory:
    # reverse_edge_id has to be -1 for geodrive and tdrive, 0 for pneuma
    def __init__(self, graph, edge_indices, edge_orientation, reverse_edge_id=-1):
        self.edge_indices = edge_indices
        self.edges = graph.edges[edge_indices]
        self.edge_orientation = edge_orientation

        for i in range(len(self.edges)):
            if edge_orientation[i] == reverse_edge_id:
                self.edges[i] = self.edges[i].flip(dims=(0,))
        
        self.node_path = torch.cat((self.edges[0, 0].unsqueeze(dim=0), self.edges[:, 1]), dim=0)
        self.chain = self.create_chain(graph)

        if not self.validate_trajectory(graph):
            raise ValueError("Invalid trajectory")
    
    def create_chain(self, graph):
        chain = torch.zeros(len(graph.edges), dtype=torch.float32)

        chain[self.edge_indices] = torch.tensor(self.edge_orientation, dtype=torch.float32)
        return chain
    
    def trajectory_to_chain(trajectory, graph, device=None):
        trajectory = trajectory.cpu().numpy()
        chain = torch.zeros(graph.edges.shape[0], dtype=torch.float, device=device)
        
        # Iterate through the trajectory
        for i in range(len(trajectory) - 1):
            u, v = trajectory[i], trajectory[i+1]
            edge = tuple(sorted((u, v)))
            
            if edge in graph.edge_to_idx:
                idx = graph.edge_to_idx[edge]
                # Set the value to 1 or -1 depending on the orientation
                chain[idx] = 1 if u < v else -1
            else:
                raise Exception(f"Edge {edge} not found in the simplicial complex")
        
        return chain
    
    def validate_trajectory(self, graph):
        valid = True
        current_node = self.node_path[0]
        for i in range(1, len(self.node_path)):
            if self.node_path[i].item() in graph.get_neighbors(current_node.item()):
                current_node = self.node_path[i]
            else:
                valid = False
                break
        
        return valid
    
    def __getitem__(self, idx):
        return self.node_path[idx]
    
    def __len__(self): 
        return len(self.edge_indices)
    
    def as_tensor(self):
        return self.node_path

    def plot(self, graph, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        nx_graph, pos = graph.get_plot_data()

        nx.draw_networkx(nx_graph, with_labels=False, pos=pos, ax=ax, node_size=10, alpha=0.4)
        nx.draw_networkx_edges(nx_graph, pos=pos, edgelist=self.edges.cpu().numpy(), ax=ax, edge_color='red', width=2, node_size=0)

class TrajectoryEncoder:
    def __init__(self, num_nodes, encoding, dim=2, device=None):
        self.max_length = 200
        self.num_nodes = num_nodes
        self.encoding = encoding
        self.device = device
        self.dim = dim

        self.precalculate_encodings()

    def precalculate_encodings(self):
        if self.encoding == 'uniform':
            self.precomputed = 1
        elif self.encoding == 'linear':
            self.precomputed = []
            for i in range(self.max_length):
                self.precomputed.append(
                    ((1+ torch.arange(i, device=self.device)) / i).unsqueeze(dim=1)
                )
        elif self.encoding == 'transformer':
            n = 1000 # fixed parameter -> open issue of you want to change it

            self.precomputed = torch.zeros(self.max_length, self.dim, device=self.device)
            for k in range(self.max_length):
                for i in torch.arange(int(self.dim/2)):
                    denominator = torch.pow(n, 2*i/self.dim)
                    self.precomputed[k, 2*i] = torch.sin(k/denominator)
                    self.precomputed[k, 2*i+1] = torch.cos(k/denominator)


    def encode_trajectory(self, path):
        if self.encoding == 'uniform':
            on_path = torch.zeros(self.num_nodes, 1, device=self.device)
            on_path[path] = self.precomputed
        elif self.encoding == 'linear':
            on_path = torch.zeros(self.num_nodes, 1, device=self.device)
            on_path[path] = self.precomputed[path.shape[0]]
        elif self.encoding == 'transformer':
            on_path = torch.zeros(self.num_nodes, self.dim, device=self.device)
            on_path[path] = self.precomputed[:path.shape[0]]
        else:
            raise NotImplementedError(f"Unknown trajectory encoding: {self.encoding}")
        
        return on_path