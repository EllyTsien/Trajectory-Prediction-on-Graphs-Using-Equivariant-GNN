import torch
import numpy as np
import h5py
from graph import Graph, Trajectory

class BaseTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.trajectories = []
        self.graph = None

    def __getitem__(self, idx):
        return self.trajectories[idx].as_tensor()
    
    def __len__(self): 
        return len(self.trajectories)
    
    def getGraph(self):
        return self.graph

class GeoLifeTrajectoryDataset(BaseTrajectoryDataset):
    def __init__(self, path, n_samples=-1, min_trajectory_length=0, max_trajectory_length=-1, device=None):
        super().__init__()
        self.path = path
        self.n_samples = n_samples
        self.min_trajectory_length = min_trajectory_length
        self.max_trajectory_length = max_trajectory_length
        self.device = device
        self.load_dataset()
    
    def load_dataset(self):
        with h5py.File(self.path, "r") as f:
            # Load graph
            graph = f['graph']
            edge_list = torch.tensor(graph['edges'][()], device=self.device)
            node_coordinates = torch.tensor(graph['node_coordinates'][()], device=self.device)
            
            # only node feature is 'traffic signal'. Since there may appear multiple values per node, we encode it with 1 if it appears for a node, and 0 otherwise. 
            node_features = graph['node_features']['highway'][()]
            node_features[node_features == b'nan'] = 0
            node_features[node_features == b'traffic_signals'] = 1
            node_features = node_features.astype(np.float32)
            node_features = torch.tensor(node_features, device=self.device).unsqueeze(dim=1)
            self.graph = Graph(node_features, node_coordinates, edge_list)
            
            # Load trajectories
            trajects = f['trajectories']
            self.trajectories = []
            for file_ix in trajects.keys():
                trajectory = torch.tensor(trajects[file_ix]['edge_idxs'][:], device=self.device)
                if trajectory.shape[0] >= self.min_trajectory_length and (self.max_trajectory_length < 0 or trajectory.shape[0] <= self.max_trajectory_length):
                    edge_orientations = trajects[file_ix]['edge_orientations'][:]
                    self.trajectories.append(Trajectory(self.graph, trajectory, edge_orientations, reverse_edge_id=-1))
                
                if self.n_samples > 0 and len(self.trajectories) >= self.n_samples:
                    return

class PneumaTrajectoryDataset(BaseTrajectoryDataset):
    def __init__(self, path, n_samples=-1, min_trajectory_length=0, max_trajectory_length=-1, device=None):
        super().__init__()
        self.path = path
        self.n_samples = n_samples
        self.min_trajectory_length = min_trajectory_length
        self.max_trajectory_length = max_trajectory_length
        self.device = device
        self.load_dataset()
    
    def load_dataset(self):
        with h5py.File(self.path, "r") as f:
            # Load graph
            graph = f['graph']
            edge_list = torch.tensor(graph['edges'][()], device=self.device)
            node_coordinates = torch.tensor(graph['node_coordinates'][()], device=self.device)

            # node features: Only one is 'highway', which can be either 'nan' (0)
            # or 'traffic_signals' (1)
            node_features = graph['node_features']['highway'][()]
            node_features[node_features == b'nan'] = 0
            node_features[node_features == b'crossing'] = 0
            node_features[node_features == b'traffic_signals'] = 1
            node_features = node_features.astype(np.float32)
            node_features = torch.tensor(node_features, device=self.device).unsqueeze(dim=1)
            self.graph = Graph(node_features, node_coordinates, edge_list)
            
            # Load trajectories
            trajects = f['trajectories']
            self.trajectories = []
            for file_ix in trajects.keys():
                for plt_ix in trajects[file_ix].keys():
                    trajectory = torch.tensor(trajects[file_ix][plt_ix]['edge_idxs'][:], device=self.device)
                        
                    #check if current trajectory is within the range of min and max trajectory length
                    if trajectory.shape[0] >= self.min_trajectory_length and (self.max_trajectory_length < 0 or trajectory.shape[0] <= self.max_trajectory_length):
                        edge_orientations = trajects[file_ix][plt_ix]['edge_orientation'][:]
                        try:
                            traj = Trajectory(self.graph, trajectory, edge_orientations, reverse_edge_id=0)
                            self.trajectories.append(traj)
                        except ValueError:
                            pass
                    
                    # benchsize; set the number of samples load for one iteration
                    if self.n_samples > 0 and len(self.trajectories) >= self.n_samples:
                        return

class TDriveTrajectoryDataset(BaseTrajectoryDataset):
    def __init__(self, path, n_samples=-1, min_trajectory_length=0, max_trajectory_length=-1, device=None):
        super().__init__()
        self.path = path
        self.n_samples = n_samples
        self.min_trajectory_length = min_trajectory_length
        self.max_trajectory_length = max_trajectory_length
        self.device = device
        self.load_dataset()
    
    def load_dataset(self):
        with h5py.File(self.path, "r") as f:
            # Load graph
            graph = f['graph']
            edge_list = torch.tensor(graph['edges'][()], device=self.device)
            node_coordinates = torch.tensor(graph['node_coordinates'][()], device=self.device)

            # node features: Only one is 'highway', which can be either 'nan' (0)
            # or 'traffic_signals' (1)
            node_features = graph['node_features']['highway'][()]
            node_features[node_features == b'nan'] = 0
            node_features[node_features == b'traffic_signals'] = 1
            node_features = node_features.astype(np.float32)
            node_features = torch.tensor(node_features, device=self.device).unsqueeze(dim=1)
            self.graph = Graph(node_features, node_coordinates, edge_list)
            
            # Load trajectories
            trajects = f['trajectories']
            self.trajectories = []
            for file_ix in trajects.keys():
                trajectory = torch.tensor(trajects[file_ix]['edge_idxs'][:], device=self.device)
                if trajectory.shape[0] >= self.min_trajectory_length and (self.max_trajectory_length < 0 or trajectory.shape[0] <= self.max_trajectory_length):
                    edge_orientations = trajects[file_ix]['edge_orientations'][:]
                    self.trajectories.append(Trajectory(self.graph, trajectory, edge_orientations, reverse_edge_id=-1))
                        
                if self.n_samples > 0 and len(self.trajectories) >= self.n_samples:
                    return
                

class Distance_evaluation():
    def __init__(self, dataset, datapath) -> None:
        #self.dataset = dataset
        self.f = h5py.File(datapath, 'a')
        self.handle = self.f[dataset]


    def add(self, key, value):
        self.handle.create_dataset(name=key, data=value)