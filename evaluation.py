import torch
import numpy as np
import networkx as nx
import pickle
from filelock import FileLock
import os

class AutoregressiveEvaluation:
    def __init__(self, dataset, node_coordinates, edges):
        """
        Initializes AutoregressiveEvaluation class.

        Args:
            dataset (String): Name of dataset ("geolife", "tdrive" or "pneuma")
        """
        self.file_path = f"/ceph/hdd/students/weea/evaluations_storage_{dataset}.pkl"
        self.lockfile_path = f"{self.file_path}.lock"
        self.lock = FileLock(self.lockfile_path)
        self.main_dict = self.load_distances()
        self.temp_dict = {}

        self.edge_distances = torch.cdist(node_coordinates, node_coordinates)[edges[:, 0], edges[:, 1]]
    
    def load_distances(self):
        with self.lock:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'rb') as f:
                    d = pickle.load(f)
            else:
                d = {}

        print(f"Loaded {len(d)} distances from file")
        return d
    
    def update_stored_dict(self):
        if len(self.temp_dict) == 0:
            return
        with self.lock:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'rb') as f:
                    d = pickle.load(f)
            else:
                d = {}
            merged_dict = d | self.temp_dict
            with open(self.file_path, 'wb') as f:
                pickle.dump(merged_dict, f)
        
        self.main_dict = merged_dict
        self.temp_dict = {}

    def eval_accuracy(self, predictions, targets):
        """
        Evaluate the average and final accuracy of a trajectory prediction model.

        Args:
            predictions (List): List of predicted node indices
            targets (Tensor): Tensor of target node indices
        """
        accuracies = (torch.stack(predictions) == targets) * 1.0

        return {
            "average_accuracy": accuracies.mean(), 
            "final_accuracy": accuracies[-1]
        }


    def eval_euclidean_distance(self, graph, predictions, targets):
        """
        Evaluate the average and final euclidean distance error of a trajectory prediction model.

        Args:
            predictions (List): List of predicted node indices
            targets (Tensor): Tensor of target node indices
        """
        distances = torch.norm(graph.node_coordinates[predictions, :] - graph.node_coordinates[targets, :], dim=1)
        
        return {
            "average_distance": distances.mean(), 
            "final_distance": distances[-1]
        }


    def eval_shortest_path(self, graph, predictions, targets):
        """
        Evaluate the average and final path length (in `nodes` and `path length`) of a trajectory prediction model.

        Args:
            predictions (List): List of predicted node indices
            targets (Tensor): Tensor of target node indices
        """
        hops, lengths = [], []
        for pred, tar in zip(predictions, targets.cpu().numpy()):
            # get unique ordering to avoid duplicate entries
            n0 = min(pred, tar).item()
            n1 = max(pred, tar).item()

            if (n0, n1) not in self.main_dict:
                if (n0, n1) not in self.temp_dict:
                    hop = nx.shortest_path_length(graph.nx_graph, n0, n1) * 1.0
                    length = nx.shortest_path_length(graph.nx_graph, n0, n1, weight=self.edge_distances)
                    self.temp_dict[(n0, n1)] = (
                        hop,
                        length
                    )
                else:
                    hop, length = self.temp_dict[(n0, n1)]
            else:
                hop, length = self.main_dict[(n0, n1)]

            hops.append(hop)
            lengths.append(length)
        
        hops = torch.tensor(hops, dtype=torch.float32, device=targets.device)
        lengths = torch.tensor(lengths, dtype=torch.float32, device=targets.device)

        return {
            "average_hops": hops.mean(), 
            "final_hops": hops[-1], 
            "average_path_length": lengths.mean(), 
            "final_path_length": lengths[-1]
        }