import torch
import torch_geometric
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx

from graph import TrajectoryEncoder
from models.gat import GATNetwork
from datasets import GeoLifeTrajectoryDataset, TDriveTrajectoryDataset, PneumaTrajectoryDataset, Distance_evaluation

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset')
parser.add_argument('--trajectory', default='transformer', type=str, help='Trajectory encoding (default: transformer)')
parser.add_argument('--name', default=None, type=str, help='Optional file name to save results (default: <dataset>_<model>)')
parser.add_argument('--plot_trajectories', default=False, action='store_true', help='Plot some sample trajectories')


def load_dataset(dataset):
    kwargs = {
        'n_samples': -1,
        'min_trajectory_length': 4,
        'max_trajectory_length': 1000,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    if dataset == "geolife":
        test_data = GeoLifeTrajectoryDataset("/ceph/hdd/students/weea/trajectory-prediction-on-graphs/datasets/geolife_test.h5", **kwargs)
    elif dataset == "tdrive":
        test_data = TDriveTrajectoryDataset("/ceph/hdd/students/weea/trajectory-prediction-on-graphs/datasets/tdrive_test.h5", **kwargs)
    elif dataset == "pneuma":
        test_data = PneumaTrajectoryDataset("/ceph/hdd/students/weea/trajectory-prediction-on-graphs/datasets/pneuma_test.h5", **kwargs)
    else:
        raise NotImplementedError("Unknown dataset")
    
    return test_data

def load_model(model, data, trajectory_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved_dict = torch.load(model, map_location=device)

    # automatically determine model type
    if 'gat_dims' in saved_dict.keys():
        # GAT
        model = GATNetwork(node_features=2 + data.graph.nodes.shape[1] + trajectory_dim, gat_dims=saved_dict['gat_dims'], fc_dims=saved_dict['fc_dims'], n_heads=saved_dict['n_heads'])
        model_type = 'gat'

    model.eval()
    model.to(device)
    return model, model_type

def evaluate(args, model, model_type, data, traject_encoder):
    def nodes2edges(nodes):
        edges = []
        last_edge = nodes[0]
        for n in nodes[1:]:
            edges.append((last_edge, n))
            last_edge = n

        return edges
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    min_history = 3
    max_length = 10
    avg_accuracy = torch.zeros(max_length - min_history, dtype=torch.float, device=device)
    avg_distance_error = torch.zeros(max_length - min_history, dtype=torch.float, device=device)
    n_samples = torch.zeros(max_length - min_history, device=device)
    dataloader = torch.utils.data.DataLoader(data, batch_size=1)

    plot_idx = -1
    fig = None
    if args.plot_trajectories:
        n_rows, n_cols = 5, 5
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
        G, pos = data.graph.get_plot_data()

        plot_idx = n_rows * n_cols -1

    for batch in tqdm(dataloader):
        batch = batch.squeeze(0)
        
        targets = batch[min_history:max_length]
        path = batch[:min_history]

        for i, t in enumerate(targets):
            on_path = traject_encoder.encode_trajectory(path)
            with torch.no_grad():
                if model_type == 'simple_gnn' or model_type == 'gat':
                    nfeatures = torch.concat([data.graph.node_coordinates, data.graph.nodes, on_path], dim=1).to(dtype=torch.float32)
                    next_node = model(nfeatures, data.graph.edge_index)
                    
                next_node = torch.softmax(next_node, dim=0)
        
                last_node = path[-1].item()
                last_node_neighbors = list(data.graph.get_neighbors(last_node))
                neighbor_mask = torch.zeros(next_node.shape[0], 1, device=device)
                neighbor_mask[last_node_neighbors] = 1

                prediction = (neighbor_mask * next_node).argmax()
                avg_accuracy[i] += (prediction == t) * 1.0
                avg_distance_error[i] += torch.norm(data.graph.node_coordinates[prediction] - data.graph.node_coordinates[t])
                n_samples[i] += 1

                path = torch.cat([path, prediction.unsqueeze(dim=0)], dim=0)
        
        if plot_idx > -1:
            edges_history = nodes2edges(batch[:min_history].cpu().numpy())
            edges_truth = nodes2edges(torch.concat([batch[min_history-1].unsqueeze(0), targets]).cpu().numpy())
            edges_pred = nodes2edges(path[min_history-1:].cpu().numpy())

            r = plot_idx  // n_cols
            c = plot_idx % n_cols
            #nx.draw_networkx(G, pos=pos, node_size=10, with_labels=False, ax=ax[r, c], alpha=0.5)
            nx.draw_networkx_edges(G, pos=pos, edgelist=edges_history, edge_color='red', width=4, node_size=0, ax=ax[r, c], alpha=0.3)
            nx.draw_networkx_edges(G, pos=pos, edgelist=edges_truth, edge_color='green', width=2, node_size=0, ax=ax[r, c], alpha=0.3)
            nx.draw_networkx_edges(G, pos=pos, edgelist=edges_pred, edge_color='blue', width=2, node_size=0, ax=ax[r, c], alpha=0.3)

            plot_idx -= 1
    
    avg_accuracy /= n_samples
    avg_distance_error /= n_samples

    return avg_accuracy.cpu(), avg_distance_error.cpu(), fig

def save_results(accuracy, distance_error, fig_trajectories, filepath):
    accuracy = torch.nan_to_num(accuracy)
    distance_error = torch.nan_to_num(distance_error)

    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    axs[0].plot(range(len(accuracy)), accuracy, marker='o', linewidth=1, markersize=10)
    axs[0].set_title("Average prediction accuracy")

    axs[1].plot(range(len(distance_error)), distance_error, marker='o', linewidth=1, markersize=10)

    fig.savefig(f'{filepath}.png')

    df = pd.DataFrame({
        'accuracy': accuracy.numpy(), 
        'distance_error': distance_error.numpy()
    })
    df.to_csv(f'{filepath}.csv', index=False)

    if not fig_trajectories is None:
        fig_trajectories.savefig(f'{filepath}_trajectories.png')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_dataset(args.dataset)
    print(f"Loaded {len(data)} samples")
    traject_encoder = TrajectoryEncoder(data.graph.node_coordinates.shape[0], args.trajectory, 2, device)
    model, model_type = load_model(args.model, data, 2 if args.trajectory == 'transformer' else 1)

    acc, dis, fig = evaluate(args, model, model_type, data, traject_encoder)

    if not args.name is None:
        filepath = f"results/{args.name}"
    else:
        filepath = f"results/{args.model}_{args.dataset}"
    save_results(acc, dis, fig, filepath)

    print("Accuracy: ", acc)
    print("Distance: ", dis)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)