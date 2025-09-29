# Trajectory prediction on Graphs
This repository implements **trajectory prediction** on graph-structured data using **Equivariant Graph Neural Networks (EGNNs)**. Evaluation and comparision with other baseline models are also included, such as:
- **MLP baseline** – non-graph, agent-wise independent prediction.
- **GCN (Graph Convolutional Network)** – a standard message-passing GNN.
- **GAT (Graph Attention Network)** – attention-based aggregation of neighbor features.
- **SCoNe / Simplicial Complex Networks** – higher-order message passing using edges and triangles.

The goal is to model the trajectory as a graph and predict their **future positions** given observed trajectories, demostrating the contribution of **equivariance** and **higher-order structure** modeling of EGNN to trajectory prediction. EGNN ensures the model is **equivariant to rotations, translations, and reflections** in Euclidean space.


```bash
# Example command
python train.py --model egnn      # run EGNN
python train.py --model gcn       # run GCN baseline
python train.py --model gat       # run GAT baseline
