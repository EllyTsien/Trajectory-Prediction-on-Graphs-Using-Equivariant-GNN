import numpy as np
import scipy.sparse as sp
from bidict import bidict
from sklearn.preprocessing import normalize
import torch
import random


class TrajectoryMarkovChain():
    
    def __init__(self, order, data):
        self.order = order
        if self.order < 1:
            raise ValueError(f'Invalid markov chain order {self.order}')
               
        paths = []
        for traj in data:
            if len(traj) >= self.order:
                for i in range(len(traj) - self.order):
                    paths.append(traj[i:i + self.order + 1])
 
        history_set, target_set = [], []
        for h in paths:
            history_set.append(frozenset(h[:-1].cpu().numpy()))
            target_set.append(h[-1].item())        
        history_set = set(history_set)
        target_set = set(target_set)

        self.history_to_idx = bidict({history : idx for idx, history in enumerate(history_set)})
        self.target_to_idx = bidict({history : idx for idx, history in enumerate(target_set)})
        i, j = [], []
        for history in paths:
            j.append(self.history_to_idx[frozenset(history[:-1].cpu().numpy())])
            i.append(self.target_to_idx[history[-1].item()])
        self.transition_counts = sp.coo_matrix((np.ones(len(paths), dtype=float), (i, j)), 
                                               shape=(len(self.target_to_idx), len(self.history_to_idx)))
        
        self.num_states = len(self.target_to_idx)

        self.transition_matrix = normalize(self.transition_counts.tocsc(), norm='l1', axis=0)
        
    
    def predict(self, history, data, device):
        #history = path[-self.order:]
        if frozenset(history.cpu().numpy()) not in self.history_to_idx:   
            neighbors = list(data.graph.get_neighbors(history[-1].item()))
            endpoint = random.choice(neighbors)
            random_flag = True
            return torch.tensor(endpoint, device=device), random_flag
        
        
        next_node_idx = torch.from_numpy(self.transition_matrix[:, self.history_to_idx[frozenset(history.cpu().numpy())]].todense()).to(device).view(-1)
        endpoints = torch.tensor([self.target_to_idx.inverse[idx] for idx in range(self.num_states)], device=device)
        endpoint_idx = np.random.choice(a=len(next_node_idx), size= 1, p= next_node_idx.cpu())
        endpoint = endpoints[endpoint_idx].squeeze()
        random_flag = False
        return endpoint, random_flag
    


    '''
    def sparse_scatter_add(A: sp.spmatrix, idxs: NDArray, axis: int=0, dim_size: int | None = None):
        """ Takes elements of a sparse array along an axis and adds them according to indices.
        
        Parameters:
        -----------
        A : spmatrix, shape [N, N]
            The sparse input matrix
        idxs : Ndarray, shape [N]
            For each element on the axis, to which index to add it to
        axis : int, 0 or 1
            Which axis `idxs` refers to. If 0, `idxs` refers to the rows of `A`
            and if 1, `idxs` refers to the cols of `A`
        dim_size : int, optional
            If given, the number of indices to reduce to. If not given, inferred as
            `idxs.max() + 1`
        """
        assert axis in (0, 1), f'sparse matrices have only two axes, {axis} is inavlid'
        if dim_size is None:
            dim_size = idxs.max() + 1
        A = A.tocoo()
        n, m = A.shape
        row, col, data = A.row, A.col, A.data
        if axis == 0:
            n, m = dim_size, m
            row = idxs[row]
        elif axis == 1:
            col = idxs[col]
            n, m = n, dim_size
        result = sp.coo_matrix((data, (row, col)), shape=(n, m))
        result.sum_duplicates() # the scatter add
        return result

        


    def markov_chain_compute_visit_probabilities(self, transitions: sp.coo_matrix, steps: int, 
                                                 state_to_node_map: NDArray, num_nodes: int | None = None, 
                                                 verbose: bool=False):
    ''' 
    '''
        Computes the probability of visiting a node in at most `steps` steps given a starting state.

        Parameters
        ----------
        transitions : sp.coo_matrix, shape [S, S]
            Markov chain transition matrix
        steps : int
            The maximal number of steps
        state_to_node_map : NDArray, shape [S]
            Maps each state in the markov chain to its node endpoint.
            
        Notes:
        ------
        The explanation for computing this probabilities: 
        Let p(s - (<=i) -> v) be the probability of reaching a state associated with node v starting at state s
        in exactly <= i steps at least once and p(s - (i) -> v) of reaching a state associated with node v *for the first time*
        after exactly i steps.
        
        p(s - (<=k) -> v) = p(s - (1) -> v) + p(s - (2) -> v) + ... p(s - (k) -> v) as all summands are disjoint events
        
        p(s - (i) -> v) can be computed recursively using DP:
            f^(1)_v,s := p(s - (1) -> v) = sum s' such that s' ends at v: p(s -> s') (note that there should only be one summand. The summands are defined by `state_to_node_map`)
            
            f^(i+1)_v,s p(s - (i + 1) -> v) = sum s' such that s' *does not* end at v: p(s -> s') p(s' - (i) -> v)
                                            = sum s' such that s' *does not* end at v: f^(i)_v,s * p(s -> s')
        
            This sum can be expressed via matrix multiplications using a matrix  \\tilde{F}^(i)_v,s = f^(i)_v,s if s does not end at v
                                                                                        = 0 if s ends at v
            F^(i+1) = \\tilde{F}^(i) P
    '''
    '''
        if num_nodes is None:
            num_nodes = state_to_node_map.max() + 1
        num_states = transitions.shape[0]
        assert transitions.shape[0] == transitions.shape[1], f'Transition matrix needs to be quadratic, not {transitions.shape}'
        F = self.sparse_scatter_add(transitions, state_to_node_map, axis=0, dim_size=num_nodes).tolil() # num_nodes x num_states
        result = F.copy()
        iterator = tqdm(range(steps - 1), desc='Computing visit probabilities') if verbose else range(steps - 1)
        for _ in iterator:
            F[state_to_node_map, np.arange(num_states)] = 0.0 # exclude paths that visit target node after 1 step
            F = (F @ transitions).tolil()
            result += F
        return result


    def visit_probabilities(self) -> sp.spmatrix:
        transition_matrix = self.transition_matrix
        state_to_endpoint = np.array([self.history_to_idx.inverse[state_idx][-1] for state_idx in range(self.num_states)])
        visit_probabilities_cache = self.markov_chain_compute_visit_probabilities(transition_matrix, 
                                                                             self.complex_probabilities_horizon, 
                                                                             state_to_endpoint, 
                                                                             num_nodes=self.num_nodes, 
                                                                             verbose=self.verbose)

        return visit_probabilities_cache
    '''
    
    
  