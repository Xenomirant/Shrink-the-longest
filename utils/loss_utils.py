import gudhi as gd
import typing
import torch
import torch_topological.nn as ttnn


_vr_complex = None


def barcode_entropy(lengths):
        return -torch.sum((lengths/torch.sum(lengths))*torch.log(lengths/torch.sum(lengths)))
    

def prominient_features_recursive(new_lens: torch.TensorType, n_bar: int, alpha: int):

    S = torch.sum(new_lens)

    for i in range(1, n_bar-2):

        Q = torch.ceil(alpha*n_bar*(alpha-1-torch.log(alpha))/(alpha-1)**2)

        S, S_prev = torch.sum(new_lens[i:]) + i*torch.sum(new_lens[i:])/torch.exp(barcode_entropy(new_lens[i:])), S

        C = S_prev/S

        if C < 1:
            break
    if Q < i:
        new_lens = torch.hstack((new_lens[:i], new_lens[-2:]))
        n_bar = i+2

        return prominient_features_recursive(new_lens, n_bar, alpha)
    else:

        return i


def prominient_features(barcodes: torch.TensorType):

    '''
    Algorithm for prominient topological feature selection. 
    Adapted from Nieves Atienza, Rocio Gonzalez-Diaz, & Matteo Rucco. (2017). 
    Persistent Entropy for Separating Topological Features from Noise in Vietoris-Rips Complexes.

    Args:
    barcodes: torch.tensor, 1-dimensional tensor, representing barcode lengths in the VR complex

    '''
    
    with torch.set_grad_enabled(False):
        inds = torch.sort(barcodes, descending=True).indices
        inds = torch.hstack((inds[1:], inds[0]))

        new_lens = barcodes[inds].clone()
        alpha = new_lens[-2]/new_lens[-1]

        n_bar = new_lens.shape[0]

        prom_feats_ind = prominient_features_recursive(new_lens, n_bar, alpha)
    # return both the differentiable barcodes and the number of features for logging
    return torch.hstack((barcodes[inds][-1], barcodes[inds][:prom_feats_ind])), barcodes[inds][prom_feats_ind:-1], prom_feats_ind


def entropy_loss(points: torch.TensorType, max_dim: int = 2, 
                 use_prominient_features=True, use_in_cluster_l2=True, l2_lambda = 0.5):
    '''
    Calculates pers. entropy maximization loss with adaptive selection of prominient features, 
    optionally adds an l2 distance minimization term acting on each found cluster.
    
    Args:
    points: torch.tensor, 2-dimensional tensor, representing the point cloud
    
    max_dim: int, maximum dimension of simplices used for persistent pairs calculation
    
    use_prominient_features: bool, whether to perform prominient feature selection 
    or to maximize entropy of distribution on pointwise distances
    
    use_in_cluser_l2: bool, whether to add an inner clusterwise l2 distance 
    minimization term to induce collapse of features inside clusters

    l2_lambda: float, hyperparameter, representing strength of inner l2 regularization
    '''

    # the second retured value can be ignored if number of features is not logged
    assert points.ndim == 2, "Points must be reshaped to a 2-dimensional tensor before proceeding"

    # default value
    ent = 0
    prom_feats = 0
    # compute persistence
    ind0, ind1 = get_persistent_pairs_gpu(points, max_dim=max_dim) if points.device != "cpu" else \
        get_persistent_pairs_cpu(points, max_dim=max_dim)
    
    lens = torch.norm(points[ind0[:, 1]] - points[ind0[:, 2]], dim=-1)
    if use_prominient_features:
        lens, in_cluster_lens, prom_feats_ind = prominient_features(lens)
        prom_feats += prom_feats_ind
    else:
        prom_feats += lens.shape[0]
    ent0 = barcode_entropy(lens)
    if (use_in_cluster_l2 and use_prominient_features):
        ent0 += l2_lambda*in_cluster_lens.mean()
    
    # compute entropy for higher dimensional simplices 
    for i in ind1:
        res = torch.norm(points[i[:, (0, 2)]] - points[i[:, (1, 3)]], dim=-1)
        lens = res[:, 1] - res[:, 0]
        if use_prominient_features:
            lens, in_cluster_lens, prom_feats_ind = prominient_features(lens)
            prom_feats += prom_feats_ind
        else:
            prom_feats += lens.shape[0]
        ent += barcode_entropy(lens)
        if (use_in_cluster_l2 and use_prominient_features):
            ent += l2_lambda*in_cluster_lens.mean()
        
    return ent + ent0, prom_feats

    
def get_persistent_pairs_cpu(points: torch.TensorType, max_dim=2):
    '''
    Uses Gudhi C++ backend for loss calculation
    Args:
    points: torch.tensor, 2-dimensional tensor, representing the point cloud
    max_dim: int, maximum dimension of simplices used for persistent pairs calculation 
    Results:
    ind0: np.array, indices of zero-dimensional persistent pairs (MST)
    ind1: List[np.array], list of higher-dimensional arrays with persistent pairs
    '''
        
    vr = gd.RipsComplex(points=points).create_simplex_tree(max_dimension=max_dim)
    vr.compute_persistence()
    # get critical simplices
    ind0, ind1 = vr.flag_persistence_generators()[:-2]

    return ind0, ind1

def get_persistent_pairs_gpu(points: torch.TensorType, max_dim=2):
    
    '''
    Uses torch_topological GPU accelerated backend for loss calculation
    Args:
    points: 2-dimensional tensor, representing the point cloud
    max_dim: maximum dimension of simplices used for persistent pairs calculation 
    (uses max_dim - 1 due to inner ttnn setup)
    Results:
    ind0: indices of zero-dimensional persistent pairs (MST)
    ind1: list of higher-dimensional arrays with persistent pairs
    '''
    
    global _vr_complex
    
    if _vr_complex is None:
        _vr_complex = ttnn.VietorisRipsComplex(dim=max_dim-1, return_generators=True)
    
    ind0, *ind1 = vr(points)
    ind0 = ind0.pairing  # Get pairing for ind0
    ind1 = [i.pairing for i in ind1]

    return ind0, ind1