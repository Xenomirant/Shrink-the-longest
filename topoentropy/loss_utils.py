import math

import torch


def barcode_entropy(lengths: torch.Tensor):
        return -torch.sum((lengths/torch.sum(lengths))*torch.log(lengths/torch.sum(lengths)))


def prominient_count(desc: torch.Tensor) -> int:
    '''
    Number of prominient features following
    Nieves Atienza, Rocio Gonzalez-Diaz, & Matteo Rucco. (2017).
    Persistent Entropy for Separating Topological Features from Noise in Vietoris-Rips Complexes.

    Vectorized form of the original sequential rule: for a fixed number of bars all candidate
    cut points i are evaluated at once on the device, so the loop only runs over the
    (few) recursive reductions of the barcode.

    Args:
    desc: torch.tensor, 1-dimensional tensor of n > 3 bar lengths sorted in descending order

    Returns k: the k + 1 longest bars are prominient
    '''
    n = desc.shape[0]
    large, small = desc[0], desc[-1]
    prefix = desc[1:-1]                      # the barcode is (prefix[:m], small, large)
    alpha = small / large
    k = torch.arange(prefix.shape[0], device=desc.device)
    xlogx = lambda x: x * torch.log(x)
    m = n - 2
    while True:
        n_bar = m + 2
        valid = k < m
        x = torch.where(valid, prefix, torch.zeros_like(prefix))
        xl = torch.where(valid, xlogx(torch.where(valid, prefix, torch.ones_like(prefix))), torch.zeros_like(prefix))
        # sums over tails (prefix[i:m], small, large) for every i
        T = torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0]) + small + large
        E = torch.flip(torch.cumsum(torch.flip(xl, [0]), 0), [0]) + xlogx(small) + xlogx(large)
        # S_i = T_i + i * T_i / exp(H_i), and T_i / exp(H_i) = exp(E_i / T_i)
        S = T + k * torch.exp(E / T)
        stop = (S[:-1] < S[1:]) & (k[1:] <= m - 1)
        i = int(torch.argmax(stop.to(torch.int8))) + 1 if bool(stop.any()) else m - 1
        a = float(alpha)
        Q = math.ceil(a * n_bar * (a - 1 - math.log(a)) / (a - 1) ** 2) if 0 < a < 1 else math.nan
        if Q < i:
            m = i
            continue
        return i


def prominient_features(barcodes: torch.Tensor):

    '''
    Algorithm for prominient topological feature selection.
    Adapted from Nieves Atienza, Rocio Gonzalez-Diaz, & Matteo Rucco. (2017).
    Persistent Entropy for Separating Topological Features from Noise in Vietoris-Rips Complexes.

    Args:
    barcodes: torch.tensor, 1-dimensional tensor, representing barcode lengths in the VR complex

    Returns (prominient bars, in-cluster bars, number of features for logging)
    '''

    n = barcodes.shape[0]
    inds = torch.sort(barcodes.detach(), descending=True).indices
    desc = barcodes[inds]
    if n <= 3:
        # too few bars for the selection rule: keep all of them as prominient
        return desc, desc[:0], n
    with torch.no_grad():
        k = prominient_count(desc.detach())
    return desc[:k + 1], desc[k + 1:], k


def mst_pairs(points: torch.Tensor):
    '''
    Zero-dimensional persistence pairs of the Vietoris-Rips filtration, i.e. the edges of the
    Euclidean minimum spanning tree, computed with Prim's algorithm on the device of `points`
    (runs on GPU for CUDA / MPS tensors).

    Returns ind0: torch.tensor of shape (n - 1, 3) in Gudhi's layout (vertex, edge end, edge end)
    '''
    n = points.shape[0]
    with torch.no_grad():
        D = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
        in_tree = torch.zeros(n, dtype=torch.bool, device=points.device)
        in_tree[0] = True
        best = D[0].clone()
        parent = torch.zeros(n, dtype=torch.long, device=points.device)
        child = torch.empty(n - 1, dtype=torch.long, device=points.device)
        par = torch.empty(n - 1, dtype=torch.long, device=points.device)
        inf = torch.tensor(float("inf"), dtype=D.dtype, device=points.device)
        for t in range(n - 1):
            j = torch.argmin(torch.where(in_tree, inf, best))
            child[t], par[t] = j, parent[j]
            in_tree[j] = True
            upd = D[j] < best
            best = torch.where(upd, D[j], best)
            parent = torch.where(upd, j, parent)
    return torch.stack((child, par, child), dim=1)


def get_persistent_pairs(points: torch.Tensor, max_dim=2):
    '''
    Computes persistence pairs of the Vietoris-Rips filtration with Gudhi (C++, on the host).
    Only the indices are computed there, bar lengths stay differentiable on the original device.
    Args:
    points: torch.tensor, 2-dimensional tensor, representing the point cloud
    max_dim: int, maximum dimension of simplices used for persistent pairs calculation
    (1 by default: H0 only, computed on the device; larger values require Gudhi)
    Results:
    ind0: torch.tensor, indices of zero-dimensional persistent pairs (MST)
    ind1: List[torch.tensor], list of higher-dimensional arrays with persistent pairs
    '''

    try:
        import gudhi as gd
    except ImportError as e:
        raise ImportError("max_dim > 1 requires Gudhi: pip install 'topoentropy[higher]'") from e

    vr_complex = gd.RipsComplex(points=points.detach().cpu().double().numpy()).create_simplex_tree(max_dimension=max_dim)
    vr_complex.compute_persistence()
    # get critical simplices
    ind0, ind1 = vr_complex.flag_persistence_generators()[:2]

    def to_index(a):
        return torch.as_tensor(a, dtype=torch.long, device=points.device)

    return to_index(ind0), [to_index(a) for a in ind1]


def get_persistent_pairs_gpu(points: torch.Tensor, max_dim=2):
    '''H0 pairs on the device of `points` for max_dim = 1, Gudhi on the host otherwise.'''
    if max_dim == 1:
        return mst_pairs(points), []
    return get_persistent_pairs(points, max_dim=max_dim)


# kept for backward compatibility
get_persistent_pairs_cpu = get_persistent_pairs


def entropy_loss(points: torch.Tensor, max_dim: int = 1,
                 use_prominient_features=True, use_in_cluster_l2=True, l2_lambda = 0.5,
                 backend: str = "auto"):
    '''
    Calculates pers. entropy maximization loss with adaptive selection of prominient features,
    optionally adds an l2 distance maximization term acting on each found cluster.
    The returned value is meant to be maximized, i.e. subtracted from the training loss.

    Args:
    points: torch.tensor, 2-dimensional tensor, representing the point cloud

    max_dim: int, maximum dimension of simplices used for persistent pairs calculation
    (1 by default: H0 only, computed on the device; larger values require Gudhi)

    use_prominient_features: bool, whether to perform prominient feature selection
    or to maximize entropy of distribution on pointwise distances

    use_in_cluster_l2: bool, whether to add an inner clusterwise l2 distance
    maximization term (mean length of non-prominient bars) to spread features inside clusters

    l2_lambda: float, hyperparameter, representing strength of inner l2 regularization

    backend: "auto" | "device" | "gudhi". "device" computes H0 pairs (max_dim = 1) on the
    device of `points`, "gudhi" computes all pairs on the host, "auto" uses "device" for
    max_dim = 1 and "gudhi" otherwise.
    '''

    # the second retured value can be ignored if number of features is not logged
    assert points.ndim == 2, "Points must be reshaped to a 2-dimensional tensor before proceeding"
    if backend == "auto":
        backend = "device" if max_dim == 1 else "gudhi"
    if backend == "device" and max_dim != 1:
        raise ValueError("backend='device' supports max_dim=1 (H0) only")

    # default value
    ent = 0
    prom_feats = 0
    # compute persistence
    ind0, ind1 = (mst_pairs(points), []) if backend == "device" else get_persistent_pairs(points, max_dim=max_dim)

    lens = torch.norm(points[ind0[:, 1]] - points[ind0[:, 2]], dim=-1)
    if backend == "device":
        # Gudhi drops zero-length bars (duplicate points), do the same
        lens = lens[lens > 0]
    if use_prominient_features:
        lens, in_cluster_lens, prom_feats_ind = prominient_features(lens)
        prom_feats += prom_feats_ind
    else:
        prom_feats += lens.shape[0]
    ent0 = barcode_entropy(lens)
    if (use_in_cluster_l2 and use_prominient_features and in_cluster_lens.numel() > 0):
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
        if (use_in_cluster_l2 and use_prominient_features and in_cluster_lens.numel() > 0):
            ent += l2_lambda*in_cluster_lens.mean()

    return ent + ent0, prom_feats
