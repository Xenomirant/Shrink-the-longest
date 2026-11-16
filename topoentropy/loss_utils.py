import math

import torch

# largest point cloud for which the minimax MST is used by default (measured crossover with Prim)
MINIMAX_MAX_POINTS = {"cpu": 128, "gpu": 512}


def barcode_entropy(lengths: torch.Tensor):
        return -torch.sum((lengths/torch.sum(lengths))*torch.log(lengths/torch.sum(lengths)))


def prominient_count(desc: torch.Tensor) -> int:
    '''
    Number of prominient features following
    Nieves Atienza, Rocio Gonzalez-Diaz, & Matteo Rucco. (2017).
    Persistent Entropy for Separating Topological Features from Noise in Vietoris-Rips Complexes.

    Vectorized form of the original sequential rule. The rule repeatedly cuts the barcode
    (prefix[:m], small, large) at some i < m and restarts with m = i. Here the cut point is
    computed for every m at once as an (n, n) table, and the chain of restarts is followed
    with log n pointer-jumping steps, so the whole selection is a fixed number of large
    operations on the device.

    Args:
    desc: torch.tensor, 1-dimensional tensor of n > 3 bar lengths sorted in descending order

    Returns k: the k + 1 longest bars are prominient
    '''
    n = desc.shape[0]
    large, small = desc[0], desc[-1]
    prefix = desc[1:-1]
    P = prefix.shape[0]                                   # n - 2
    k = torch.arange(P, device=desc.device)
    m = torch.arange(P + 1, device=desc.device)           # row m: the barcode (prefix[:m], small, large)
    xlogx = lambda x: x * torch.log(x)

    valid = k[None, :] < m[:, None]
    x = torch.where(valid, prefix[None, :], torch.zeros_like(prefix)[None, :])
    xl = torch.where(valid, xlogx(prefix)[None, :], torch.zeros_like(prefix)[None, :])
    # sums over tails (prefix[i:m], small, large) for every m and i
    T = torch.flip(torch.cumsum(torch.flip(x, [1]), 1), [1]) + small + large
    E = torch.flip(torch.cumsum(torch.flip(xl, [1]), 1), [1]) + xlogx(small) + xlogx(large)
    # S_i = T_i + i * T_i / exp(H_i), and T_i / exp(H_i) = exp(E_i / T_i)
    S = T + k[None, :] * torch.exp(E / T)
    stop = (S[:, :-1] < S[:, 1:]) & (k[None, 1:] <= m[:, None] - 1)
    cut = torch.where(stop.any(dim=1), torch.argmax(stop.to(torch.int8), dim=1) + 1, m - 1).clamp(min=0)

    alpha = small / large
    Q = torch.ceil(alpha * (m + 2) * (alpha - 1 - torch.log(alpha)) / (alpha - 1) ** 2)
    restart = Q < cut                                     # False for nan (alpha = 1)
    nxt = torch.where(restart, cut, m)
    for _ in range(max(1, math.ceil(math.log2(P + 1)))):
        nxt = nxt[nxt]
    return int(cut[nxt[P]])


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


def _mst_prim(D: torch.Tensor):
    '''Prim's algorithm: n - 1 sequential steps of O(n) work. Returns (child, parent) index tensors.'''
    n = D.shape[0]
    in_tree = torch.zeros(n, dtype=torch.bool, device=D.device)
    in_tree[0] = True
    best = D[0].clone()
    parent = torch.zeros(n, dtype=torch.long, device=D.device)
    child = torch.empty(n - 1, dtype=torch.long, device=D.device)
    par = torch.empty(n - 1, dtype=torch.long, device=D.device)
    inf = torch.tensor(float("inf"), dtype=D.dtype, device=D.device)
    for t in range(n - 1):
        j = torch.argmin(torch.where(in_tree, inf, best))
        child[t], par[t] = j, parent[j]
        in_tree[j] = True
        upd = D[j] < best
        best = torch.where(upd, D[j], best)
        parent = torch.where(upd, j, parent)
    return child, par


def _mst_minimax(D: torch.Tensor, max_elements: int = 2 ** 26):
    '''
    MST through the minimax (bottleneck) path matrix. An edge belongs to the MST iff its weight
    equals the minimax distance between its ends. The minimax matrix is the closure of the weight
    matrix under the (min, max) matrix product, obtained with ceil(log2(n - 1)) squarings, each
    of which is one large parallel operation. Weights are replaced by their ranks, so ties are
    broken consistently and exactly n - 1 edges are returned.

    D: (n, n) or a batch (B, n, n) of symmetric weight matrices.
    Returns (child, parent) index tensors of shape (n - 1,) or (B, n - 1).
    '''
    batched = D.dim() == 3
    D = D if batched else D[None]
    B, n = D.shape[0], D.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=D.device)
    order = torch.argsort(D[:, iu[0], iu[1]], dim=1, stable=True)
    rank = torch.empty_like(order, dtype=torch.int32)
    rank.scatter_(1, order, torch.arange(1, order.shape[1] + 1, dtype=torch.int32, device=D.device).expand(B, -1))
    R = torch.zeros((B, n, n), dtype=torch.int32, device=D.device)
    R[:, iu[0], iu[1]] = rank
    R = R + R.transpose(1, 2)
    M = R
    rows = max(1, max_elements // (B * n * n))
    for _ in range(max(1, math.ceil(math.log2(max(n - 1, 2))))):
        if rows >= n:
            M = torch.maximum(M[:, :, :, None], M[:, None, :, :]).amin(dim=2)
        else:
            M = torch.cat([torch.maximum(M[:, r:r + rows, :, None], M[:, None, :, :]).amin(dim=2)
                           for r in range(0, n, rows)], dim=1)
    edges = torch.nonzero(torch.triu(R == M, diagonal=1))        # sorted by batch, n - 1 rows each
    a, b = edges[:, 1].reshape(B, n - 1), edges[:, 2].reshape(B, n - 1)
    return (a, b) if batched else (a[0], b[0])


def mst_edges(D: torch.Tensor, method: str = "auto"):
    '''
    Edges of the minimum spanning tree of a complete weighted graph, computed on the device of `D`.

    D: (n, n) or a batch (B, n, n) of symmetric weight matrices (gradients are not needed here)
    method: "minimax", "prim" or "auto", see `mst_pairs`

    Returns a long tensor of shape (n - 1, 2) or (B, n - 1, 2)
    '''
    D = D.detach()
    n = D.shape[-1]
    if method == "auto":
        limit = MINIMAX_MAX_POINTS["gpu" if D.device.type == "cuda" else "cpu"]
        method = "minimax" if n <= limit else "prim"
    if method == "minimax":
        a, b = _mst_minimax(D)
        return torch.stack((a, b), dim=-1)

    def prim(d):
        # many tiny sequential steps are slow on accelerators: run them on CPU, only indices travel back
        return torch.stack([i.to(d.device) for i in _mst_prim(d.cpu())], dim=-1)

    return prim(D) if D.dim() == 2 else torch.stack([prim(d) for d in D])


def mst_pairs(points: torch.Tensor, method: str = "auto"):
    '''
    Zero-dimensional persistence pairs of the Vietoris-Rips filtration, i.e. the edges of the
    Euclidean minimum spanning tree, computed on the device of `points`
    (runs on GPU for CUDA / MPS tensors).

    method: "minimax" (log n large parallel steps, O(n^3) work), "prim" (n small sequential
    steps, O(n^2) work) or "auto" (minimax up to MINIMAX_MAX_POINTS points for the device, prim above)

    Returns ind0: torch.tensor of shape (n - 1, 3) in Gudhi's layout (vertex, edge end, edge end)
    '''
    with torch.no_grad():
        D = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
    e = mst_edges(D, method=method)
    return torch.stack((e[:, 0], e[:, 1], e[:, 0]), dim=1)


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
