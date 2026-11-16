'''
JAX implementation of the persistent entropy loss.

For max_dim = 1 (H0) everything runs on the accelerator: the persistence pairs are the edges
of the Euclidean minimum spanning tree (Prim's algorithm in `lax.fori_loop`) and the selection
of prominient features is a vectorized `lax.while_loop`.

For max_dim > 1 persistence pairs and the selection do not need gradients, so they are
computed on the host (Gudhi + NumPy) through `jax.pure_callback`. The callback returns
fixed-size, padded index arrays with masks. In both cases the loss is differentiable
w.r.t. the points and works under `jax.jit` and `jax.grad`.
'''

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _require_gudhi():
    try:
        import gudhi  # noqa: F401
    except ImportError as e:
        raise ImportError("max_dim > 1 requires Gudhi: pip install 'topoentropy[higher]'") from e


def _entropy_np(lens):
    p = lens / np.sum(lens)
    return -np.sum(p * np.log(p))


def _prominient_count_recursive(new_lens, n_bar, alpha):
    S = np.sum(new_lens)
    for i in range(1, n_bar - 2):
        Q = np.ceil(alpha * n_bar * (alpha - 1 - np.log(alpha)) / (alpha - 1) ** 2)
        tail = new_lens[i:]
        S, S_prev = np.sum(tail) + i * np.sum(tail) / np.exp(_entropy_np(tail)), S
        if S_prev / S < 1:
            break
    if Q < i:
        new_lens = np.concatenate((new_lens[:i], new_lens[-2:]))
        return _prominient_count_recursive(new_lens, i + 2, alpha)
    return i


def prominient_count(lens):
    '''
    NumPy port of `loss_utils.prominient_features`.
    Returns the number k reported by the torch version; the prominient bars are
    the k + 1 longest ones (all bars if there are at most 3).
    '''
    n = lens.shape[0]
    if n <= 3:
        return n
    desc = np.sort(lens)[::-1]
    new_lens = np.concatenate((desc[1:], desc[:1]))
    alpha = new_lens[-2] / new_lens[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return int(_prominient_count_recursive(new_lens, n, alpha))


def _masks(lens, use_prominient_features):
    '''Masks (prominient, in-cluster) over the given bars and the reported count.'''
    n = lens.shape[0]
    if not use_prominient_features:
        return np.ones(n, bool), np.zeros(n, bool), n
    k = prominient_count(lens)
    n_prom = n if n <= 3 else k + 1
    order = np.argsort(-lens, kind="stable")
    prom = np.zeros(n, bool)
    prom[order[:n_prom]] = True
    return prom, ~prom, k


def _host_pairs(points, max_dim, max_pairs, use_prominient_features):
    import gudhi as gd
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    st = gd.RipsComplex(points=points).create_simplex_tree(max_dimension=max_dim)
    st.compute_persistence()
    gens0, gens_hi = st.flag_persistence_generators()[:2]

    out = []
    count = 0

    # H0: bar length is the length of the edge that kills the component
    idx0 = np.zeros((n - 1, 2), np.int32)
    prom0 = np.zeros(n - 1, bool)
    clus0 = np.zeros(n - 1, bool)
    m = gens0.shape[0]
    if m > 0:
        e = gens0[:, 1:3]
        lens = np.linalg.norm(points[e[:, 0]] - points[e[:, 1]], axis=-1)
        p, c, k = _masks(lens, use_prominient_features)
        idx0[:m], prom0[:m], clus0[:m] = e, p, c
        count += k
    out += [idx0, prom0, clus0]

    # H1, H2, ...: death edge length minus birth edge length
    for d in range(max_dim - 1):
        idx = np.zeros((max_pairs, 4), np.int32)
        prom = np.zeros(max_pairs, bool)
        clus = np.zeros(max_pairs, bool)
        g = gens_hi[d] if d < len(gens_hi) else np.zeros((0, 4), np.int64)
        m = g.shape[0]
        if m > max_pairs:
            raise ValueError(f"H{d + 1} has {m} pairs, more than max_pairs={max_pairs}. Increase max_pairs.")
        if m > 0:
            birth = np.linalg.norm(points[g[:, 0]] - points[g[:, 1]], axis=-1)
            death = np.linalg.norm(points[g[:, 2]] - points[g[:, 3]], axis=-1)
            p, c, k = _masks(death - birth, use_prominient_features)
            idx[:m], prom[:m], clus[:m] = g, p, c
            count += k
        out += [idx, prom, clus]

    return tuple(out) + (np.int32(count),)


# largest point cloud for which the minimax MST is used by default (measured crossover with Prim,
# the compiled Prim loop is faster on CPU at every size)
MINIMAX_MAX_POINTS = {"cpu": 0, "gpu": 512}


def _mst_prim(D):
    '''Prim's algorithm: n - 1 sequential steps of O(n) work.'''
    n = D.shape[0]

    def step(t, state):
        in_tree, best, parent, edges = state
        j = jnp.argmin(jnp.where(in_tree, jnp.inf, best)).astype(jnp.int32)
        edges = edges.at[t].set(jnp.stack((parent[j], j)))
        in_tree = in_tree.at[j].set(True)
        upd = D[j] < best
        return in_tree, jnp.where(upd, D[j], best), jnp.where(upd, j, parent), edges

    init = (jnp.zeros(n, bool).at[0].set(True), D[0], jnp.zeros(n, jnp.int32), jnp.zeros((n - 1, 2), jnp.int32))
    return jax.lax.fori_loop(0, n - 1, step, init)[3]


def _mst_minimax(D):
    '''
    MST through the minimax (bottleneck) path matrix, see `loss_utils._mst_minimax`:
    ceil(log2(n - 1)) squarings in the (min, max) semiring on the ranks of the weights.
    '''
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    order = jnp.argsort(D[iu], stable=True)
    rank = jnp.zeros(order.shape[0], jnp.int32).at[order].set(jnp.arange(1, order.shape[0] + 1, dtype=jnp.int32))
    R = jnp.zeros((n, n), jnp.int32).at[iu].set(rank)
    R = R + R.T
    M = R
    for _ in range(max(1, int(np.ceil(np.log2(max(n - 1, 2)))))):
        M = jnp.min(jnp.maximum(M[:, :, None], M[None, :, :]), axis=1)
    a, b = jnp.nonzero(jnp.triu(R == M, 1), size=n - 1)
    return jnp.stack((a, b), axis=1).astype(jnp.int32)


def mst_pairs(points, method: str = "auto"):
    '''
    Edges (n - 1, 2) of the Euclidean minimum spanning tree, computed on the device.
    method: "minimax" (log n large parallel steps, O(n^3) work), "prim" (n small sequential
    steps, O(n^2) work) or "auto" (minimax up to MINIMAX_MAX_POINTS points for the backend, prim above)
    '''
    x = jax.lax.stop_gradient(points)
    n = x.shape[0]
    if method == "auto":
        limit = MINIMAX_MAX_POINTS["cpu" if jax.default_backend() == "cpu" else "gpu"]
        method = "minimax" if n <= limit else "prim"
    sq = jnp.sum(x * x, axis=-1)
    D = jnp.maximum(sq[:, None] + sq[None, :] - 2.0 * x @ x.T, 0.0)   # squared distances, same ordering
    return _mst_minimax(D) if method == "minimax" else _mst_prim(D)


def _xlogx(x):
    return x * jnp.log(x)


def prominient_masks(lens, valid):
    '''
    Vectorized selection of prominient features (see `loss_utils.prominient_count`) for a
    fixed-size array of bars with a validity mask: an (N, N) table of cut points followed
    by log N pointer-jumping steps.
    Returns (prominient mask, in-cluster mask, reported number of features).
    '''
    N = lens.shape[0]
    v = jnp.sum(valid).astype(jnp.int32)
    order = jnp.argsort(jnp.where(valid, -lens, jnp.inf))
    desc = jnp.where(valid, lens, 0.0)[order]                 # valid bars first, descending
    large = desc[0]
    small = desc[jnp.maximum(v - 1, 0)]
    prefix = desc[1:]                                         # the barcode is (prefix[:m], small, large)
    k = jnp.arange(N - 1, dtype=jnp.int32)
    alpha = small / large
    safe_alpha = jnp.where((alpha > 0) & (alpha < 1), alpha, 0.5)

    P = N - 1
    m = jnp.arange(P + 1, dtype=jnp.int32)                    # row m: the barcode (prefix[:m], small, large)
    ok = k[None, :] < m[:, None]
    x = jnp.where(ok, prefix[None, :], 0.0)
    xl = jnp.where(ok, _xlogx(jnp.where(prefix > 0, prefix, 1.0))[None, :], 0.0)
    T = jnp.cumsum(x[:, ::-1], axis=1)[:, ::-1] + small + large
    E = jnp.cumsum(xl[:, ::-1], axis=1)[:, ::-1] + _xlogx(jnp.where(small > 0, small, 1.0)) \
        + _xlogx(jnp.where(large > 0, large, 1.0))
    S = T + k[None, :] * jnp.exp(E / jnp.where(T > 0, T, 1.0))
    stop = (S[:, :-1] < S[:, 1:]) & (k[None, 1:] <= m[:, None] - 1)
    cut = jnp.maximum(jnp.where(jnp.any(stop, axis=1), jnp.argmax(stop, axis=1) + 1, m - 1), 0).astype(jnp.int32)
    Q = jnp.ceil(safe_alpha * (m + 2) * (safe_alpha - 1 - jnp.log(safe_alpha)) / (safe_alpha - 1) ** 2)
    restart = (Q < cut) & (alpha > 0) & (alpha < 1)
    nxt = jnp.where(restart, cut, m)
    for _ in range(max(1, int(np.ceil(np.log2(P + 1))))):
        nxt = nxt[nxt]
    i = cut[nxt[jnp.clip(v - 2, 0, P)]]

    few = v <= 3
    n_prom = jnp.where(few, v, i + 1)
    rank = jnp.zeros(N, jnp.int32).at[order].set(jnp.arange(N, dtype=jnp.int32))
    prom = valid & (rank < n_prom)
    return prom, valid & ~prom, jnp.where(few, v, i).astype(jnp.int32)


def _dist(points, a, b, valid):
    # padded rows index (0, 0); give them a non-zero difference so the norm has a finite gradient
    diff = jnp.where(valid[:, None], points[a] - points[b], 1.0)
    return jnp.linalg.norm(diff, axis=-1)


def _masked_entropy(lens, mask):
    safe = jnp.where(mask, lens, 1.0)
    total = jnp.sum(jnp.where(mask, lens, 0.0))
    # guard against an empty mask so that gradients stay finite
    p = safe / jnp.where(total > 0, total, 1.0)
    return -jnp.sum(jnp.where(mask, p * jnp.log(p), 0.0))


def _masked_mean(lens, mask):
    cnt = jnp.sum(mask)
    return jnp.where(cnt > 0, jnp.sum(jnp.where(mask, lens, 0.0)) / jnp.maximum(cnt, 1), 0.0)


@partial(jax.jit, static_argnames=("max_dim", "use_prominient_features", "use_in_cluster_l2", "max_pairs"))
def entropy_loss(points, max_dim: int = 1, use_prominient_features: bool = True,
                 use_in_cluster_l2: bool = True, l2_lambda=0.5, max_pairs: int | None = None):
    '''
    JAX counterpart of `topoentropy.entropy_loss` with the same arguments and return values.
    The returned value is meant to be maximized, i.e. subtracted from the training loss.

    Args:
    points: jax.Array, 2-dimensional array, representing the point cloud

    max_dim: int, maximum dimension of simplices used for persistent pairs calculation
    (1 by default: H0 only, computed on the device; larger values require Gudhi)

    use_prominient_features: bool, whether to perform prominient feature selection
    or to maximize entropy of distribution on pointwise distances

    use_in_cluster_l2: bool, whether to add an inner clusterwise l2 distance
    maximization term (mean length of non-prominient bars) to spread features inside clusters

    l2_lambda: float, hyperparameter, representing strength of inner l2 regularization

    max_pairs: int, static size of the padded buffer for each homology dimension above 0
    (defaults to 2 * number of points), an error is raised if it is exceeded

    Returns:
    (loss, number of prominient features)
    '''
    assert points.ndim == 2, "Points must be reshaped to a 2-dimensional array before proceeding"
    n = points.shape[0]
    use_l2 = use_in_cluster_l2 and use_prominient_features

    if max_dim == 1:
        # H0 only: everything stays on the device
        e = mst_pairs(points)
        diff = points[e[:, 0]] - points[e[:, 1]]
        valid = jnp.sum(jax.lax.stop_gradient(diff) ** 2, axis=-1) > 0   # drop zero-length bars as Gudhi does
        lens = jnp.linalg.norm(jnp.where(valid[:, None], diff, 1.0), axis=-1)
        if use_prominient_features and n - 1 > 3:
            prom, clus, count = prominient_masks(jax.lax.stop_gradient(lens), valid)
        else:
            prom, clus, count = valid, jnp.zeros_like(valid), jnp.sum(valid).astype(jnp.int32)
        loss = _masked_entropy(lens, prom)
        if use_l2:
            loss = loss + l2_lambda * _masked_mean(lens, clus)
        return loss, count

    max_pairs = 2 * n if max_pairs is None else max_pairs
    _require_gudhi()

    shapes = [jax.ShapeDtypeStruct((n - 1, 2), jnp.int32),
              jax.ShapeDtypeStruct((n - 1,), jnp.bool_),
              jax.ShapeDtypeStruct((n - 1,), jnp.bool_)]
    for _ in range(max_dim - 1):
        shapes += [jax.ShapeDtypeStruct((max_pairs, 4), jnp.int32),
                   jax.ShapeDtypeStruct((max_pairs,), jnp.bool_),
                   jax.ShapeDtypeStruct((max_pairs,), jnp.bool_)]
    shapes.append(jax.ShapeDtypeStruct((), jnp.int32))

    res = jax.pure_callback(
        partial(_host_pairs, max_dim=max_dim, max_pairs=max_pairs,
                use_prominient_features=use_prominient_features),
        tuple(shapes), jax.lax.stop_gradient(points))
    *groups, count = res

    loss = 0.0
    for d in range(max_dim):
        idx, prom, clus = groups[3 * d: 3 * d + 3]
        valid = prom | clus
        if d == 0:
            lens = _dist(points, idx[:, 0], idx[:, 1], valid)
        else:
            lens = _dist(points, idx[:, 2], idx[:, 3], valid) - _dist(points, idx[:, 0], idx[:, 1], valid)
        loss = loss + _masked_entropy(lens, prom)
        if use_l2:
            loss = loss + l2_lambda * _masked_mean(lens, clus)
    return loss, count
