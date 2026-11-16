'''NumPy reference: the original sequential selection rule and the H0 loss through Gudhi.'''
import gudhi as gd
import numpy as np


def cloud(seed, n=60, d=8):
    return np.random.default_rng(seed).normal(size=(n, d))


def ref_entropy(lens):
    p = lens / lens.sum()
    return -(p * np.log(p)).sum()


def ref_recursive(new_lens, n_bar, alpha):
    S = new_lens.sum()
    for i in range(1, n_bar - 2):
        Q = np.ceil(alpha * n_bar * (alpha - 1 - np.log(alpha)) / (alpha - 1) ** 2)
        S, S_prev = new_lens[i:].sum() + i * new_lens[i:].sum() / np.exp(ref_entropy(new_lens[i:])), S
        if S_prev / S < 1:
            break
    if Q < i:
        return ref_recursive(np.concatenate((new_lens[:i], new_lens[-2:])), i + 2, alpha)
    return i


def ref_count(lens):
    desc = np.sort(lens)[::-1]
    new_lens = np.concatenate((desc[1:], desc[:1]))
    return int(ref_recursive(new_lens, len(lens), new_lens[-2] / new_lens[-1]))


def ref_h0_loss(x, use_prominient_features=True, use_in_cluster_l2=True, l2_lambda=0.5):
    st = gd.RipsComplex(points=x).create_simplex_tree(max_dimension=1)
    st.compute_persistence()
    e = st.flag_persistence_generators()[0][:, 1:3]
    desc = np.sort(np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=-1))[::-1]
    if not use_prominient_features:
        return ref_entropy(desc), len(desc)
    if len(desc) <= 3:
        return ref_entropy(desc), len(desc)
    k = ref_count(desc)
    loss = ref_entropy(desc[:k + 1])
    if use_in_cluster_l2 and len(desc) > k + 1:
        loss += l2_lambda * desc[k + 1:].mean()
    return loss, k


def barcodes():
    rng = np.random.default_rng(0)
    out = []
    for n in [4, 5, 6, 8, 13, 32, 63, 127, 300]:
        for _ in range(40):
            kind = rng.integers(4)
            if kind == 0:
                l = rng.uniform(0.1, 1.0, n)
            elif kind == 1:
                l = rng.pareto(1.5, n) + 0.01
            elif kind == 2:  # a few long bars over noise
                l = np.concatenate((rng.uniform(5, 10, max(1, n // 8)), rng.uniform(0.01, 0.3, n - max(1, n // 8))))
            else:
                l = np.abs(rng.normal(1.0, 0.05, n)) + 1e-3
            out.append(l)
    return out


H0_CONFIGS = [dict(), dict(use_prominient_features=False), dict(use_in_cluster_l2=False)]
CONFIGS = [
    dict(max_dim=1),
    dict(max_dim=2),
    dict(max_dim=2, use_prominient_features=False),
    dict(max_dim=2, use_in_cluster_l2=False),
    dict(max_dim=3),
]

