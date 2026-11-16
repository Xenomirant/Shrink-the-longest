import numpy as np
import pytest
import torch

from topoentropy import loss_utils as L

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from topoentropy import jax_loss as J
except ImportError:
    jax = None

needs_jax = pytest.mark.skipif(jax is None, reason="jax not installed")

CONFIGS = [
    dict(max_dim=1),
    dict(max_dim=2),
    dict(max_dim=2, use_prominient_features=False),
    dict(max_dim=2, use_in_cluster_l2=False),
    dict(max_dim=3),
]


def cloud(seed, n=60, d=8):
    return np.random.default_rng(seed).normal(size=(n, d))


def torch_loss(x, backend, **kw):
    pts = torch.tensor(x, requires_grad=True)
    loss, k = L.entropy_loss(pts, backend=backend, **kw)
    loss.backward()
    return float(loss.detach()), k, pts.grad.numpy()


# ---- reference: the original sequential selection rule, ported to NumPy ----
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


def test_vectorized_selection_matches_original_torch():
    for l in barcodes():
        desc = torch.tensor(np.sort(l)[::-1].copy())
        assert L.prominient_count(desc) == ref_count(l)


@needs_jax
def test_vectorized_selection_matches_original_jax():
    for l in barcodes():
        pad = np.concatenate((l, np.zeros(5)))
        valid = np.arange(len(pad)) < len(l)
        prom, clus, k = J.prominient_masks(jnp.asarray(pad), jnp.asarray(valid))
        r = ref_count(l)
        assert int(k) == r
        assert int(prom.sum()) == r + 1 and int(clus.sum()) == len(l) - r - 1
        assert np.asarray(pad)[np.asarray(prom)].min() >= np.asarray(pad)[np.asarray(clus)].max(initial=0)


@pytest.mark.parametrize("kw", [dict(), dict(use_prominient_features=False), dict(use_in_cluster_l2=False)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_device_mst_matches_gudhi(kw, seed):
    x = cloud(seed)
    a, ka, ga = torch_loss(x, "gudhi", max_dim=1, **kw)
    b, kb, gb = torch_loss(x, "device", max_dim=1, **kw)
    assert ka == kb
    np.testing.assert_allclose(a, b, rtol=1e-9)
    np.testing.assert_allclose(ga, gb, rtol=1e-7, atol=1e-10)


def test_duplicate_points():
    x = cloud(0)
    x[5] = x[7]
    a, ka, _ = torch_loss(x, "gudhi", max_dim=1)
    b, kb, gb = torch_loss(x, "device", max_dim=1)
    assert ka == kb and np.isfinite(gb).all()
    np.testing.assert_allclose(a, b, rtol=1e-9)


@pytest.mark.parametrize("dev", ["cuda", "mps"])
def test_accelerator(dev):
    if not getattr(torch, dev if dev == "cuda" else "backends").__dict__.get("is_available", lambda: False)() \
            and not (dev == "mps" and torch.backends.mps.is_available()):
        pytest.skip(f"{dev} not available")
    x = cloud(0).astype(np.float32)
    ref, kr, gr = torch_loss(x, "gudhi", max_dim=1)
    pts = torch.tensor(x, device=dev, requires_grad=True)
    loss, k = L.entropy_loss(pts, max_dim=1)
    loss.backward()
    assert loss.device.type == dev and k == kr
    np.testing.assert_allclose(float(loss.detach()), ref, rtol=1e-4)
    np.testing.assert_allclose(pts.grad.cpu().numpy(), gr, rtol=1e-2, atol=1e-5)


@needs_jax
@pytest.mark.parametrize("kw", CONFIGS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_jax_matches_torch(kw, seed):
    x = cloud(seed)
    a, ka, ga = torch_loss(x, "gudhi", **kw)
    (b, kb), gb = jax.value_and_grad(lambda p: J.entropy_loss(p, **kw), has_aux=True)(jnp.asarray(x))
    assert ka == int(kb)
    np.testing.assert_allclose(a, float(b), rtol=1e-6)
    np.testing.assert_allclose(ga, np.asarray(gb), rtol=1e-5, atol=1e-8)


def test_small_diagram_does_not_crash():
    x = cloud(0, n=12, d=3)
    loss, _ = L.entropy_loss(torch.tensor(x, requires_grad=True), max_dim=3)
    assert torch.isfinite(loss)


@needs_jax
def test_jax_small_diagram_does_not_crash():
    jl, _ = J.entropy_loss(jnp.asarray(cloud(0, n=12, d=3)), max_dim=3)
    assert np.isfinite(float(jl))


@needs_jax
def test_jax_duplicate_points():
    x = cloud(0)
    x[5] = x[7]
    a, ka, ga = torch_loss(x, "gudhi", max_dim=1)
    (b, kb), gb = jax.value_and_grad(lambda p: J.entropy_loss(p, max_dim=1), has_aux=True)(jnp.asarray(x))
    assert ka == int(kb) and np.isfinite(np.asarray(gb)).all()
    np.testing.assert_allclose(a, float(b), rtol=1e-9)


@needs_jax
def test_jax_grad_under_jit_is_finite():
    x = jnp.asarray(cloud(3))
    for md in (1, 2):
        g = jax.jit(jax.grad(lambda p: J.entropy_loss(p, max_dim=md)[0]))(x)
        assert np.all(np.isfinite(np.asarray(g)))
