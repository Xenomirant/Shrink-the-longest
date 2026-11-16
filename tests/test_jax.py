import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from topoentropy import jax_loss as J  # noqa: E402
from _reference import H0_CONFIGS, barcodes, cloud, ref_count, ref_h0_loss  # noqa: E402


def test_vectorized_selection_matches_original():
    for l in barcodes():
        pad = np.concatenate((l, np.zeros(5)))
        valid = np.arange(len(pad)) < len(l)
        prom, clus, k = J.prominient_masks(jnp.asarray(pad), jnp.asarray(valid))
        r = ref_count(l)
        assert int(k) == r
        assert int(prom.sum()) == r + 1 and int(clus.sum()) == len(l) - r - 1
        assert pad[np.asarray(prom)].min() >= pad[np.asarray(clus)].max(initial=0)


@pytest.mark.parametrize("kw", H0_CONFIGS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_h0_matches_reference(kw, seed):
    x = cloud(seed)
    ref, kr = ref_h0_loss(x, **kw)
    (b, kb), gb = jax.value_and_grad(lambda p: J.entropy_loss(p, max_dim=1, **kw), has_aux=True)(jnp.asarray(x))
    assert int(kb) == kr and np.isfinite(np.asarray(gb)).all()
    np.testing.assert_allclose(float(b), ref, rtol=1e-9)


@pytest.mark.parametrize("n", [4, 5, 17, 60, 200])
def test_minimax_mst_matches_prim(n):
    x = cloud(n, n=n)

    def weights(method):
        e = np.asarray(J.mst_pairs(jnp.asarray(x), method=method))
        assert e.shape == (n - 1, 2)
        return np.sort(np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=-1))

    np.testing.assert_array_equal(weights("prim"), weights("minimax"))


@pytest.mark.parametrize("method", ["prim", "minimax"])
def test_methods_give_same_loss(method, monkeypatch):
    x = jnp.asarray(cloud(1))
    ref, _ = ref_h0_loss(np.asarray(x))
    monkeypatch.setattr(J, "MINIMAX_MAX_POINTS", {"cpu": 10 ** 6, "gpu": 10 ** 6} if method == "minimax" else {"cpu": 0, "gpu": 0})
    J.entropy_loss.clear_cache() if hasattr(J.entropy_loss, "clear_cache") else None
    b, _ = J.entropy_loss(x, max_dim=1)
    np.testing.assert_allclose(float(b), ref, rtol=1e-9)


def test_duplicate_points():
    x = cloud(0)
    x[5] = x[7]
    ref, kr = ref_h0_loss(x)
    (b, kb), gb = jax.value_and_grad(lambda p: J.entropy_loss(p, max_dim=1), has_aux=True)(jnp.asarray(x))
    assert int(kb) == kr and np.isfinite(np.asarray(gb)).all()
    np.testing.assert_allclose(float(b), ref, rtol=1e-9)


def test_small_diagram_does_not_crash():
    jl, _ = J.entropy_loss(jnp.asarray(cloud(0, n=12, d=3)), max_dim=3)
    assert np.isfinite(float(jl))


def test_grad_under_jit_is_finite():
    x = jnp.asarray(cloud(3))
    for md in (1, 2):
        g = jax.jit(jax.grad(lambda p: J.entropy_loss(p, max_dim=md)[0]))(x)
        assert np.all(np.isfinite(np.asarray(g)))
