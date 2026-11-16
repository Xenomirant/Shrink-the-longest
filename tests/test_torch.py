import numpy as np
import pytest

torch = pytest.importorskip("torch")

from topoentropy import loss_utils as L  # noqa: E402
from _reference import H0_CONFIGS, barcodes, cloud, ref_count, ref_h0_loss  # noqa: E402


def torch_loss(x, backend, **kw):
    pts = torch.tensor(x, requires_grad=True)
    loss, k = L.entropy_loss(pts, backend=backend, **kw)
    loss.backward()
    return float(loss.detach()), k, pts.grad.numpy()


def test_lazy_import():
    import topoentropy
    assert topoentropy.entropy_loss is L.entropy_loss


def test_vectorized_selection_matches_original():
    for l in barcodes():
        desc = torch.tensor(np.sort(l)[::-1].copy())
        assert L.prominient_count(desc) == ref_count(l)


@pytest.mark.parametrize("kw", H0_CONFIGS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_h0_matches_reference_and_gudhi_backend(kw, seed):
    x = cloud(seed)
    ref, kr = ref_h0_loss(x, **kw)
    a, ka, ga = torch_loss(x, "gudhi", max_dim=1, **kw)
    b, kb, gb = torch_loss(x, "device", max_dim=1, **kw)
    assert ka == kb == kr
    np.testing.assert_allclose(a, ref, rtol=1e-9)
    np.testing.assert_allclose(b, ref, rtol=1e-9)
    np.testing.assert_allclose(ga, gb, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("n", [4, 5, 17, 60, 200])
def test_minimax_mst_matches_prim(n):
    x = torch.tensor(cloud(n, n=n))

    def weights(method):
        e = L.mst_pairs(x, method=method)
        assert e.shape == (n - 1, 3)
        return torch.sort((x[e[:, 1]] - x[e[:, 2]]).norm(dim=-1)).values

    assert torch.equal(weights("prim"), weights("minimax"))


def test_minimax_mst_chunked():
    D = torch.cdist(*(torch.tensor(cloud(0, n=50)),) * 2)
    a = torch.stack(L._mst_minimax(D), 1)
    b = torch.stack(L._mst_minimax(D, max_elements=50 * 50 * 7), 1)
    assert torch.equal(a, b)


def test_tied_distances():
    # integer grid: many equal distances, any MST has the same multiset of weights
    g = np.stack(np.meshgrid(np.arange(6.0), np.arange(6.0)), -1).reshape(-1, 2)
    ref, _ = ref_h0_loss(g, use_prominient_features=False)
    for method in ("prim", "minimax"):
        pts = torch.tensor(g)
        e = L.mst_pairs(pts, method=method)
        assert e.shape[0] == len(g) - 1
        lens = (pts[e[:, 1]] - pts[e[:, 2]]).norm(dim=-1)
        np.testing.assert_allclose(float(L.barcode_entropy(lens)), ref, rtol=1e-12)


def test_duplicate_points():
    x = cloud(0)
    x[5] = x[7]
    ref, kr = ref_h0_loss(x)
    b, kb, gb = torch_loss(x, "device", max_dim=1)
    assert kb == kr and np.isfinite(gb).all()
    np.testing.assert_allclose(b, ref, rtol=1e-9)


def test_small_diagram_does_not_crash():
    x = cloud(0, n=12, d=3)
    loss, _ = L.entropy_loss(torch.tensor(x, requires_grad=True), max_dim=3)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("dev", ["cuda", "mps"])
def test_accelerator(dev):
    available = torch.cuda.is_available() if dev == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{dev} not available")
    x = cloud(0).astype(np.float32)
    ref, kr = ref_h0_loss(x.astype(np.float64))
    pts = torch.tensor(x, device=dev, requires_grad=True)
    loss, k = L.entropy_loss(pts, max_dim=1)
    loss.backward()
    assert loss.device.type == dev and k == kr
    np.testing.assert_allclose(float(loss.detach()), ref, rtol=1e-4)
    assert torch.isfinite(pts.grad).all()
