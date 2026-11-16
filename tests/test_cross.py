'''PyTorch and JAX implementations agree in value and gradient (needs both installed).'''
import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from topoentropy import jax_loss as J  # noqa: E402
from topoentropy import loss_utils as L  # noqa: E402
from _reference import CONFIGS, cloud  # noqa: E402


@pytest.mark.parametrize("kw", CONFIGS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_jax_matches_torch(kw, seed):
    x = cloud(seed)
    pts = torch.tensor(x, requires_grad=True)
    a, ka = L.entropy_loss(pts, **kw)
    a.backward()
    (b, kb), gb = jax.value_and_grad(lambda p: J.entropy_loss(p, **kw), has_aux=True)(jnp.asarray(x))
    assert ka == int(kb)
    np.testing.assert_allclose(float(a.detach()), float(b), rtol=1e-6)
    np.testing.assert_allclose(pts.grad.numpy(), np.asarray(gb), rtol=1e-5, atol=1e-8)
