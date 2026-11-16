'''Times the full H0 loss + backward: torch on-device, torch through Gudhi, and jitted JAX.'''
import time

import torch

import topoentropy.loss_utils as L

dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
sync = {"cuda": lambda: torch.cuda.synchronize(), "mps": lambda: torch.mps.synchronize(), "cpu": lambda: None}[dev]
try:
    import jax
    import jax.numpy as jnp
    import topoentropy.jax_loss as J
except ImportError:
    jax = None

print("torch", torch.__version__, dev, torch.cuda.get_device_name(0) if dev == "cuda" else "",
      "| jax", (jax.__version__, str(jax.devices()[0])) if jax else None)
print(f"{'n':>5}{'torch device':>14}{'torch gudhi':>13}{'jax jit':>10}   [ms, loss + backward, 768-d]")
for n in (32, 64, 128, 256, 512, 1024):
    p = torch.randn(n, 768, device=dev, requires_grad=True)

    def run(backend, reps):
        def f():
            loss, _ = L.entropy_loss(p, max_dim=1, backend=backend)
            loss.backward()
        for _ in range(3):
            f()
        sync()
        s = time.perf_counter()
        for _ in range(reps):
            f()
        sync()
        return (time.perf_counter() - s) / reps * 1e3

    tj = float("nan")
    if jax:
        x = jnp.asarray(p.detach().cpu().numpy())
        g = jax.jit(jax.grad(lambda q: J.entropy_loss(q, max_dim=1)[0]))
        g(x).block_until_ready()
        s = time.perf_counter()
        for _ in range(10):
            g(x).block_until_ready()
        tj = (time.perf_counter() - s) / 10 * 1e3
    print(f"{n:>5}{run('device', 10):>14.2f}{run('gudhi', 3):>13.1f}{tj:>10.2f}")
