'''Times loss + backward for the H0 path with the Prim and minimax MST. Usage: python benchmarks/bench_mst.py'''
import sys
import time

import torch

import topoentropy.loss_utils as L

SIZES = (32, 64, 128, 256, 512)
DIM = 768


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def timeit(f, dev, reps):
    for _ in range(3):
        f()
    sync(dev)
    s = time.perf_counter()
    for _ in range(reps):
        f()
    sync(dev)
    return (time.perf_counter() - s) / reps * 1e3


def torch_bench():
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if torch.backends.mps.is_available() else [])
    print(f"torch {torch.__version__}", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    print(f"{'device':8}{'n':>5}{'prim ms':>10}{'minimax ms':>12}{'speedup':>9}")
    for dev in devs:
        for n in SIZES:
            p = torch.randn(n, DIM, device=dev, requires_grad=True)
            res = {}
            for method in ("prim", "minimax"):
                def f():
                    e = L.mst_pairs(p, method=method)
                    lens = (p[e[:, 1]] - p[e[:, 2]]).norm(dim=-1)
                    l, _ = L.prominient_features(lens)[0], None
                    L.barcode_entropy(l).backward()
                res[method] = timeit(f, dev, 20 if n <= 128 else 5)
            print(f"{dev:8}{n:>5}{res['prim']:>10.2f}{res['minimax']:>12.2f}{res['prim'] / res['minimax']:>8.1f}x")


def jax_bench():
    try:
        import jax
        import jax.numpy as jnp
        import topoentropy.jax_loss as J
    except ImportError:
        return
    print(f"\njax {jax.__version__} on {jax.devices()[0]}")
    print(f"{'n':>5}{'prim ms':>10}{'minimax ms':>12}{'speedup':>9}")
    for n in SIZES:
        x = jax.random.normal(jax.random.PRNGKey(0), (n, DIM))
        res = {}
        for method in ("prim", "minimax"):
            def loss(p):
                e = J.mst_pairs(p, method=method)
                lens = jnp.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=-1)
                q = lens / lens.sum()
                return -(q * jnp.log(q)).sum()
            g = jax.jit(jax.grad(loss))
            g(x).block_until_ready()
            s = time.perf_counter()
            reps = 20 if n <= 128 else 5
            for _ in range(reps):
                g(x).block_until_ready()
            res[method] = (time.perf_counter() - s) / reps * 1e3
        print(f"{n:>5}{res['prim']:>10.2f}{res['minimax']:>12.2f}{res['prim'] / res['minimax']:>8.1f}x")


if __name__ == "__main__":
    torch_bench()
    if "--no-jax" not in sys.argv:
        jax_bench()
