import time, torch, os
import topoentropy.loss_utils as L
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else ((lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None))
def t(f, reps=30):
    for _ in range(5): f()
    sync(); s = time.perf_counter()
    for _ in range(reps): f()
    sync(); return (time.perf_counter() - s) / reps * 1e3
print(dev, "load", os.getloadavg(), "cpus", os.cpu_count(), "threads", torch.get_num_threads())
print(f"{'n':>5}{'cdist':>8}{'prim':>8}{'minimax':>9}{'select':>8}{'gudhi':>8}{'full/minimax':>14}{'full/gudhi':>12}")
for n in (32, 64, 128, 256, 512):
    p = torch.randn(n, 768, device=dev, requires_grad=True)
    D = torch.cdist(p.detach(), p.detach(), compute_mode="donot_use_mm_for_euclid_dist")
    e = L.mst_pairs(p, "minimax"); lens = (p[e[:, 1]] - p[e[:, 2]]).norm(dim=-1).detach()
    def full(be):
        def f():
            l, _ = L.entropy_loss(p, max_dim=1, backend=be); l.backward()
        return f
    r = [t(lambda: torch.cdist(p.detach(), p.detach(), compute_mode="donot_use_mm_for_euclid_dist")),
         t(lambda: L._mst_prim(D), 5), t(lambda: L._mst_minimax(D)), t(lambda: L.prominient_features(lens)),
         t(lambda: L.get_persistent_pairs(p, 1), 5), t(full("device")), t(full("gudhi"), 5)]
    print(f"{n:>5}" + "".join(f"{v:>{w}.2f}" for v, w in zip(r, (8, 8, 9, 8, 8, 14, 12))))
