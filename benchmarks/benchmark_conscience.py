#!/usr/bin/env python3
"""20类混合 — 良心机制 (修复数据覆盖bug)"""
import sys, os, time, numpy as np, gzip, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from train_multiscale import ColdEye, load_mnist, low_contrast

def load_fashion_mnist(d="data/fashion"):
    """Fashion-MNIST 用独立目录"""
    base = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    files = {"ti":"train-images-idx3-ubyte.gz","tl":"train-labels-idx1-ubyte.gz",
             "ei":"t10k-images-idx3-ubyte.gz","el":"t10k-labels-idx1-ubyte.gz"}
    os.makedirs(d, exist_ok=True)
    for f in files.values():
        p = os.path.join(d, f)
        if not os.path.exists(p): urllib.request.urlretrieve(base + f, p)
    def li(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32) / 255
    def ll(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return (li(os.path.join(d, files["ti"])), ll(os.path.join(d, files["tl"])),
            li(os.path.join(d, files["ei"])), ll(os.path.join(d, files["el"])))

np.random.seed(42)
Xm_tr, ym_tr, Xm_te, ym_te = load_mnist()
Xf_tr, yf_tr, Xf_te, yf_te = load_fashion_mnist()

# Verify they're different
print(f"MNIST[0] mean={Xm_tr[0].mean():.3f} max={Xm_tr[0].max():.3f}")
print(f"Fashion[0] mean={Xf_tr[0].mean():.3f} max={Xf_tr[0].max():.3f}")
assert abs(Xm_tr[0].mean() - Xf_tr[0].mean()) > 0.01, "DATA STILL IDENTICAL!"

X_tr = np.concatenate([Xm_tr, Xf_tr]); y_tr = np.concatenate([ym_tr, yf_tr + 10])
X_te = np.concatenate([Xm_te, Xf_te]); y_te = np.concatenate([ym_te, yf_te + 10])
p = np.random.permutation(len(X_tr)); X_tr, y_tr = X_tr[p], y_tr[p]
p = np.random.permutation(len(X_te)); X_te, y_te = X_te[p], y_te[p]

print(f"\n20类混合: train {X_tr.shape}, test {X_te.shape}")
print("良心机制 sweep: 200+100=300d, 50K×5ep")

for beta in [0.0, 0.2, 0.5, 1.0]:
    model = ColdEye(eye_specs=[{"type": "global", "n": 200}, {"type": "patch", "ps": 16, "st": 8, "n": 100}])
    model.init_templates(X_tr[:5000])
    t0 = time.time()
    model.train(X_tr, y_tr, epochs=5, n_train=50000, contrast_aug=True, conscience_beta=beta)
    model.build_memory(X_tr, y_tr, size=5000)

    acc = model.evaluate(X_te[:500], y_te[:500])

    # Node specialization
    ga = model._activate_batch(Xm_te[:500])
    fa = model._activate_batch(Xf_te[:500])
    nn = model.eyes[0].n
    m_mnist = ga[:, :nn].mean(0); m_fash = fa[:, :nn].mean(0)
    pref = (m_fash - m_mnist) / (m_fash + m_mnist + 1e-8)
    n_spec = int(((pref < -0.05) | (pref > 0.05)).sum())

    print(f"  β={beta:.1f}: acc={acc:.1%}  spec_nodes={n_spec}/{nn}  pref_range=[{pref.min():.3f},{pref.max():.3f}]  ({time.time()-t0:.0f}s)")

print("\n=== DONE ===")
