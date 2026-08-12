#!/usr/bin/env python3
"""20类混合: MNIST(0-9) + Fashion-MNIST(10-19) — 复杂场景探自组织"""
import sys, os, time, numpy as np, gzip, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from train_multiscale import ColdEye, GlobalEye, PatchEye, low_contrast

def load_fashion(d="data"):
    base = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    files = {"ti":"train-images-idx3-ubyte.gz","tl":"train-labels-idx1-ubyte.gz",
             "ei":"t10k-images-idx3-ubyte.gz","el":"t10k-labels-idx1-ubyte.gz"}
    os.makedirs(d,exist_ok=True)
    for f in files.values():
        p=os.path.join(d,f)
        if not os.path.exists(p): urllib.request.urlretrieve(base+f,p)
    def li(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255
    def ll(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(),np.uint8,offset=8).astype(np.int64)
    return li(os.path.join(d,files["ti"])),ll(os.path.join(d,files["tl"])),li(os.path.join(d,files["ei"])),ll(os.path.join(d,files["el"]))

# Load both
from train_multiscale import load_mnist
np.random.seed(42)

Xm_tr, ym_tr, Xm_te, ym_te = load_mnist()
Xf_tr, yf_tr, Xf_te, yf_te = load_fashion()

# Combine: MNIST labels 0-9, Fashion labels 10-19
X_tr = np.concatenate([Xm_tr, Xf_tr])
y_tr = np.concatenate([ym_tr, yf_tr + 10])
X_te = np.concatenate([Xm_te, Xf_te])
y_te = np.concatenate([ym_te, yf_te + 10])

# Shuffle
p = np.random.permutation(len(X_tr)); X_tr, y_tr = X_tr[p], y_tr[p]
p = np.random.permutation(len(X_te)); X_te, y_te = X_te[p], y_te[p]

print(f"20类混合: train {X_tr.shape}, test {X_te.shape}")
print(f"  0-9=MNIST  10-19=Fashion")

# 大容量: 300d
print("\nColdEye — global(200n) + coarse(100n) = 300d, 5ep")
model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
])
model.init_templates(X_tr[:5000])
model.train(X_tr, y_tr, epochs=5, n_train=50000, contrast_aug=True)
model.build_memory(X_tr, y_tr, size=5000)

n_test = 500
print(f"\n  ── 整体准确率 ──")
for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]:
    test_batch = X_te[:n_test] if c==1.0 else low_contrast(X_te[:n_test], c)
    acc = model.evaluate(test_batch, y_te[:n_test])
    print(f"  c={c:.2f}  {acc*100:.1f}%")

# Per-domain breakdown
print(f"\n  ── 域分解 (c=1.0) ──")
te_batch = X_te[:n_test]
for domain, mask_fn in [("MNIST", lambda l: l<10), ("Fashion", lambda l: l>=10)]:
    idxs = [i for i in range(n_test) if mask_fn(y_te[i])]
    if not idxs: continue
    correct = sum(1 for i in idxs if model.predict(te_batch[i])[0]==y_te[i])
    print(f"  {domain:>8s}: {correct}/{len(idxs)} = {correct/len(idxs):.1%}")

# Node specialization analysis
print(f"\n  ── 节点类型专精 ──")
# Activate on first 500 MNIST + 500 Fashion test images
mnist_acts = model._activate_batch(Xm_te[:500])
fash_acts = model._activate_batch(Xf_te[:500])

# For each node, compute mean activation on MNIST vs Fashion
for eye_idx, eye_name in enumerate(["global","coarse"]):
    eye = model.eyes[eye_idx]
    n_nodes = eye.n if isinstance(eye, GlobalEye) else len(eye.nids)
    offset = sum(e.n if isinstance(e,GlobalEye) else len(e.nids) for e in model.eyes[:eye_idx])

    mnist_mean = mnist_acts[:, offset:offset+n_nodes].mean(axis=0)
    fash_mean = fash_acts[:, offset:offset+n_nodes].mean(axis=0)

    # Node preference: >0 = prefers Fashion, <0 = prefers MNIST
    pref = (fash_mean - mnist_mean) / (fash_mean + mnist_mean + 1e-8)
    n_digit = (pref < -0.1).sum()
    n_fashion = (pref > 0.1).sum()
    n_shared = n_nodes - n_digit - n_fashion
    print(f"  {eye_name:>8s} ({n_nodes}n): digit={n_digit}  fashion={n_fashion}  shared={n_shared}")
    # Top-3 most specialized per domain
    top_digit = np.argsort(pref)[:3]
    top_fash = np.argsort(pref)[-3:][::-1]
    print(f"    top-digit nodes: {list(top_digit)}")
    print(f"    top-fashion nodes: {list(top_fash)}")

print(f"\n  随机基线 (20类) = 5%")
print("=== DONE ===")
