#!/usr/bin/env python3
"""
缺口3-B 真正实现: 全局模板替代 fine 4×4
MultiScaleEye v2: global(28×28) + mid(8×8) + coarse(16×16)
对比 baseline MultiScaleEye v1: fine(4×4) + mid(8×8) + coarse(16×16)
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import gzip, urllib.request

def load_mnist(d="data"):
    fs = {"ti":"train-images-idx3-ubyte.gz","tl":"train-labels-idx1-ubyte.gz",
          "ei":"t10k-images-idx3-ubyte.gz","el":"t10k-labels-idx1-ubyte.gz"}
    os.makedirs(d,exist_ok=True)
    url="https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
    for f in fs.values():
        p=os.path.join(d,f)
        if not os.path.exists(p): urllib.request.urlretrieve(url+f,p)
    def li(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255
    def ll(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(),np.uint8,offset=8).astype(np.int64)
    return li(os.path.join(d,fs["ti"])),ll(os.path.join(d,fs["tl"])),li(os.path.join(d,fs["ei"])),ll(os.path.join(d,fs["el"]))

def low_contrast(X, c):
    Xc = X.copy()
    m = Xc.mean(axis=(1,2), keepdims=True)
    return m + (Xc - m) * c

from graph import VisionGraph
from vision import VisionInterface

class KNN:
    def __init__(s, k=5): s.k = k
    def fit(s, X, y): s.X, s.y = X, y
    def score(s, X, y):
        c = 0
        for i in range(len(X)):
            d = np.sum((s.X-X[i])**2, axis=1)
            nn = np.argpartition(d, s.k)[:s.k]
            if np.bincount(s.y[nn].astype(int)).argmax() == y[i]: c += 1
        return c/len(X)

def train_eye(images, ps, st, n_nodes, epochs=3, n_train=10000):
    ts = ps * ps
    g = VisionGraph(n_nodes=n_nodes, template_size=ts)
    v = VisionInterface(g, patch_size=ps, stride=st)
    nids = sorted(g.nodes.keys())

    idxs = np.random.choice(len(images), min(200, len(images)), replace=False)
    ap = [p.astype(np.float32) for i in idxs for p in v.extractor.extract(images[i])]
    for k, nid in enumerate(nids):
        p = ap[k % len(ap)]; p /= np.linalg.norm(p)+1e-8
        g.nodes[nid].template = p

    lr = 0.1
    for ep in range(epochs):
        for idx in np.random.permutation(min(n_train, len(images))):
            v.set_image(images[idx])
            for nid, aps in v.node_assignments.items():
                node = g.nodes[nid]
                t = np.mean(aps, axis=0)
                n = np.linalg.norm(t)
                if n > 0: t /= n
                node.template += lr*(t-node.template)
                node.template /= np.linalg.norm(node.template)+1e-8
    return g, v, nids

def train_global_eye(images, n_nodes=100, epochs=3, n_train=10000):
    """全局模板: 28×28 整图做 competitive routing
    不用 patch extraction — 直接把整个 flatten 的图当"一个大 patch"
    手动实现路由 (VisionInterface 最小 patch=4, 不支持 28×28)
    """
    flat_dim = 28*28
    # 初始化模板
    idxs = np.random.choice(len(images), min(200, len(images)), replace=False)
    init_patches = images[idxs].reshape(len(idxs), -1).astype(np.float32)
    nids = list(range(n_nodes))
    templates = np.zeros((n_nodes, flat_dim), dtype=np.float32)
    for i, nid in enumerate(nids):
        templates[nid] = init_patches[i % len(init_patches)]
        templates[nid] /= np.linalg.norm(templates[nid]) + 1e-8

    lr = 0.1
    for ep in range(epochs):
        for idx in np.random.permutation(min(n_train, len(images))):
            img_flat = images[idx].reshape(-1).astype(np.float32)
            img_flat /= np.linalg.norm(img_flat) + 1e-8

            # 竞争路由: 找最匹配的 top-1 模板
            sims = templates @ img_flat  # [n_nodes]
            best = int(np.argmax(sims))

            # Hebbian: 赢者模板向输入靠近
            templates[best] += lr * (img_flat - templates[best])
            templates[best] /= np.linalg.norm(templates[best]) + 1e-8

    return templates, nids

def global_activation(templates, images):
    """计算全局模板的激活值"""
    N = len(images)
    flat = images.reshape(N, -1).astype(np.float32)
    flat_norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
    acts = np.clip(flat_norm @ templates.T, 0, 1)  # [N, n_nodes]
    return acts.astype(np.float32)

def patch_activation(g, v, nids, images):
    feats = np.zeros((len(images), len(nids)), dtype=np.float32)
    for i, img in enumerate(images):
        v.set_image(img)
        feats[i] = np.array([g.nodes[nid].activation for nid in nids], dtype=np.float32)
    return feats

# ── main ──
np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
n_tr, n_tr_knn, n_te = 10000, 2000, 500

# Train all components
print("训练全局眼 (28×28, 100 nodes)...")
t0 = time.time()
glob_templates, glob_nids = train_global_eye(X_tr, n_nodes=100, n_train=n_tr)
print(f"  {time.time()-t0:.0f}s")

print("训练中眼 (8×8, 100 nodes)...")
mid_g, mid_v, mid_nids = train_eye(X_tr, 8, 4, 100, n_train=n_tr)

print("训练粗眼 (16×16, 50 nodes)...")
coarse_g, coarse_v, coarse_nids = train_eye(X_tr, 16, 8, 50, n_train=n_tr)

# Fine 4×4 for baseline comparison
print("训练细眼 (4×4, 100 nodes) — baseline...")
fine_g, fine_v, fine_nids = train_eye(X_tr, 4, 4, 100, n_train=n_tr)

# Build reference features
def build_features(config_name, eye_specs, ref_imgs):
    """eye_specs: list of (name, feat_fn)"""
    parts = []
    for name, fn in eye_specs:
        parts.append(fn(ref_imgs))
    return np.concatenate(parts, axis=1)

# Feature extractors
fine_fn   = lambda imgs: patch_activation(fine_g, fine_v, fine_nids, imgs)
mid_fn    = lambda imgs: patch_activation(mid_g, mid_v, mid_nids, imgs)
coarse_fn = lambda imgs: patch_activation(coarse_g, coarse_v, coarse_nids, imgs)
global_fn = lambda imgs: global_activation(glob_templates, imgs)

# v1: fine + mid + coarse (baseline)
ref_imgs = np.array([X_tr[i] for i in range(n_tr_knn)])
Xref_v1 = build_features("v1", [("fine", fine_fn), ("mid", mid_fn), ("coarse", coarse_fn)], ref_imgs)
knn_v1 = KNN(k=5); knn_v1.fit(Xref_v1, y_tr[:n_tr_knn])

# v2: global + mid + coarse (fine replaced)
Xref_v2 = build_features("v2", [("global", global_fn), ("mid", mid_fn), ("coarse", coarse_fn)], ref_imgs)
knn_v2 = KNN(k=5); knn_v2.fit(Xref_v2, y_tr[:n_tr_knn])

# v3: global + coarse only (drop mid too — profiling showed mid crashes at c=0.5)
Xref_v3 = build_features("v3", [("global", global_fn), ("coarse", coarse_fn)], ref_imgs)
knn_v3 = KNN(k=5); knn_v3.fit(Xref_v3, y_tr[:n_tr_knn])

print(f"\n{'='*65}")
print(f"  v1: fine(4×4) + mid(8×8) + coarse(16×16)  [250d, baseline]")
print(f"  v2: global(28×28) + mid(8×8) + coarse(16×16)  [250d]")
print(f"  v3: global(28×28) + coarse(16×16)  [150d]")
print(f"{'='*65}")

print(f"\n  {'c':>6s}  {'v1(f+m+c)':>10s}  {'v2(g+m+c)':>10s}  {'v3(g+c)':>8s}")
print(f"  {'-'*42}")
for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07]:
    if c == 1.0:
        test_batch = X_te[:n_te]
    else:
        test_batch = low_contrast(X_te[:n_te], c)

    a1 = knn_v1.score(build_features("v1", [("f", fine_fn), ("m", mid_fn), ("c", coarse_fn)], test_batch), y_te[:n_te])
    a2 = knn_v2.score(build_features("v2", [("g", global_fn), ("m", mid_fn), ("c", coarse_fn)], test_batch), y_te[:n_te])
    a3 = knn_v3.score(build_features("v3", [("g", global_fn), ("c", coarse_fn)], test_batch), y_te[:n_te])

    best = max(a1, a2, a3)
    marker = " ←" if a2 > a1 or a3 > a1 else ""
    print(f"  {c:5.2f}  {a1*100:9.1f}%  {a2*100:9.1f}%  {a3*100:7.1f}%{marker}")

print("\n=== DONE ===")
