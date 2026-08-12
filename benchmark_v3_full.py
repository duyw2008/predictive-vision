#!/usr/bin/env python3
"""
v3 架构全配验证: global(28×28) + coarse(16×16) → KNN
三组对比:
  A: 10K, no aug, 3ep
  B: 10K, +contrast aug, 3ep
  C: 60K, +contrast aug, 5ep
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

def train_global_eye(images, n_nodes=100, epochs=3, n_train=10000, contrast_aug=False):
    flat_dim = 28*28
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
            img = images[idx].copy()
            if contrast_aug and np.random.random() < 0.5:
                m = img.mean(); c = 0.3 + np.random.random() * 0.7
                img = m + (img - m) * c
            img_flat = img.reshape(-1).astype(np.float32)
            img_flat /= np.linalg.norm(img_flat) + 1e-8
            sims = templates @ img_flat
            best = int(np.argmax(sims))
            templates[best] += lr * (img_flat - templates[best])
            templates[best] /= np.linalg.norm(templates[best]) + 1e-8
    return templates, nids

def train_coarse_eye(images, ps=16, st=8, n_nodes=50, epochs=3, n_train=10000, contrast_aug=False):
    ts = ps * ps
    g = VisionGraph(n_nodes=n_nodes, template_size=ts)
    v = VisionInterface(g, patch_size=ps, stride=st)
    nids = sorted(g.nodes.keys())

    idxs = np.random.choice(len(images), min(200, len(images)), replace=False)
    ap = [p.astype(np.float32) for i in idxs for p in v.extractor.extract(images[i])]
    for k, nid in enumerate(nids):
        p = ap[k % len(ap)]; p /= np.linalg.norm(p) + 1e-8
        g.nodes[nid].template = p

    lr = 0.1
    for ep in range(epochs):
        for idx in np.random.permutation(min(n_train, len(images))):
            img = images[idx].copy()
            if contrast_aug and np.random.random() < 0.5:
                m = img.mean(); c = 0.3 + np.random.random() * 0.7
                img = m + (img - m) * c
            v.set_image(img)
            for nid, aps in v.node_assignments.items():
                node = g.nodes[nid]
                t = np.mean(aps, axis=0)
                n = np.linalg.norm(t)
                if n > 0: t /= n
                node.template += lr*(t-node.template)
                node.template /= np.linalg.norm(node.template)+1e-8
    return g, v, nids

def global_acts(templates, images):
    N = len(images)
    flat = images.reshape(N, -1).astype(np.float32)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    return np.clip(flat @ templates.T, 0, 1).astype(np.float32)

def coarse_acts(g, v, nids, images):
    feats = np.zeros((len(images), len(nids)), dtype=np.float32)
    for i, img in enumerate(images):
        v.set_image(img)
        feats[i] = np.array([g.nodes[nid].activation for nid in nids], dtype=np.float32)
    return feats

def features(glob_t, g, v, nids, images):
    return np.concatenate([global_acts(glob_t, images), coarse_acts(g, v, nids, images)], axis=1)


np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
n_tr_knn, n_te = 5000, 500

configs = [
    ("A: 10K, no aug, 3ep",  10000, 3, False),
    ("B: 10K, +aug, 3ep",    10000, 3, True),
    ("C: 60K, +aug, 5ep",    60000, 5, True),
]

results = {}
for name, n_tr, n_ep, aug in configs:
    t0 = time.time()
    glob_t, _ = train_global_eye(X_tr, n_nodes=100, epochs=n_ep, n_train=n_tr, contrast_aug=aug)
    g, v, nids = train_coarse_eye(X_tr, n_nodes=50, epochs=n_ep, n_train=n_tr, contrast_aug=aug)

    # Build KNN reference
    ref_imgs = np.array([X_tr[i] for i in range(n_tr_knn)])
    Xref = features(glob_t, g, v, nids, ref_imgs)
    knn = KNN(k=5); knn.fit(Xref, y_tr[:n_tr_knn])

    accs = {}
    for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07]:
        if c == 1.0:
            test_batch = X_te[:n_te]
        else:
            test_batch = low_contrast(X_te[:n_te], c)
        Xte = features(glob_t, g, v, nids, test_batch)
        accs[c] = knn.score(Xte, y_te[:n_te])

    results[name] = accs
    print(f"  ✓ {name}: {time.time()-t0:.0f}s — c1.0={accs[1.0]*100:.1f}% c0.2={accs[0.2]*100:.1f}% c0.1={accs[0.1]*100:.1f}%")

print(f"\n{'='*70}")
print(f"  v3: global(28×28,100n) + coarse(16×16,50n) → 150d KNN")
print(f"{'='*70}")
print(f"  {'c':>6s}  {'A(10K,no aug)':>13s}  {'B(10K,+aug)':>12s}  {'C(60K,+aug)':>12s}")
print(f"  {'-'*50}")

prev_v1 = {1.0:77.0, 0.5:75.0, 0.3:67.0, 0.2:53.4, 0.15:36.0, 0.1:10.6, 0.07:5.4}
for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07]:
    a = results["A: 10K, no aug, 3ep"][c]
    b = results["B: 10K, +aug, 3ep"][c]
    cc = results["C: 60K, +aug, 5ep"][c]
    vs_v1 = cc - prev_v1[c]/100
    print(f"  {c:5.2f}  {a*100:12.1f}%  {b*100:11.1f}%  {cc*100:11.1f}%  (vs v1: {vs_v1*100:+.1f}%)")

print("\n=== DONE ===")
