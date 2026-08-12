#!/usr/bin/env python3
"""
缺口3诊断: c≤0.1 崩塌 — 三个尺度各自的表现
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

def train_single_scale(images, labels, ps, st, n_nodes, epochs=3, n_train=10000):
    ts = ps * ps
    g = VisionGraph(n_nodes=n_nodes, template_size=ts)
    v = VisionInterface(g, patch_size=ps, stride=st)
    nids = sorted(g.nodes.keys())
    idxs = np.random.choice(len(images), min(200, len(images)), replace=False)
    ap = [p.astype(np.float32) for i in idxs for p in v.extractor.extract(images[i])]
    for k, nid in enumerate(nids):
        p = ap[k % len(ap)]
        g.nodes[nid].template = p
        g.nodes[nid].template /= np.linalg.norm(g.nodes[nid].template) + 1e-8
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

def get_features(g, v, nids, images):
    feats = np.zeros((len(images), len(nids)), dtype=np.float32)
    for i, img in enumerate(images):
        v.set_image(img)
        feats[i] = np.array([g.nodes[nid].activation for nid in nids], dtype=np.float32)
    return feats


np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
n_tr, n_tr_knn, n_te = 10000, 2000, 500

scales = [
    ("fine   4x4",   4, 4, 100),
    ("mid    8x8",   8, 4, 100),
    ("coarse 16x16", 16, 8, 50),
]

models = {}
print(f"三个尺度独立训练 ({n_tr} 样本, 3 epochs)\n")

for name, ps, st, nn in scales:
    t0 = time.time()
    g, v, nids = train_single_scale(X_tr, y_tr, ps, st, nn, n_train=n_tr)
    Xref = get_features(g, v, nids, [X_tr[i] for i in range(n_tr_knn)])
    knn = KNN(k=5); knn.fit(Xref, y_tr[:n_tr_knn])
    models[name] = (g, v, nids, knn)
    print(f"  {name} ({nn}n) — train {time.time()-t0:.0f}s")

# 对比度扫描
print(f"\n{'='*60}")
print(f"  c        fine-4x4   mid-8x8    coarse-16x16    (各尺度的相对保留率)")
print(f"{'='*60}")

acc_table = {}
for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07]:
    if c == 1.0:
        test_batch = X_te[:n_te]
    else:
        test_batch = low_contrast(X_te[:n_te], c)
    row = []
    for name in [s[0] for s in scales]:
        g, v, nids, knn = models[name]
        Xte = get_features(g, v, nids, test_batch)
        acc = knn.score(Xte, y_te[:n_te])
        row.append(acc)
    acc_table[c] = row

# 打印
ref = acc_table[1.0]
print(f"  {'c':>6s}  {'fine':>8s}({'rel':>5s})    {'mid':>8s}({'rel':>5s})    {'coarse':>8s}({'rel':>5s})")
print(f"  {'-'*55}")
for c, row in acc_table.items():
    f_r = row[0] / ref[0]; m_r = row[1] / ref[1]; c_r = row[2] / ref[2]
    print(f"  {c:5.2f}  {row[0]*100:7.1f}%({f_r:4.0%})  {row[1]*100:7.1f}%({m_r:4.0%})  {row[2]*100:7.1f}%({c_r:4.0%})")

# 图像 SNR
print(f"\n{'='*60}")
print(f"  图像像素 SNR (跨 {n_te} 张)")
print(f"  {'c':>6s}  {'mean':>8s}  {'std':>8s}  {'CV(std/mean)':>12s}")
print(f"  {'-'*35}")
for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07]:
    if c == 1.0:
        batch = X_te[:n_te]
    else:
        batch = low_contrast(X_te[:n_te], c)
    m = batch.mean()
    s = batch.std()
    print(f"  {c:5.2f}  {m:7.4f}  {s:7.4f}  {s/m:11.2%}")

print("\n=== DONE ===")
