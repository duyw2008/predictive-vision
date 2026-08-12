#!/usr/bin/env python3
"""
验证: per-patch centering 是否修复 c≤0.1 崩塌
只测 GlobalEye — 速度最快
"""
import sys, os, time, numpy as np, gzip, urllib.request
sys.path.insert(0, os.path.dirname(__file__))

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
    Xc = X.copy(); m = Xc.mean(axis=(1,2), keepdims=True)
    return m + (Xc - m) * c

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

def train_global(images, n_nodes=100, epochs=3, n_train=10000, contrast_aug=True, center=True):
    """center=True: 模板和输入都 per-sample 减均值，L2 归一化"""
    rng = np.random.RandomState(42)
    idxs = rng.choice(len(images), min(200, len(images)), replace=False)
    init = images[idxs].reshape(len(idxs), -1).astype(np.float32)
    if center: init = init - init.mean(axis=1, keepdims=True)
    templates = np.zeros((n_nodes, 784), dtype=np.float32)
    for i in range(n_nodes):
        templates[i] = init[i % len(init)]
        templates[i] /= np.linalg.norm(templates[i]) + 1e-8

    lr = 0.1
    for ep in range(epochs):
        for idx in rng.permutation(min(n_train, len(images))):
            img = images[idx].copy()
            if contrast_aug and rng.random() < 0.5:
                m = img.mean(); img = m + (img - m) * (0.3 + rng.random() * 0.7)
            flat = img.reshape(-1).astype(np.float32)
            if center: flat = flat - flat.mean()
            flat /= np.linalg.norm(flat) + 1e-8
            best = int(np.argmax(templates @ flat))
            templates[best] += lr * (flat - templates[best])
            templates[best] /= np.linalg.norm(templates[best]) + 1e-8
    return templates

def activate(templates, images, center=True):
    N = len(images)
    flat = images.reshape(N, -1).astype(np.float32)
    if center: flat = flat - flat.mean(axis=1, keepdims=True)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    return np.clip(flat @ templates.T, 0, 1).astype(np.float32)


np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
n_tr, n_tr_knn, n_te = 10000, 2000, 500

for center in [False, True]:
    name = "centered" if center else "raw"
    t0 = time.time()
    temps = train_global(X_tr, n_nodes=100, epochs=3, n_train=n_tr, contrast_aug=True, center=center)
    Xref = activate(temps, np.array([X_tr[i] for i in range(n_tr_knn)]), center=center)
    knn = KNN(k=5); knn.fit(Xref, y_tr[:n_tr_knn])
    train_t = time.time()-t0

    print(f"\n{'='*50}")
    print(f"  GlobalEye 100n | {name} | train {train_t:.0f}s")
    print(f"  {'c':>6s}  {'acc':>7s}")
    print(f"  {'-'*15}")
    for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]:
        test_batch = X_te[:n_te] if c==1.0 else low_contrast(X_te[:n_te], c)
        Xte = activate(temps, test_batch, center=center)
        acc = knn.score(Xte, y_te[:n_te])
        print(f"  {c:5.2f}  {acc*100:6.1f}%")

print("\n=== DONE ===")
