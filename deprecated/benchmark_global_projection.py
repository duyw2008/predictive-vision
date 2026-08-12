#!/usr/bin/env python3
"""
缺口3-B: 全局投影替代 patch 匹配
思路: 整张图做 Hadamard 投影，低对比度下全局积分 > 局部匹配
对比: global projection vs MultiScale patch matching at c≤0.1
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

# ── Hadamard basis ──
def hadamard_basis(n_dim=28*28, n_keep=200):
    """生成 Hadamard-like 正交基 (用 Walsh 序 Hadamard, 需 2^k 大小)
    对于 28x28=784，扩展到 1024，取前 784 个非 DC 分量"""
    # 找到最小的 2^k >= 784
    k = int(np.ceil(np.log2(n_dim)))
    N = 2**k  # 1024
    # 递归生成 Hadamard
    H = np.array([[1]], dtype=np.float32)
    for _ in range(k):
        H = np.block([[H, H], [H, -H]])
    # 取前 784 行和列 (跳过 DC)
    H_sub = H[1:n_dim+1, :n_dim]  # [784, 784]
    # 取前 n_keep 个非 DC 基向量 (低频优先)
    H_out = H[1:n_keep+1, :n_dim]  # [n_keep, 784]
    return H_out / np.sqrt(n_dim)

def hadamard_features(images, basis):
    """images [N,28,28] → features [N, n_basis]"""
    N = len(images)
    flat = images.reshape(N, -1)  # [N, 784]
    # 去均值
    flat = flat - flat.mean(axis=1, keepdims=True)
    return flat @ basis.T  # [N, n_basis]

# ── KNN ──
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

# ── Simple global (raw pixels, for baseline) ──
def raw_pixel_features(images):
    return images.reshape(len(images), -1)


np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
n_tr_knn, n_te = 2000, 500

print("生成 Hadamard 基...")
t0 = time.time()
basis = hadamard_basis(n_dim=784, n_keep=200)
print(f"  {basis.shape} — {time.time()-t0:.1f}s")

print(f"\n{'='*55}")
print(f"  Global projection baselines ({n_tr_knn} ref, {n_te} test)")
print(f"{'='*55}")

configs = [
    ("raw pixels (784d)", lambda imgs: raw_pixel_features(imgs)),
    ("Hadamard (200d)",   lambda imgs: hadamard_features(imgs, basis)),
]

for name, feat_fn in configs:
    t0 = time.time()
    Xref = feat_fn(np.array([X_tr[i] for i in range(n_tr_knn)]))
    knn = KNN(k=5); knn.fit(Xref, y_tr[:n_tr_knn])
    print(f"\n  {name}")
    print(f"  {'c':>6s}  {'acc':>7s}")
    print(f"  {'-'*15}")
    for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07]:
        if c == 1.0:
            test_batch = X_te[:n_te]
        else:
            test_batch = low_contrast(X_te[:n_te], c)
        Xte = feat_fn(test_batch)
        acc = knn.score(Xte, y_te[:n_te])
        print(f"  {c:5.2f}  {acc*100:6.1f}%")
    print(f"  ({time.time()-t0:.1f}s)")

print("\n=== DONE ===")
