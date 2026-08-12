#!/usr/bin/env python3
"""
retina_normalize.py — 视网膜预处理 + 冷眼对比

加一层局部对比度归一化 (Difference of Gaussians):
  1. LoG-like 高通滤波 (去全局亮度)
  2. Local std normalization (局部方差归一化)
  → 边缘特征在 c=0.1 和 c=1.0 下输出一致

对比: raw pixels vs retina-normalized → 竞争路由
"""

import gzip, numpy as np, time
np.random.seed(42)

def load_mnist():
    import os
    base='/home/duyw/predictive-vision'
    for kind in ['train','t10k']:
        for suf in ['images-idx3-ubyte.gz','labels-idx1-ubyte.gz']:
            fpath=os.path.join(base,f'{kind}-{suf}')
            if not os.path.exists(fpath):
                import urllib.request
                urllib.request.urlretrieve(f"https://storage.googleapis.com/cvdf-datasets/mnist/{kind}-{suf}",fpath)
    with gzip.open(os.path.join(base,'train-images-idx3-ubyte.gz'),'rb') as f:
        Xt=np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255.0
    with gzip.open(os.path.join(base,'train-labels-idx1-ubyte.gz'),'rb') as f:
        yt=np.frombuffer(f.read(),np.uint8,offset=8)
    with gzip.open(os.path.join(base,'t10k-images-idx3-ubyte.gz'),'rb') as f:
        Xv=np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255.0
    with gzip.open(os.path.join(base,'t10k-labels-idx1-ubyte.gz'),'rb') as f:
        yv=np.frombuffer(f.read(),np.uint8,offset=8)
    return Xt,yt,Xv,yv

def low_contrast(X,c):
    Xc=X.copy(); m=Xc.mean(axis=(1,2),keepdims=True); return m+(Xc-m)*c

# ═══ 视网膜处理 ═══

def retina_normalize(images):
    """快速视网膜: box-avg center-surround + local std norm (全 numpy 广播)"""
    N, H, W = images.shape
    # Center: 3×3 box avg
    center = np.zeros_like(images)
    center[:,1:-1,1:-1] = images[:,1:-1,1:-1]
    kernel_c = np.ones((1,3,3), dtype=np.float32) / 9.0
    # Surround: 7×7 box avg  
    kernel_s = np.ones((1,7,7), dtype=np.float32) / 49.0
    # 用简单的 pad + stride-1 conv
    def box_conv(x, kernel, ksize):
        p = ksize // 2
        xp = np.pad(x, ((0,0),(p,p),(p,p)), mode='reflect')
        out = np.zeros_like(x)
        for dy in range(ksize):
            for dx in range(ksize):
                out += xp[:, dy:dy+H, dx:dx+W]
        return out / (ksize*ksize)
    
    center = box_conv(images, None, 3)
    surround = box_conv(images, None, 7)
    dog = center - surround
    
    # Local std: box-avg of (dog - box_avg(dog))^2
    dog_mean = box_conv(dog, None, 7)
    dog_var = box_conv((dog - dog_mean)**2, None, 7)
    local_std = np.sqrt(dog_var + 0.001)
    normalized = dog / local_std
    
    normalized = np.clip(normalized, -4, 4)
    return ((normalized + 4) / 8.0).astype(np.float32)

# ═══ 竞争路由 (复用现有架构) ═══

def build_and_train(images, labels, n_nodes=100, epochs=3):
    from graph import VisionGraph
    from vision import VisionInterface
    
    g = VisionGraph(n_nodes=n_nodes, template_size=64)
    v = VisionInterface(g, patch_size=8, stride=4)
    
    # 初始化
    ex = v.extractor
    nids = sorted(g.nodes.keys())
    idxs = np.random.choice(min(200,len(images)), min(200,len(images)), replace=False)
    ap = [p for i in idxs for p in ex.extract(images[i])]
    for k, nid in enumerate(nids):
        p = ap[k%len(ap)]
        g.nodes[nid].template = p.astype(np.float32)
        g.nodes[nid].template /= np.linalg.norm(g.nodes[nid].template) + 1e-8
    
    # Hebbian
    lr = 0.1
    for ep in range(epochs):
        for idx in np.random.permutation(min(10000, len(images))):
            img = images[idx].copy()
            if np.random.random() < 0.5:
                m = img.mean(); c = 0.3 + np.random.random() * 0.7
                img = m + (img - m) * c
            v.set_image(img)
            for nid, aps in v.node_assignments.items():
                t = np.mean(aps, axis=0)
                n = np.linalg.norm(t)
                if n > 0: t /= n
                g.nodes[nid].template += lr * (t - g.nodes[nid].template)
                g.nodes[nid].template /= np.linalg.norm(g.nodes[nid].template) + 1e-8
    return g, v

def get_features(g, v, images):
    nids = sorted(g.nodes.keys())
    feats = np.zeros((len(images), len(nids)), dtype=np.float32)
    for i, img in enumerate(images):
        v.set_image(img)
        feats[i] = [g.nodes[nid].activation for nid in nids]
    return feats

# ═══ 主测试 ═══

print("="*60)
print("Retinal preprocessing — DoG + local normalization")
print("="*60)

Xt, yt, Xv, yv = load_mnist()

# Preprocess training images
print("\n[1] Retina normalization on training data...")
t0 = time.time()
Xt_retina = retina_normalize(Xt)
print(f"  done ({time.time()-t0:.1f}s)")

# Train both
print("\n[2] Training raw pixel model...")
g_raw, v_raw = build_and_train(Xt, yt, n_nodes=100, epochs=3)

print("\n[3] Training retina-normalized model...")
g_ret, v_ret = build_and_train(Xt_retina, yt, n_nodes=100, epochs=3)

# Test at all contrasts
print("\n[4] Testing...")
n_tr = 2000; n_te = 500

# Raw model
Xtr_raw = get_features(g_raw, v_raw, [Xt[i] for i in range(n_tr)])
Xtr_ret = get_features(g_ret, v_ret, [Xt_retina[i] for i in range(n_tr)])

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

knn_raw = KNN(); knn_raw.fit(Xtr_raw, yt[:n_tr])
knn_ret = KNN(); knn_ret.fit(Xtr_ret, yt[:n_tr])

print(f"{'c':>6s}  {'raw':>8s}  {'retina':>8s}  {'Δ':>7s}")
print("-"*34)
for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]:
    Xc = Xv[:n_te] if c==1.0 else low_contrast(Xv[:n_te], c)
    
    # Raw model: test on raw low-contrast
    Xte = get_features(g_raw, v_raw, Xc)
    acc_raw = knn_raw.score(Xte, yv[:n_te])
    
    # Retina model: normalize test images then test
    Xc_ret = retina_normalize(Xc)
    Xte_ret = get_features(g_ret, v_ret, Xc_ret)
    acc_ret = knn_ret.score(Xte_ret, yv[:n_te])
    
    print(f"{c:5.2f}  {acc_raw*100:7.1f}%  {acc_ret*100:7.1f}%  {acc_ret-acc_raw:+6.1%}")

print("\n=== DONE ===")
