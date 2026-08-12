#!/usr/bin/env python3
"""
retina_bio.py — 生物视网膜预处理 + 冷眼竞争路由

视网膜层（给方法）：
  ON 通道: 中心亮周围暗 → 正响应
  OFF 通道: 中心暗周围亮 → 负响应
  半波整流: ON = max(0, dog), OFF = max(0, -dog)
  除性归一化: 局域能量归一化（模拟皮层下增益控制）

皮层路由（给容量）：
  双通道输入 → 竞争路由 → Hebbian → KNN
"""

import gzip, numpy as np, time
np.random.seed(42)

def load_mnist():
    import os
    base = '/home/duyw/predictive-vision'
    for kind in ['train','t10k']:
        for suf in ['images-idx3-ubyte.gz','labels-idx1-ubyte.gz']:
            fpath = os.path.join(base, f'{kind}-{suf}')
            if not os.path.exists(fpath):
                import urllib.request
                urllib.request.urlretrieve(f"https://storage.googleapis.com/cvdf-datasets/mnist/{kind}-{suf}", fpath)
    with gzip.open(os.path.join(base,'train-images-idx3-ubyte.gz'),'rb') as f:
        Xt=np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255.0
    with gzip.open(os.path.join(base,'train-labels-idx1-ubyte.gz'),'rb') as f:
        yt=np.frombuffer(f.read(),np.uint8,offset=8)
    with gzip.open(os.path.join(base,'t10k-images-idx3-ubyte.gz'),'rb') as f:
        Xv=np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255.0
    with gzip.open(os.path.join(base,'t10k-labels-idx1-ubyte.gz'),'rb') as f:
        yv=np.frombuffer(f.read(),np.uint8,offset=8)
    return Xt, yt, Xv, yv

def low_contrast(X, c):
    Xc = X.copy()
    m = Xc.mean(axis=(1,2), keepdims=True)
    return m + (Xc - m) * c

# ═══ 视网膜（给方法） ═══

def retinal_processing(images):
    """
    模拟视网膜输出到 LGN:
      1. DoG 中心-周围拮抗
      2. ON/OFF 半波整流 → 双通道
      3. 局域除性归一化
    返回: (batch, 2, 28, 28) — ON/OFF 双通道
    """
    N, H, W = images.shape

    def box_conv(x, ksize):
        p = ksize // 2
        xp = np.pad(x, ((0,0),(p,p),(p,p)), mode='reflect')
        out = np.zeros_like(x)
        for dy in range(ksize):
            for dx in range(ksize):
                out += xp[:, dy:dy+H, dx:dx+W]
        return out / (ksize * ksize)

    # Center-surround
    center = box_conv(images, 3)
    surround = box_conv(images, 7)
    dog = center - surround

    # ON/OFF 半波整流
    on_channel = np.maximum(0, dog)   # 亮中心
    off_channel = np.maximum(0, -dog) # 暗中心

    # 局域除性归一化 (local energy pool)
    energy = box_conv(on_channel**2 + off_channel**2, 7)
    on_norm = on_channel / (np.sqrt(energy) + 0.1)
    off_norm = off_channel / (np.sqrt(energy) + 0.1)

    # Stack: (N, 2, H, W)
    out = np.stack([on_norm, off_norm], axis=1).astype(np.float32)
    return out

# ═══ 竞争路由（给容量） ═══

def train_routing(images, labels, epochs=3, n_train=60000):
    """
    输入: (N, 2, 28, 28) ON/OFF 双通道
    用 VisionGraph 对每个通道独立做竞争路由
    """
    from graph import VisionGraph
    from vision import VisionInterface

    configs = [
        {"ps": 4,  "st": 4, "n": 100},
        {"ps": 8,  "st": 4, "n": 100},
        {"ps": 16, "st": 8, "n": 50},
    ]

    # 双通道三尺度 = 6 个 eye
    eyes = []  # [(graph, vision, ch_name, patch_size, stride)]
    for ch in range(2):
        ch_name = "ON" if ch == 0 else "OFF"
        for cf in configs:
            ts = cf["ps"] * cf["ps"]
            g = VisionGraph(n_nodes=cf["n"], template_size=ts)
            v = VisionInterface(g, patch_size=cf["ps"], stride=cf["st"])
            eyes.append((g, v, ch, ch_name, cf["ps"], cf["st"]))

    # Init from real patches
    for g, v, ch, name, ps, st in eyes:
        ex = v.extractor
        nids = sorted(g.nodes.keys())
        idxs = np.random.choice(min(500, len(images)), min(500, len(images)), replace=False)
        ap = [p for i in idxs for p in ex.extract(images[i, ch])]
        for k, nid in enumerate(nids):
            p = ap[k % len(ap)]
            g.nodes[nid].template = p.astype(np.float32)
            g.nodes[nid].template /= np.linalg.norm(g.nodes[nid].template) + 1e-8

    # Hebbian on each channel independently
    lr = 0.1
    for ep in range(epochs):
        for idx in np.random.permutation(min(n_train, len(images))):
            for g, v, ch, name, ps, st in eyes:
                img = images[idx, ch].copy()
                # No contrast aug — retina handles it
                v.set_image(img)
                for nid, aps in v.node_assignments.items():
                    t = np.mean(aps, axis=0)
                    n = np.linalg.norm(t)
                    if n > 0: t /= n
                    g.nodes[nid].template += lr * (t - g.nodes[nid].template)
                    g.nodes[nid].template /= np.linalg.norm(g.nodes[nid].template) + 1e-8
        print(f"  epoch {ep+1}/{epochs}")

    return eyes

def extract_features(eyes, images):
    """images: (N, 2, H, W) → features: (N, total_nodes)"""
    nids_list = [sorted(g.nodes.keys()) for g, v, ch, name, ps, st in eyes]
    dim = sum(len(n) for n in nids_list)
    feats = np.zeros((len(images), dim), dtype=np.float32)
    for i in range(len(images)):
        off = 0
        for (g, v, ch, name, ps, st), nids in zip(eyes, nids_list):
            v.set_image(images[i, ch])
            feats[i, off:off+len(nids)] = [g.nodes[nid].activation for nid in nids]
            off += len(nids)
    return feats

# ═══ Main ═══

print("=" * 60)
print("Retina (ON/OFF dual-channel) → ColdEye routing")
print("=" * 60)

Xt, yt, Xv, yv = load_mnist()

print("\n[1] Retinal processing (ON/OFF + divisive norm)...")
t0 = time.time()
Xt_ret = retinal_processing(Xt)
Xv_ret = retinal_processing(Xv)
print(f"  Xt: {Xt_ret.shape}  Xv: {Xv_ret.shape}  ({time.time()-t0:.1f}s)")

print(f"\n[2] Training 6-channel competitive routing (60K × 3 ep)...")
t0 = time.time()
eyes = train_routing(Xt_ret, yt, epochs=3, n_train=60000)
print(f"  done ({time.time()-t0:.0f}s)")

print(f"\n[3] Extract features + KNN...")
n_tr = 5000; n_te = 2000
Xtr = extract_features(eyes, Xt_ret[:n_tr])
Xte_full = extract_features(eyes, Xv_ret[:n_te])

class KNN:
    def __init__(s, k=5): s.k = k
    def fit(s, X, y): s.X, s.y = X, y
    def score(s, X, y):
        c = 0
        for i in range(len(X)):
            d = np.sum((s.X-X[i])**2, axis=1)
            nn = np.argpartition(d, s.k)[:s.k]
            if np.bincount(s.y[nn].astype(int)).argmax() == y[i]: c += 1
        return c / len(X)

knn = KNN(k=5)
knn.fit(Xtr, yt[:n_tr])

# Test: normalize LOW-CONTRAST images too
print(f"\n  {'c':>6s}  {'accuracy':>8s}")
print(f"  {'-'*16}")
for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]:
    Xc_raw = Xv[:n_te] if c == 1.0 else low_contrast(Xv[:n_te], c)
    Xc_ret = retinal_processing(Xc_raw)
    Xte = extract_features(eyes, Xc_ret)
    acc = knn.score(Xte, yv[:n_te])
    print(f"  {c:5.2f}  {acc*100:7.1f}%")

print(f"\n=== DONE ===")
