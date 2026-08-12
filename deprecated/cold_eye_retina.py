#!/usr/bin/env python3
"""
cold_eye_retina.py — 完整冷眼 + 视网膜预处理

Pipeline:
  1. Retina normalize (DoG + local variance norm)
  2. MultiScaleEye routing (3 scales)
  3. Colony (cell walking + tier weights)
  4. FB error correction (α=0.8 for low contrast)
  5. KNN classification

对比: retina vs raw, α=0 vs α=0.8
"""

import gzip, numpy as np, time
np.random.seed(42)
import random; random.seed(42)

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
    return Xt,yt,Xv,yv

def low_contrast(X, c):
    Xc = X.copy()
    m = Xc.mean(axis=(1,2), keepdims=True)
    return m + (Xc - m) * c

# ═══ Retina ═══
def retina_normalize(images):
    """DoG + local variance normalization — contrast invariant edge map"""
    N, H, W = images.shape

    def box_conv(x, ksize):
        p = ksize // 2
        xp = np.pad(x, ((0,0),(p,p),(p,p)), mode='reflect')
        out = np.zeros_like(x)
        for dy in range(ksize):
            for dx in range(ksize):
                out += xp[:, dy:dy+H, dx:dx+W]
        return out / (ksize * ksize)

    center = box_conv(images, 3)
    surround = box_conv(images, 7)
    dog = center - surround
    dog_mean = box_conv(dog, 7)
    dog_var = box_conv((dog - dog_mean)**2, 7)
    local_std = np.sqrt(dog_var + 0.001)
    normalized = dog / local_std
    normalized = np.clip(normalized, -4, 4)
    return ((normalized + 4) / 8.0).astype(np.float32)

# ═══ MultiScaleEye ═══
def train_multiscale(images, labels, epochs=3, n_train=10000, contrast_aug=True):
    from graph import VisionGraph
    from vision import VisionInterface

    configs = [
        {"ps": 4,  "st": 4, "n": 100},
        {"ps": 8,  "st": 4, "n": 100},
        {"ps": 16, "st": 8, "n": 50},
    ]
    eyes = []
    for cfg in configs:
        ts = cfg["ps"] * cfg["ps"]
        g = VisionGraph(n_nodes=cfg["n"], template_size=ts)
        v = VisionInterface(g, patch_size=cfg["ps"], stride=cfg["st"])
        eyes.append((g, v, cfg["ps"], cfg["st"]))

    # Init templates
    for g, v, ps, st in eyes:
        ex = v.extractor
        nids = sorted(g.nodes.keys())
        idxs = np.random.choice(min(n_train, len(images)), min(200, n_train), replace=False)
        ap = [p for i in idxs for p in ex.extract(images[i])]
        for k, nid in enumerate(nids):
            p = ap[k % len(ap)]
            g.nodes[nid].template = p.astype(np.float32)
            g.nodes[nid].template /= np.linalg.norm(g.nodes[nid].template) + 1e-8

    lr = 0.1
    for ep in range(epochs):
        for idx in np.random.permutation(min(n_train, len(images))):
            img = images[idx].copy()
            if contrast_aug and np.random.random() < 0.5:
                m = img.mean(); c = 0.3 + np.random.random() * 0.7
                img = m + (img - m) * c
            for g, v, ps, st in eyes:
                v.set_image(img)
                for nid, aps in v.node_assignments.items():
                    t = np.mean(aps, axis=0)
                    n = np.linalg.norm(t)
                    if n > 0: t /= n
                    g.nodes[nid].template += lr * (t - g.nodes[nid].template)
                    g.nodes[nid].template /= np.linalg.norm(g.nodes[nid].template) + 1e-8
    return eyes

# ═══ Colony ═══
def build_colony(eyes, images, labels, n=2000, gens=80, act_thresh=0.05):
    from synapse import SynapticLayer
    from vision_colony import VisionColony

    colonies = []
    for g, v, ps, st in eyes:
        g.synapse = SynapticLayer()
        coactive = {}
        for i in range(min(n, len(images))):
            v.set_image(images[i].copy())
            active = [nid for nid, nd in g.nodes.items() if nd.activation > act_thresh]
            for j, src in enumerate(active):
                for dst in active[j+1:]:
                    coactive[(src, dst)] = coactive.get((src, dst), 0) + 1
        for (src, dst), cnt in coactive.items():
            if cnt >= 3:
                g.adjacency[src][dst] = min(1.0, cnt / 20.0)
                g.adjacency[dst][src] = min(1.0, cnt / 20.0)
        colony = VisionColony(g)
        colony.seed_cells(n_per_node=2, max_cells=200)
        colony.breathe(n_generations=gens, verbose=False)
        colonies.append(colony)
    return colonies

# ═══ Test ═══
def test_pipeline(name, train_images, train_labels, test_images, test_labels,
                  n_tr=2000, n_te=500, contrast_aug=True, act_thresh=0.05):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    t0 = time.time()
    eyes = train_multiscale(train_images, train_labels, epochs=3, contrast_aug=contrast_aug)
    print(f"  MultiScaleEye: {time.time()-t0:.1f}s")

    t0 = time.time()
    colonies = build_colony(eyes, train_images, train_labels, n=3000, gens=80, act_thresh=act_thresh)
    print(f"  Colony: {time.time()-t0:.1f}s")

    # Tier weights
    tier_w = []
    nids_list = []
    for (g, v, ps, st), col in zip(eyes, colonies):
        nids = sorted(g.nodes.keys())
        nids_list.append(nids)
        tw = np.zeros(len(nids), dtype=np.float32)
        for j, nid in enumerate(nids):
            c2 = c3 = 0
            for (s, d), tier in col.synapse.tiers.items():
                if s == nid or d == nid:
                    if tier == 2: c2 += 1
                    elif tier == 3: c3 += 1
            tw[j] = c3 * 0.3 + c2 * 0.1
        tier_w.append(tw)

    def extract(images, alpha=0):
        feats = []
        for img in images:
            parts = []
            for (g, v, ps, st), col, nids, tw in zip(eyes, colonies, nids_list, tier_w):
                v.set_image(img)
                act = np.array([g.nodes[nid].activation for nid in nids], dtype=np.float32)
                if alpha > 0:
                    error = v.predictive_boost(col, strength=0.3)
                    act = np.clip(act + error * alpha, 0, 1)
                parts.append(np.concatenate([act, tw]))
            feats.append(np.concatenate(parts))
        return np.array(feats, dtype=np.float32)

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

    # Train KNN
    Xtr = extract([train_images[i] for i in range(n_tr)], alpha=0)
    knn0 = KNN(k=5); knn0.fit(Xtr, train_labels[:n_tr])
    knn_fb = KNN(k=5)
    Xtr_fb = extract([train_images[i] for i in range(n_tr)], alpha=0.8)
    knn_fb.fit(Xtr_fb, train_labels[:n_tr])

    # Test
    print(f"  {'c':>6s}  {'α=0':>7s}  {'α=0.8':>7s}")
    print(f"  {'-'*24}")
    for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]:
        Xc = test_images[:n_te] if c == 1.0 else low_contrast(test_images[:n_te], c)
        Xte = extract(Xc, alpha=0)
        acc0 = knn0.score(Xte, test_labels[:n_te])
        Xte_fb = extract(Xc, alpha=0.8)
        acc_fb = knn_fb.score(Xte_fb, test_labels[:n_te])
        best = max(acc0, acc_fb)
        print(f"  {c:5.2f}  {acc0*100:6.1f}%  {acc_fb*100:6.1f}%")

    return best

# ═══ Main ═══
print("=" * 55)
print("ColdEye + Retina — Full Pipeline")
print("=" * 55)

Xt_raw, yt, Xv_raw, yv = load_mnist()

# Preprocess
print("\n[1] Retina normalization...")
t0 = time.time()
Xt_ret = retina_normalize(Xt_raw)
Xv_ret = retina_normalize(Xv_raw)
print(f"  done ({time.time()-t0:.1f}s)")

# Test both
test_pipeline("RAW pixels (baseline)", Xt_raw, yt, Xv_raw, yv)
test_pipeline("RETINA normalized", Xt_ret, yt, Xv_ret, yv)

print("\n=== DONE ===")
