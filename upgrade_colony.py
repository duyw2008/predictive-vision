#!/usr/bin/env python3
"""
upgrade_colony.py — 把 Colony+FB 接到 MultiScaleEye 上

MultiScaleEye: 3 尺度 (4×4, 8×8, 16×16) → 250 维激活 → K-means 形状
Colony: 在每个尺度的 VisionGraph 上建 KG → 细胞行走 → 突触投票
FB: 预测误差校正激活

对比: MultiScaleEye (baseline) vs MultiScaleEye+Colony+FB
"""

import sys, os, time, gzip, numpy as np
np.random.seed(42)
import random; random.seed(42)

_avg_error_mag = 0

# ═══ 数据 ═══

def load_mnist():
    base = os.path.dirname(__file__) or '.'
    for kind in ['train', 't10k']:
        for suffix in ['images-idx3-ubyte.gz', 'labels-idx1-ubyte.gz']:
            fpath = os.path.join(base, f'{kind}-{suffix}')
            if not os.path.exists(fpath):
                import urllib.request
                urllib.request.urlretrieve(
                    f"https://storage.googleapis.com/cvdf-datasets/mnist/{kind}-{suffix}", fpath)
    with gzip.open(os.path.join(base, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        Xt = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)/255.0
    with gzip.open(os.path.join(base, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        yt = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(os.path.join(base, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        Xv = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)/255.0
    with gzip.open(os.path.join(base, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        yv = np.frombuffer(f.read(), np.uint8, offset=8)
    return Xt, yt, Xv, yv


def low_contrast(X, c):
    Xc = X.copy()
    mean = Xc.mean(axis=(1,2), keepdims=True)
    return mean + (Xc - mean) * c


# ═══ MultiScaleEye ═══

def train_multiscale(images, labels, epochs=3, n_train=10000):
    """训练 MultiScaleEye, 返回 eyes 列表"""
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
        eyes.append({"graph": g, "vision": v, "ps": cfg["ps"], "st": cfg["st"]})

    # 初始化模板
    for eye in eyes:
        ex = eye["vision"].extractor
        nids = sorted(eye["graph"].nodes.keys())
        idxs = np.random.choice(min(n_train, len(images)), min(200, n_train), replace=False)
        ap = []
        for i in idxs:
            ap.extend(ex.extract(images[i]))
        if not ap:
            continue
        for k, nid in enumerate(nids):
            p = ap[k % len(ap)]
            eye["graph"].nodes[nid].template = p.astype(np.float32)
            eye["graph"].nodes[nid].template += np.random.randn(len(p)).astype(np.float32) * 0.01
            eye["graph"].nodes[nid].template /= np.linalg.norm(eye["graph"].nodes[nid].template) + 1e-8

    # Hebbian 训练
    lr = 0.1
    for ep in range(epochs):
        perm = np.random.permutation(min(n_train, len(images)))
        for idx in perm:
            img = images[idx].copy()
            # 对比度增强
            if np.random.random() < 0.5:
                m = img.mean()
                c = 0.3 + np.random.random() * 0.7
                img = m + (img - m) * c
            for eye in eyes:
                eye["vision"].set_image(img)
                for nid, aps in eye["vision"].node_assignments.items():
                    node = eye["graph"].nodes[nid]
                    t = np.mean(aps, axis=0)
                    n = np.linalg.norm(t)
                    if n > 0:
                        t /= n
                    node.template += lr * (t - node.template)
                    node.template /= np.linalg.norm(node.template) + 1e-8
        print(f"  epoch {ep+1} done", flush=True)

    return eyes


def build_k_shapes(eyes, images, labels, k=100, n_samples=5000):
    """K-means 形状节点"""
    n = min(n_samples, len(images))
    idxs = np.random.choice(len(images), n, replace=False)

    vecs = []
    for i in idxs:
        parts = []
        for eye in eyes:
            eye["vision"].set_image(images[i])
            nids = sorted(eye["graph"].nodes.keys())
            parts.append(np.array([eye["graph"].nodes[nid].activation for nid in nids], dtype=np.float32))
        vecs.append(np.concatenate(parts))
    X = np.array(vecs, dtype=np.float32)

    # K-means
    rng = np.random.RandomState(42)
    indices = rng.choice(len(X), k, replace=False)
    centers = X[indices].copy()
    for it in range(50):
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        km = np.argmin(dists, axis=1)
        new_c = np.zeros_like(centers)
        for c in range(k):
            mask = km == c
            new_c[c] = X[mask].mean(axis=0) if mask.sum() > 0 else X[rng.choice(len(X))]
        shift = np.sum((centers - new_c) ** 2)
        centers = new_c
        if shift < 1e-6:
            break

    # 形状→类别分布
    dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    km = np.argmin(dists, axis=1)
    labels_arr = np.array([labels[i] for i in idxs])
    shape_dist = {}
    for c in range(k):
        mask = km == c
        if mask.sum() > 0:
            shape_dist[c] = np.bincount(labels_arr[mask], minlength=10)
    return centers.astype(np.float32), shape_dist


def predict_ms(images, eyes, shape_centers, shape_dist, k=5):
    """MultiScaleEye 预测 — 1-NN on shape + KNN 回退"""
    nids_list = [sorted(eye["graph"].nodes.keys()) for eye in eyes]

    preds = []
    for img in images:
        parts = []
        for eye in eyes:
            eye["vision"].set_image(img)
            nids = sorted(eye["graph"].nodes.keys())
            parts.append(np.array([eye["graph"].nodes[nid].activation for nid in nids], dtype=np.float32))
        vec = np.concatenate(parts)

        # 找最近形状
        dists = np.sum((shape_centers - vec) ** 2, axis=1)
        sid = int(np.argmin(dists))

        if sid in shape_dist:
            counts = shape_dist[sid]
            if counts.sum() >= 5:
                preds.append(int(np.argmax(counts)))
            else:
                preds.append(0)  # fallback
        else:
            preds.append(0)
    return np.array(preds)


def accuracy(preds, labels):
    return (preds == labels).mean()


# ═══ Colony + FB on each eye ═══

def build_kg_and_train_colony(eye, images, labels, n_train=3000, colony_gens=100):
    """对一个 VisionGraph: 建 KG → 跑 Colony"""
    graph = eye["graph"]
    vision = eye["vision"]

    from synapse import SynapticLayer
    graph.synapse = SynapticLayer()

    # KG: 共激活边
    coactive = {}
    for i in range(min(n_train, len(images))):
        vision.set_image(images[i].copy())
        active = [nid for nid, n in graph.nodes.items() if n.activation > 0.05]
        for j, src in enumerate(active):
            for dst in active[j+1:]:
                coactive[(src, dst)] = coactive.get((src, dst), 0) + 1

    for (src, dst), count in coactive.items():
        if count >= 3:
            graph.adjacency[src][dst] = min(1.0, count / 20.0)
            graph.adjacency[dst][src] = min(1.0, count / 20.0)

    # 跑 Colony
    from vision_colony import VisionColony
    colony = VisionColony(graph)
    colony.seed_cells(n_per_node=2, max_cells=200)
    colony.breathe(n_generations=colony_gens, verbose=False)

    return colony


def extract_activation_vector(eyes, image):
    """250 维激活向量"""
    parts = []
    for eye in eyes:
        eye["vision"].set_image(image)
        nids = sorted(eye["graph"].nodes.keys())
        parts.append(np.array([eye["graph"].nodes[nid].activation for nid in nids], dtype=np.float32))
    return np.concatenate(parts)


def extract_fb_vector(eyes, colonies, image, alpha=0.3):
    """250 维误差校正激活向量"""
    global _avg_error_mag
    parts = []
    for eye, colony in zip(eyes, colonies):
        vision = eye["vision"]
        vision.set_image(image)
        nids = sorted(eye["graph"].nodes.keys())
        act = np.array([eye["graph"].nodes[nid].activation for nid in nids], dtype=np.float32)
        error = vision.predictive_boost(colony, strength=0.3)
        act_corrected = np.clip(act + error * alpha, 0.0, 1.0)
        parts.append(act_corrected)
    return np.concatenate(parts)


# ═══ 主测试 ═══

def main():
    print("=" * 55)
    print("MultiScaleEye vs MultiScaleEye+Colony+FB")
    print("=" * 55)

    print("\n[1] Loading MNIST...")
    Xt, yt, Xv, yv = load_mnist()
    print(f"  train: {Xt.shape}, test: {Xv.shape}")

    # 训练 MultiScaleEye
    print("\n[2] Training MultiScaleEye (250 nodes, Hebbian, contrast aug)...")
    t0 = time.time()
    eyes = train_multiscale(Xt[:10000], yt[:10000], epochs=3, n_train=10000)
    print(f"  done ({time.time()-t0:.1f}s)")

    # K-means 形状
    print("\n[3] Building K-means shapes...")
    t0 = time.time()
    shape_centers, shape_dist = build_k_shapes(eyes, Xt[:10000], yt[:10000], k=100, n_samples=5000)
    print(f"  {len(shape_centers)} shapes done ({time.time()-t0:.1f}s)")

    # Baseline 准确率
    print("\n[4] MultiScaleEye baseline...")
    t0 = time.time()
    for c in [1.0, 0.5, 0.3, 0.2]:
        Xc = Xv[:2000] if c == 1.0 else low_contrast(Xv[:2000], c)
        preds = predict_ms(Xc, eyes, shape_centers, shape_dist)
        acc = accuracy(preds, yv[:2000])
        print(f"  c={c:.1f}: {acc*100:.1f}%")
    print(f"  ({time.time()-t0:.1f}s)")

    # 在每个眼的图上跑 Colony
    print("\n[5] Building KG + Colony on each eye...")
    t0 = time.time()
    colonies = []
    for i, eye in enumerate(eyes):
        colony = build_kg_and_train_colony(eye, Xt[:3000], yt[:3000], n_train=2000, colony_gens=80)
        colonies.append(colony)
        stats = colony.stats()
        print(f"  eye[{i}]: t2={stats['t2']} t3={stats['t3']} syn={stats['synapses']}")
    print(f"  done ({time.time()-t0:.1f}s)")

    # 用 FB 特征重新分类 (KNN 投票)
    print("\n[6] Colony+FB: error-corrected activations → KNN...")
    t0 = time.time()

    # 构建训练特征
    n_train = 2000
    X_fb_train = np.array([extract_fb_vector(eyes, colonies, Xt[i]) for i in range(n_train)], dtype=np.float32)

    # KNN
    class SimpleKNN:
        def __init__(self, k=5): self.k = k
        def fit(self, X, y): self.X, self.y = X, y
        def predict(self, X):
            preds = []
            for i in range(len(X)):
                dists = np.sum((self.X - X[i])**2, axis=1)
                nn = np.argpartition(dists, self.k)[:self.k]
                preds.append(np.bincount(self.y[nn].astype(int)).argmax())
            return np.array(preds)

    knn = SimpleKNN(k=5)
    knn.fit(X_fb_train, yt[:n_train])

    for c in [1.0, 0.5, 0.3, 0.2]:
        Xc = Xv[:2000] if c == 1.0 else low_contrast(Xv[:2000], c)
        X_fb_test = np.array([extract_fb_vector(eyes, colonies, Xc[i]) for i in range(500)], dtype=np.float32)
        preds = knn.predict(X_fb_test)
        acc = accuracy(preds, yv[:500])
        print(f"  c={c:.1f}: {acc*100:.1f}%  (err mag={_avg_error_mag:.4f})")

    print("\n=== DONE ===")


if __name__ == '__main__':
    main()
