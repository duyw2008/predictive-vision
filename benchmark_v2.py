#!/usr/bin/env python3
"""
benchmark_v2.py — 冷眼预测反馈 v3: 误差校正激活 (同维度融合)

对比:
  1. KNN (raw pixels)
  2. ColdEye baseline (竞争路由)
  3. ColdEye+Colony (tier 特征)
  4. ColdEye+Colony+FB (误差校正激活 — 同维度)

改进:
  - 预测误差校正激活而非拼接: act_corrected = act + error × α
  - 固定 seed, 更大节点数
  - 对比度增强训练 (细胞在全对比度和低对比度上都走)
"""

import sys, os, time, gzip, numpy as np

_avg_error_mag = 0

# ═══ 数据 ═══

def load_mnist(limit=4000):
    base = os.path.dirname(__file__) or '.'
    url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    for kind in ['train', 't10k']:
        for suffix in ['images-idx3-ubyte.gz', 'labels-idx1-ubyte.gz']:
            fpath = os.path.join(base, f'{kind}-{suffix}')
            if not os.path.exists(fpath):
                import urllib.request
                urllib.request.urlretrieve(url + f'{kind}-{suffix}', fpath)
    with gzip.open(os.path.join(base, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        X_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
    with gzip.open(os.path.join(base, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(os.path.join(base, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
    with gzip.open(os.path.join(base, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)
    return X_train[:limit], y_train[:limit], X_test[:2000], y_test[:2000]


def low_contrast(X, c):
    Xc = X.copy()
    mean = Xc.mean(axis=1, keepdims=True)
    return mean + (Xc - mean) * c


# ═══ KNN ═══

class SimpleKNN:
    def __init__(self, k=5): self.k = k
    def fit(self, X, y): self.X, self.y = X, y
    def score(self, X, y):
        correct = 0
        for i in range(len(X)):
            dists = np.sum((self.X - X[i])**2, axis=1)
            nn = np.argpartition(dists, self.k)[:self.k]
            pred = np.bincount(self.y[nn].astype(int)).argmax()
            if pred == y[i]: correct += 1
        return correct / len(X)


# ═══ CNN ═══

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(32 * 7 * 7, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def train_cnn(X_train, y_train, X_test, y_test, epochs=10, lr=0.01):
    """训练简单 CNN, 返回各对比度准确率"""
    if not HAS_TORCH:
        return {'c1.0': None, 'c0.5': None, 'c0.3': None, 'c0.2': None}

    device = torch.device('cpu')
    Xt = torch.tensor(X_train[:3000].reshape(-1, 1, 28, 28), dtype=torch.float32)
    yt = torch.tensor(y_train[:3000], dtype=torch.long)
    Xv = torch.tensor(X_test[:2000].reshape(-1, 1, 28, 28), dtype=torch.float32)
    yv = torch.tensor(y_test[:2000], dtype=torch.long)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for i in range(0, len(Xt), 64):
            bx, by = Xt[i:i+64].to(device), yt[i:i+64].to(device)
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()

    model.eval()
    results = {}
    for c in [1.0, 0.5, 0.3, 0.2]:
        Xc = Xv.clone()
        if c < 1.0:
            mean = Xc.mean(dim=(2,3), keepdim=True)
            Xc = mean + (Xc - mean) * c
        with torch.no_grad():
            pred = model(Xc.to(device)).argmax(1)
            acc = (pred == yv.to(device)).float().mean().item()
        results[f'c{c}'] = acc
    return results

def build_cold_eye(images, labels, n_nodes=100, n_train=2000, contrast=1.0):
    sys.path.insert(0, os.path.dirname(__file__))
    from graph import VisionGraph
    from vision import VisionInterface

    graph = VisionGraph(n_nodes=n_nodes, template_size=64)
    vision = VisionInterface(graph, patch_size=8, stride=4)

    # 模板初始化
    patch_samples = []
    for i in range(min(500, n_train)):
        patches = vision.extractor.extract(images[i].reshape(28, 28))
        patch_samples.append(patches[np.random.randint(len(patches))])
    for nid in graph.nodes:
        idx = np.random.randint(len(patch_samples))
        graph.nodes[nid].template = patch_samples[idx].copy()
        graph.nodes[nid].template /= np.linalg.norm(graph.nodes[nid].template) + 1e-8

    from synapse import SynapticLayer
    graph.synapse = SynapticLayer()

    # 知识图谱: 共激活边
    coactive = {}
    for i in range(min(n_train, len(images))):
        img = images[i].reshape(28, 28).copy()
        vision.set_image(img, contrast=contrast)
        active = [nid for nid, n in graph.nodes.items() if n.activation > 0.05]
        for j, src in enumerate(active):
            for dst in active[j+1:]:
                coactive[(src, dst)] = coactive.get((src, dst), 0) + 1

    kg_edge_count = 0
    for (src, dst), count in coactive.items():
        if count >= 3:
            graph.adjacency[src][dst] = min(1.0, count / 20.0)
            graph.adjacency[dst][src] = min(1.0, count / 20.0)
            kg_edge_count += 2
    print(f"    KG: {kg_edge_count} edges, synapse: empty (cells will build)")

    return graph, vision


def extract_activations(vision, images, limit):
    node_ids = sorted(vision.graph.nodes.keys())
    features = []
    for i in range(min(limit, len(images))):
        vision.set_image(images[i].reshape(28, 28).copy())
        features.append(np.array([vision.graph.nodes[nid].activation for nid in node_ids], dtype=np.float32))
    return np.array(features), node_ids


def extract_tier_weights(colony, node_ids):
    """每个节点的 tier 权重 (静态, 所有图片共享)"""
    w = np.zeros(len(node_ids), dtype=np.float32)
    for j, nid in enumerate(node_ids):
        c2 = c3 = 0
        for (src, dst), tier in colony.synapse.tiers.items():
            if src == nid or dst == nid:
                if tier == 2: c2 += 1
                elif tier == 3: c3 += 1
        w[j] = c3 * 0.3 + c2 * 0.1
    return w


def extract_colony_features(vision, colony, images, limit, node_ids, tier_w):
    """激活 + tier 权重 (160 维)"""
    features = []
    for i in range(min(limit, len(images))):
        vision.set_image(images[i].reshape(28, 28).copy())
        act = np.array([vision.graph.nodes[nid].activation for nid in node_ids], dtype=np.float32)
        features.append(np.concatenate([act, tier_w]))
    return np.array(features)


def extract_feedback_features(vision, colony, images, limit, node_ids, tier_w, alpha=0.25):
    """
    预测反馈 v3: 误差校正激活 (同 160 维, 不拼接)

    act_corrected = act + error × α
    满对比度: error≈0 → 不变
    低对比度: error 修正被弱化的特征
    """
    global _avg_error_mag
    features = []
    total_err = 0
    for i in range(min(limit, len(images))):
        vision.set_image(images[i].reshape(28, 28).copy())
        act = np.array([vision.graph.nodes[nid].activation for nid in node_ids], dtype=np.float32)
        error = vision.predictive_boost(colony, strength=0.3)
        act_corrected = np.clip(act + error * alpha, 0.0, 1.0)
        total_err += float(np.mean(np.abs(error)))
        features.append(np.concatenate([act_corrected, tier_w]))
    _avg_error_mag = total_err / max(1, limit)
    return np.array(features)


# ═══ 主测试 ═══

def main():
    np.random.seed(42)
    random.seed(42)

    print("=" * 60)
    print("冷眼预测反馈 v3 — 误差校正激活")
    print("=" * 60)

    print("\n[1] Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist(limit=4000)
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    print("\n[2] KNN baseline...")
    t0 = time.time()
    knn = SimpleKNN(k=5)
    knn.fit(X_train, y_train)
    acc_knn = knn.score(X_test[:300], y_test[:300])
    print(f"  KNN (raw):  {acc_knn*100:.1f}%  ({time.time()-t0:.1f}s)")

    print("\n[3] CNN baseline...")
    t0 = time.time()
    cnn_results = train_cnn(X_train, y_train, X_test, y_test, epochs=10)
    print(f"  CNN:  c1.0={cnn_results.get('c1.0',0)*100:.1f}%  "
          f"c0.5={cnn_results.get('c0.5',0)*100:.1f}%  "
          f"c0.3={cnn_results.get('c0.3',0)*100:.1f}%  "
          f"c0.2={cnn_results.get('c0.2',0)*100:.1f}%  ({time.time()-t0:.1f}s)")

    print("\n[4] ColdEye baseline...")
    t0 = time.time()
    ce_graph, ce_vision = build_cold_eye(
        X_train.reshape(-1, 28, 28), y_train, n_nodes=100, n_train=3000)
    print(f"  built: {sum(len(v) for v in ce_graph.adjacency.values())} KG edges ({time.time()-t0:.1f}s)")

    # 提取特征
    X_act_train, node_ids = extract_activations(ce_vision, X_train.reshape(-1, 28, 28), limit=2000)
    X_act_test, _ = extract_activations(ce_vision, X_test.reshape(-1, 28, 28), limit=300)

    clf_ce = SimpleKNN(k=5)
    clf_ce.fit(X_act_train, y_train[:2000])
    acc_ce = clf_ce.score(X_act_test, y_test[:300])
    results = {'KNN (raw)': {'c1.0': acc_knn},
               'CNN': {f'c{k}': v for k, v in cnn_results.items()},
               'ColdEye': {'c1.0': acc_ce}}
    print(f"  ColdEye:  {acc_ce*100:.1f}%  ({time.time()-t0:.1f}s total)")

    # 5. Colony
    print("\n[5] ColdEye + Colony...")
    t0 = time.time()
    from vision_colony import VisionColony
    colony = VisionColony(ce_graph)
    colony.seed_cells(n_per_node=2, max_cells=200)
    colony.breathe(n_generations=100, verbose=False)
    stats = colony.stats()
    print(f"  t2={stats['t2']} t3={stats['t3']} ({time.time()-t0:.1f}s)")

    tier_w = extract_tier_weights(colony, node_ids)
    X_col_train = extract_colony_features(ce_vision, colony, X_train.reshape(-1, 28, 28), 2000, node_ids, tier_w)
    X_col_test = extract_colony_features(ce_vision, colony, X_test.reshape(-1, 28, 28), 300, node_ids, tier_w)

    clf_col = SimpleKNN(k=5)
    clf_col.fit(X_col_train, y_train[:2000])
    acc_col = clf_col.score(X_col_test, y_test[:300])
    results['Colony'] = {'c1.0': acc_col}
    print(f"  Colony:  {acc_col*100:.1f}%")

    # 5. Feedback
    X_fb_train = extract_feedback_features(ce_vision, colony, X_train.reshape(-1, 28, 28), 2000, node_ids, tier_w, alpha=0.5)
    X_fb_test = extract_feedback_features(ce_vision, colony, X_test.reshape(-1, 28, 28), 300, node_ids, tier_w, alpha=0.5)

    clf_fb = SimpleKNN(k=5)
    clf_fb.fit(X_fb_train, y_train[:2000])
    acc_fb = clf_fb.score(X_fb_test, y_test[:300])
    results['FB (error-correct)'] = {'c1.0': acc_fb}
    print(f"  FB:  {acc_fb*100:.1f}%  (error mag {_avg_error_mag:.3f})")

    # 6. 低对比度
    print("\n[5] Low contrast...")
    for c in [0.5, 0.3, 0.2]:
        X_low = low_contrast(X_test, c).reshape(-1, 28, 28)

        X_act_low, _ = extract_activations(ce_vision, X_low, limit=300)
        X_col_low = extract_colony_features(ce_vision, colony, X_low, 300, node_ids, tier_w)
        X_fb_low = extract_feedback_features(ce_vision, colony, X_low, 300, node_ids, tier_w, alpha=0.5)

        acc_ce_low = clf_ce.score(X_act_low, y_test[:300])
        acc_col_low = clf_col.score(X_col_low, y_test[:300])
        acc_fb_low = clf_fb.score(X_fb_low, y_test[:300])

        results['ColdEye'][f'c{c}'] = acc_ce_low
        results['Colony'][f'c{c}'] = acc_col_low
        results['FB (error-correct)'][f'c{c}'] = acc_fb_low

        print(f"  c={c:.1f}:  CE={acc_ce_low*100:.1f}%  Col={acc_col_low*100:.1f}%  "
              f"FB={acc_fb_low*100:.1f}%  (err mag {_avg_error_mag:.3f})")

    # ═══ 总结 ═══
    print("\n" + "=" * 62)
    print("RESULTS (误差校正 v3)")
    print("=" * 62)
    header = f"{'Method':<22s}"
    for cl in ['c1.0', 'c0.5', 'c0.3', 'c0.2']:
        header += f" {cl:>7s}"
    print(header)
    print("-" * 54)
    for name in ['KNN (raw)', 'CNN', 'ColdEye', 'Colony', 'FB (error-correct)']:
        scores = results.get(name)
        if scores is None: continue
        row = f"{name:<22s}"
        for cl in ['c1.0', 'c0.5', 'c0.3', 'c0.2']:
            val = scores.get(cl, '-')
            row += f" {val*100:>6.1f}%" if isinstance(val, float) else f" {'-':>6s}"
        print(row)

    # 反馈增益
    print("\nFeedback GAIN (FB - Colony):")
    for cl in ['c1.0', 'c0.5', 'c0.3', 'c0.2']:
        gain = results.get('FB (error-correct)', {}).get(cl, 0) - results.get('Colony', {}).get(cl, 0)
        sign = '+' if gain > 0 else ''
        print(f"  {cl}: {sign}{gain*100:.1f}%")

    print("\n=== DONE ===")

import random
if __name__ == '__main__':
    main()
