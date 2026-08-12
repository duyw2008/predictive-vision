#!/usr/bin/env python3
"""
benchmark_colony.py — 冷眼 vs 传统算法 + 预测反馈对比

对比:
  1. Raw pixel KNN (k=5)
  2. Raw pixel SVM (linear)
  3. 2-layer MLP (sklearn)
  4. ColdEye baseline (竞争路由 → KNN on activations)
  5. ColdEye + Colony (Stage 2 细胞行走 → tier 特征 → KNN)
  6. ColdEye + Colony + FB (预测反馈增强)

评价: 准确率 (full + low contrast 0.5/0.3/0.2)
"""

import sys, os, time, gzip, numpy as np

_avg_boosted = 0


def load_mnist(limit=4000):
    base = os.path.dirname(__file__) or '.'
    # 优先用本地已有数据
    url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    for kind in ['train', 't10k']:
        for suffix in ['images-idx3-ubyte.gz', 'labels-idx1-ubyte.gz']:
            fname = f'{kind}-{suffix}'
            fpath = os.path.join(base, fname)
            if not os.path.exists(fpath):
                import urllib.request
                urllib.request.urlretrieve(url + fname, fpath)

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


# ═══ 传统基线 (纯 numpy, 不依赖 sklearn) ═══

def bench_knn(X_train, y_train, X_test, y_test, k=5):
    """纯 numpy KNN"""
    correct = 0
    for i in range(len(X_test)):
        dists = np.sum((X_train - X_test[i]) ** 2, axis=1)
        neighbors = np.argpartition(dists, k)[:k]
        pred = np.bincount(y_train[neighbors].astype(int)).argmax()
        if pred == y_test[i]:
            correct += 1
    return correct / len(X_test)


def bench_svm(X_train, y_train, X_test, y_test):
    """跳过 SVM (无 sklearn)"""
    return None


def bench_mlp(X_train, y_train, X_test, y_test):
    """跳过 MLP (无 sklearn)"""
    return None


# ═══ ColdEye 管线 ═══

def build_cold_eye(images, labels, n_nodes=80, n_train=2000, contrast=1.0):
    sys.path.insert(0, os.path.dirname(__file__))
    from graph import VisionGraph
    from vision import VisionInterface

    ts = 8 * 8
    graph = VisionGraph(n_nodes=n_nodes, template_size=ts)
    vision = VisionInterface(graph, patch_size=8, stride=4)

    patch_samples = []
    for i in range(min(500, n_train)):
        patches = vision.extractor.extract(images[i].reshape(28, 28))
        patch_samples.append(patches[np.random.randint(len(patches))])
    for nid in graph.nodes:
        idx = np.random.randint(len(patch_samples))
        graph.nodes[nid].template = patch_samples[idx].copy()
        graph.nodes[nid].template /= np.linalg.norm(graph.nodes[nid].template) + 1e-8

    from synapse import SynapticLayer
    synapse = SynapticLayer()
    graph.synapse = synapse

    # 知识图谱: 共激活边写入 VisionGraph.adjacency (不是突触层!)
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
            # 写入知识图谱 (VisionGraph.adjacency)
            graph.adjacency[src][dst] = min(1.0, count / 20.0)
            graph.adjacency[dst][src] = min(1.0, count / 20.0)
            kg_edge_count += 2
    print(f"    KG edges: {kg_edge_count} in VisionGraph.adjacency")

    # 突触层: 初始为空 — 细胞行走时自己建
    # (SynapticLayer 已创建在上面, 不预填)

    return graph, vision


def cold_eye_features(vision, images, limit=2000):
    features = []
    for i in range(min(limit, len(images))):
        img = images[i].reshape(28, 28).copy()
        vision.set_image(img)
        vec = np.array([vision.graph.nodes[nid].activation
                        for nid in sorted(vision.graph.nodes.keys())], dtype=np.float32)
        features.append(vec)
    return np.array(features)


def cold_eye_colony_features(colony, vision, images, limit=2000):
    features = []
    node_ids = sorted(vision.graph.nodes.keys())
    for i in range(min(limit, len(images))):
        img = images[i].reshape(28, 28).copy()
        vision.set_image(img)
        act_vec = np.array([vision.graph.nodes[nid].activation for nid in node_ids], dtype=np.float32)
        tier_boost = np.zeros(len(node_ids), dtype=np.float32)
        for j, nid in enumerate(node_ids):
            c2 = c3 = 0
            for (src, dst), tier in colony.synapse.tiers.items():
                if src == nid or dst == nid:
                    if tier == 2: c2 += 1
                    elif tier == 3: c3 += 1
            tier_boost[j] = c3 * 0.3 + c2 * 0.1
        features.append(np.concatenate([act_vec, tier_boost]))
    return np.array(features)

def cold_eye_feedback_features(colony, vision, images, limit=300):
    """预测误差特征: 对每个节点, 比较高 tier 入边预测激活 vs 实际激活"""
    global _avg_boosted
    features = []
    node_ids = sorted(vision.graph.nodes.keys())
    total_error_sum = 0
    for i in range(min(limit, len(images))):
        img = images[i].reshape(28, 28).copy()
        vision.set_image(img)

        # 实际激活
        act_vec = np.array([vision.graph.nodes[nid].activation for nid in node_ids], dtype=np.float32)

        # 预测误差 (不修改激活值)
        error_vec = vision.predictive_boost(colony, strength=0.5)
        total_error_sum += float(np.mean(np.abs(error_vec)))

        # Tier 权重
        tier_boost = np.zeros(len(node_ids), dtype=np.float32)
        for j, nid in enumerate(node_ids):
            c2 = c3 = 0
            for (src, dst), tier in colony.synapse.tiers.items():
                if src == nid or dst == nid:
                    if tier == 2: c2 += 1
                    elif tier == 3: c3 += 1
            tier_boost[j] = c3 * 0.3 + c2 * 0.1

        # 特征 = 激活 + 预测误差 + tier权重
        feat = np.concatenate([act_vec, error_vec, tier_boost])
        features.append(feat)

    _avg_boosted = total_error_sum / max(1, limit)
    return np.array(features)


# ═══ 主测试 ═══

def main():
    print("=" * 60)
    print("冷眼 + Colony + 预测反馈 vs 传统算法")
    print("=" * 60)

    print("\n[1] Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist(limit=4000)
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    print("\n[2] Traditional baselines...")
    results = {}

    t0 = time.time()
    # 纯 numpy KNN 较慢, 只用 300 测试样本
    acc = bench_knn(X_train, y_train, X_test[:300], y_test[:300])
    if acc is not None:
        results['KNN (raw pixels)'] = {'c1.0': acc}
        print(f"  KNN:  {acc*100:.1f}%  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    acc = bench_svm(X_train, y_train, X_test, y_test)
    if acc is not None:
        results['SVM (linear)'] = {'c1.0': acc}
        print(f"  SVM:  {acc*100:.1f}%  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    acc = bench_mlp(X_train, y_train, X_test, y_test)
    if acc is not None:
        results['MLP (128-64)'] = {'c1.0': acc}
        print(f"  MLP:  {acc*100:.1f}%  ({time.time()-t0:.1f}s)")

    print("\n[3] ColdEye baseline...")
    t0 = time.time()
    ce_graph, ce_vision = build_cold_eye(
        X_train.reshape(-1, 28, 28), y_train, n_nodes=80, n_train=2000)
    print(f"  graph: {sum(len(v) for v in ce_graph.adjacency.values())} KG edges ({time.time()-t0:.1f}s)")

    # Use pure numpy KNN
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
    KNeighborsClassifier = SimpleKNN
    X_ce_train = cold_eye_features(ce_vision, X_train.reshape(-1, 28, 28), limit=2000)
    X_ce_test = cold_eye_features(ce_vision, X_test.reshape(-1, 28, 28), limit=300)

    clf = KNeighborsClassifier(k=5)
    clf.fit(X_ce_train, y_train[:2000])
    acc = clf.score(X_ce_test, y_test[:300])
    results['ColdEye (激活)'] = {'c1.0': acc}
    print(f"  ColdEye:  {acc*100:.1f}%  ({time.time()-t0:.1f}s total)")

    print("\n[4] ColdEye + Colony...")
    t0 = time.time()
    from vision_colony import VisionColony
    colony = VisionColony(ce_graph)
    colony.seed_cells(n_per_node=3, max_cells=200)
    print(f"  running colony (100 gens)...")
    colony.breathe(n_generations=100, verbose=False)
    stats = colony.stats()
    print(f"  t2={stats['t2']} t3={stats['t3']} multi={stats['multi_pct']:.0f}% ({time.time()-t0:.1f}s)")

    X_col_train = cold_eye_colony_features(colony, ce_vision, X_train.reshape(-1, 28, 28), limit=2000)
    X_col_test = cold_eye_colony_features(colony, ce_vision, X_test.reshape(-1, 28, 28), limit=300)
    clf2 = KNeighborsClassifier(k=5)
    clf2.fit(X_col_train, y_train[:2000])
    acc = clf2.score(X_col_test, y_test[:300])
    results['ColdEye+Colony'] = {'c1.0': acc}
    print(f"  Colony:  {acc*100:.1f}%")

    X_fb_train = cold_eye_feedback_features(colony, ce_vision, X_train.reshape(-1, 28, 28), limit=500)
    X_fb_test = cold_eye_feedback_features(colony, ce_vision, X_test.reshape(-1, 28, 28), limit=300)
    clf_fb = KNeighborsClassifier(k=5)
    clf_fb.fit(X_fb_train, y_train[:500])
    acc = clf_fb.score(X_fb_test, y_test[:300])
    results['Colony+Feedback'] = {'c1.0': acc}
    print(f"  Colony+FB:  {acc*100:.1f}%  (avg error {_avg_boosted:.3f})")

    print("\n[5] Low-contrast + predictive feedback...")
    for c in [0.5, 0.3, 0.2]:
        X_low = low_contrast(X_test, c).reshape(-1, 28, 28)
        X_ce_low = cold_eye_features(ce_vision, X_low, limit=300)
        X_col_low = cold_eye_colony_features(colony, ce_vision, X_low, limit=300)
        X_fb_low = cold_eye_feedback_features(colony, ce_vision, X_low, limit=300)

        acc_ce = clf.score(X_ce_low, y_test[:300])
        acc_col = clf2.score(X_col_low, y_test[:300])
        acc_fb = clf_fb.score(X_fb_low, y_test[:300])

        results['ColdEye (激活)'][f'c{c}'] = acc_ce
        results['ColdEye+Colony'][f'c{c}'] = acc_col
        results['Colony+Feedback'][f'c{c}'] = acc_fb
        print(f"  c={c:.1f}:  ColdEye={acc_ce*100:.1f}%  Colony={acc_col*100:.1f}%  "
              f"Colony+FB={acc_fb*100:.1f}%  (err {_avg_boosted:.3f})")

    # ═══ 总结 ═══
    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)
    header = f"{'Method':<22s}"
    for cl in ['c1.0', 'c0.5', 'c0.3', 'c0.2']:
        header += f" {cl:>7s}"
    print(header)
    print("-" * 54)
    for name in ['KNN (raw pixels)', 'SVM (linear)', 'MLP (128-64)',
                 'ColdEye (激活)', 'ColdEye+Colony', 'Colony+Feedback']:
        scores = results.get(name)
        if scores is None:
            continue
        row = f"{name:<22s}"
        for cl in ['c1.0', 'c0.5', 'c0.3', 'c0.2']:
            val = scores.get(cl, '-')
            if isinstance(val, float):
                row += f" {val*100:>6.1f}%"
            else:
                row += f" {'-':>6s}"
        print(row)

    print("\n=== DONE ===")


if __name__ == '__main__':
    main()
