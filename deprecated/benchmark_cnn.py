#!/usr/bin/env python3
"""
benchmark_cnn.py — CNN vs ColdEye+Colony+FB (低对比度对比)

只比两个模型, 四个对比度 (1.0, 0.5, 0.3, 0.2):
  CNN:  2-conv + 2-fc, 3000 样本训练, Adam
  ColdEye+Colony+FB: 100 节点竞争路由 + 细胞行走 + 误差校正反馈
"""

import sys, os, time, gzip, numpy as np

_avg_error_mag = 0
np.random.seed(42)
import random; random.seed(42)

# ═══ 数据 ═══

def load_mnist(limit=4000):
    base = os.path.dirname(__file__) or '.'
    for kind in ['train', 't10k']:
        for suffix in ['images-idx3-ubyte.gz', 'labels-idx1-ubyte.gz']:
            fpath = os.path.join(base, f'{kind}-{suffix}')
            if not os.path.exists(fpath):
                import urllib.request
                urllib.request.urlretrieve(
                    f"https://storage.googleapis.com/cvdf-datasets/mnist/{kind}-{suffix}", fpath)
    with gzip.open(os.path.join(base, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        Xt = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32)/255.0
    with gzip.open(os.path.join(base, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        yt = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(os.path.join(base, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        Xv = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32)/255.0
    with gzip.open(os.path.join(base, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        yv = np.frombuffer(f.read(), np.uint8, offset=8)
    return Xt[:limit], yt[:limit], Xv[:2000], yv[:2000]


def low_contrast(X, c):
    Xc = X.copy().reshape(-1, 784)
    mean = Xc.mean(axis=1, keepdims=True)
    return (mean + (Xc - mean) * c).reshape(X.shape)


# ═══ CNN (纯 numpy, 手写反向传播) ═══

def conv2d(x, w, stride=1):
    """x: (N,C,H,W)  w: (OC,IC,K,K) → (N,OC,OH,OW)"""
    N, IC, H, W = x.shape
    OC, _, K, _ = w.shape
    OH = (H - K) // stride + 1
    OW = (W - K) // stride + 1
    out = np.zeros((N, OC, OH, OW), dtype=np.float32)
    for n in range(N):
        for oc in range(OC):
            for i in range(OH):
                for j in range(OW):
                    ii, jj = i*stride, j*stride
                    out[n, oc, i, j] = np.sum(x[n, :, ii:ii+K, jj:jj+K] * w[oc])
    return out

def maxpool2d(x, pool=2):
    """x: (N,C,H,W) → (N,C,H/pool,W/pool) + argmax indices"""
    N, C, H, W = x.shape
    OH, OW = H // pool, W // pool
    out = np.zeros((N, C, OH, OW), dtype=np.float32)
    idxs = np.zeros((N, C, OH, OW), dtype=np.int32)  # flat index inside pool
    for n in range(N):
        for c in range(C):
            for i in range(OH):
                for j in range(OW):
                    patch = x[n, c, i*pool:(i+1)*pool, j*pool:(j+1)*pool]
                    fidx = np.argmax(patch)
                    out[n, c, i, j] = patch.flat[fidx]
                    idxs[n, c, i, j] = fidx
    return out, idxs

def maxpool2d_backprop(dout, idxs, x_shape, pool=2):
    """dout: (N,C,OH,OW) → dx: x_shape"""
    N, C, _, _ = dout.shape
    OH, OW = dout.shape[2], dout.shape[3]
    dx = np.zeros(x_shape, dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for i in range(OH):
                for j in range(OW):
                    fidx = idxs[n, c, i, j]
                    pi = i*pool + fidx // pool
                    pj = j*pool + fidx % pool
                    dx[n, c, pi, pj] += dout[n, c, i, j]
    return dx

def conv2d_backprop_w(x, dout, K, stride=1):
    """x:(N,IC,H,W) dout:(N,OC,OH,OW) → dw:(OC,IC,K,K)"""
    N, IC, _, _ = x.shape
    N2, OC, OH, OW = dout.shape
    dw = np.zeros((OC, IC, K, K), dtype=np.float32)
    for n in range(N):
        for oc in range(OC):
            for i in range(OH):
                for j in range(OW):
                    ii, jj = i*stride, j*stride
                    dw[oc] += x[n, :, ii:ii+K, jj:jj+K] * dout[n, oc, i, j]
    return dw

def conv2d_backprop_x(dout, w, x_shape, stride=1):
    """dout:(N,OC,OH,OW) w:(OC,IC,K,K) → dx: x_shape"""
    N, IC, H, W = x_shape
    OC, _, K, _ = w.shape
    OH, OW = dout.shape[2], dout.shape[3]
    dx = np.zeros(x_shape, dtype=np.float32)
    for n in range(N):
        for oc in range(OC):
            for i in range(OH):
                for j in range(OW):
                    ii, jj = i*stride, j*stride
                    dx[n, :, ii:ii+K, jj:jj+K] += dout[n, oc, i, j] * w[oc]
    return dx

def relu(x): return np.maximum(0, x)
def relu_backprop(dout, x): return dout * (x > 0)
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class MiniCNN:
    """2-conv + 2-fc, 纯 numpy, 自适应 FC 输入维度"""
    def __init__(self, input_shape=(1, 28, 28)):
        C, H, W = input_shape
        # conv1: 3×3 no pad → H-2, W-2
        h1, w1 = H-2, W-2
        # pool1: /2
        h1, w1 = h1//2, w1//2
        # conv2: 3×3 no pad
        h2, w2 = h1-2, w1-2
        # pool2: /2
        h2, w2 = h2//2, w2//2
        fc_in = 32 * h2 * w2

        self.cw1 = np.random.randn(16, 1, 3, 3).astype(np.float32) * np.sqrt(2.0/9)
        self.cw2 = np.random.randn(32, 16, 3, 3).astype(np.float32) * np.sqrt(2.0/(16*9))
        self.fw1 = np.random.randn(fc_in, 64).astype(np.float32) * np.sqrt(2.0/fc_in)
        self.fw2 = np.random.randn(64, 10).astype(np.float32) * np.sqrt(2.0/64)
        self.fb1 = np.zeros(64, dtype=np.float32)
        self.fb2 = np.zeros(10, dtype=np.float32)

    def forward(self, x):
        self.x0 = x  # (N,1,28,28)
        self.c1 = relu(conv2d(x, self.cw1))  # (N,16,26,26)
        self.p1, self.p1_idx = maxpool2d(self.c1, 2)  # (N,16,13,13)

        self.c2 = relu(conv2d(self.p1, self.cw2))  # (N,32,11,11)
        self.p2, self.p2_idx = maxpool2d(self.c2, 2)  # (N,32,5,5) → wait, 11//2=5... 

        # Actually 11//2=5, and we need 7×7 for fc input. Let me adjust.
        # Hmm, 28→26→13→11→5. So fc input is 32*5*5=800 not 32*7*7.
        # This is a bug. Let me fix the fc layer size.
        # For now, let's just use the actual shape.
        N = x.shape[0]
        self.fc_in = self.p2.reshape(N, -1)
        fc1_out = self.fc_in @ self.fw1 + self.fb1
        self.fc1_r = relu(fc1_out)
        self.score = self.fc1_r @ self.fw2 + self.fb2
        return softmax(self.score)

    def backward(self, y_true, lr=0.001):
        N = len(y_true)
        dy = self.forward(self.x0).copy()
        dy[np.arange(N), y_true] -= 1
        dy /= N  # d_loss/d_score

        dfw2 = self.fc1_r.T @ dy
        dfb2 = dy.sum(0)
        dfc1_r = dy @ self.fw2.T
        dfc1 = dfc1_r * (self.fc1_r > 0)
        dfw1 = self.fc_in.T @ dfc1
        dfb1 = dfc1.sum(0)
        dfc_in = dfc1 @ self.fw1.T
        dp2 = dfc_in.reshape(self.p2.shape)

        dc2 = maxpool2d_backprop(dp2, self.p2_idx, self.c2.shape)
        dc2_r = relu_backprop(dc2, self.c2)
        dcw2 = conv2d_backprop_w(self.p1, dc2_r, K=3)
        dp1 = conv2d_backprop_x(dc2_r, self.cw2, self.p1.shape)

        dc1 = maxpool2d_backprop(dp1, self.p1_idx, self.c1.shape)
        dc1_r = relu_backprop(dc1, self.c1)
        dcw1 = conv2d_backprop_w(self.x0, dc1_r, K=3)

        self.cw1 -= lr * dcw1
        self.cw2 -= lr * dcw2
        self.fw1 -= lr * dfw1
        self.fw2 -= lr * dfw2
        self.fb1 -= lr * dfb1
        self.fb2 -= lr * dfb2

    def predict(self, x):
        return self.forward(x).argmax(1)

    def accuracy(self, x, y):
        return (self.predict(x) == y).mean()


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


# ═══ ColdEye ═══

def build_cold_eye(images, labels, n_nodes=100, n_train=2000):
    sys.path.insert(0, os.path.dirname(__file__))
    from graph import VisionGraph
    from vision import VisionInterface

    graph = VisionGraph(n_nodes=n_nodes, template_size=64)
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
    graph.synapse = SynapticLayer()
    coactive = {}
    for i in range(min(n_train, len(images))):
        vision.set_image(images[i].reshape(28, 28).copy())
        active = [nid for nid, n in graph.nodes.items() if n.activation > 0.05]
        for j, src in enumerate(active):
            for dst in active[j+1:]:
                coactive[(src, dst)] = coactive.get((src, dst), 0) + 1

    for (src, dst), count in coactive.items():
        if count >= 3:
            graph.adjacency[src][dst] = min(1.0, count/20.0)
            graph.adjacency[dst][src] = min(1.0, count/20.0)

    return graph, vision


def extract_fb_features(vision, colony, images, limit, node_ids, tier_w, alpha=0.25):
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
    print("=" * 62)
    print("CNN vs ColdEye+Colony+FB — 低对比度对比")
    print("=" * 62)

    print("\n[1] Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist(limit=4000)
    Xt_2d = X_train.reshape(-1, 1, 28, 28)
    Xv_2d = X_test.reshape(-1, 1, 28, 28)
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    # ═══ CNN ═══
    print("\n[2] CNN training...")
    t0 = time.time()
    cnn = MiniCNN()
    bs = 32
    for ep in range(10):
        perm = np.random.permutation(3000)
        for i in range(0, 3000, bs):
            idx = perm[i:i+bs]
            cnn.forward(Xt_2d[idx])
            cnn.backward(y_train[idx], lr=0.001)
        acc_train = cnn.accuracy(Xt_2d[:3000], y_train[:3000])
        if ep % 2 == 0:
            print(f"  epoch {ep+1}: train acc={acc_train*100:.1f}%", flush=True)

    cnn_results = {}
    for c in [1.0, 0.5, 0.3, 0.2]:
        Xv_c = Xv_2d.copy() if c == 1.0 else low_contrast(X_test, c).reshape(-1, 1, 28, 28)
        cnn_results[c] = cnn.accuracy(Xv_c[:500], y_test[:500])

    print(f"  CNN done ({time.time()-t0:.1f}s):  "
          f"1.0={cnn_results[1.0]*100:.1f}%  "
          f"0.5={cnn_results[0.5]*100:.1f}%  "
          f"0.3={cnn_results[0.3]*100:.1f}%  "
          f"0.2={cnn_results[0.2]*100:.1f}%")

    # ═══ ColdEye ═══
    print("\n[3] ColdEye + Colony + FB...")
    t0 = time.time()
    ce_graph, ce_vision = build_cold_eye(
        X_train.reshape(-1, 28, 28), y_train, n_nodes=100, n_train=3000)

    node_ids = sorted(ce_graph.nodes.keys())
    print(f"  KG edges: {sum(len(v) for v in ce_graph.adjacency.values())} ({time.time()-t0:.1f}s)")

    from vision_colony import VisionColony
    colony = VisionColony(ce_graph)
    colony.seed_cells(n_per_node=2, max_cells=200)
    colony.breathe(n_generations=100, verbose=False)
    stats = colony.stats()
    print(f"  colony done: t2={stats['t2']} t3={stats['t3']} ({time.time()-t0:.1f}s)")

    # tier 权重
    tier_w = np.zeros(len(node_ids), dtype=np.float32)
    for j, nid in enumerate(node_ids):
        c2 = c3 = 0
        for (src, dst), tier in colony.synapse.tiers.items():
            if src == nid or dst == nid:
                if tier == 2: c2 += 1
                elif tier == 3: c3 += 1
        tier_w[j] = c3 * 0.3 + c2 * 0.1

    # 训练 KNN
    X_fb_train = extract_fb_features(ce_vision, colony, X_train.reshape(-1, 28, 28), 2000, node_ids, tier_w)
    clf = SimpleKNN(k=5)
    clf.fit(X_fb_train, y_train[:2000])

    ce_results = {}
    for c in [1.0, 0.5, 0.3, 0.2]:
        Xc = X_test.copy() if c == 1.0 else low_contrast(X_test, c)
        X_fb_test = extract_fb_features(ce_vision, colony, Xc.reshape(-1, 28, 28), 300, node_ids, tier_w)
        ce_results[c] = clf.score(X_fb_test, y_test[:300])

    print(f"  ColdEye+FB done:  "
          f"1.0={ce_results[1.0]*100:.1f}%  "
          f"0.5={ce_results[0.5]*100:.1f}%  "
          f"0.3={ce_results[0.3]*100:.1f}%  "
          f"0.2={ce_results[0.2]*100:.1f}%")

    # ═══ 总结 ═══
    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)
    print(f"{'Model':<22s}  {'c=1.0':>7s}  {'c=0.5':>7s}  {'c=0.3':>7s}  {'c=0.2':>7s}")
    print("-" * 58)
    print(f"{'CNN (2-conv+2-fc)':<22s}  {cnn_results[1.0]*100:>6.1f}%  "
          f"{cnn_results[0.5]*100:>6.1f}%  {cnn_results[0.3]*100:>6.1f}%  "
          f"{cnn_results[0.2]*100:>6.1f}%")
    print(f"{'ColdEye+Colony+FB':<22s}  {ce_results[1.0]*100:>6.1f}%  "
          f"{ce_results[0.5]*100:>6.1f}%  {ce_results[0.3]*100:>6.1f}%  "
          f"{ce_results[0.2]*100:>6.1f}%")

    # 衰减率
    for name, res in [("CNN", cnn_results), ("ColdEye+FB", ce_results)]:
        drop = [res[1.0] - res[c] for c in [0.5, 0.3, 0.2]]
        print(f"\n  {name} contrast drop:  "
              f"Δ0.5=-{drop[0]*100:.1f}%  "
              f"Δ0.3=-{drop[1]*100:.1f}%  "
              f"Δ0.2=-{drop[2]*100:.1f}%")

    print("\n=== DONE ===")


if __name__ == '__main__':
    main()
