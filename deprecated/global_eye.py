#!/usr/bin/env python3
"""
global_eye.py — 全局模板冷眼 v2

Architecture:
  每个节点存 28×28 模板 (不是 8×8 patch)
  滑动窗口匹配 → 找最佳位置 → 取 max 相似度作为激活
  竞争路由 Top-K → Hebbian 模板更新
  → KNN 分类

vs 老冷眼: patch 8×8 → 模板 28×28, 信号量 12x 提升
"""

import gzip, numpy as np, time
np.random.seed(42)

def load_mnist():
    base = '/home/duyw/predictive-vision'
    import os
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

def low_contrast(X,c):
    Xc=X.copy(); m=Xc.mean(axis=(1,2),keepdims=True); return m+(Xc-m)*c

class GlobalNode:
    def __init__(self, nid, template_size=28):
        self.id = nid
        self.template = np.random.randn(template_size, template_size).astype(np.float32)
        self.template /= np.linalg.norm(self.template) + 1e-8
        self.activation = 0.0
        self.match_pos = (0,0)  # best match position
        self.match_val = 0.0
        self.win_count = 0
        self.class_votes = np.zeros(10)

class GlobalEye:
    def __init__(self, n_nodes=200, template_size=12, stride=2):
        """
        template_size: 滑动窗口大小 (12×12 能跑, 28×28 太慢)
        stride: 滑动步长
        """
        self.nodes = {i: GlobalNode(i, template_size) for i in range(n_nodes)}
        self.n_nodes = n_nodes
        self.template_size = template_size
        self.stride = stride
        
        # 预计算所有窗口位置
        self.positions = []
        for y in range(0, 28 - template_size + 1, stride):
            for x in range(0, 28 - template_size + 1, stride):
                self.positions.append((y, x))
        self.n_positions = len(self.positions)
    
    def init_templates(self, images, n_samples=200):
        """从 random crops 初始化模板"""
        nids = list(self.nodes.keys())
        for i, nid in enumerate(nids):
            img = images[i % len(images)]
            y = np.random.randint(0, 28 - self.template_size)
            x = np.random.randint(0, 28 - self.template_size)
            patch = img[y:y+self.template_size, x:x+self.template_size].copy()
            self.nodes[nid].template = patch.astype(np.float32)
            self.nodes[nid].template /= np.linalg.norm(self.nodes[nid].template) + 1e-8
    
    def route(self, image, top_k=5):
        """全局滑动窗口匹配 → Top-K 竞争路由"""
        H, W = image.shape
        
        # 预提取所有窗口
        patches = np.zeros((self.n_positions, self.template_size*self.template_size), dtype=np.float32)
        for pi, (y, x) in enumerate(self.positions):
            patches[pi] = image[y:y+self.template_size, x:x+self.template_size].ravel()
        
        nids = list(self.nodes.keys())
        templates = np.array([self.nodes[nid].template.ravel() for nid in nids], dtype=np.float32)
        
        # 计算所有节点 × 所有位置的相似度矩阵
        # scores[节点, 位置] = template @ patch
        scores = templates @ patches.T  # (N_nodes, N_positions)
        
        # 每个节点取最大匹配位置
        best_scores = scores.max(axis=1)  # (N_nodes,)
        best_pos = scores.argmax(axis=1)  # (N_nodes,)
        
        # 竞争: 每个位置只给最高分的节点激活
        pos_best_node = np.argmax(scores, axis=0)  # 每个位置的最佳节点
        pos_best_score = scores.max(axis=0)
        
        # Top-K: K 个最高分的 (位置→节点) 配对
        flat_indices = np.argpartition(-pos_best_score, min(top_k, len(pos_best_score)))[:top_k]
        
        # 重置激活
        for nid in nids:
            self.nodes[nid].activation = 0.0
        
        # 激活最好的 K 个位置上对应的节点
        for idx in flat_indices:
            nid = int(pos_best_node[idx])
            score_val = float(pos_best_score[idx])
            y, x = self.positions[idx]
            self.nodes[nid].activation += score_val  # 累加（一个节点可能在不同位置赢）
            self.nodes[nid].match_pos = (y, x)
            self.nodes[nid].match_val = score_val
        
        # Cap activation
        for nid in nids:
            self.nodes[nid].activation = min(1.0, self.nodes[nid].activation)
    
    def update_hebbian(self, image, lr=0.05):
        """Hebbian: 激活的节点模板向 winning patch 移动"""
        for nid, node in self.nodes.items():
            if node.activation > 0.01:
                y, x = node.match_pos
                patch = image[y:y+self.template_size, x:x+self.template_size].copy()
                node.template += lr * (patch - node.template)
                node.template /= np.linalg.norm(node.template) + 1e-8
                node.win_count += 1
    
    def get_activation_vector(self):
        nids = sorted(self.nodes.keys())
        return np.array([self.nodes[nid].activation for nid in nids], dtype=np.float32)

# ═══ 主测试 ═══
print("="*55)
print("GlobalEye v2 — 全图滑动窗口模板匹配")
print("="*55)

Xt, yt, Xv, yv = load_mnist()

# 配置: 12×12 窗口 stride 3 → 约 36 个位置 × 200 节点
eye = GlobalEye(n_nodes=200, template_size=12, stride=3)

print(f"\n[1] {eye.n_nodes} nodes, {eye.template_size}×{eye.template_size} templates")
print(f"    {eye.n_positions} sliding positions (stride {eye.stride})")
eye.init_templates(Xt, n_samples=200)

print("\n[2] Hebbian training (10K, 3 epochs)...")
t0 = time.time()
for ep in range(3):
    for i in range(10000):
        img = Xt[i].copy()
        if np.random.random() < 0.5:
            m = img.mean(); c = 0.3 + np.random.random() * 0.7
            img = m + (img - m) * c
        eye.route(img, top_k=10)
        eye.update_hebbian(img, lr=0.05)
    print(f"  epoch {ep+1} done ({time.time()-t0:.1f}s)")

# Node specialization check
nids = sorted(eye.nodes.keys())
win_counts = [eye.nodes[nid].win_count for nid in nids]
print(f"  nodes: median wins={np.median(win_counts):.0f}, min={np.min(win_counts)}, max={np.max(win_counts)}")

# ═══ Feature extraction ═══
print("\n[3] Extracting features...")
t0 = time.time()
n_train = 2000
X_train_feat = np.zeros((n_train, eye.n_nodes), dtype=np.float32)
for i in range(n_train):
    eye.route(Xt[i], top_k=10)
    X_train_feat[i] = eye.get_activation_vector()
print(f"  {n_train} train ({time.time()-t0:.1f}s)")

# KNN
class KNN:
    def __init__(s, k=5): s.k = k
    def fit(s, X, y): s.X, s.y = X, y
    def score(s, X, y):
        c = 0
        for i in range(len(X)):
            d = np.sum((s.X - X[i])**2, axis=1)
            nn = np.argpartition(d, s.k)[:s.k]
            if np.bincount(s.y[nn].astype(int)).argmax() == y[i]: c += 1
        return c / len(X)

knn = KNN(k=5)
knn.fit(X_train_feat, yt[:n_train])

# ═══ Test at all contrasts ═══
print("\n[4] Testing...")
n_test = 500
for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]:
    Xc = Xv[:n_test] if c == 1.0 else low_contrast(Xv[:n_test], c)
    X_test_feat = np.zeros((n_test, eye.n_nodes), dtype=np.float32)
    for i in range(n_test):
        eye.route(Xc[i], top_k=10)
        X_test_feat[i] = eye.get_activation_vector()
    acc = knn.score(X_test_feat, yv[:n_test])
    print(f"  c={c:.2f}: {acc*100:.1f}%")

print("\n=== DONE ===")
