#!/usr/bin/env python3
"""
缺口3-C: 两步推理 — 粗定位 + 精细验证
思路: 16x16先粗筛候选类 → 8x8/4x4只验证ROI区域
对比: two-step vs standard MultiScaleEye at c≤0.1
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

from graph import VisionGraph
from vision import VisionInterface

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

class TwoStepEye:
    """两步推理: 粗尺度定位 + 细尺度验证"""

    def __init__(self):
        # 粗眼: 16x16 (coarse location)
        self.coarse_g = VisionGraph(n_nodes=50, template_size=256)
        self.coarse_v = VisionInterface(self.coarse_g, patch_size=16, stride=8)
        self.coarse_nids = sorted(self.coarse_g.nodes.keys())

        # 细眼: 8x8 (fine verification)
        self.fine_g = VisionGraph(n_nodes=100, template_size=64)
        self.fine_v = VisionInterface(self.fine_g, patch_size=8, stride=4)
        self.fine_nids = sorted(self.fine_g.nodes.keys())

        self.ref_vecs = []   # [(coarse_vec, fine_vec, label), ...]

    def train_coarse(self, images, labels, epochs=3, n_train=10000):
        ts = 256
        idxs = np.random.choice(len(images), min(200, len(images)), replace=False)
        ap = [p.astype(np.float32) for i in idxs for p in self.coarse_v.extractor.extract(images[i])]
        for k, nid in enumerate(self.coarse_nids):
            p = ap[k % len(ap)]
            self.coarse_g.nodes[nid].template = p
            self.coarse_g.nodes[nid].template /= np.linalg.norm(self.coarse_g.nodes[nid].template) + 1e-8

        lr = 0.1
        for ep in range(epochs):
            for idx in np.random.permutation(min(n_train, len(images))):
                self.coarse_v.set_image(images[idx])
                for nid, aps in self.coarse_v.node_assignments.items():
                    node = self.coarse_g.nodes[nid]
                    t = np.mean(aps, axis=0)
                    n = np.linalg.norm(t)
                    if n > 0: t /= n
                    node.template += lr*(t-node.template)
                    node.template /= np.linalg.norm(node.template)+1e-8

    def train_fine(self, images, labels, epochs=3, n_train=10000):
        ts = 64
        idxs = np.random.choice(len(images), min(200, len(images)), replace=False)
        ap = [p.astype(np.float32) for i in idxs for p in self.fine_v.extractor.extract(images[i])]
        for k, nid in enumerate(self.fine_nids):
            p = ap[k % len(ap)]
            self.fine_g.nodes[nid].template = p
            self.fine_g.nodes[nid].template /= np.linalg.norm(self.fine_g.nodes[nid].template) + 1e-8

        lr = 0.1
        for ep in range(epochs):
            for idx in np.random.permutation(min(n_train, len(images))):
                self.fine_v.set_image(images[idx])
                for nid, aps in self.fine_v.node_assignments.items():
                    node = self.fine_g.nodes[nid]
                    t = np.mean(aps, axis=0)
                    n = np.linalg.norm(t)
                    if n > 0: t /= n
                    node.template += lr*(t-node.template)
                    node.template /= np.linalg.norm(node.template)+1e-8

    def build_memory(self, images, labels, n=2000):
        """存参考向量: 粗+细拼接"""
        self.ref_vecs.clear()
        for i in range(min(n, len(images))):
            c_vec, f_vec = self._extract_both(images[i])
            self.ref_vecs.append((c_vec, f_vec, labels[i]))

    def _extract_both(self, image):
        """提取粗+细激活向量"""
        self.coarse_v.set_image(image)
        c_vec = np.array([self.coarse_g.nodes[nid].activation for nid in self.coarse_nids], dtype=np.float32)
        self.fine_v.set_image(image)
        f_vec = np.array([self.fine_g.nodes[nid].activation for nid in self.fine_nids], dtype=np.float32)
        return c_vec, f_vec

    def predict_two_step(self, image, top_k=3):
        """两步推理:
        Step1: 粗尺度筛选 top_k 候选类
        Step2: 细尺度只在候选类中选最佳
        """
        c_vec, f_vec = self._extract_both(image)

        # Step 1: 粗筛选
        coarse_scores = []
        for ref_c, ref_f, ref_label in self.ref_vecs:
            sim = np.dot(c_vec, ref_c) / (np.linalg.norm(c_vec)*np.linalg.norm(ref_c)+1e-8)
            coarse_scores.append((sim, ref_label, ref_f))
        coarse_scores.sort(key=lambda x: x[0], reverse=True)

        # 取 top_k 候选类的所有ref (不只是 top_k 个ref)
        candidates = set()
        for _, lbl, _ in coarse_scores[:top_k*20]:  # 多取一些粗匹配
            candidates.add(lbl)
            if len(candidates) >= top_k:
                break

        # Step 2: 细尺度在候选类中选最佳
        best_sim, best_label = -1, -1
        for ref_c, ref_f, ref_label in self.ref_vecs:
            if ref_label not in candidates:
                continue
            sim = np.dot(f_vec, ref_f) / (np.linalg.norm(f_vec)*np.linalg.norm(ref_f)+1e-8)
            if sim > best_sim:
                best_sim, best_label = sim, ref_label

        return best_label, best_sim

    def predict_baseline(self, image):
        """基线: 粗+细拼接的 KNN (非两步)"""
        c_vec, f_vec = self._extract_both(image)
        combined = np.concatenate([c_vec, f_vec])
        best_sim, best_label = -1, -1
        for ref_c, ref_f, ref_label in self.ref_vecs:
            ref_combined = np.concatenate([ref_c, ref_f])
            sim = np.dot(combined, ref_combined) / (
                np.linalg.norm(combined)*np.linalg.norm(ref_combined)+1e-8)
            if sim > best_sim:
                best_sim, best_label = sim, ref_label
        return best_label, best_sim


np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
n_tr, n_mem, n_te = 10000, 2000, 500

print("训练粗眼 (16x16)...")
t0 = time.time()
eye = TwoStepEye()
eye.train_coarse(X_tr, y_tr, epochs=3, n_train=n_tr)
print(f"  {time.time()-t0:.0f}s")

print("训练细眼 (8x8)...")
t0 = time.time()
eye.train_fine(X_tr, y_tr, epochs=3, n_train=n_tr)
print(f"  {time.time()-t0:.0f}s")

print(f"建记忆 ({n_mem} 样本)...")
t0 = time.time()
eye.build_memory(X_tr, y_tr, n=n_mem)
print(f"  {time.time()-t0:.0f}s")

print(f"\n{'='*60}")
print(f"  Two-step Coarse→Fine vs Baseline ({n_mem} ref, {n_te} test)")
print(f"{'='*60}")

for top_k in [1, 3, 5]:
    print(f"\n  ── Two-step (top_k={top_k} candidates) ──")
    print(f"  {'c':>6s}  {'2-step':>7s}  {'baseline':>8s}  {'Δ':>7s}")
    print(f"  {'-'*30}")
    for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07]:
        if c == 1.0:
            test_batch = X_te[:n_te]
        else:
            test_batch = low_contrast(X_te[:n_te], c)

        correct_2step = correct_base = 0
        for i in range(n_te):
            pred2, _ = eye.predict_two_step(test_batch[i], top_k=top_k)
            if pred2 == y_te[i]: correct_2step += 1
            predb, _ = eye.predict_baseline(test_batch[i])
            if predb == y_te[i]: correct_base += 1

        a2 = correct_2step/n_te
        ab = correct_base/n_te
        print(f"  {c:5.2f}  {a2*100:6.1f}%  {ab*100:7.1f}%  {a2-ab:+6.1%}")

print("\n=== DONE ===")
