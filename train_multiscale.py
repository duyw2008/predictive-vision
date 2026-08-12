#!/usr/bin/env python3
"""
冷眼 v3 — 可配置多尺度预测反馈视觉系统
架构: 全局眼(28×28) + 任意数量 patch 眼 → 激活拼接 → KNN
默认: global(100n) + coarse 16×16(50n) = 150d
"""

import sys, os, time, numpy as np, gzip, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from graph import VisionGraph
from vision import VisionInterface


# ═══════════════════════════════════════════
#  Global Eye: 整图竞争路由 (不用 VisionGraph)
# ═══════════════════════════════════════════

class GlobalEye:
    """28×28 整图 → 竞争路由 → 激活向量。信号积分 784 维，低对比度鲁棒。"""

    def __init__(self, n_nodes=100, seed=None):
        self.n = n_nodes
        self.templates = np.empty((n_nodes, 784), dtype=np.float32)
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def init_templates(self, images):
        idxs = self.rng.choice(len(images), min(200, len(images)), replace=False)
        patches = images[idxs].reshape(len(idxs), -1).astype(np.float32)
        patches = patches - patches.mean(axis=1, keepdims=True)
        for i in range(self.n):
            self.templates[i] = patches[i % len(patches)]
            self.templates[i] /= np.linalg.norm(self.templates[i]) + 1e-8

    def train(self, images, epochs=3, n_train=None, contrast_aug=True, conscience_beta=0.0):
        if n_train is None: n_train = len(images)
        lr = 0.1
        rng = self.rng
        # 良心机制: 节点赢的频率 → 抑制常胜节点，促进生态位分化
        freq = np.zeros(self.n, dtype=np.float32) if conscience_beta > 0 else None
        for ep in range(epochs):
            for idx in rng.permutation(min(n_train, len(images))):
                img = images[idx].copy()
                if contrast_aug and rng.random() < 0.5:
                    m = img.mean()
                    img = m + (img - m) * (0.3 + rng.random() * 0.7)
                flat = img.reshape(-1).astype(np.float32)
                flat = flat - flat.mean()
                flat /= np.linalg.norm(flat) + 1e-8
                scores = self.templates @ flat
                if freq is not None:
                    scores = scores - conscience_beta * freq
                best = int(np.argmax(scores))
                self.templates[best] += lr * (flat - self.templates[best])
                self.templates[best] /= np.linalg.norm(self.templates[best]) + 1e-8
                if freq is not None:
                    freq[best] += 1.0
                    freq *= 0.999  # 缓慢衰减

    def activate(self, images):
        N = len(images)
        flat = images.reshape(N, -1).astype(np.float32)
        flat = flat - flat.mean(axis=1, keepdims=True)
        flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
        return np.clip(flat @ self.templates.T, 0, 1).astype(np.float32)

    def activate_one(self, image):
        flat = image.reshape(-1).astype(np.float32)
        flat = flat - flat.mean()
        flat /= np.linalg.norm(flat) + 1e-8
        return np.clip(self.templates @ flat, 0, 1).astype(np.float32)


# ═══════════════════════════════════════════
#  Patch Eye: VisionGraph 竞争路由 (任意尺度)
# ═══════════════════════════════════════════

class PatchEye:
    """patch 尺度竞争路由，封装 VisionGraph + VisionInterface"""

    def __init__(self, patch_size, stride, n_nodes, seed=None):
        ts = patch_size * patch_size
        self.g = VisionGraph(n_nodes=n_nodes, template_size=ts)
        self.v = VisionInterface(self.g, patch_size=patch_size, stride=stride)
        self.nids = sorted(self.g.nodes.keys())
        self.ps = patch_size
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def init_templates(self, images):
        rng = self.rng
        idxs = rng.choice(len(images), min(200, len(images)), replace=False)
        ap = [p.astype(np.float32) for i in idxs for p in self.v.extractor.extract(images[i] - images[i].mean())]
        for k, nid in enumerate(self.nids):
            p = ap[k % len(ap)]; p /= np.linalg.norm(p) + 1e-8
            self.g.nodes[nid].template = p

    def train(self, images, epochs=3, n_train=None, contrast_aug=True, conscience_beta=0.0):
        if n_train is None: n_train = len(images)
        lr = 0.1
        rng = self.rng
        n_nodes = len(self.nids)
        nid_to_idx = {nid: i for i, nid in enumerate(self.nids)}
        freq = np.zeros(n_nodes, dtype=np.float32) if conscience_beta > 0 else None
        for ep in range(epochs):
            for idx in rng.permutation(min(n_train, len(images))):
                img = images[idx].copy()
                if contrast_aug and rng.random() < 0.5:
                    m = img.mean()
                    img = m + (img - m) * (0.3 + rng.random() * 0.7)
                img = img - img.mean()
                self.v.set_image(img)
                if freq is not None:
                    best_nid = self.nids[0]; best_score = -float('inf')
                    for nid in self.nids:
                        score = self.g.nodes[nid].activation - conscience_beta * freq[nid_to_idx[nid]]
                        if score > best_score: best_score, best_nid = score, nid
                else:
                    best_nid = max(self.nids, key=lambda n: self.g.nodes[n].activation)
                node = self.g.nodes[best_nid]
                t = np.mean(self.v.node_assignments.get(best_nid, [self.v.extractor.extract(img)[0]]), axis=0)
                n = np.linalg.norm(t); t = t/(n+1e-8) if n>0 else t
                node.template += lr*(t-node.template)
                node.template /= np.linalg.norm(node.template)+1e-8
                if freq is not None:
                    freq[nid_to_idx[best_nid]] += 1.0
                    freq *= 0.999

    def activate(self, images):
        feats = np.zeros((len(images), len(self.nids)), dtype=np.float32)
        for i, img in enumerate(images):
            # zero-mean before patch extraction → approximate per-patch centering
            img_c = img - img.mean()
            self.v.set_image(img_c)
            feats[i] = np.array([self.g.nodes[nid].activation for nid in self.nids], dtype=np.float32)
        return feats

    def activate_one(self, image):
        img_c = image - image.mean()
        self.v.set_image(img_c)
        return np.array([self.g.nodes[nid].activation for nid in self.nids], dtype=np.float32)


# ═══════════════════════════════════════════
#  ColdEye v3
# ═══════════════════════════════════════════

class ColdEye:
    """可配置多尺度预测反馈系统。

    eye_specs 格式:
      [{"type": "global", "n": 100},
       {"type": "patch", "ps": 16, "st": 8, "n": 50}]

    默认 v3: global(100) + coarse 16×16(50) = 150d
    """

    DEFAULT_SPECS = [
        {"type": "global", "n": 100},
        {"type": "patch", "ps": 16, "st": 8, "n": 50},
    ]

    def __init__(self, eye_specs=None, k_knn=5, seed=42):
        specs = eye_specs or self.DEFAULT_SPECS
        self.eyes = []
        self.seed = seed
        rng = np.random.RandomState(seed)
        for i, s in enumerate(specs):
            eye_seed = rng.randint(0, 2**31)
            if s["type"] == "global":
                self.eyes.append(GlobalEye(n_nodes=s["n"], seed=eye_seed))
            elif s["type"] == "patch":
                self.eyes.append(PatchEye(s["ps"], s["st"], s["n"], seed=eye_seed))
            else:
                raise ValueError(f"Unknown eye type: {s['type']}")

        self.k = k_knn
        self.memory = []   # [(vec, label), ...]
        self.shape_centers = None   # optional K-means layer
        self.object_centers = None
        self.dim = sum(e.n if isinstance(e, GlobalEye) else len(e.nids) for e in self.eyes)

    # ── training ──

    def init_templates(self, images):
        for eye in self.eyes: eye.init_templates(images)

    def train(self, images, labels, epochs=3, n_train=None, contrast_aug=True, conscience_beta=0.0):
        n = n_train or len(images)
        for i, eye in enumerate(self.eyes):
            t0 = time.time()
            eye.train(images, epochs=epochs, n_train=n, contrast_aug=contrast_aug,
                      conscience_beta=conscience_beta)
            t = eye.n if isinstance(eye, GlobalEye) else len(eye.nids)
            print(f"  eye[{i}]: {type(eye).__name__} {t}d — {time.time()-t0:.0f}s")

    # ── feature extraction ──

    def _activate_one(self, image):
        parts = [eye.activate_one(image) for eye in self.eyes]
        return np.concatenate(parts)

    def _activate_batch(self, images):
        parts = [eye.activate(images) for eye in self.eyes]
        return np.concatenate(parts, axis=1)

    # ── optional K-means layers (legacy, not needed for v3) ──

    def build_shapes(self, images, labels, n_samples=5000, n_shape=100):
        """K-means on activation vectors → shape layer"""
        n = min(n_samples, len(images))
        idxs = np.random.choice(len(images), n, replace=False)
        X = self._activate_batch(images[idxs])
        rng = np.random.RandomState(42)
        idx = rng.choice(n, n_shape, replace=False)
        centers = X[idx].copy()
        for it in range(50):
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels_k = np.argmin(dists, axis=1)
            new_c = np.zeros_like(centers)
            for c in range(n_shape):
                mask = labels_k == c
                new_c[c] = X[mask].mean(axis=0) if mask.sum() > 0 else X[rng.choice(n)]
            if np.sum((centers - new_c)**2) < 1e-6: break
            centers = new_c
        self.shape_centers = centers.astype(np.float32)

    def build_objects(self, images, labels, n_samples=5000, n_object=50):
        """K-means on shape vectors → object layer"""
        if self.shape_centers is None: raise RuntimeError("build_shapes() first")
        n = min(n_samples, len(images))
        idxs = np.random.choice(len(images), n, replace=False)
        shape_vecs = np.array([self.get_shape_vector(images[i]) for i in idxs], dtype=np.float32)
        rng = np.random.RandomState(42)
        idx = rng.choice(n, n_object, replace=False)
        centers = shape_vecs[idx].copy()
        for it in range(50):
            dists = np.sum((shape_vecs[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels_k = np.argmin(dists, axis=1)
            new_c = np.zeros_like(centers)
            for c in range(n_object):
                mask = labels_k == c
                new_c[c] = shape_vecs[mask].mean(axis=0) if mask.sum() > 0 else shape_vecs[rng.choice(n)]
            if np.sum((centers - new_c)**2) < 1e-6: break
            centers = new_c
        self.object_centers = centers.astype(np.float32)

    def get_shape_vector(self, image):
        if self.shape_centers is None: raise RuntimeError("build_shapes() first")
        base = self._activate_one(image)
        s = self.shape_centers @ base / (np.linalg.norm(self.shape_centers, axis=1)*np.linalg.norm(base) + 1e-8)
        return np.clip(s, 0, 1).astype(np.float32)

    def get_object_vector(self, image):
        if self.shape_centers is None or self.object_centers is None:
            raise RuntimeError("build_shapes() + build_objects() first")
        sv = self.get_shape_vector(image)
        s = self.object_centers @ sv / (np.linalg.norm(self.object_centers, axis=1)*np.linalg.norm(sv) + 1e-8)
        return np.clip(s, 0, 1).astype(np.float32)

    # ── memory / prediction ──

    def build_memory(self, images, labels, size=5000, use_shapes=False):
        """存 KNN 参考向量。use_shapes=True 走 object_vector, False 走原始激活。"""
        self.memory.clear()
        n = min(size, len(images))
        idxs = np.random.choice(len(images), n, replace=False)
        for idx in idxs:
            vec = self.get_object_vector(images[idx]) if use_shapes else self._activate_one(images[idx])
            self.memory.append((vec, labels[idx]))

    def predict(self, image, use_shapes=False):
        """KNN 预测。返回 (label, confidence)。"""
        if not self.memory: return -1, 0.0
        q = self.get_object_vector(image) if use_shapes else self._activate_one(image)
        best_sim, best_label = -1, -1
        for mvec, mlbl in self.memory:
            sim = np.dot(mvec, q) / (np.linalg.norm(mvec)*np.linalg.norm(q) + 1e-8)
            if sim > best_sim: best_sim, best_label = sim, mlbl
        return best_label, best_sim

    def evaluate(self, images, labels, use_shapes=False):
        correct = 0
        for i in range(len(images)):
            pred, _ = self.predict(images[i], use_shapes=use_shapes)
            if pred == labels[i]: correct += 1
        return correct / len(images)

    # ═══ 预测反馈 + 脑补 ═══

    def predict_with_feedback(self, image, alpha=0.3, max_iter=5, tol=0.01):
        """迭代闭环反馈: 猜类→找最相似同类记忆→修正→重猜→收敛"""
        if not self.memory: return -1, 0.0
        act = self._activate_one(image)
        prev_act = act.copy()

        for it in range(max_iter):
            # 1. 找最佳匹配的邻居 (不是类平均，是最相似的具体记忆)
            best_sim, best_vec, best_class = -1, None, -1
            for mvec, mlbl in self.memory:
                sim = np.dot(mvec, act) / (np.linalg.norm(mvec)*np.linalg.norm(act)+1e-8)
                if sim > best_sim:
                    best_sim, best_vec, best_class = sim, mvec, int(mlbl)

            # 2. 只用同类的最相似记忆修正
            same_class = [(sim, mvec) for mvec, mlbl in self.memory
                          if int(mlbl) == best_class]
            if len(same_class) < 2:
                break
            # 加权同类记忆平均 (不只是单一最相似)
            same_class.sort(key=lambda x: x[0], reverse=True)
            top = same_class[:5]
            total_sim = sum(s for s, _ in top) + 1e-8
            act_avg = sum(mvec * (s/total_sim) for s, mvec in top)

            # 3. 融合
            act = act * (1 - alpha) + act_avg * alpha

            # 4. 收敛检查
            if np.max(np.abs(act - prev_act)) < tol:
                break
            prev_act = act.copy()

        return self._knn_search(act)

    def _knn_search(self, vec):
        """KNN 搜索，返回 (label, confidence)"""
        if not self.memory: return -1, 0.0
        best_sim, best_label = -1, -1
        for mvec, mlbl in self.memory:
            sim = np.dot(mvec, vec) / (np.linalg.norm(mvec) * np.linalg.norm(vec) + 1e-8)
            if sim > best_sim: best_sim, best_label = sim, mlbl
        return best_label, best_sim

    def build_decoder(self, images, n_samples=10000):
        """学习线性解码器 W[150,784]: 激活 → 重建图像"""
        n = min(n_samples, len(images))
        idxs = np.random.choice(len(images), n, replace=False)
        acts = self._activate_batch(images[idxs])  # [N, 150]
        flat = images[idxs].reshape(n, -1).astype(np.float32)  # [N, 784]
        # 最小二乘: W = (A^T A)^-1 A^T I → pinv approach
        ATA = acts.T @ acts
        ATI = acts.T @ flat
        self.W = (np.linalg.pinv(ATA) @ ATI).astype(np.float32)  # [150, 784]

    def reconstruct(self, image):
        """脑补: 激活 → 重建。自动适配图像尺寸。"""
        if not hasattr(self, 'W'):
            raise RuntimeError("先 build_decoder()")
        act = self._activate_one(image)
        h, w = image.shape[:2]
        return (act @ self.W).reshape(h, w)

    def evaluate_with_feedback(self, images, labels, alpha=0.5):
        correct = 0
        for i in range(len(images)):
            pred, _ = self.predict_with_feedback(images[i], alpha=alpha)
            if pred == labels[i]: correct += 1
        return correct / len(images)


# ═══════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════

def load_mnist(path="data"):
    fs = {"ti":"train-images-idx3-ubyte.gz","tl":"train-labels-idx1-ubyte.gz",
          "ei":"t10k-images-idx3-ubyte.gz","el":"t10k-labels-idx1-ubyte.gz"}
    os.makedirs(path, exist_ok=True)
    url = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
    for f in fs.values():
        p = os.path.join(path, f)
        if not os.path.exists(p): urllib.request.urlretrieve(url+f, p)

    def limg(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)/255
    def llbl(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return (limg(os.path.join(path, fs["ti"])), llbl(os.path.join(path, fs["tl"])),
            limg(os.path.join(path, fs["ei"])), llbl(os.path.join(path, fs["el"])))

def low_contrast(X, c):
    Xc = X.copy(); m = Xc.mean(axis=(1,2), keepdims=True)
    return m + (Xc - m) * c


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    X_tr, y_tr, X_te, y_te = load_mnist()

    print(f"ColdEye v3 — global(100n) + coarse 16×16(50n) = 150d")
    print(f"  训练: 60K, 5ep, contrast_aug=True")

    model = ColdEye()
    model.init_templates(X_tr[:5000])
    model.train(X_tr, y_tr, epochs=5, n_train=60000, contrast_aug=True)
    model.build_memory(X_tr[:15000], y_tr[:15000], size=5000)

    n_test = 500
    print(f"\n  {'c':>6s}  {'acc':>7s}")
    print(f"  {'-'*15}")
    for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01]:
        if c == 1.0:
            test_batch = X_te[:n_test]
        else:
            test_batch = low_contrast(X_te[:n_test], c)
        acc = model.evaluate(test_batch, y_te[:n_test])
        print(f"  {c:5.2f}  {acc*100:6.1f}%")

    print(f"\n  架构: {model.dim}d ({' + '.join(f'{type(e).__name__}({e.n if isinstance(e,GlobalEye) else len(e.nids)}n)' for e in model.eyes)})")
    print("=== DONE ===")
