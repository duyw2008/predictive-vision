#!/usr/bin/env python3
"""
冷眼 A+B: 多尺度 patch + 中层形状节点
  尺度1: 4×4 stride=4 → 49 patches → 100 节点
  尺度2: 8×8 stride=4 → 36 patches → 100 节点
  尺度3: 16×16 stride=8 → 4 patches → 50 节点
  → 拼接 → 250 维激活向量
  → K-means 聚类 → 形状节点 (中层)
  → NN 分类
"""

import sys, os, time, numpy as np, gzip, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from graph import VisionGraph
from vision import VisionInterface


class MultiScaleEye:
    """多尺度 + 中层形状"""

    def __init__(self):
        # 三个尺度的子眼
        self.scales = [
            {"name": "fine",   "ps": 4,  "st": 4, "n": 100},
            {"name": "mid",    "ps": 8,  "st": 4, "n": 100},
            {"name": "coarse", "ps": 16, "st": 8, "n": 50},
        ]
        self.eyes = []
        for s in self.scales:
            ts = s["ps"] * s["ps"]
            g = VisionGraph(n_nodes=s["n"], template_size=ts)
            v = VisionInterface(g, patch_size=s["ps"], stride=s["st"])
            self.eyes.append({"graph": g, "vision": v, "name": s["name"]})

        # 中层形状节点 (第1层 K-means)
        self.shape_centers = None
        self.n_shape = 100

        # 高级物体节点 (第2层 K-means → 形状的聚类)
        self.object_centers = None
        self.n_object = 50

        self.memory = []

    def init_templates(self, images):
        for eye in self.eyes:
            ex = eye["vision"].extractor
            nids = sorted(eye["graph"].nodes.keys())
            idxs = np.random.choice(len(images), min(200, len(images)), replace=False)
            ap = []
            for i in idxs: ap.extend(ex.extract(images[i]))
            if not ap: continue
            for i, nid in enumerate(nids):
                p = ap[i % len(ap)]
                eye["graph"].nodes[nid].template = p.astype(np.float32) + np.random.randn(len(p)).astype(np.float32)*0.01
                eye["graph"].nodes[nid].template /= np.linalg.norm(eye["graph"].nodes[nid].template)+1e-8

    def train(self, imgs, lbls, epochs=5, contrast_aug=True):
        """训练 + 可选的对比度增强 (50% 概率降低对比度)"""
        lr = 0.1
        for ep in range(epochs):
            perm = np.random.permutation(len(imgs))
            for idx in perm:
                img = imgs[idx].copy()
                if contrast_aug and np.random.random() < 0.5:
                    mean = img.mean()
                    c = 0.3 + np.random.random() * 0.7  # [0.3, 1.0]
                    img = mean + (img - mean) * c
                for eye in self.eyes:
                    eye["vision"].set_image(img)
                    for nid, aps in eye["vision"].node_assignments.items():
                        node = eye["graph"].nodes[nid]
                        t = np.mean(aps, axis=0)
                        n = np.linalg.norm(t)
                        if n > 0: t /= n
                        node.template += lr*(t-node.template)
                        node.template /= np.linalg.norm(node.template)+1e-8

    def get_activation_vector(self, image):
        """提取多尺度激活向量并拼接"""
        vecs = []
        for eye in self.eyes:
            eye["vision"].set_image(image)
            nids = sorted(eye["graph"].nodes.keys())
            v = np.array([eye["graph"].nodes[nid].activation for nid in nids], dtype=np.float32)
            vecs.append(v)
        return np.concatenate(vecs)

    def build_shapes(self, imgs, labels, n_samples=5000):
        """K-means 聚类 → 中层形状节点"""
        n = min(n_samples, len(imgs), len(labels))
        idxs = np.random.choice(min(len(imgs), len(labels)), n, replace=False)

        # 提取所有训练图像的激活向量
        print(f"  提取 {n} 个激活向量... (imgs={len(imgs)}, labels={len(labels)})")
        vecs = []
        for i in idxs:
            vecs.append(self.get_activation_vector(imgs[i]))
        X = np.array(vecs, dtype=np.float32)

        # K-means (手写, 无依赖)
        print(f"  K-means (k={self.n_shape})...")
        # 随机初始化
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X), self.n_shape, replace=False)
        centers = X[indices].copy()

        for it in range(50):
            # 分配
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            km_labels = np.argmin(dists, axis=1)
            # 更新
            new_centers = np.zeros_like(centers)
            for c in range(self.n_shape):
                mask = km_labels == c
                if mask.sum() > 0:
                    new_centers[c] = X[mask].mean(axis=0)
                else:
                    new_centers[c] = X[rng.choice(len(X))]
            shift = np.sum((centers - new_centers) ** 2)
            centers = new_centers
            if shift < 1e-6:
                print(f"   收敛 @ iter {it}")
                break

        self.shape_centers = centers.astype(np.float32)

        # 统计每个形状的类别分布
        dists = np.sum((X[:, None, :] - self.shape_centers[None, :, :]) ** 2, axis=2)
        cluster_labels = np.argmin(dists, axis=1)
        labels_arr = np.array([labels[i] for i in idxs])
        shape_dist = {}
        for c in range(self.n_shape):
            mask = cluster_labels == c
            if mask.sum() > 0:
                lbls_c = labels_arr[mask]
                counts = np.bincount(lbls_c, minlength=10)
                shape_dist[c] = counts
        return shape_dist

    def build_objects(self, imgs, labels, n_samples=5000):
        """第2层 K-means: 形状向量 → 物体节点"""
        if self.shape_centers is None:
            raise RuntimeError("先 build_shapes()")
        n = min(n_samples, len(imgs))
        idxs = np.random.choice(len(imgs), n, replace=False)

        # 提取形状向量 (100维), 不是形状中心
        print(f"  物体层: 提取 {n} 个形状向量...")
        shape_vecs = []
        for i in idxs:
            shape_vecs.append(self.get_shape_vector(imgs[i]))
        X = np.array(shape_vecs, dtype=np.float32)  # [n, 100]

        print(f"  物体层 K-means (k={self.n_object})...")
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X), self.n_object, replace=False)
        centers = X[indices].copy()

        for it in range(50):
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            km_labels = np.argmin(dists, axis=1)
            new_centers = np.zeros_like(centers)
            for c in range(self.n_object):
                mask = km_labels == c
                if mask.sum() > 0:
                    new_centers[c] = X[mask].mean(axis=0)
                else:
                    new_centers[c] = X[rng.choice(len(X))]
            shift = np.sum((centers - new_centers) ** 2)
            centers = new_centers
            if shift < 1e-6:
                print(f"   收敛 @ iter {it}")
                break
        self.object_centers = centers.astype(np.float32)  # [50, 100]

    def get_object_vector(self, image):
        """图像 → 形状向量 → 物体向量"""
        shape_vec = self.get_shape_vector(image)
        sims = self.object_centers @ shape_vec / (
            np.linalg.norm(self.object_centers, axis=1) * np.linalg.norm(shape_vec) + 1e-8)
        return np.clip(sims, 0, 1).astype(np.float32)

    def get_shape_vector(self, image):
        """图像 → 中层形状激活向量"""
        if self.shape_centers is None:
            raise RuntimeError("先 build_shapes()")
        base = self.get_activation_vector(image)
        sims = self.shape_centers @ base / (
            np.linalg.norm(self.shape_centers, axis=1) * np.linalg.norm(base) + 1e-8)
        return np.clip(sims, 0, 1).astype(np.float32)

    def build_memory(self, imgs, lbls, size=5000):
        self.memory.clear()
        n = min(size, len(imgs))
        idxs = np.random.choice(len(imgs), n, replace=False)
        for idx in idxs:
            vec = self.get_object_vector(imgs[idx])
            self.memory.append((vec, lbls[idx]))
        print(f"  记忆: {n} 样本 (物体向量, {len(vec)} 维)")

    def predict(self, image):
        q = self.get_object_vector(image)
        if not self.memory: return -1, 0.0
        best_sim, best_label = -1, -1
        for mvec, mlbl in self.memory:
            sim = np.dot(mvec, q) / (np.linalg.norm(mvec)*np.linalg.norm(q) + 1e-8)
            if sim > best_sim:
                best_sim, best_label = sim, mlbl
        return best_label, best_sim


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


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_mnist()

    model = MultiScaleEye()
    model.init_templates(X_tr[:5000])
    print("训练多尺度特征...")
    t0 = time.time()
    model.train(X_tr[:10000], y_tr[:10000], epochs=3, contrast_aug=False)
    print(f"训练: {time.time()-t0:.0f}s")

    print("构建中层形状节点...")
    model.build_shapes(X_tr[:15000].copy(), y_tr[:15000].copy(), n_samples=10000)
    model.build_objects(X_tr[:15000].copy(), y_tr[:15000].copy(), n_samples=5000)

    model.build_memory(X_tr[:15000], y_tr[:15000], size=8000)

    print(f"\n📊 三层: Patch → Shape → Object (A+B+C) 评估 (500 测试):")
    print(f"  基线 (单尺度 200节点): 61.6% (网格搜索最佳)")
    n_test = 500
    for c in [1.0, 0.5, 0.3, 0.2]:
        correct = 0
        for i in range(n_test):
            img = X_te[i].copy()
            mean = img.mean()
            img = mean + (img-mean)*c
            pred, _ = model.predict(img)
            if pred == y_te[i]: correct += 1
        print(f"  c={c:.2f}: {correct}/{n_test} = {correct/n_test:.1%}")
