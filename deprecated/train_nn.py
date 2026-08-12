#!/usr/bin/env python3
"""
冷眼 — 简化验证
节点模板作为特征提取器, 最近邻分类

先验证机制正确 (模板学到有用特征), 再逐步把分类逻辑还给涌现
"""

import sys, os, time, json, gzip, urllib.request
import numpy as np
from collections import defaultdict, Counter
from typing import Tuple

sys.path.insert(0, os.path.dirname(__file__))
from graph import VisionGraph
from vision import VisionInterface, PatchExtractor


class SimpleColdEye:
    """最简冷眼: 竞争学习模板 + NN分类"""

    def __init__(self, n_nodes: int = 500):
        self.graph = VisionGraph(n_nodes=n_nodes)
        self.vision = VisionInterface(self.graph, top_k=1)
        self.generation = 0

        # 最近邻记忆: [(activation_vector, label), ...]
        self.memory: list = []

    def init_templates(self, images: np.ndarray):
        """用真实 patches 初始化"""
        extractor = self.vision.extractor
        node_ids = sorted(self.graph.nodes.keys())
        indices = np.random.choice(len(images), min(50, len(images)), replace=False)
        all_patches = []
        for idx in indices:
            all_patches.extend(extractor.extract(images[idx]))

        if not all_patches:
            return
        for i, nid in enumerate(node_ids):
            p = all_patches[i % len(all_patches)]
            self.graph.nodes[nid].template = p.astype(np.float32)
            self.graph.nodes[nid].template += np.random.randn(
                len(p)).astype(np.float32) * 0.01
            self.graph.nodes[nid].template /= np.linalg.norm(
                self.graph.nodes[nid].template) + 1e-8

    def train(self, images: np.ndarray, labels: np.ndarray,
              n_epochs: int = 5, contrast_fn=None):
        """在线竞争学习: 每个 patch → 最近节点 → 模板移向 patch"""
        n = len(images)
        lr = 0.1

        for epoch in range(n_epochs):
            perm = np.random.permutation(n)
            for i, idx in enumerate(perm):
                img = images[idx]
                label = labels[idx]
                contrast = contrast_fn(self.generation) if contrast_fn else 1.0

                self.vision.set_image(img, contrast=contrast)

                # 更新模板: 节点 → 匹配 patch 的平均
                for nid, patches in self.vision.node_assignments.items():
                    node = self.graph.nodes[nid]
                    target = np.mean(patches, axis=0)
                    norm = np.linalg.norm(target)
                    if norm > 0:
                        target = target / norm
                    node.template += lr * (target - node.template)
                    node.template /= np.linalg.norm(node.template) + 1e-8

                self.generation += 1

                if self.generation % 2000 == 0:
                    print(f"  gen={self.generation:5d} epoch={epoch+1}/{n_epochs}")

    def build_memory(self, images: np.ndarray, labels: np.ndarray):
        """存储训练样本的激活向量"""
        self.memory.clear()
        for i in range(len(images)):
            self.vision.set_image(images[i])
            vec = np.array([self.graph.nodes[nid].activation
                           for nid in sorted(self.graph.nodes.keys())],
                          dtype=np.float32)
            self.memory.append((vec, labels[i]))

    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """最近邻: 找激活向量最相似的训练样本"""
        self.vision.set_image(image)
        query = np.array([self.graph.nodes[nid].activation
                         for nid in sorted(self.graph.nodes.keys())],
                        dtype=np.float32)

        if not self.memory:
            return -1, 0.0

        best_sim = -1
        best_label = -1
        for vec, label in self.memory:
            # 余弦相似度
            dot = np.dot(query, vec)
            nq = np.linalg.norm(query)
            nv = np.linalg.norm(vec)
            if nq == 0 or nv == 0:
                continue
            sim = dot / (nq * nv)
            if sim > best_sim:
                best_sim = sim
                best_label = label

        return best_label, best_sim


def load_mnist(data_dir: str = "data"):
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    base_url = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
    os.makedirs(data_dir, exist_ok=True)

    for name, fname in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  下载 {fname}...")
            urllib.request.urlretrieve(base_url + fname, path)

    def load_im(path):
        with gzip.open(path, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28,28).astype(np.float32)/255.0

    def load_lb(path):
        with gzip.open(path, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

    return (load_im(os.path.join(data_dir, files["train_images"])),
            load_lb(os.path.join(data_dir, files["train_labels"])),
            load_im(os.path.join(data_dir, files["test_images"])),
            load_lb(os.path.join(data_dir, files["test_labels"])))


if __name__ == "__main__":
    print("🧊 冷眼 — 简化验证 (NN)")
    print("=" * 50)

    X_train, y_train, X_test, y_test = load_mnist()
    print(f"MNIST: train={len(X_train)}, test={len(X_test)}")

    model = SimpleColdEye(n_nodes=500)
    model.init_templates(X_train[:1000])

    def contrast_fn(gen):
        if gen < 5000: return 1.0
        elif gen < 20000: return 0.5
        else: return 0.25

    # 训练
    n_train = 10000
    t0 = time.time()
    model.train(X_train[:n_train], y_train[:n_train], n_epochs=5,
                contrast_fn=contrast_fn)
    print(f"训练: {time.time()-t0:.0f}s, gen={model.generation}")

    # 建记忆
    model.build_memory(X_train[:n_train], y_train[:n_train])

    # 评估
    print("\n📊 对比度评估:")
    n_test = min(1000, len(X_test))
    for contrast in [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]:
        correct = 0
        for i in range(n_test):
            img = X_test[i].copy()
            mean = img.mean()
            img_low = mean + (img - mean) * contrast
            pred, conf = model.predict(img_low)
            if pred == y_test[i]:
                correct += 1
        print(f"  contrast={contrast:.2f}: {correct}/{n_test} = {correct/n_test:.1%}")
