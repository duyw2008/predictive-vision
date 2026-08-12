#!/usr/bin/env python3
"""
冷眼 — 提示引导 demo
"图里有只猫" → 把猫从低对比度里找出来
"""

import sys, os, time, urllib.request, gzip
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from graph import VisionGraph
from vision import VisionInterface


class HintEye:
    """冷眼 + 提示引导"""

    def __init__(self, n=500, patch_size=8, stride=4):
        self.graph = VisionGraph(n_nodes=n, template_size=patch_size*patch_size)
        self.vision = VisionInterface(self.graph, patch_size=patch_size, stride=stride)
        self.node_votes = defaultdict(lambda: defaultdict(int))
        self.generation = 0

    def init_templates(self, images):
        ex = self.vision.extractor
        node_ids = sorted(self.graph.nodes.keys())
        idxs = np.random.choice(len(images), min(50, len(images)), replace=False)
        all_p = []
        for i in idxs:
            all_p.extend(ex.extract(images[i]))
        if not all_p:
            return
        for i, nid in enumerate(node_ids):
            p = all_p[i % len(all_p)]
            self.graph.nodes[nid].template = p.astype(np.float32) + np.random.randn(
                len(p)).astype(np.float32) * 0.01
            self.graph.nodes[nid].template /= np.linalg.norm(
                self.graph.nodes[nid].template) + 1e-8

    def train(self, images, labels, n_epochs=5):
        lr = 0.1
        n = len(images)
        for epoch in range(n_epochs):
            perm = np.random.permutation(n)
            for idx in perm:
                img, label = images[idx], labels[idx]
                self.vision.set_image(img)
                for nid, patches in self.vision.node_assignments.items():
                    node = self.graph.nodes[nid]
                    target = np.mean(patches, axis=0)
                    norm = np.linalg.norm(target)
                    if norm > 0:
                        target = target / norm
                    node.template += lr * (target - node.template)
                    node.template /= np.linalg.norm(node.template) + 1e-8
                    self.node_votes[nid][label] += 1
                self.generation += 1
            print(f"  epoch {epoch+1}/{n_epochs} gen={self.generation}")

        # 分配 domain_tag
        for nid, votes in self.node_votes.items():
            if votes:
                best = max(votes, key=votes.get)
                self.graph.nodes[nid].domain_tag = str(best)
        tag_dist = defaultdict(int)
        for n in self.graph.nodes.values():
            if n.domain_tag:
                tag_dist[n.domain_tag] += 1
        print(f"  domain_tag 分布: {dict(sorted(tag_dist.items()))}")


def load_mnist(d="data"):
    fs = {"tr_i": "train-images-idx3-ubyte.gz", "tr_l": "train-labels-idx1-ubyte.gz",
          "te_i": "t10k-images-idx3-ubyte.gz", "te_l": "t10k-labels-idx1-ubyte.gz"}
    os.makedirs(d, exist_ok=True)
    url = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
    for f in fs.values():
        p = os.path.join(d, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(url + f, p)

    def li(p):
        with gzip.open(p) as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28,28).astype(np.float32)/255
    def ll(p):
        with gzip.open(p) as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return (li(os.path.join(d, fs["tr_i"])), ll(os.path.join(d, fs["tr_l"])),
            li(os.path.join(d, fs["te_i"])), ll(os.path.join(d, fs["te_l"])))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X_train, y_train, X_test, y_test = load_mnist()
    print(f"MNIST: {len(X_train)} train, {len(X_test)} test")

    # 训练
    model = HintEye(n=500, patch_size=16, stride=8)
    model.init_templates(X_train[:1000])
    t0 = time.time()
    model.train(X_train[:10000], y_train[:10000], n_epochs=3)
    print(f"训练: {time.time()-t0:.0f}s")

    # 挑一个正确提示和一个错误提示做对比
    sample = X_test[0]
    true_label = y_test[0]
    wrong_label = (true_label + 3) % 10

    contrasts = [0.12, 0.06]
    fig, axes = plt.subplots(len(contrasts), 4, figsize=(16, 4 * len(contrasts)))

    for row, contrast in enumerate(contrasts):
        # 原始
        axes[row][0].imshow(sample, cmap="gray")
        axes[row][0].set_title(f"原始 (label={true_label})")
        axes[row][0].axis("off")

        # 低对比度
        mean = sample.mean()
        img_low = mean + (sample - mean) * contrast
        axes[row][1].imshow(img_low, cmap="gray")
        axes[row][1].set_title(f"低对比度 ({contrast})")
        axes[row][1].axis("off")

        # 正确提示
        model.vision.set_image_with_hint(img_low, true_label, contrast=1.0, boost=10.0)
        recon_correct = model.vision.reconstruct(hint_label=true_label, boost=10.0)
        axes[row][2].imshow(recon_correct, cmap="hot")
        axes[row][2].set_title(f'Hint "{true_label}" ✓')
        axes[row][2].axis("off")

        # 错误提示
        model.vision.set_image_with_hint(img_low, wrong_label, contrast=1.0, boost=10.0)
        recon_wrong = model.vision.reconstruct(hint_label=wrong_label, boost=10.0)
        axes[row][3].imshow(recon_wrong, cmap="hot")
        axes[row][3].set_title(f'Hint "{wrong_label}" ✗')
        axes[row][3].axis("off")

    plt.suptitle("冷眼 — 提示引导: 把目标从噪声里挑出来", fontsize=14)
    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/hint_demo.png", dpi=150, bbox_inches="tight")
    print("\n✅ 保存到 data/hint_demo.png")
    plt.close()
    print(f"MEDIA:data/hint_demo.png")
