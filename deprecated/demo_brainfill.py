#!/usr/bin/env python3
"""
冷眼 — 脑补 demo v4: 空间热点 + 水平连接
每个节点知道自己在图像中的"主场"位置，脑补时只在主场增强
"""

import sys, os, time, numpy as np, gzip, urllib.request
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from graph import VisionGraph
from vision import VisionInterface


class SpatialBrain:
    """空间索引脑: 节点有位置感 + 邻居只连空间近的"""

    def __init__(self, n=500, patch_size=8, stride=4):
        ts = patch_size * patch_size
        self.graph = VisionGraph(n_nodes=n, template_size=ts)
        self.vision = VisionInterface(self.graph, patch_size=patch_size, stride=stride)
        self.patch_size = patch_size
        self.stride = stride

        # 空间热点: {nid: (cy, cx, count)}
        self.hotspots: dict = {}
        # 预测边 (空间约束)
        self.pred_edges: dict = defaultdict(float)
        self.node_votes = defaultdict(lambda: defaultdict(int))
        self.gen = 0

    def init_templates(self, images):
        ex = self.vision.extractor
        nids = sorted(self.graph.nodes.keys())
        idxs = np.random.choice(len(images), min(200, len(images)), replace=False)
        ap = []
        for i in idxs: ap.extend(ex.extract(images[i]))
        if not ap: return
        for i, nid in enumerate(nids):
            p = ap[i % len(ap)]
            self.graph.nodes[nid].template = p.astype(np.float32) + np.random.randn(len(p)).astype(np.float32)*0.01
            self.graph.nodes[nid].template /= np.linalg.norm(self.graph.nodes[nid].template)+1e-8

    def train(self, imgs, lbls, epochs=5):
        lr = 0.1
        for ep in range(epochs):
            perm = np.random.permutation(len(imgs))
            for idx in perm:
                img, label = imgs[idx], lbls[idx]
                self.vision.set_image(img)
                patches = self.vision._last_patches
                positions = self.vision.extractor.patch_positions
                nids = sorted(self.graph.nodes.keys())
                templates = np.array([self.graph.nodes[n].template for n in nids], dtype=np.float32)

                # 每个 patch 的最佳节点
                patch_to_node = {}
                for pi in range(len(patches)):
                    scores = templates @ patches[pi]
                    bi = int(np.argmax(scores))
                    if float(scores[bi]) > 0:
                        patch_to_node[pi] = nids[bi]

                # 更新模板 + 追踪空间热点
                for nid, aps in self.vision.node_assignments.items():
                    node = self.graph.nodes[nid]
                    t = np.mean(aps, axis=0)
                    n = np.linalg.norm(t)
                    if n > 0: t /= n
                    node.template += lr*(t-node.template)
                    node.template /= np.linalg.norm(node.template)+1e-8
                    self.node_votes[nid][label] += 1

                # 记录每个节点匹配的 patch 位置 → 空间热点
                for pi, nid in patch_to_node.items():
                    y1, x1, y2, x2 = positions[pi]
                    cy, cx = (y1+y2)/2, (x1+x2)/2
                    if nid not in self.hotspots:
                        self.hotspots[nid] = [cy, cx, 1]
                    else:
                        hy, hx, hc = self.hotspots[nid]
                        alpha = 1.0 / (hc + 1)
                        self.hotspots[nid] = [
                            hy + alpha*(cy - hy),
                            hx + alpha*(cx - hx),
                            hc + 1,
                        ]

                # 空间约束预测边: 只连相邻位置
                for pi in patch_to_node:
                    y1, x1, y2, x2 = positions[pi]
                    cy, cx = (y1+y2)/2, (x1+x2)/2
                    for pj in patch_to_node:
                        if pj <= pi: continue
                        y3, x3, y4, x4 = positions[pj]
                        cy2, cx2 = (y3+y4)/2, (x3+x4)/2
                        if np.sqrt((cy-cy2)**2+(cx-cx2)**2) > 16: continue
                        a, b = patch_to_node[pi], patch_to_node[pj]
                        s = min(self.graph.nodes[a].activation, self.graph.nodes[b].activation)
                        if s > 0.05:
                            self.pred_edges[(a,b)] += s
                            self.pred_edges[(b,a)] += s
                self.gen += 1

        mx = max(self.pred_edges.values()) if self.pred_edges else 1
        for k in self.pred_edges: self.pred_edges[k] /= mx
        for nid, votes in self.node_votes.items():
            if votes: self.graph.nodes[nid].domain_tag = str(max(votes, key=votes.get))
        print(f"  热点: {len(self.hotspots)} 节点, 预测边: {len(self.pred_edges)//2} 对")

    def brain_fill(self, image, contrast=0.3, n_iter=15, lr=0.5):
        """低对比度 → 竞争路由 → 预测传播增强 → 空间热点重建"""
        self.vision.set_image(image, contrast=contrast)

        for _ in range(n_iter):
            boosts = {}
            for nid, node in self.graph.nodes.items():
                ps, pw = 0.0, 0.0
                for (s, d), w in self.pred_edges.items():
                    if d == nid and self.graph.nodes[s].activation > 0.05:
                        ps += self.graph.nodes[s].activation * w; pw += w
                if pw > 0:
                    boost = lr * (ps/pw) * (1.0 - node.activation)
                    boosts[nid] = min(1.0, node.activation + boost)
                elif node.activation > 0.1:
                    # 活跃但无邻居预测 → 保持并缓慢衰减
                    boosts[nid] = max(0.0, node.activation * 0.95)
            for nid, na in boosts.items():
                self.graph.nodes[nid].activation = na

        return self.spatial_reconstruct(image.shape)

    def spatial_reconstruct(self, shape):
        """空间热点重建: 每个节点在其主场位置渲染激活值 (高斯斑点)"""
        H, W = shape
        canvas = np.zeros((H, W), dtype=np.float32)

        for nid, node in self.graph.nodes.items():
            if node.activation < 0.05:
                continue
            if nid not in self.hotspots:
                continue
            cy, cx, _ = self.hotspots[nid]

            # 在热点位置生成高斯斑点，sigma = patch_size/2
            sigma = self.patch_size / 2.0
            yy, xx = np.mgrid[0:H, 0:W]
            g = np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*sigma**2))
            canvas += g * node.activation

        canvas = np.clip(canvas, 0, 1)
        # 归一化到 [0,1]
        mx = canvas.max()
        if mx > 0:
            canvas /= mx
        return canvas


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
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    X_tr, y_tr, X_te, y_te = load_mnist()
    model = SpatialBrain(n=500, patch_size=8, stride=4)
    model.init_templates(X_tr[:1000])
    t0 = time.time()
    model.train(X_tr[:10000], y_tr[:10000], epochs=5)
    print(f"训练: {time.time()-t0:.0f}s")

    sample = X_te[0]
    print(f"样本: digit {y_te[0]}")

    contrasts = [1.0, 0.5, 0.3, 0.2]
    fig, axes = plt.subplots(len(contrasts), 4, figsize=(16, 4*len(contrasts)))

    for row, c in enumerate(contrasts):
        img_low = sample.copy()
        mean = img_low.mean()
        img_low = mean + (img_low-mean)*c

        axes[row][0].imshow(img_low, cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_title(f"c={c}")
        axes[row][0].axis("off")

        # 无脑补: 直接空间重建
        model.vision.set_image(img_low.copy())
        r_no = model.spatial_reconstruct(img_low.shape)
        axes[row][1].imshow(r_no, cmap="hot", vmin=0, vmax=1)
        axes[row][1].set_title("No fill")
        axes[row][1].axis("off")

        # 脑补
        r_fill = model.brain_fill(img_low.copy(), contrast=1.0, n_iter=30, lr=0.8)
        axes[row][2].imshow(r_fill, cmap="hot", vmin=0, vmax=1)
        axes[row][2].set_title("Brain fill")
        axes[row][2].axis("off")

        diff = np.abs(r_fill - r_no)
        axes[row][3].imshow(diff, cmap="hot", vmin=0, vmax=diff.max() or 1)
        axes[row][3].set_title(f"|Diff|")
        axes[row][3].axis("off")

    plt.suptitle(f"Spatial Brain Fill (digit {y_te[0]})", fontsize=14)
    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/brain_fill_spatial.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ data/brain_fill_spatial.png")
