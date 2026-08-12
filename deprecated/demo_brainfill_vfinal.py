#!/usr/bin/env python3
"""
冷眼 脑补 vFinal: 64×64 + 空间热点 + 预测传播 + 自适应渲染
"""

import sys, os, time, numpy as np, gzip, urllib.request
from collections import defaultdict
from scipy.ndimage import zoom
sys.path.insert(0, os.path.dirname(__file__))
from graph import VisionGraph
from vision import VisionInterface


class SpatialBrainFill:
    def __init__(self):
        # 多尺度: fine + coarse
        self.eyes = [
            {"ps": 8,  "st": 4,  "n": 200, "name": "fine"},
            {"ps": 16, "st": 8,  "n": 100, "name": "coarse"},
        ]
        self.graphs = []
        self.visions = []
        for e in self.eyes:
            ts = e["ps"] * e["ps"]
            g = VisionGraph(n_nodes=e["n"], template_size=ts)
            v = VisionInterface(g, patch_size=e["ps"], stride=e["st"])
            self.graphs.append(g)
            self.visions.append(v)

        # 空间热点
        self.hotspots: dict = {}        # {nid: (cy, cx, count)}
        self.all_edges: dict = defaultdict(float)
        self.gen = 0
        self.img_h, self.img_w = 64, 64

    def init(self, images):
        for gi, g in enumerate(self.graphs):
            ex = self.visions[gi].extractor
            nids = sorted(g.nodes.keys())
            idxs = np.random.choice(len(images), min(200, len(images)), replace=False)
            ap = []
            for i in idxs: ap.extend(ex.extract(images[i]))
            if not ap: continue
            for i, nid in enumerate(nids):
                p = ap[i % len(ap)]
                g.nodes[nid].template = p.astype(np.float32) + np.random.randn(len(p)).astype(np.float32)*0.01
                g.nodes[nid].template /= np.linalg.norm(g.nodes[nid].template)+1e-8

    def train(self, images, epochs=5):
        lr = 0.1
        for ep in range(epochs):
            perm = np.random.permutation(len(images))
            for idx in perm:
                img = images[idx]
                for gi in range(len(self.eyes)):
                    g, v = self.graphs[gi], self.visions[gi]
                    v.set_image(img)
                    patches = v._last_patches
                    positions = v.extractor.patch_positions
                    nids = sorted(g.nodes.keys())
                    templates = np.array([g.nodes[n].template for n in nids], dtype=np.float32)

                    # 每个 patch 的最佳节点 + 热点追踪
                    for pi in range(len(patches)):
                        scores = templates @ patches[pi]
                        bi = int(np.argmax(scores))
                        nid = nids[bi]
                        # 追踪热点
                        y1, x1, y2, x2 = positions[pi]
                        cy, cx = (y1+y2)/2, (x1+x2)/2
                        key = f"{gi}:{nid}"
                        if key not in self.hotspots:
                            self.hotspots[key] = [cy, cx, 1]
                        else:
                            hy, hx, hc = self.hotspots[key]
                            a = 1.0/(hc+1)
                            self.hotspots[key] = [hy+a*(cy-hy), hx+a*(cx-hx), hc+1]

                    # Hebbian 模板更新
                    for nid, aps in v.node_assignments.items():
                        node = g.nodes[nid]
                        t = np.mean(aps, axis=0)
                        n = np.linalg.norm(t)
                        if n > 0: t /= n
                        node.template += lr*(t-node.template)
                        node.template /= np.linalg.norm(node.template)+1e-8

                    # 空间约束边: 同一 eye 内相邻 patch 的节点
                    patch_nodes = {}
                    for pi in range(len(patches)):
                        scores = templates @ patches[pi]
                        bi = int(np.argmax(scores))
                        patch_nodes[pi] = nids[bi]

                    for pi in patch_nodes:
                        y1,x1,y2,x2 = positions[pi]; cy,cx = (y1+y2)/2,(x1+x2)/2
                        for pj in patch_nodes:
                            if pj <= pi: continue
                            y3,x3,y4,x4 = positions[pj]; cy2,cx2 = (y3+y4)/2,(x3+x4)/2
                            d = np.sqrt((cy-cy2)**2 + (cx-cx2)**2)
                            limit = self.eyes[gi]["ps"] * 3  # 3x patch_size
                            if d > limit: continue
                            a, b = patch_nodes[pi], patch_nodes[pj]
                            self.all_edges[(f"{gi}:{a}", f"{gi}:{b}")] += 0.1
                            self.all_edges[(f"{gi}:{b}", f"{gi}:{a}")] += 0.1

                self.gen += 1
                if self.gen % 10000 == 0:
                    print(f"  gen={self.gen} ep={ep+1}/{epochs}")

        # 按目标节点归一化入边 (ps/pw 变为加权平均)
        node_in_sum = defaultdict(float)
        for (s, d), w in self.all_edges.items():
            node_in_sum[d] += w
        for (s, d), w in list(self.all_edges.items()):
            if node_in_sum[d] > 0:
                self.all_edges[(s, d)] = w / node_in_sum[d]
        print(f"  热点: {len(self.hotspots)}, 边: {len(self.all_edges)//2} 对")

    def brain_fill(self, image, contrast=0.3, n_iter=30, lr=0.8):
        """低对比度 → 竞争路由 → 预测传播 → 热点重建"""
        # 底向上路由
        for gi in range(len(self.eyes)):
            self.visions[gi].set_image(image.copy(), contrast=contrast)

        # 迭代传播
        for _ in range(n_iter):
            boosts = {}
            # 收集所有节点的当前状态
            all_nodes = {}
            for gi, g in enumerate(self.graphs):
                for nid, node in g.nodes.items():
                    all_nodes[f"{gi}:{nid}"] = node.activation

            for key, act in all_nodes.items():
                ps, pw = 0.0, 0.0
                for (s, d), w in self.all_edges.items():
                    if d == key and all_nodes.get(s, 0) > 0.01:
                        ps += all_nodes[s] * w
                        pw += w
                if pw > 0:
                    boost = lr * (ps/pw) * (1.0 - act)
                    boosts[key] = min(1.0, act + boost)
                # 无预测支持也不衰减——保持原值，等邻居传播过来

            # 应用更新
            for key, na in boosts.items():
                gi_str, nid = key.split(":", 1)
                gi = int(gi_str)
                self.graphs[gi].nodes[nid].activation = na

            # 竞争归一化: 总激活受限 → 节点间产生差异
            total_act = sum(n.activation for g in self.graphs for n in g.nodes.values())
            if total_act > 1:
                for g in self.graphs:
                    for n in g.nodes.values():
                        n.activation /= total_act

        return self.render()

    def render(self):
        """热点渲染: 活跃节点高斯斑点 (无热点则用邻居插值)"""
        H, W = self.img_h, self.img_w
        canvas = np.zeros((H, W), dtype=np.float32)

        for gi, g in enumerate(self.graphs):
            ps = self.eyes[gi]["ps"]
            sigma = ps / 1.5
            for nid, node in g.nodes.items():
                key = f"{gi}:{nid}"
                if node.activation < 0.01:
                    continue

                if key in self.hotspots:
                    cy, cx, _ = self.hotspots[key]
                else:
                    # 无热点: 用平均位置 (图像中心)
                    cy, cx = H/2, W/2

                cy = np.clip(int(cy), 0, H-1)
                cx = np.clip(int(cx), 0, W-1)

                yy, xx = np.mgrid[0:H, 0:W]
                gauss = np.exp(-((yy-cy)**2 + (xx-cx)**2)/(2*sigma**2))
                canvas += gauss * node.activation

        canvas = np.clip(canvas, 0, 1)
        mx = canvas.max()
        return canvas / mx if mx > 0 else canvas

    def nofill_render(self, image, contrast=0.3):
        """无脑补的底向上重建"""
        for gi in range(len(self.eyes)):
            self.visions[gi].set_image(image.copy(), contrast=contrast)
        return self.render()


def load_mnist_64(n_train=5000, n_test=50):
    os.makedirs("data",exist_ok=True)
    url = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
    files = {
        "ti": "train-images-idx3-ubyte.gz", "tl": "train-labels-idx1-ubyte.gz",
        "ei": "t10k-images-idx3-ubyte.gz", "el": "t10k-labels-idx1-ubyte.gz"
    }
    for f in files.values():
        p = os.path.join("data", f)
        if not os.path.exists(p): urllib.request.urlretrieve(url+f, p)
    def li(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255
    def ll(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(),np.uint8,offset=8).astype(np.int64)
    X_tr = li(os.path.join("data",files["ti"]))
    y_tr = ll(os.path.join("data",files["tl"]))
    X_te = li(os.path.join("data",files["ei"]))
    y_te = ll(os.path.join("data",files["el"]))

    # Resize to 64×64
    X_tr64 = np.zeros((min(n_train, len(X_tr)), 64, 64), dtype=np.float32)
    for i in range(len(X_tr64)):
        X_tr64[i] = zoom(X_tr[i], 64/28, order=1)
    X_te64 = np.zeros((min(n_test, len(X_te)), 64, 64), dtype=np.float32)
    for i in range(len(X_te64)):
        X_te64[i] = zoom(X_te[i], 64/28, order=1)
    return X_tr64, y_tr[:len(X_tr64)], X_te64, y_te[:len(X_te64)]


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    X_tr, y_tr, X_te, y_te = load_mnist_64(n_train=5000, n_test=50)

    model = SpatialBrainFill()
    model.init(X_tr[:500])
    t0 = time.time()
    model.train(X_tr, epochs=5)
    print(f"训练: {time.time()-t0:.0f}s")

    # 测试多个样本
    for si in range(3):
        sample = X_te[si]
        label = y_te[si]
        print(f"\n样本 {si}: digit {label}")

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        for row, c in enumerate([1.0, 0.3, 0.15]):
            img_low = sample.copy()
            mean = img_low.mean()
            img_low = mean + (img_low-mean)*c

            # 原图
            axes[row][0].imshow(img_low, cmap="gray", vmin=0, vmax=1)
            axes[row][0].set_title(f"Input c={c}")
            axes[row][0].axis("off")

            # 无脑补
            r_no = model.nofill_render(img_low, contrast=1.0)
            vmn, vmx = r_no.min(), r_no.max()
            axes[row][1].imshow(r_no, cmap="hot", vmin=vmn, vmax=vmx if vmx > vmn else vmn+1e-8)
            axes[row][1].set_title("No fill")
            axes[row][1].axis("off")

            # 脑补 — 用实际数据范围，不做归一化到 [0,1]
            r_fill = model.brain_fill(img_low, contrast=1.0, n_iter=80, lr=1.0)
            axes[row][2].imshow(r_fill, cmap="hot", vmin=r_fill.min(), vmax=r_fill.max())
            axes[row][2].set_title("Brain fill")
            axes[row][2].axis("off")

            # 差异
            diff = np.abs(r_fill - r_no)
            axes[row][3].imshow(diff, cmap="hot", vmin=0, vmax=diff.max() or 1)
            axes[row][3].set_title("|Diff|")
            axes[row][3].axis("off")

        plt.suptitle(f"Brain Fill — digit {label}", fontsize=14)
        plt.tight_layout()
        os.makedirs("data", exist_ok=True)
        plt.savefig(f"data/brainfill_{si}.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  ✅ data/brainfill_{si}.png")

    print("\n✅ 完成")
