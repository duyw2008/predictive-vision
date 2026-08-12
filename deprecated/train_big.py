#!/usr/bin/env python3
"""A线: 堆 NN 分类 — 2000节点 → 推 85%+"""
import sys, os, time, numpy as np, gzip, urllib.request
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from graph import VisionGraph
from vision import VisionInterface

class BigEye:
    def __init__(self, n=2000, patch_size=8, stride=4):
        ts = patch_size*patch_size
        self.graph = VisionGraph(n_nodes=n, template_size=ts)
        self.vision = VisionInterface(self.graph, patch_size=patch_size, stride=stride)
        self.memory = []
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
            for i, idx in enumerate(perm):
                img, label = imgs[idx], lbls[idx]
                self.vision.set_image(img)
                for nid, aps in self.vision.node_assignments.items():
                    node = self.graph.nodes[nid]
                    t = np.mean(aps, axis=0)
                    n = np.linalg.norm(t)
                    if n > 0: t /= n
                    node.template += lr*(t-node.template)
                    node.template /= np.linalg.norm(node.template)+1e-8
                self.gen += 1
                if self.gen % 5000 == 0:
                    print(f"  gen={self.gen} ep={ep+1}/{epochs}")

    def build_memory(self, imgs, lbls, sample_size=None):
        self.memory.clear()
        n = len(imgs) if sample_size is None else min(sample_size, len(imgs))
        idxs = np.random.choice(len(imgs), n, replace=False)
        for idx in idxs:
            self.vision.set_image(imgs[idx])
            vec = np.array([self.graph.nodes[nid].activation
                           for nid in sorted(self.graph.nodes.keys())], dtype=np.float32)
            self.memory.append((vec, lbls[idx]))
        print(f"  记忆: {len(self.memory)} 样本")

    def predict(self, image):
        self.vision.set_image(image)
        q = np.array([self.graph.nodes[nid].activation
                     for nid in sorted(self.graph.nodes.keys())], dtype=np.float32)
        if not self.memory: return -1, 0.0

        best_sim, best_label = -1, -1
        # Fast batch NN
        mem_vecs = np.array([m[0] for m in self.memory], dtype=np.float32)
        mem_lbls = np.array([m[1] for m in self.memory])
        sims = mem_vecs @ q / (np.linalg.norm(mem_vecs, axis=1) * np.linalg.norm(q) + 1e-8)
        best_idx = int(np.argmax(sims))
        return mem_lbls[best_idx], float(sims[best_idx])


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

    model = BigEye(n=2000)
    model.init_templates(X_tr[:5000])
    print(f"初始化完成, 开始训练...")

    t0 = time.time()
    model.train(X_tr[:30000], y_tr[:30000], epochs=3)
    print(f"训练: {time.time()-t0:.0f}s, gen={model.gen}")

    model.build_memory(X_tr[:10000], y_tr[:10000])

    print("\n📊 对比度评估 (1000 测试):")
    n_test = min(1000, len(X_te))
    for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05]:
        correct = 0
        for i in range(n_test):
            img = X_te[i].copy()
            mean = img.mean()
            img = mean + (img-mean)*c
            pred, _ = model.predict(img)
            if pred == y_te[i]: correct += 1
        print(f"  c={c:.2f}: {correct}/{n_test} = {correct/n_test:.1%}")
