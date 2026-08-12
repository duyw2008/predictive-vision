#!/usr/bin/env python3
"""冷眼 — 网格搜索最优参数"""
import sys, os, time, numpy as np, gzip, urllib.request, json
sys.path.insert(0, os.path.dirname(__file__))
from graph import VisionGraph
from vision import VisionInterface


class QuickEye:
    def __init__(self, n=500):
        self.graph = VisionGraph(n_nodes=n, template_size=64)
        self.vision = VisionInterface(self.graph, patch_size=8, stride=4)
        self.memory = []

    def init_templates(self, images):
        ex = self.vision.extractor
        nids = sorted(self.graph.nodes.keys())
        idxs = np.random.choice(len(images), min(200, len(images)), replace=False)
        ap = []
        for i in idxs: ap.extend(ex.extract(images[i]))
        if not ap: return
        for i, nid in enumerate(nids):
            p = ap[i % len(ap)]
            self.graph.nodes[nid].template = p.astype(np.float32) + np.random.randn(64).astype(np.float32)*0.01
            self.graph.nodes[nid].template /= np.linalg.norm(self.graph.nodes[nid].template)+1e-8

    def train(self, imgs, lbls, epochs=5):
        lr = 0.1
        for ep in range(epochs):
            perm = np.random.permutation(len(imgs))
            for idx in perm:
                img = imgs[idx]
                self.vision.set_image(img)
                for nid, aps in self.vision.node_assignments.items():
                    node = self.graph.nodes[nid]
                    t = np.mean(aps, axis=0)
                    n = np.linalg.norm(t)
                    if n > 0: t /= n
                    node.template += lr*(t-node.template)
                    node.template /= np.linalg.norm(node.template)+1e-8

    def build_memory(self, imgs, lbls, size=5000):
        self.memory.clear()
        n = min(size, len(imgs))
        idxs = np.random.choice(len(imgs), n, replace=False)
        for idx in idxs:
            self.vision.set_image(imgs[idx])
            vec = np.array([self.graph.nodes[nid].activation
                           for nid in sorted(self.graph.nodes.keys())], dtype=np.float32)
            self.memory.append((vec, lbls[idx]))

    def predict_batch(self, images):
        mem_vecs = np.array([m[0] for m in self.memory], dtype=np.float32)
        mem_norms = np.linalg.norm(mem_vecs, axis=1) + 1e-8
        mem_lbls = np.array([m[1] for m in self.memory])

        results = []
        for img in images:
            self.vision.set_image(img)
            q = np.array([self.graph.nodes[nid].activation
                         for nid in sorted(self.graph.nodes.keys())], dtype=np.float32)
            qn = np.linalg.norm(q) + 1e-8
            sims = mem_vecs @ q / (mem_norms * qn)
            best = int(np.argmax(sims))
            results.append((mem_lbls[best], float(sims[best])))
        return results


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
    n_test = 500

    # 测试参数网格
    grid = []
    for n_nodes in [200, 350, 500, 700]:
        for n_epochs in [3, 5]:
            for n_train in [10000, 20000]:
                for mem_size in [3000, 5000]:
                    grid.append((n_nodes, n_epochs, n_train, mem_size))

    print(f"{'节点':>5} {'epochs':>6} {'train':>6} {'memory':>6} {'acc@1.0':>8} {'acc@0.5':>8} {'time':>6}")
    print("-" * 60)

    results = []
    for n_nodes, n_epochs, n_train, mem_size in grid:
        t0 = time.time()
        model = QuickEye(n=n_nodes)
        model.init_templates(X_tr[:min(5000, n_train)])

        # 使用 n_train 样本训练
        model.train(X_tr[:n_train], y_tr[:n_train], epochs=n_epochs)
        model.build_memory(X_tr[:n_train], y_tr[:n_train], size=mem_size)

        # 评估全对比度
        correct_1 = 0
        for i in range(n_test):
            pred, _ = model.predict_batch([X_te[i]])[0]
            if pred == y_te[i]: correct_1 += 1

        # 评估 0.5 对比度
        X_te_low = X_te[:n_test].copy()
        for i in range(n_test):
            m = X_te_low[i].mean()
            X_te_low[i] = m + (X_te_low[i]-m)*0.5
        correct_05 = sum(1 for i in range(n_test)
                        if model.predict_batch([X_te_low[i]])[0][0] == y_te[i])

        t = time.time()-t0
        print(f"{n_nodes:5d} {n_epochs:6d} {n_train:6d} {mem_size:6d} {correct_1/n_test:8.1%} {correct_05/n_test:8.1%} {t:5.0f}s")
        results.append({
            "n_nodes": n_nodes, "epochs": n_epochs, "n_train": n_train,
            "mem": mem_size, "acc1": correct_1/n_test, "acc05": correct_05/n_test, "time": t
        })

    # 最佳组合
    best = max(results, key=lambda r: r["acc1"])
    print(f"\n🏆 最佳: nodes={best['n_nodes']} epochs={best['epochs']} "
          f"train={best['n_train']} mem={best['mem']} → {best['acc1']:.1%}")

    with open("data/grid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("结果已保存到 data/grid_results.json")
