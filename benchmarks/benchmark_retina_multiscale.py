#!/usr/bin/env python3
"""
缺口1: 视网膜归一化 + MultiScaleEye
对比 raw vs retina 在 MultiScaleEye 架构上的表现
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

# ── retina normalize (from retina_normalize.py) ──
def retina_normalize(images):
    N, H, W = images.shape
    def box_conv(x, ksize):
        p = ksize // 2
        xp = np.pad(x, ((0,0),(p,p),(p,p)), mode='reflect')
        out = np.zeros_like(x)
        for dy in range(ksize):
            for dx in range(ksize):
                out += xp[:, dy:dy+H, dx:dx+W]
        return out / (ksize * ksize)
    center = box_conv(images, 3)
    surround = box_conv(images, 7)
    dog = center - surround
    dog_mean = box_conv(dog, 7)
    dog_var = box_conv((dog - dog_mean)**2, 7)
    local_std = np.sqrt(dog_var + 0.001)
    normalized = dog / local_std
    normalized = np.clip(normalized, -4, 4)
    return ((normalized + 4) / 8.0).astype(np.float32)

# ── MultiScaleEye (from train_multiscale.py) ──
from graph import VisionGraph
from vision import VisionInterface

class MultiScaleEye:
    def __init__(self):
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
        self.shape_centers = None
        self.n_shape = 100
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
        lr = 0.1
        for ep in range(epochs):
            perm = np.random.permutation(len(imgs))
            for idx in perm:
                img = imgs[idx].copy()
                if contrast_aug and np.random.random() < 0.5:
                    mean = img.mean()
                    c = 0.3 + np.random.random() * 0.7
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
        vecs = []
        for eye in self.eyes:
            eye["vision"].set_image(image)
            nids = sorted(eye["graph"].nodes.keys())
            v = np.array([eye["graph"].nodes[nid].activation for nid in nids], dtype=np.float32)
            vecs.append(v)
        return np.concatenate(vecs)

    def build_shapes(self, imgs, labels, n_samples=5000):
        n = min(n_samples, len(imgs))
        idxs = np.random.choice(len(imgs), n, replace=False)
        vecs = []
        for i in idxs:
            vecs.append(self.get_activation_vector(imgs[i]))
        X = np.array(vecs, dtype=np.float32)
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X), self.n_shape, replace=False)
        centers = X[indices].copy()
        for it in range(50):
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            km_labels = np.argmin(dists, axis=1)
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
                break
        self.shape_centers = centers.astype(np.float32)

    def build_objects(self, imgs, labels, n_samples=5000):
        if self.shape_centers is None:
            raise RuntimeError("先 build_shapes()")
        n = min(n_samples, len(imgs))
        idxs = np.random.choice(len(imgs), n, replace=False)
        shape_vecs = []
        for i in idxs:
            shape_vecs.append(self.get_shape_vector(imgs[i]))
        X = np.array(shape_vecs, dtype=np.float32)
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
                break
        self.object_centers = centers.astype(np.float32)

    def get_shape_vector(self, image):
        base = self.get_activation_vector(image)
        sims = self.shape_centers @ base / (
            np.linalg.norm(self.shape_centers, axis=1) * np.linalg.norm(base) + 1e-8)
        return np.clip(sims, 0, 1).astype(np.float32)

    def get_object_vector(self, image):
        shape_vec = self.get_shape_vector(image)
        sims = self.object_centers @ shape_vec / (
            np.linalg.norm(self.object_centers, axis=1) * np.linalg.norm(shape_vec) + 1e-8)
        return np.clip(sims, 0, 1).astype(np.float32)

    def build_memory(self, imgs, lbls, size=5000):
        self.memory.clear()
        n = min(size, len(imgs))
        idxs = np.random.choice(len(imgs), n, replace=False)
        for idx in idxs:
            vec = self.get_object_vector(imgs[idx])
            self.memory.append((vec, lbls[idx]))

    def predict(self, image):
        q = self.get_object_vector(image)
        if not self.memory: return -1, 0.0
        best_sim, best_label = -1, -1
        for mvec, mlbl in self.memory:
            sim = np.dot(mvec, q) / (np.linalg.norm(mvec)*np.linalg.norm(q) + 1e-8)
            if sim > best_sim:
                best_sim, best_label = sim, mlbl
        return best_label, best_sim

# ── data loading ──
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

# ── main ──
np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()

# retina normalize
print("视网膜归一化...")
t0 = time.time()
X_tr_ret = retina_normalize(X_tr)
X_te_ret = retina_normalize(X_te)
print(f"  {time.time()-t0:.1f}s")

n_train = 10000
n_mem = 8000
n_test = 500

for name, tr_imgs, te_imgs in [
    ("RAW",  X_tr, X_te),
    ("RETINA", X_tr_ret, X_te_ret),
]:
    print(f"\n{'='*50}")
    print(f"  MultiScaleEye + {name}")
    print(f"{'='*50}")

    t0 = time.time()
    model = MultiScaleEye()
    model.init_templates(tr_imgs[:5000])
    model.train(tr_imgs[:n_train], y_tr[:n_train], epochs=3, contrast_aug=False)
    print(f"  训练: {time.time()-t0:.0f}s")

    t0 = time.time()
    model.build_shapes(tr_imgs[:15000], y_tr[:15000], n_samples=10000)
    model.build_objects(tr_imgs[:15000], y_tr[:15000], n_samples=5000)
    model.build_memory(tr_imgs[:15000], y_tr[:15000], size=n_mem)
    print(f"  K-means+记忆: {time.time()-t0:.0f}s")

    print(f"  {'c':>6s}  {'acc':>7s}")
    print(f"  {'-'*15}")
    for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]:
        if c == 1.0:
            test_batch = te_imgs[:n_test]
        else:
            test_batch = low_contrast(te_imgs[:n_test], c)
        correct = 0
        for i in range(n_test):
            pred, _ = model.predict(test_batch[i])
            if pred == y_te[i]: correct += 1
        print(f"  {c:5.2f}  {correct/n_test:6.1%}")

print("\n=== DONE ===")
