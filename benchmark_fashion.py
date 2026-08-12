#!/usr/bin/env python3
"""ColdEye v3 on Fashion-MNIST — 零代码改动，只换数据"""
import sys, os, time, numpy as np, gzip, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from train_multiscale import ColdEye, low_contrast

FASHION_CLASSES = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def load_fashion_mnist(d="data"):
    os.makedirs(d, exist_ok=True)
    base = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    files = {
        "ti": "train-images-idx3-ubyte.gz",
        "tl": "train-labels-idx1-ubyte.gz",
        "ei": "t10k-images-idx3-ubyte.gz",
        "el": "t10k-labels-idx1-ubyte.gz",
    }
    for fname in files.values():
        p = os.path.join(d, fname)
        if not os.path.exists(p):
            urllib.request.urlretrieve(base + fname, p)

    def limg(p):
        with gzip.open(p) as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32) / 255
    def llbl(p):
        with gzip.open(p) as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

    return (limg(os.path.join(d, files["ti"])), llbl(os.path.join(d, files["tl"])),
            limg(os.path.join(d, files["ei"])), llbl(os.path.join(d, files["el"])))


np.random.seed(42)
print("Loading Fashion-MNIST...")
X_tr, y_tr, X_te, y_te = load_fashion_mnist()
print(f"  train: {X_tr.shape}, test: {X_te.shape}")

print("\nColdEye v3 — Fashion-MNIST")
print(f"  训练: 60K, 5ep, contrast_aug=True")

model = ColdEye()
model.init_templates(X_tr[:5000])
model.train(X_tr, y_tr, epochs=5, n_train=60000, contrast_aug=True)
model.build_memory(X_tr, y_tr, size=5000)

n_test = 500
print(f"\n  {'c':>6s}  {'acc':>7s}")
print(f"  {'-'*15}")
for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05]:
    if c == 1.0:
        test_batch = X_te[:n_test]
    else:
        test_batch = low_contrast(X_te[:n_test], c)
    acc = model.evaluate(test_batch, y_te[:n_test])
    print(f"  {c:5.2f}  {acc*100:6.1f}%")

print("\n=== DONE ===")
