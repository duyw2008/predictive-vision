#!/usr/bin/env python3
"""ColdEye v3 on CIFAR-10 — 灰度 + 增容量 + contrast_aug"""
import sys, os, time, numpy as np, pickle, tarfile
sys.path.insert(0, os.path.dirname(__file__))
from train_multiscale import ColdEye, low_contrast

CIFAR_CLASSES = ['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck']

def load_cifar10(d="data"):
    tgz = os.path.join(d, "cifar-10-python.tar.gz")
    extracted = os.path.join(d, "cifar-10-batches-py")
    if not os.path.exists(extracted):
        with tarfile.open(tgz) as tf: tf.extractall(d)

    X_tr, y_tr = [], []
    for i in range(1, 6):
        with open(os.path.join(extracted, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="bytes")
            X_tr.append(batch[b"data"])
            y_tr.extend(batch[b"labels"])
    X_tr = np.vstack(X_tr).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255

    with open(os.path.join(extracted, "test_batch"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        X_te = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255
        y_te = np.array(batch[b"labels"], dtype=np.int64)

    # 灰度
    gray = lambda x: (x[...,0]*0.299 + x[...,1]*0.587 + x[...,2]*0.114).astype(np.float32)
    return gray(X_tr), np.array(y_tr, dtype=np.int64), gray(X_te), y_te


np.random.seed(42)
if not os.path.exists("data/cifar-10-batches-py"):
    print("CIFAR-10 未下载，等待...")
    sys.exit(1)

X_tr, y_tr, X_te, y_te = load_cifar10()
print(f"CIFAR-10 grayscale: train {X_tr.shape}, test {X_te.shape}")

# 增容量: 200 global + 100 coarse = 300d
print("\nColdEye — global(200n) + coarse 16×16(100n) = 300d")
print("  训练: 50K, 5ep, contrast_aug=True")

model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
])
model.init_templates(X_tr[:5000])
model.train(X_tr, y_tr, epochs=5, n_train=50000, contrast_aug=True)
model.build_memory(X_tr, y_tr, size=5000)

n_test = 500
print(f"\n  {'c':>6s}  {'acc':>7s}")
print(f"  {'-'*15}")
for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]:
    if c == 1.0:
        test_batch = X_te[:n_test]
    else:
        test_batch = low_contrast(X_te[:n_test], c)
    acc = model.evaluate(test_batch, y_te[:n_test])
    print(f"  {c:5.2f}  {acc*100:6.1f}%")

print(f"\n  架构: {model.dim}d")
print(f"  CIFAR-10 CNN baseline (2-conv): ~70%")
print("=== DONE ===")
