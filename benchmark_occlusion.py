#!/usr/bin/env python3
"""迭代闭环FB: 遮挡场景验证"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from train_multiscale import ColdEye, load_mnist, low_contrast

def occlude(images, block_size=8):
    occluded = images.copy()
    H, W = images.shape[1], images.shape[2]
    for i in range(len(images)):
        y = np.random.randint(0, H - block_size)
        x = np.random.randint(0, W - block_size)
        occluded[i, y:y+block_size, x:x+block_size] = images[i].mean()
    return occluded

np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
n_test = 500

print("训练 v3 (10K, 3ep)...")
model = ColdEye()
model.init_templates(X_tr[:5000])
model.train(X_tr, y_tr, epochs=3, n_train=10000)
model.build_memory(X_tr, y_tr, size=2000)
model.build_class_memory()

base = model.evaluate(X_te[:n_test], y_te[:n_test])
print(f"  干净: {base*100:.1f}%")

# ── 迭代FB: α + max_iter sweep ──
print(f"\n{'='*60}")
print("  迭代闭环FB — 遮挡 sweep")
print(f"{'='*60}")
print(f"  {'block':>6s}  {'no FB':>7s}  {'α.2/i3':>8s}  {'α.3/i3':>8s}  {'α.3/i5':>8s}  {'α.5/i3':>8s}")
print(f"  {'-'*48}")

for bs in [4, 6, 8, 10, 12]:
    test_occ = occlude(X_te[:n_test], block_size=bs)
    base = model.evaluate(test_occ, y_te[:n_test])
    row = f"  {bs:3d}×{bs:<3d}  {base*100:6.1f}%"
    best = base
    for a, mi in [(0.2,3),(0.3,3),(0.3,5),(0.5,3)]:
        correct = sum(1 for i in range(n_test)
                      if model.predict_with_feedback(test_occ[i], alpha=a, max_iter=mi)[0] == y_te[i])
        acc = correct / n_test
        best = max(best, acc)
        mark = "↑" if acc > base+0.005 else ("↓" if acc < base-0.005 else "≈")
        row += f"  {acc*100:7.1f}%{mark}"
    row += f"  best:{best*100:.1f}%"
    print(row)

# ── 遮挡+对比度 ──
print(f"\n{'='*60}")
print("  遮挡(10×10) + 低对比度 — 迭代FB (α=0.3, max_iter=3)")
print(f"{'='*60}")
print(f"  {'c':>6s}  {'no FB':>7s}  {'FB':>8s}  {'Δ':>7s}")
print(f"  {'-'*30}")
for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]:
    test_occ = occlude(X_te[:n_test], block_size=10)
    if c < 1.0: test_occ = low_contrast(test_occ, c)
    base = model.evaluate(test_occ, y_te[:n_test])
    correct = sum(1 for i in range(n_test)
                  if model.predict_with_feedback(test_occ[i], alpha=0.3, max_iter=3)[0] == y_te[i])
    fb = correct / n_test
    print(f"  {c:5.2f}  {base*100:6.1f}%  {fb*100:7.1f}%  {fb-base:+6.1%}")

print("\n=== DONE ===")
