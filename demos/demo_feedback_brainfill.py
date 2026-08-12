#!/usr/bin/env python3
"""预测反馈 v2: KNN记忆平均 + 脑补 demo"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from train_multiscale import ColdEye, load_mnist, low_contrast

np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
n_test = 500

print("训练 v3 (10K, 3ep)...")
model = ColdEye()
model.init_templates(X_tr[:5000])
model.train(X_tr, y_tr, epochs=3, n_train=10000)
model.build_memory(X_tr, y_tr, size=2000)

# ── FB: α + k sweep ──
print(f"\n{'='*65}")
print("  FB v2: KNN记忆平均 — α × k sweep")
print(f"{'='*65}")

test_batch_full = X_te[:n_test]
base = model.evaluate(test_batch_full, y_te[:n_test])
print(f"  no FB: {base*100:.1f}%")
print(f"  {'k':>3s}  {'α=0.3':>7s}  {'α=0.5':>7s}  {'α=0.8':>7s}")
print(f"  {'-'*28}")

for k in [3, 5, 10, 20]:
    print(f"  {k:3d}", end="")
    for a in [0.3, 0.5, 0.8]:
        correct = 0
        for i in range(n_test):
            pred, _ = model.predict_with_feedback(test_batch_full[i], alpha=a, k=k)
            if pred == y_te[i]: correct += 1
        acc = correct / n_test
        mark = "↑" if acc > base + 0.005 else ("↓" if acc < base - 0.005 else "≈")
        print(f"  {acc*100:6.1f}%{mark}", end="")
    print()

# ── Per-contrast best FB ──
print(f"\n{'='*65}")
print("  Best FB per contrast (k=5, α sweep)")
print(f"{'='*65}")
print(f"  {'c':>6s}  {'no FB':>7s}  {'α=0.1':>7s}  {'α=0.2':>7s}  {'α=0.3':>7s}  {'α=0.5':>7s}")
print(f"  {'-'*42}")

for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]:
    test_batch = X_te[:n_test] if c == 1.0 else low_contrast(X_te[:n_test], c)
    base = model.evaluate(test_batch, y_te[:n_test])
    print(f"  {c:5.2f}  {base*100:6.1f}%", end="")
    for a in [0.1, 0.2, 0.3, 0.5]:
        correct = sum(1 for i in range(n_test)
                      if model.predict_with_feedback(test_batch[i], alpha=a, k=5)[0] == y_te[i])
        acc = correct / n_test
        mark = "↑" if acc > base + 0.003 else ("↓" if acc < base - 0.003 else "≈")
        print(f"  {acc*100:6.1f}%{mark}", end="")
    print()

# ── 脑补 ──
print(f"\n{'='*65}")
print("  脑补: 低对比度重建 (decoder on 5K samples)")
print(f"{'='*65}")
model.build_decoder(X_tr, n_samples=5000)

img = X_te[0]
for c in [1.0, 0.1, 0.05, 0.03]:
    img_lc = img.copy() if c == 1.0 else low_contrast(img.reshape(1,28,28), c)[0]
    recon = model.reconstruct(img_lc)
    mse = np.mean((img_lc - recon)**2)
    print(f"  c={c:.2f}  MSE={mse:.6f}  std(in)={img_lc.std():.4f}  std(out)={recon.std():.4f}")

print("\n=== DONE ===")
