#!/usr/bin/env python3
"""配对脑补 + 闭环推理: 遮挡 → 重建干净图 → 重路由 → 修正分类"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from train_multiscale import ColdEye, load_mnist

def occlude_block(images, bs=8):
    oc = images.copy()
    for i in range(len(oc)):
        y = np.random.randint(0, 28 - bs)
        x = np.random.randint(0, 28 - bs)
        oc[i, y:y+bs, x:x+bs] = oc[i].mean()
    return oc

np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
n_test = 200

print("训练 v3 (10K, 3ep)...")
model = ColdEye()
model.init_templates(X_tr[:5000])
model.train(X_tr, y_tr, epochs=3, n_train=10000)
model.build_memory(X_tr, y_tr, size=2000)

# 配对解码器: 遮挡(8x8)激活 → 干净图
print("训练配对解码器: act(遮挡) → clean...")
model.build_paired_decoder(X_tr, degrade_fn=lambda imgs: occlude_block(imgs, bs=8), n_samples=5000)

# ── 遮挡 sweep ──
print(f"\n{'='*65}")
print("  闭环脑补: 遮挡 → 重建 → 重路由 → KNN")
print(f"  (decoder trained on 8x8 occlusion)")
print(f"{'='*65}")
print(f"  {'block':>6s}  {'no BF':>7s}  {'α=0.3':>7s}  {'α=0.5':>7s}  {'α=0.8':>7s}  {'best Δ':>8s}")
print(f"  {'-'*50}")

for bs in [6, 8, 10, 12, 14]:
    test_occ = occlude_block(X_te[:n_test], bs=bs)
    base = model.evaluate(test_occ, y_te[:n_test])
    row = f"  {bs:3d}×{bs:<3d}  {base*100:6.1f}%"
    best_d = 0
    best_alpha = 0
    for a in [0.3, 0.5, 0.8]:
        correct = 0
        for i in range(n_test):
            pred, _ = model.predict_brainfill(test_occ[i], alpha=a, n_iter=2)
            if pred == y_te[i]: correct += 1
        acc = correct / n_test
        d = acc - base
        if d > best_d: best_d, best_alpha = d, a
        mark = "↑" if d > 0.005 else ("↓" if d < -0.005 else "≈")
        row += f"  {acc*100:6.1f}%{mark}"
    row += f"  {best_d:+.1%}(α={best_alpha})"
    print(row)

# ── 重建质量 ──
print(f"\n{'='*65}")
print("  脑补重建质量: 遮挡 → 重建")
print(f"{'='*65}")

img = X_te[0]  # digit 7
for bs in [6, 8, 10, 12]:
    occ = occlude_block(img.reshape(1, 28, 28), bs=bs)[0]
    recon = model.reconstruct_paired(occ)
    mse_occ = np.mean((img - occ) ** 2)
    mse_recon = np.mean((img - recon) ** 2)
    gain = mse_occ - mse_recon
    print(f"  block={bs}×{bs}: MSE(occ)={mse_occ:.4f}  MSE(recon)={mse_recon:.4f}  gain={gain:+.4f}")

print("\n=== DONE ===")
