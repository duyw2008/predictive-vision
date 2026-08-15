#!/usr/bin/env python3
"""迭代闭环推理 v2: 置信度门控 + 自适应融合 + 收敛检测"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye, load_mnist

def occlude(images, bs=8):
    oc = images.copy()
    for i in range(len(oc)):
        y = np.random.randint(0, 28 - bs); x = np.random.randint(0, 28 - bs)
        oc[i, y:y+bs, x:x+bs] = oc[i].mean()
    return oc

def add_noise(images, sigma=0.3):
    noisy = images.copy()
    noisy += np.random.randn(*noisy.shape).astype(np.float32) * sigma
    return np.clip(noisy, 0, 1)

np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
n_test = 200

print("训练 v3 (10K, 3ep)...")
model = ColdEye()
model.init_templates(X_tr[:5000])
model.train(X_tr, y_tr, epochs=3, n_train=10000)
model.build_memory(X_tr, y_tr, size=2000)

# 配对解码器: 综合退化 → 干净图
def mixed_degrade(imgs):
    # 50% occlusion, 50% noise
    mask = np.random.random(len(imgs)) < 0.5
    degraded = imgs.copy()
    degraded[mask] = occlude(imgs[mask], bs=8)
    degraded[~mask] = add_noise(imgs[~mask], sigma=0.2)
    return degraded

print("训练配对解码器: act(occlusion+noise) → clean...")
model.build_paired_decoder(X_tr, degrade_fn=mixed_degrade, n_samples=5000)

# ── 迭代闭环 sweep ──
print(f"\n{'='*70}")
print("  迭代闭环推理 v2: 遮挡 → 置信度门控 → 重建 → 重路由")
print(f"{'='*70}")
print(f"  {'block':>6s}  {'no BF':>7s}  {'α=0.3':>7s}  {'α=0.5':>7s}  {'α=0.8':>7s}  {'best Δ':>8s}")
print(f"  {'-'*50}")

for bs in [6, 8, 10, 12, 14]:
    test_occ = occlude(X_te[:n_test], bs=bs)
    base = model.evaluate(test_occ, y_te[:n_test])
    row = f"  {bs:3d}×{bs:<3d}  {base*100:6.1f}%"
    best_d = 0; best_a = 0
    for a in [0.3, 0.5, 0.8]:
        correct = 0
        for i in range(n_test):
            pred, _ = model.predict_brainfill(test_occ[i], alpha=a)
            if pred == y_te[i]: correct += 1
        acc = correct / n_test
        d = acc - base
        if d > best_d: best_d, best_a = d, a
        mark = "↑" if d > 0.005 else ("↓" if d < -0.005 else "≈")
        row += f"  {acc*100:6.1f}%{mark}"
    row += f"  {best_d:+.1%}(α={best_a})"
    print(row)

# ── 噪声 sweep ──
print(f"\n{'='*70}")
print("  迭代闭环推理 v2: 高斯噪声 → 重建去噪")
print(f"{'='*70}")
print(f"  {'σ':>6s}  {'no BF':>7s}  {'α=0.3':>7s}  {'α=0.5':>7s}  {'α=0.8':>7s}  {'best Δ':>8s}")
print(f"  {'-'*50}")

for sigma in [0.1, 0.2, 0.3, 0.4]:
    test_noisy = add_noise(X_te[:n_test], sigma=sigma)
    base = model.evaluate(test_noisy, y_te[:n_test])
    row = f"  {sigma:.1f}   {base*100:6.1f}%"
    best_d = 0
    for a in [0.3, 0.5, 0.8]:
        correct = sum(1 for i in range(n_test)
                      if model.predict_brainfill(test_noisy[i], alpha=a)[0] == y_te[i])
        acc = correct / n_test
        d = acc - base
        if d > best_d: best_d = d
        mark = "↑" if d > 0.005 else ("↓" if d < -0.005 else "≈")
        row += f"  {acc*100:6.1f}%{mark}"
    row += f"  {best_d:+.1%}"
    print(row)

# ── 迭代统计 ──
print(f"\n  ── 平均迭代次数 (block=12, α=0.5) ──")
iters = []
test_occ = occlude(X_te[:100], bs=12)
for i in range(100):
    # monkey-patch to count iterations
    act = model._activate_one(test_occ[i])
    n = 0
    for it in range(8):
        best_sim = max(np.dot(mv, act) / (np.linalg.norm(mv)*np.linalg.norm(act)+1e-8)
                       for mv, _ in model.memory)
        if best_sim >= 0.7: break
        recon = (act @ model.W_paired).reshape(28, 28)
        act_recon = model._activate_one(recon)
        adaptive_alpha = 0.5 * (1.0 - min(best_sim, 0.9))
        act = act * (1-adaptive_alpha) + act_recon * adaptive_alpha
        n += 1
    iters.append(n)
print(f"  mean={np.mean(iters):.1f}  median={np.median(iters):.0f}  "
      f"early-stop={(np.array(iters)<8).sum()}/{len(iters)}")

print("\n=== DONE ===")
