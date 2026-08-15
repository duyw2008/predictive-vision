#!/usr/bin/env python3
"""CIFAR-10 灰度: ColdEye v3 (GlobalEye + PatchEye) — 冷眼核心路线: 形状基元 + 对比度不变性."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye, load_cifar10, low_contrast

np.random.seed(42)
print("加载 CIFAR-10 (灰度, 冷眼核心路线: 形状基元)...")
t0 = time.time()
X_tr, y_tr, X_te, y_te = load_cifar10(gray=True, n_train_per_class=2000, n_test_per_class=500)
print(f"  训练 {X_tr.shape} 测试 {X_te.shape} — {time.time()-t0:.0f}s")

# ColdEye v3: GlobalEye(100) + PatchEye(16×16/50) — 纯灰度形状
model = ColdEye()
print("\n训练 ColdEye (灰度)...")
t0 = time.time()
model.init_templates(X_tr[:2000])
model.train(X_tr, y_tr, epochs=5, n_train=20000, contrast_aug=True)
model.build_memory(X_tr, y_tr, size=5000)
print(f"  训练完成 {time.time()-t0:.0f}s")

# 分类 + 对比度不变性
print("\nCIFAR-10 RGB (10类) 结果:")
accs = []
for c in [1.0, 0.5, 0.1]:
    t = X_te if c == 1.0 else low_contrast(X_te, c)
    acc = model.evaluate(t, y_te)
    accs.append(acc)
    print(f"  c={c:.2f}: {acc:.1%}")

print(f"\n对比度衰减: {accs[0]-accs[-1]:.2%}  (应≈0)")
print(f"架构: {model.dim}d")
