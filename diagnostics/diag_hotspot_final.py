#!/usr/bin/env python3
"""端到端验证: ColdEye v3 固化热点重建 — 分类不变 + 热点脑补优于全局."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye, load_mnist, low_contrast

np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()

print("训练 ColdEye v3 (global100 + patch16×16/50)...")
t0 = time.time()
model = ColdEye()
model.init_templates(X_tr[:5000])
model.train(X_tr, y_tr, epochs=3, n_train=10000, contrast_aug=True)
model.build_memory(X_tr, y_tr, size=5000)
print(f"  训练完成 {time.time()-t0:.0f}s")

# 1. 分类准确率不变 (c=1.0 应 ~84%)
acc = model.evaluate(X_te[:1000], y_te[:1000])
print(f"\n[1] 分类 c=1.0: {acc:.1%}  (基线 84.4%, 热点固化不应破坏)")

# 2. 空间热点已累积 (PatchEye 节点)
patch_eye = [e for e in model.eyes if type(e).__name__ == 'PatchEye'][0]
n_hot = sum(1 for nid in patch_eye.nids if patch_eye.g.nodes[nid].spatial_hotspot is not None)
print(f"[2] PatchEye 节点空间热点: {n_hot}/{len(patch_eye.nids)} 已累积")

# 3. 脑补对比: 全局 vs 热点
print("\n[3] 脑补重建 MSE 对比 (56×56 放大, 降质 c=0.2)...")
# 用 28×28 直接重建 (ColdEye 标准尺寸)
model.build_decoder(X_tr[:5000], n_samples=5000)
model.build_decoder_hotspot(X_tr[:5000], n_samples=5000, n_regions=4)

# 测试几个数字
test_idx = [3, 5, 18, 0, 1, 2, 7]
for t_idx in test_idx:
    img = X_te[t_idx]; d = int(y_te[t_idx])
    m = img.mean(); deg = m + (img - m) * 0.2
    r_glob = model.reconstruct(deg)
    r_hot = model.reconstruct_hotspot(deg, n_regions=4)
    mse_g = np.mean((r_glob - img)**2)
    mse_h = np.mean((r_hot - img)**2)
    tag = "✅" if mse_h < mse_g else "  "
    print(f"  digit={d}: 全局={mse_g:.4f}  热点={mse_h:.4f}  {tag}")

print("\n=== 验证完成 ===")
