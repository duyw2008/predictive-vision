#!/usr/bin/env python3
"""合成数据脑补训练 + 评估"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye

np.random.seed(42)
d = np.load("data/synthetic/scenes.npz")
clean, degraded = d['clean'], d['degraded']
print(f"加载: clean={clean.shape}, degraded={degraded.shape}")

# 拆分
n_train = 8000
tr_clean, tr_deg = clean[:n_train], degraded[:n_train]
te_clean, te_deg = clean[n_train:], degraded[n_train:]

# 大容量 ColdEye (合成场景更复杂)
model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
])
model.init_templates(tr_clean[:5000])
model.train(tr_clean, np.zeros(n_train), epochs=5, n_train=n_train, contrast_aug=True)
print(f"模板训练完成 ({model.dim}d)")

# 配对脑补解码器: act(退化) → clean (直接配对, 不用 degrade_fn)
print("训练配对解码器: act(退化场景) → 干净场景...")
# 手动配对: 直接用 tr_deg 和 tr_clean 的对应关系
acts = model._activate_batch(tr_deg)  # [N, 300] from degraded
flat_clean = tr_clean.reshape(len(tr_clean), -1).astype(np.float32)
ATA = acts.T @ acts
ATI = acts.T @ flat_clean
model.W_paired = (np.linalg.pinv(ATA) @ ATI).astype(np.float32)

# 评估重建质量
print(f"\n{'='*60}")
print("  脑补重建质量 (MSE: 退化 vs 脑补重建)")
print(f"{'='*60}")

n_eval = 200
mse_deg_total = 0
mse_recon_total = 0
for i in range(n_eval):
    deg = te_deg[i]
    clean_img = te_clean[i]
    recon = model.reconstruct_paired(deg)
    mse_deg_total += np.mean((clean_img - deg) ** 2)
    mse_recon_total += np.mean((clean_img - recon) ** 2)

mse_deg = mse_deg_total / n_eval
mse_recon = mse_recon_total / n_eval
print(f"  MSE(退化→干净) = {mse_deg:.5f}")
print(f"  MSE(脑补→干净) = {mse_recon:.5f}")
print(f"  改善 = {mse_deg - mse_recon:+.5f} ({(mse_deg-mse_recon)/mse_deg*100:.1f}%)")

# 可视化几个重建
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i in range(3):
    axes[i,0].imshow(te_clean[i], cmap='gray'); axes[i,0].set_title('Clean')
    axes[i,1].imshow(te_deg[i], cmap='gray'); axes[i,1].set_title('Degraded')
    axes[i,2].imshow(model.reconstruct_paired(te_deg[i]), cmap='gray'); axes[i,2].set_title('Brainfill')
    for j in range(3):
        axes[i,j].axis('off')
plt.tight_layout()
os.makedirs("data/synthetic", exist_ok=True)
plt.savefig("data/synthetic/brainfill_demo.png", dpi=100)
print(f"\n保存 data/synthetic/brainfill_demo.png")
print("=== DONE ===")
