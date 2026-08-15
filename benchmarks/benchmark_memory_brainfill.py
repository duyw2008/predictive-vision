#!/usr/bin/env python3
"""记忆式脑补 vs 线性解码器 — 合成复杂场景"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye

np.random.seed(42)
d = np.load("data/synthetic/scenes.npz")
clean, degraded = d['clean'], d['degraded']

n_train = 8000
tr_clean, tr_deg = clean[:n_train], degraded[:n_train]
te_clean, te_deg = clean[n_train:], degraded[n_train:]

model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
])
model.init_templates(tr_clean[:5000])
model.train(tr_clean, np.zeros(n_train), epochs=5, n_train=n_train, contrast_aug=True)

# ── 记忆式脑补: 存 (激活, 干净图) 配对 ──
print("建记忆: (clean_activation, clean_image) 配对...")
mem_acts = model._activate_batch(tr_clean)   # [N, 300] from clean
mem_imgs = tr_clean.reshape(n_train, -1)     # [N, 784] clean images

def memory_brainfill(deg_image, k=5):
    """退化图 → 激活 → 找K近邻干净激活 → 加权干净图 = 重建"""
    act = model._activate_one(deg_image)
    sims = mem_acts @ act / (np.linalg.norm(mem_acts, axis=1) * np.linalg.norm(act) + 1e-8)
    top_k = np.argsort(sims)[-k:]
    weights = sims[top_k]
    weights = weights / (weights.sum() + 1e-8)
    recon = (mem_imgs[top_k].T @ weights).reshape(28, 28)
    return recon

# ── 线性解码器 (对照) ──
acts_deg = model._activate_batch(tr_deg)
ATA = acts_deg.T @ acts_deg
ATI = acts_deg.T @ tr_clean.reshape(n_train, -1).astype(np.float32)
W_linear = (np.linalg.pinv(ATA) @ ATI).astype(np.float32)

def linear_brainfill(deg_image):
    act = model._activate_one(deg_image)
    return (act @ W_linear).reshape(28, 28)

# ── 评估 ──
n_eval = 200
print(f"\n{'='*65}")
print("  重建质量对比 (MSE 越低越好, 相对退化改善)")
print(f"{'='*65}")

for name, fn in [("线性解码器", linear_brainfill), ("记忆式(k=5)", memory_brainfill)]:
    mse_total = 0
    mse_deg_total = 0
    for i in range(n_eval):
        recon = fn(te_deg[i])
        mse_total += np.mean((te_clean[i] - recon) ** 2)
        mse_deg_total += np.mean((te_clean[i] - te_deg[i]) ** 2)
    mse = mse_total / n_eval
    mse_deg = mse_deg_total / n_eval
    print(f"  {name}: MSE={mse:.5f}  (退化基线 {mse_deg:.5f}, 改善 {mse_deg-mse:+.5f} / {(mse_deg-mse)/mse_deg*100:+.1f}%)")

# 记忆式不同 k
for k in [1, 3, 5, 10, 20]:
    mse_total = 0
    for i in range(n_eval):
        recon = memory_brainfill(te_deg[i], k=k)
        mse_total += np.mean((te_clean[i] - recon) ** 2)
    print(f"  记忆式 k={k:2d}: MSE={mse_total/n_eval:.5f}")

# 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 3, figsize=(9, 12))
titles = ['Clean', 'Degraded', 'Linear', 'Memory(k=5)']
for i in range(3):
    axes[0,i].imshow(te_clean[i], cmap='gray'); axes[0,i].set_title('Clean')
    axes[1,i].imshow(te_deg[i], cmap='gray'); axes[1,i].set_title('Degraded')
    axes[2,i].imshow(linear_brainfill(te_deg[i]), cmap='gray'); axes[2,i].set_title('Linear')
    axes[3,i].imshow(memory_brainfill(te_deg[i], k=5), cmap='gray'); axes[3,i].set_title('Memory(k=5)')
    for j in range(4):
        axes[j,i].axis('off')
plt.tight_layout()
plt.savefig("data/synthetic/brainfill_compare.png", dpi=100)
print(f"\n保存 data/synthetic/brainfill_compare.png")
print("=== DONE ===")
