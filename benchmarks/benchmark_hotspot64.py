#!/usr/bin/env python3
"""
64×64 空间热点脑补 vs 全局 — 高分辨率下热点是否打破线性天花板
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye

np.random.seed(42)
d = np.load("data/synthetic/scenes64.npz")
clean, degraded = d['clean'], d['degraded']
n_train = 5000
tr_clean, tr_deg = clean[:n_train], degraded[:n_train]
te_clean, te_deg = clean[n_train:n_train+150], degraded[n_train:n_train+150]
print(f"加载: clean={clean.shape}, train={n_train}, test=150")

# 64×64: 大 patch 眼 (16×16 和 32×32) + 全局眼
model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},              # 4096-dim
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
    {"type": "patch", "ps": 32, "st": 16, "n": 50},
])
model.init_templates(tr_clean[:3000])
model.train(tr_clean, np.zeros(n_train), epochs=4, n_train=n_train, contrast_aug=True)
print(f"模板训练完成 ({model.dim}d)")

# ── 全局解码器 (baseline) ──
acts_deg = model._activate_batch(tr_deg)
flat_clean = tr_clean.reshape(n_train, -1).astype(np.float32)
W_global = np.linalg.pinv(acts_deg.T @ acts_deg) @ (acts_deg.T @ flat_clean)

def global_bf(img):
    return (model._activate_one(img) @ W_global).reshape(64, 64)

# ── 空间热点解码器 ──
# 用 patch 眼的热点激活 (6×6 区域 — 9600→5400 维, 避免 pinv 欠定系统慢)
patch_eyes = [e for e in model.eyes if hasattr(e, 'hotspot_activation')]
acts_hot = np.array([np.concatenate([e.hotspot_activation(tr_deg[i], n_regions=6) for e in patch_eyes])
                     for i in range(n_train)], np.float32)
print(f"热点激活维度: {acts_hot.shape[1]} (vs 全局 {acts_deg.shape[1]})")
# lstsq: 对欠定系统稳定 (thin SVD of [N, D], 不是 pinv of [D, D])
W_hot = np.linalg.lstsq(acts_hot, flat_clean, rcond=1e-4)[0].astype(np.float32)

def hotspot_bf(img):
    act = np.concatenate([e.hotspot_activation(img, n_regions=6) for e in patch_eyes])
    return (act @ W_hot).reshape(64, 64)

# ── 评估 ──
n_eval = len(te_deg)
mse_deg = np.mean([np.mean((te_clean[i]-te_deg[i])**2) for i in range(n_eval)])
print(f"\n{'='*60}")
print(f"  64×64 空间热点 vs 全局脑补 (重度擦除 40%)")
print(f"{'='*60}")
print(f"  退化基线 MSE: {mse_deg:.5f}\n")

for name, fn in [("全局 (200+150d)", global_bf), ("空间热点 (8×8)", hotspot_bf)]:
    mse = np.mean([np.mean((te_clean[i]-fn(te_deg[i]))**2) for i in range(n_eval)])
    print(f"  {name:>18s}: MSE={mse:.5f}  改善 {(mse_deg-mse)/mse_deg*100:+.1f}%")

# 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 3, figsize=(9, 12))
for i in range(3):
    axes[0,i].imshow(te_clean[i], cmap='gray'); axes[0,i].set_title('Clean 64×64')
    axes[1,i].imshow(te_deg[i], cmap='gray'); axes[1,i].set_title('Erased 40%')
    axes[2,i].imshow(global_bf(te_deg[i]), cmap='gray'); axes[2,i].set_title('Global')
    axes[3,i].imshow(hotspot_bf(te_deg[i]), cmap='gray'); axes[3,i].set_title('Hotspot 8×8')
    for j in range(4): axes[j,i].axis('off')
plt.tight_layout()
os.makedirs("data/synthetic", exist_ok=True)
plt.savefig("data/synthetic/hotspot64_compare.png", dpi=100)
print(f"\n保存 data/synthetic/hotspot64_compare.png")
print("=== DONE ===")
