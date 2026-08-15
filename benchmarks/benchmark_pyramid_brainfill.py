#!/usr/bin/env python3
"""
多尺度金字塔脑补 — 打破线性解码器天花板

核心洞察: 竞争路由 activate() 是非线性的 (hard assignment),
所以 decode→activate→decode 级联引入非线性, 突破单层线性解码器的上限。

对比:
  1. 单层线性 (baseline):   act → 28×28
  2. Laplacian 残差:        coarse(14×14) + residual(28×28)  [纯线性]
  3. 金字塔重路由:           act → coarse(14×14) → upsample → re-activate → fine(28×28)
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye

def downsample(img, k=2):
    return img[::k, ::k]

def upsample(img, k=2):
    return np.kron(img, np.ones((k, k), dtype=img.dtype))

def severe_erase(images, frac=0.4, seed=0):
    rng = np.random.RandomState(seed)
    out = images.copy()
    H, W = images.shape[1], images.shape[2]
    for i in range(len(images)):
        bh = int(H * rng.uniform(0.3, 0.5)); bw = int(W * rng.uniform(0.3, 0.5))
        y = rng.randint(0, H-bh+1); x = rng.randint(0, W-bw+1)
        out[i, y:y+bh, x:x+bw] = 0
        for _ in range(2):
            sh = rng.randint(3, 8); sw = rng.randint(3, 8)
            y2 = rng.randint(0, H-sh); x2 = rng.randint(0, W-sw)
            out[i, y2:y2+sh, x2:x2+sw] = 0
    return out

np.random.seed(42)
d = np.load("data/synthetic/scenes.npz")
clean, _ = d['clean'], d['degraded']
n_train = 8000
tr_clean = clean[:n_train]
te_clean = clean[n_train:n_train+200]
te_deg = severe_erase(te_clean, seed=123)

model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
])
model.init_templates(tr_clean[:5000])
model.train(tr_clean, np.zeros(n_train), epochs=5, n_train=n_train, contrast_aug=True)

tr_deg = severe_erase(tr_clean, seed=0)
acts = model._activate_batch(tr_deg)          # [N, 300] from degraded
flat_clean = tr_clean.reshape(n_train, -1)     # [N, 784]

# ── 1. 单层线性 (baseline) ──
W_single = np.linalg.pinv(acts.T @ acts) @ (acts.T @ flat_clean)

def single_bf(img):
    return (model._activate_one(img) @ W_single).reshape(28, 28)

# ── 2. Laplacian 残差 (纯线性) ──
clean_coarse = downsample(tr_clean, 2).reshape(n_train, -1)  # [N, 196]
W_coarse = np.linalg.pinv(acts.T @ acts) @ (acts.T @ clean_coarse)

coarse_recon = acts @ W_coarse                        # [N, 196]
coarse_recon_28 = np.array([upsample(c.reshape(14,14)).reshape(-1) for c in coarse_recon])
residual = flat_clean - coarse_recon_28               # [N, 784]
W_residual = np.linalg.pinv(acts.T @ acts) @ (acts.T @ residual)

def laplacian_bf(img):
    act = model._activate_one(img)
    coarse = upsample((act @ W_coarse).reshape(14, 14)).reshape(-1)
    resid = act @ W_residual
    return (coarse + resid).reshape(28, 28)

# ── 3. 金字塔重路由 (非线性) ──
# re-activate coarse reconstruction (非线性 via hard assignment)
coarse_recon_imgs = np.array([upsample((a @ W_coarse).reshape(14,14)) for a in acts])
acts2 = model._activate_batch(coarse_recon_imgs)     # [N, 300] re-routed
W_fine = np.linalg.pinv(acts2.T @ acts2) @ (acts2.T @ flat_clean)

def pyramid_bf(img):
    act = model._activate_one(img)
    coarse = upsample((act @ W_coarse).reshape(14, 14))
    act2 = model._activate_one(coarse)
    return (act2 @ W_fine).reshape(28, 28)

# ── 评估 ──
n_eval = len(te_deg)
mse_deg = np.mean([np.mean((te_clean[i]-te_deg[i])**2) for i in range(n_eval)])
print(f"{'='*60}")
print(f"  多尺度金字塔脑补 vs 单层线性 (重度擦除 40%)")
print(f"{'='*60}")
print(f"  退化基线 MSE: {mse_deg:.5f}\n")

for name, fn in [("单层线性", single_bf), ("Laplacian残差", laplacian_bf), ("金字塔重路由", pyramid_bf)]:
    mse = np.mean([np.mean((te_clean[i]-fn(te_deg[i]))**2) for i in range(n_eval)])
    print(f"  {name:>12s}: MSE={mse:.5f}  改善 {(mse_deg-mse)/mse_deg*100:+.1f}%")

# 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(5, 3, figsize=(9, 15))
for i in range(3):
    axes[0,i].imshow(te_clean[i], cmap='gray'); axes[0,i].set_title('Clean')
    axes[1,i].imshow(te_deg[i], cmap='gray'); axes[1,i].set_title('Erased 40%')
    axes[2,i].imshow(single_bf(te_deg[i]), cmap='gray'); axes[2,i].set_title('Single linear')
    axes[3,i].imshow(laplacian_bf(te_deg[i]), cmap='gray'); axes[3,i].set_title('Laplacian')
    axes[4,i].imshow(pyramid_bf(te_deg[i]), cmap='gray'); axes[4,i].set_title('Pyramid re-route')
    for j in range(5): axes[j,i].axis('off')
plt.tight_layout()
os.makedirs("data/synthetic", exist_ok=True)
plt.savefig("data/synthetic/pyramid_compare.png", dpi=100)
print(f"\n保存 data/synthetic/pyramid_compare.png")
print("=== DONE ===")
