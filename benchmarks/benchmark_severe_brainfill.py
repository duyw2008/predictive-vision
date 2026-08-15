#!/usr/bin/env python3
"""重度退化脑补: 大面积擦除 → 脑补能否恢复被摧毁的信息"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye

np.random.seed(42)
d = np.load("data/synthetic/scenes.npz")
clean, _ = d['clean'], d['degraded']

n_train = 8000
tr_clean = clean[:n_train]
te_clean = clean[n_train:]

model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
])
model.init_templates(tr_clean[:5000])
model.train(tr_clean, np.zeros(n_train), epochs=5, n_train=n_train, contrast_aug=True)

def severe_erase(images, frac=0.4):
    """擦除图像 40% 区域 (整块擦除, 摧毁信息)"""
    out = images.copy()
    H, W = images.shape[1], images.shape[2]
    for i in range(len(images)):
        # 随机擦除一大块 + 两小块
        # 大块
        bh = int(H * np.random.uniform(0.3, 0.5))
        bw = int(W * np.random.uniform(0.3, 0.5))
        y = np.random.randint(0, H-bh+1); x = np.random.randint(0, W-bw+1)
        out[i, y:y+bh, x:x+bw] = 0
        # 两小块
        for _ in range(2):
            sh = np.random.randint(3, 8); sw = np.random.randint(3, 8)
            y2 = np.random.randint(0, H-sh); x2 = np.random.randint(0, W-sw)
            out[i, y2:y2+sh, x2:x2+sw] = 0
    return out

# 训练配对解码器 (重度擦除)
tr_deg = severe_erase(tr_clean)
acts_deg = model._activate_batch(tr_deg)
ATA = acts_deg.T @ acts_deg
ATI = acts_deg.T @ tr_clean.reshape(n_train, -1).astype(np.float32)
W_linear = (np.linalg.pinv(ATA) @ ATI).astype(np.float32)

# 记忆式
mem_acts = model._activate_batch(tr_clean)
mem_imgs = tr_clean.reshape(n_train, -1)

def linear_bf(img):
    return (model._activate_one(img) @ W_linear).reshape(28, 28)

def memory_bf(img, k=10):
    act = model._activate_one(img)
    sims = mem_acts @ act / (np.linalg.norm(mem_acts, axis=1)*np.linalg.norm(act)+1e-8)
    top = np.argsort(sims)[-k:]
    w = sims[top]; w = w/(w.sum()+1e-8)
    return (mem_imgs[top].T @ w).reshape(28, 28)

# 评估
n_eval = 200
te_deg = severe_erase(te_clean[:n_eval])
print(f"\n{'='*60}")
print("  重度擦除 (40% 信息摧毁) 脑补对比")
print(f"{'='*60}")

mse_deg = np.mean([np.mean((te_clean[i]-te_deg[i])**2) for i in range(n_eval)])
print(f"  退化基线 MSE: {mse_deg:.5f}")

for name, fn in [("线性解码器", linear_bf), ("记忆式 k=5", lambda x: memory_bf(x,5)), ("记忆式 k=20", lambda x: memory_bf(x,20))]:
    mse = np.mean([np.mean((te_clean[i]-fn(te_deg[i]))**2) for i in range(n_eval)])
    print(f"  {name}: MSE={mse:.5f}  改善 {(mse_deg-mse)/mse_deg*100:+.1f}%")

# 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 3, figsize=(9, 12))
for i in range(3):
    axes[0,i].imshow(te_clean[i], cmap='gray'); axes[0,i].set_title('Clean')
    axes[1,i].imshow(te_deg[i], cmap='gray'); axes[1,i].set_title('Erased 40%')
    axes[2,i].imshow(linear_bf(te_deg[i]), cmap='gray'); axes[2,i].set_title('Linear')
    axes[3,i].imshow(memory_bf(te_deg[i], k=20), cmap='gray'); axes[3,i].set_title('Memory k=20')
    for j in range(4): axes[j,i].axis('off')
plt.tight_layout()
plt.savefig("data/synthetic/brainfill_severe.png", dpi=100)
print(f"\n保存 data/synthetic/brainfill_severe.png")
print("=== DONE ===")
