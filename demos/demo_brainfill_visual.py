#!/usr/bin/env python3
"""
脑补图像级 demo — 干净 | 降质 | 脑补重建 三列对比
覆盖: 低对比度 / 遮挡 / 重度擦除
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye, load_mnist, low_contrast

np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()

# 训练 (10K, 3ep 足够)
print("训练 ColdEye...")
model = ColdEye()
model.init_templates(X_tr[:5000])
model.train(X_tr, y_tr, epochs=3, n_train=10000)

# 配对解码器: 遮挡 → 干净
def occlude(images, bs=8):
    o = images.copy()
    for i in range(len(o)):
        y = np.random.randint(0, 28-bs); x = np.random.randint(0, 28-bs)
        o[i, y:y+bs, x:x+bs] = o[i].mean()
    return o

def severe_erase(images):
    rng = np.random.RandomState(0)
    o = images.copy()
    for i in range(len(o)):
        bh = rng.randint(8, 14); bw = rng.randint(8, 14)
        y = rng.randint(0, 28-bh); x = rng.randint(0, 28-bw)
        o[i, y:y+bh, x:x+bw] = 0
    return o

print("训练配对解码器 (遮挡→干净)...")
model.build_paired_decoder(X_tr, degrade_fn=lambda imgs: occlude(imgs, bs=8), n_samples=8000)

# 挑几张好看的测试图 (不同数字)
np.random.seed(7)
demo_idx = []
for digit in [0, 3, 7, 9, 2, 5]:
    candidates = np.where(y_te == digit)[0]
    demo_idx.append(candidates[np.random.randint(len(candidates))])

# 生成对比图
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n_rows = len(demo_idx) * 2  # 每个数字两行: 低对比度 + 遮挡
fig, axes = plt.subplots(n_rows, 3, figsize=(6, 2*n_rows))

for r, idx in enumerate(demo_idx):
    img = X_te[idx]
    # ── 行1: 低对比度 c=0.1 ──
    lc = low_contrast(img.reshape(1,28,28), 0.1)[0]
    lc_recon = model.reconstruct_paired(lc)
    axes[r*2, 0].imshow(img, cmap='gray', vmin=0, vmax=1); axes[r*2, 0].set_title(f'clean {y_te[idx]}')
    axes[r*2, 1].imshow(lc, cmap='gray', vmin=0, vmax=1); axes[r*2, 1].set_title('c=0.1')
    axes[r*2, 2].imshow(lc_recon, cmap='gray', vmin=0, vmax=1); axes[r*2, 2].set_title('brainfill')
    # ── 行2: 遮挡 8×8 ──
    occ = occlude(img.reshape(1,28,28), bs=8)[0]
    occ_recon = model.reconstruct_paired(occ)
    axes[r*2+1, 0].imshow(img, cmap='gray', vmin=0, vmax=1); axes[r*2+1, 0].set_title(f'clean {y_te[idx]}')
    axes[r*2+1, 1].imshow(occ, cmap='gray', vmin=0, vmax=1); axes[r*2+1, 1].set_title('occluded 8x8')
    axes[r*2+1, 2].imshow(occ_recon, cmap='gray', vmin=0, vmax=1); axes[r*2+1, 2].set_title('brainfill')
    for c in range(3):
        axes[r*2, c].axis('off'); axes[r*2+1, c].axis('off')

plt.tight_layout()
os.makedirs("data/demo", exist_ok=True)
plt.savefig("data/demo/brainfill_demo.png", dpi=120)
print(f"\n保存 data/demo/brainfill_demo.png")

# ── 重度擦除单独一张 ──
model2 = ColdEye()
model2.init_templates(X_tr[:5000])
model2.train(X_tr, y_tr, epochs=3, n_train=10000)
model2.build_paired_decoder(X_tr, degrade_fn=lambda imgs: severe_erase(imgs), n_samples=8000)

fig2, axes2 = plt.subplots(3, 3, figsize=(7, 7))
for i in range(3):
    idx = demo_idx[i]
    img = X_te[idx]
    er = severe_erase(img.reshape(1,28,28))[0]
    er_recon = model2.reconstruct_paired(er)
    axes2[0, i].imshow(img, cmap='gray', vmin=0, vmax=1); axes2[0, i].set_title(f'clean {y_te[idx]}')
    axes2[1, i].imshow(er, cmap='gray', vmin=0, vmax=1); axes2[1, i].set_title('erased')
    axes2[2, i].imshow(er_recon, cmap='gray', vmin=0, vmax=1); axes2[2, i].set_title('brainfill')
    for j in range(3): axes2[j, i].axis('off')
plt.tight_layout()
plt.savefig("data/demo/brainfill_erase_demo.png", dpi=120)
print("保存 data/demo/brainfill_erase_demo.png")
print("=== DONE ===")
