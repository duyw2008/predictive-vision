#!/usr/bin/env python3
"""
邮轮适应 demo — 冷眼如何"几发样本学会任意形状"

展示:
  1. 几何基类 (圆/方/三角/梯形/矩形/细线/圆环) — 形状基元库
  2. 合成邮轮 (船体梯形 + 上层建筑 + 烟囱 + 桅杆)
  3. 5-shot 适应后: 邮轮识别 + 几何形状正确拒绝
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
from train_multiscale import ColdEye
from benchmark_ship_fewshot import gen_geometric_shapes, gen_cruise_ship

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK HK', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

np.random.seed(42)
size = 28

# ── 1. 几何基类 ──
geo, geo_labels = gen_geometric_shapes(n_each=20)
print(f"几何基类: {len(geo)} 张 (圆/方/三角/长矩形/竖矩形/梯形/细线/圆环)")

# ── 2. 合成邮轮 ──
ships = gen_cruise_ship(n=50, rng=np.random.RandomState(0))
print(f"合成邮轮: {len(ships)} 张")

# ── 3. 训练 + 5-shot 适应 ──
print("训练 ColdEye (几何基类, 5ep)...")
model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
])
model.init_templates(geo[:200])
model.train(geo, np.zeros(len(geo)), epochs=5, n_train=len(geo), contrast_aug=True)

n_shot = 5
print(f"{n_shot}-shot 适应邮轮...")
for img in ships[:n_shot]:
    model.memory.append((model._activate_one(img), 1))   # 邮轮
for img in geo[:n_shot]:
    model.memory.append((model._activate_one(img), 0))   # 非邮轮

# ── 测试: 邮轮 vs 几何 ──
test_ships = ships[n_shot:n_shot+20]
test_geo = geo[n_shot:n_shot+20]
preds_ship = [model.predict(test_ships[i])[0] for i in range(20)]
preds_geo = [model.predict(test_geo[i])[0] for i in range(20)]
tp = sum(p == 1 for p in preds_ship)
tn = sum(p == 0 for p in preds_geo)
acc = (tp + tn) / 40
print(f"识别: 邮轮 {tp}/20, 几何正确拒绝 {tn}/20, 总 {acc:.0%}")

# ── 可视化 ──
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 6)

# 面板1: 几何基类
ax1 = fig.add_subplot(gs[0, 0:2])
grid = np.vstack([np.hstack([geo[i*5+j] for j in range(5)]) for i in range(4)])
ax1.imshow(grid, cmap='gray'); ax1.set_title('几何基类 (形状基元库)'); ax1.axis('off')

# 面板2: 合成邮轮
ax2 = fig.add_subplot(gs[0, 2:4])
ship_grid = np.vstack([np.hstack([ships[i*5+j] for j in range(5)]) for i in range(4)])
ax2.imshow(ship_grid, cmap='gray'); ax2.set_title('合成邮轮 (目标形状)'); ax2.axis('off')

# 面板3: 5-shot 记忆 (5 邮轮 + 5 几何)
ax3 = fig.add_subplot(gs[0, 4:6])
mem_ships = np.hstack(ships[:5])
mem_geo = np.hstack(geo[:5])
ax3.imshow(np.vstack([mem_ships, mem_geo]), cmap='gray')
ax3.set_title(f'{n_shot}-shot 记忆 (上=邮轮 下=几何)'); ax3.axis('off')

# 面板4: 识别结果 (6 个测试样本, 绿框=正确, 红框=错)
show_ships = test_ships[:3]
show_geo = test_geo[:3]
all_imgs = list(show_ships) + list(show_geo)
all_preds = preds_ship[:3] + preds_geo[:3]
all_true = [1]*3 + [0]*3

for i, (img, p, t) in enumerate(zip(all_imgs, all_preds, all_true)):
    ax = fig.add_subplot(gs[1:3, i])
    ax.imshow(img, cmap='gray')
    color = 'green' if p == t else 'red'
    label = '船' if p == 1 else '非船'
    ax.set_title(f'预测:{label}', color=color, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle(f'冷眼邮轮适应 demo — 几何基类 → {n_shot}-shot 学会识别邮轮 (准确率 {acc:.0%})', fontsize=13)
plt.tight_layout()
os.makedirs("data/demo", exist_ok=True)
plt.savefig("data/demo/ship_adapt_demo.png", dpi=120)
print(f"\n保存 data/demo/ship_adapt_demo.png")

# ── ASCII 展示一张邮轮 ──
chars = ' .:-=+*#%@'
print(f"\n一张合成邮轮 (ASCII):")
for row in ships[0]:
    print(''.join(chars[min(9, int(v*9.9))] for v in row))

print("\n=== DONE ===")
