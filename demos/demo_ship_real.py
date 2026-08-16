#!/usr/bin/env python3
"""
ship.png → 灰度样本 → 冷眼识别测试

1. RGBA → 灰度, 保存 data/ship_gray.png
2. 缩放 64×64 (拉伸, centering 会处理), 保存 data/ship_gray_64.png
3. 用几何基类 + 合成邮轮 5-shot 的模型, 测试真船照片能否识别
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
from train_multiscale import ColdEye
from benchmark_ship_fewshot import gen_geometric_shapes, gen_cruise_ship
from PIL import Image

np.random.seed(42)

# ── 1. 转灰度 ──
img = Image.open("data/ship.png")
gray = np.array(img.convert('L')).astype(np.float32) / 255
Image.fromarray((gray*255).astype(np.uint8)).save("data/ship_gray.png")
print(f"灰度图: {gray.shape} → data/ship_gray.png")

# ── 2. 缩放 64×64 (直接拉伸, 不用 letterbox 保留形状) ──
gray64 = np.array(Image.fromarray((gray*255).astype(np.uint8)).resize((64,64), Image.LANCZOS)).astype(np.float32)/255
Image.fromarray((gray64*255).astype(np.uint8)).save("data/ship_gray_64.png")
# 也存一份 28×28 (MNIST 尺度)
gray28 = np.array(Image.fromarray((gray*255).astype(np.uint8)).resize((28,28), Image.LANCZOS)).astype(np.float32)/255
Image.fromarray((gray28*255).astype(np.uint8)).save("data/ship_gray_28.png")
print(f"缩放: 64×64 → data/ship_gray_64.png, 28×28 → data/ship_gray_28.png")

# ── 3. 训练 + 5-shot 适应 (和 demo 一样) ──
print("\n训练 ColdEye (几何基类 + 合成邮轮 5-shot)...")
geo, _ = gen_geometric_shapes(n_each=150)
ships = gen_cruise_ship(n=50, rng=np.random.RandomState(0))

model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
])
model.init_templates(geo[:200])
model.train(geo, np.zeros(len(geo)), epochs=5, n_train=len(geo), contrast_aug=True)

for img in ships[:5]:
    model.memory.append((model._activate_one(img), 1))   # 邮轮
for img in geo[:5]:
    model.memory.append((model._activate_one(img), 0))   # 非邮轮

# ── 4. 测试真船照片 ──
print("\n测试真船照片 ship.png (28×28 匹配模型):")
for name, sample in [("28×28", gray28)]:
    pred, conf = model.predict(sample)
    label = '船' if pred == 1 else '非船'
    print(f"  {name}: 预测={label} (置信度 {conf:.2f})")

    pred_inv, conf_inv = model.predict(1.0 - sample)
    label_inv = '船' if pred_inv == 1 else '非船'
    print(f"  {name} 反相: 预测={label_inv} (置信度 {conf_inv:.2f})")

# ── 5. 对比: 合成邮轮的正确预测置信度 ──
print("\n对照 — 合成邮轮 (应预测为船):")
for i in range(3):
    pred, conf = model.predict(ships[5+i])
    print(f"  合成邮轮{i}: 预测={'船' if pred==1 else '非船'} (置信度 {conf:.2f})")

print("\n=== DONE ===")
