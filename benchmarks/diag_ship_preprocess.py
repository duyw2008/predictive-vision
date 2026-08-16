#!/usr/bin/env python3
"""
真船照片预处理实验 — 让脏照片变成几何基类能认的干净形状

方案对比:
  A. 原始缩放 (baseline, 失败)
  B. 裁剪船区域 + 缩放
  C. 边缘检测 (Sobel) + 缩放  ← 提取轮廓, 不管绝对亮度/颜色
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
from train_multiscale import ColdEye
from benchmark_ship_fewshot import gen_geometric_shapes, gen_cruise_ship
from PIL import Image

np.random.seed(42)
g = np.array(Image.open('data/ship_gray.png').convert('L')).astype(np.float32)/255
h, w = g.shape

def resize_to(img, size=28):
    return np.array(Image.fromarray((img*255).astype(np.uint8)).resize((size,size), Image.LANCZOS)).astype(np.float32)/255

# ── 预处理方案 ──
# B: 裁剪 (找亮像素 bounding box, 假设船比背景亮)
def crop_object(img, thresh_frac=0.7):
    thresh = np.percentile(img, thresh_frac*100)
    mask = img > thresh
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img
    return img[ys.min():ys.max()+1, xs.min():xs.max()+1]

# C: 边缘检测 (Sobel 梯度幅值)
def sobel_edges(img):
    gy = np.gradient(img, axis=0)
    gx = np.gradient(img, axis=1)
    mag = np.sqrt(gx**2 + gy**2)
    mag = mag / (mag.max() + 1e-8)
    return mag

# ── 训练模型 (同 demo) ──
geo, _ = gen_geometric_shapes(n_each=150)
ships = gen_cruise_ship(n=50, rng=np.random.RandomState(0))
model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
])
model.init_templates(geo[:200])
model.train(geo, np.zeros(len(geo)), epochs=5, n_train=len(geo), contrast_aug=True)
for img in ships[:5]: model.memory.append((model._activate_one(img), 1))
for img in geo[:5]:  model.memory.append((model._activate_one(img), 0))

# ── 测试 ──
chars = ' .:-=+*#%@'
def ascii_render(img, w=28):
    return '\n'.join(''.join(chars[min(9,int(v*9.9))] for v in row) for row in img)

preprocess = {
    'A 原始缩放':      resize_to(g, 28),
    'B 裁剪+缩放':     resize_to(crop_object(g), 28),
    'C Sobel边缘':     resize_to(sobel_edges(g), 28),
}

print(f"{'='*60}")
print("  真船照片预处理: 原始 vs 裁剪 vs 边缘检测")
print(f"{'='*60}")
for name, sample in preprocess.items():
    pred, conf = model.predict(sample)
    label = '船' if pred == 1 else '非船'
    print(f"\n{name}: 预测={label} (置信度 {conf:.2f})")
    print(ascii_render(sample))

print("\n=== DONE ===")
