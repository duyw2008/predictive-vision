#!/usr/bin/env python3
"""
横向邮轮 few-shot — 匹配真船的侧视朝向
真船照片是横向长条, 之前的合成邮轮是竖直, 朝向不匹配导致识别失败
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
from train_multiscale import ColdEye
from benchmark_ship_fewshot import gen_geometric_shapes
from PIL import Image

def norm(x): return np.clip(x, 0, 1).astype(np.float32)

def gen_horizontal_ship(size=28, n=50, rng=None):
    """横向邮轮 (侧视): 长平船体 + 上层建筑 + 烟囱 + 桅杆"""
    rng = rng or np.random
    ships = []
    for _ in range(n):
        img = np.zeros((size, size), np.float32)
        # 长平船体 (横卧, 占据下部, 两端略收窄)
        hull_len = rng.uniform(size*0.7, size-2)
        hull_h = rng.uniform(3, 5)
        hull_y = rng.uniform(size*0.6, size*0.75)  # 船体在下
        x0 = (size - hull_len) / 2
        for x in range(size):
            # 船体两端收窄 (船头船尾)
            dist_from_center = abs(x - size/2) / (hull_len/2)
            taper = 1.0 if dist_from_center < 0.85 else 0.5
            half_h = hull_h/2 * taper
            img[int(hull_y-half_h):int(hull_y+half_h), x] = 0.8
        # 上层建筑 (船体上方, 靠船尾 2/3 处, 2-3 层)
        super_x = size/2 + hull_len*0.15  # 靠后
        super_w = rng.uniform(size*0.15, size*0.25)
        super_y = hull_y - hull_h/2
        for deck in range(rng.randint(2, 4)):
            deck_h = rng.uniform(2, 3)
            img[int(super_y-deck_h):int(super_y), int(super_x-super_w/2):int(super_x+super_w/2)] = 0.7
            super_y -= deck_h
            super_w *= rng.uniform(0.7, 0.85)
        # 烟囱 (顶部小矩形)
        fun_w = rng.uniform(2, 4)
        img[int(super_y-3):int(super_y), int(super_x-fun_w/2):int(super_x+fun_w/2)] = 0.9
        # 桅杆 (细竖线, 船头方向)
        mast_x = int(super_x - super_w*1.5)
        img[2:int(super_y-1), mast_x:mast_x+1] = 0.6
        ships.append(norm(img))
    return np.array(ships, np.float32)

np.random.seed(42)
# 真船灰度
g = np.array(Image.open('data/ship_gray.png').convert('L')).astype(np.float32)/255
g28 = np.array(Image.fromarray((g*255).astype(np.uint8)).resize((28,28), Image.LANCZOS)).astype(np.float32)/255

geo, _ = gen_geometric_shapes(n_each=150)
hships = gen_horizontal_ship(n=50, rng=np.random.RandomState(0))

model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
])
model.init_templates(geo[:200])
model.train(geo, np.zeros(len(geo)), epochs=5, n_train=len(geo), contrast_aug=True)

# 横向邮轮 few-shot
for img in hships[:5]: model.memory.append((model._activate_one(img), 1))
for img in geo[:5]:    model.memory.append((model._activate_one(img), 0))

chars = ' .:-=+*#%@'
print("横向合成邮轮 (few-shot 样本):")
for k in range(2):
    print(f'--- 横向邮轮{k} ---')
    for row in hships[k]:
        print(''.join(chars[min(9,int(v*9.9))] for v in row))

print("\n真船照片识别:")
pred, conf = model.predict(g28)
print(f"  真船 (28x28): 预测={'船' if pred==1 else '非船'} (置信度 {conf:.2f})")

print("\n对照:")
for k in range(3):
    p, c = model.predict(hships[5+k])
    print(f"  横向合成邮轮{k}: 预测={'船' if p==1 else '非船'} (置信度 {c:.2f})")

print("\n=== DONE ===")
