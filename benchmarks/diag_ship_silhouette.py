#!/usr/bin/env python3
"""
真船照片 → 裁剪 → 二值化剪影 → 识别

完整管线: 低对比度(centering定位) → 裁剪(去背景) → 二值化(去纹理)
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
from train_multiscale import ColdEye
from benchmark_ship_fewshot import gen_geometric_shapes
from benchmark_ship_horizontal import gen_horizontal_ship
from PIL import Image

np.random.seed(42)
g = np.array(Image.open('data/ship_gray.png').convert('L')).astype(np.float32)/255

def otsu_threshold(img):
    """numpy Otsu 二值化阈值"""
    hist, _ = np.histogram(img, bins=256, range=(0,1))
    total = img.size
    sum_total = (img * 255).sum()
    sum_bg = 0.0; w_bg = 0
    max_var = -1; best_t = 0.5
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0: continue
        w_fg = total - w_bg
        if w_fg == 0: break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg
        var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var > max_var:
            max_var = var; best_t = t / 255.0
    return best_t

# 1. centering 定位 + 裁剪
c = g - g.mean()
rowp = c.mean(axis=1); colp = c.mean(axis=0)
sr = np.where(rowp > rowp.mean()+1.5*rowp.std())[0]
sc = np.where(colp > colp.mean()+1.0*colp.std())[0]
crop = g[sr.min():sr.max()+1, sc.min():sc.max()+1]
print(f"裁剪: {crop.shape}")

# 2. Otsu 二值化 (船=亮, 海=暗)
t = otsu_threshold(crop)
binarized = (crop > t).astype(np.float32)
print(f"Otsu 阈值: {t:.2f}, 船像素占比: {binarized.mean():.1%}")

bin28 = np.array(Image.fromarray((binarized*255).astype(np.uint8)).resize((28,28), Image.LANCZOS)).astype(np.float32)/255

chars = ' .:-=+*#%@'
def render(img, label):
    print(f'--- {label} ---')
    for row in img:
        print(''.join(chars[min(9,int(v*9.9))] for v in row))

render(bin28, '二值化剪影 28x28')

# 3. 测试 (横向邮轮 few-shot)
geo, _ = gen_geometric_shapes(n_each=150)
hships = gen_horizontal_ship(n=50, rng=np.random.RandomState(0))
m = ColdEye(eye_specs=[{"type":"global","n":200},{"type":"patch","ps":16,"st":8,"n":100}])
m.init_templates(geo[:200])
m.train(geo, np.zeros(len(geo)), epochs=5, n_train=len(geo), contrast_aug=True)
for img in hships[:5]: m.memory.append((m._activate_one(img), 1))
for img in geo[:5]:    m.memory.append((m._activate_one(img), 0))

pred, conf = m.predict(bin28)
print(f"\n二值化剪影识别: 预测={'船' if pred==1 else '非船'} (置信度 {conf:.2f})")
print("\n=== DONE ===")
