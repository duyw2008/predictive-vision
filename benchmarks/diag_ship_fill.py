#!/usr/bin/env python3
"""
真船 → 填平内部细节 → 干净剪影 → 识别

用户思路: 去掉船内部复杂几何特征(窗户/甲板/栏杆), 用灰度填平, 只留外轮廓。
实现: 阈值找船区域 → binary_fill_holes 填洞 → binary_closing 平滑 → 实心剪影。
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
from train_multiscale import ColdEye
from benchmark_ship_fewshot import gen_geometric_shapes
from benchmark_ship_horizontal import gen_horizontal_ship
from PIL import Image
from scipy import ndimage

np.random.seed(42)
g = np.array(Image.open('data/ship_gray.png').convert('L')).astype(np.float32)/255

def otsu(img):
    hist, _ = np.histogram(img, bins=256, range=(0,1))
    total = img.size; s = (img*255).sum(); sb = 0.0; wb = 0; best = 0.5; mv = -1
    for t in range(256):
        wb += hist[t]
        if wb == 0: continue
        wf = total - wb
        if wf == 0: break
        sb += t*hist[t]
        mb = sb/wb; mf = (s-sb)/wf
        v = wb*wf*(mb-mf)**2
        if v > mv: mv = v; best = t/255.0
    return best

# 1. centering 定位 + 裁剪
c = g - g.mean()
rowp = c.mean(axis=1); colp = c.mean(axis=0)
sr = np.where(rowp > rowp.mean()+1.5*rowp.std())[0]
sc = np.where(colp > colp.mean()+1.0*colp.std())[0]
crop = g[sr.min():sr.max()+1, sc.min():sc.max()+1]

# 2. 阈值 + 填洞 + 闭运算
t = otsu(crop)
mask = crop > t                    # 船=亮
filled = ndimage.binary_fill_holes(mask)     # 填平窗户/甲板暗孔
closed = ndimage.binary_closing(filled, structure=np.ones((3,3)), iterations=2)  # 平滑边缘
silhouette = closed.astype(np.float32)

chars = ' .:-=+*#%@'
def render(img, label):
    print(f'--- {label} ---')
    for row in img:
        print(''.join(chars[min(9,int(v*9.9))] for v in row))

sil28 = np.array(Image.fromarray((silhouette*255).astype(np.uint8)).resize((28,28), Image.LANCZOS)).astype(np.float32)/255
render(sil28, '填平后剪影 28x28')

# 3. 测试
geo, _ = gen_geometric_shapes(n_each=150)
hships = gen_horizontal_ship(n=50, rng=np.random.RandomState(0))
m = ColdEye(eye_specs=[{"type":"global","n":200},{"type":"patch","ps":16,"st":8,"n":100}])
m.init_templates(geo[:200])
m.train(geo, np.zeros(len(geo)), epochs=5, n_train=len(geo), contrast_aug=True)
for img in hships[:5]: m.memory.append((m._activate_one(img), 1))
for img in geo[:5]:    m.memory.append((m._activate_one(img), 0))

pred, conf = m.predict(sil28)
print(f"\n填平剪影识别: 预测={'船' if pred==1 else '非船'} (置信度 {conf:.2f})")

# 对照: 合成横向邮轮
print("对照:")
for k in range(3):
    p, cc = m.predict(hships[5+k])
    print(f"  合成横向邮轮{k}: {'船' if p==1 else '非船'} ({cc:.2f})")
print("\n=== DONE ===")
