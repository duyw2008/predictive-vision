#!/usr/bin/env python3
"""
颜色定位 + 灰度填充: 用 R 通道找船边界, 填平内部复杂结构, 得到干净灰度剪影

管线: R通道阈值(船52 vs 海1, 51x对比) → 填洞 → 裁剪 → 灰度剪影 → 冷眼识别
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
img = np.array(Image.open('data/ship.png').convert('RGBA'))
R = img[:,:,0].astype(np.float32) / 255   # 船=0.204, 海=0.004

# ── 1. R 通道找船 (阈值 Otsu, 51x 对比度) ──
def otsu(x):
    hist,_ = np.histogram(x, bins=256, range=(0,1))
    total=x.size; s=(x*255).sum(); sb=0.0; wb=0; best=0.5; mv=-1
    for t in range(256):
        wb+=hist[t]
        if wb==0: continue
        wf=total-wb
        if wf==0: break
        sb+=t*hist[t]
        mb=sb/wb; mf=(s-sb)/wf
        v=wb*wf*(mb-mf)**2
        if v>mv: mv=v; best=t/255.0
    return best

t = otsu(R)
ship_mask = R > t
print(f"R通道 Otsu 阈值 {t:.3f}, 船像素占比 {ship_mask.mean():.1%}")

# ── 2. 填平内部 (去窗户/甲板复杂结构) ──
filled = ndimage.binary_fill_holes(ship_mask)
closed = ndimage.binary_closing(filled, np.ones((3,3)), iterations=2)

# 保留最大连通区域 (船本体, 去掉散落的浪花高光)
labeled, n = ndimage.label(closed)
if n > 1:
    sizes = ndimage.sum(closed, labels=labeled, index=range(1, n+1))
    largest = int(np.argmax(sizes)) + 1
    closed = (labeled == largest)

print(f"填平后船像素占比 {closed.mean():.1%}")

# ── 3. 裁剪到船 bounding box ──
ys, xs = np.where(closed)
crop = closed[ys.min():ys.max()+1, xs.min():xs.max()+1]
print(f"裁剪: {crop.shape}")

# ── 4. 缩放成灰度剪影 ──
sil = crop.astype(np.float32)
sil28 = np.array(Image.fromarray((sil*255).astype(np.uint8)).resize((28,28), Image.LANCZOS)).astype(np.float32)/255

chars = ' .:-=+*#%@'
print("--- 灰度剪影 28x28 (颜色定位+填平) ---")
for row in sil28:
    print(''.join(chars[min(9,int(v*9.9))] for v in row))

# 保存
Image.fromarray((sil*255).astype(np.uint8)).save('data/ship_silhouette.png')
Image.fromarray((sil28*255).astype(np.uint8)).save('data/ship_silhouette_28.png')

# ── 5. 冷眼识别 (横向邮轮 few-shot) ──
geo, _ = gen_geometric_shapes(n_each=150)
hships = gen_horizontal_ship(n=50, rng=np.random.RandomState(0))
m = ColdEye(eye_specs=[{"type":"global","n":200},{"type":"patch","ps":16,"st":8,"n":100}])
m.init_templates(geo[:200])
m.train(geo, np.zeros(len(geo)), epochs=5, n_train=len(geo), contrast_aug=True)
for img in hships[:5]: m.memory.append((m._activate_one(img), 1))
for img in geo[:5]:    m.memory.append((m._activate_one(img), 0))

pred, conf = m.predict(sil28)
print(f"\n灰度剪影识别: 预测={'船' if pred==1 else '非船'} (置信度 {conf:.2f})")
print("对照:")
for k in range(3):
    p, cc = m.predict(hships[5+k])
    print(f"  合成横向邮轮{k}: {'船' if p==1 else '非船'} ({cc:.2f})")
print("\n=== DONE ===")
