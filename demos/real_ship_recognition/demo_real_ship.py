#!/usr/bin/env python3
"""
冷眼真实照片识别 demo — 完整管线

真船照片 → 颜色定位(R通道) → 填平内部 → 灰度剪影 → few-shot → 识别

颜色不是语义捷径(非"红=船"), 是图形-背景分离(figure-ground, 如视网膜色拮抗细胞)
—— 只用于找边界, 识别仍走灰度, 不违背灰度-first 哲学。
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from train_multiscale import ColdEye
from PIL import Image
from scipy import ndimage

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK HK', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

np.random.seed(42)

# ═══ 几何基类 (形状基元库) ═══
def norm(x): return np.clip(x, 0, 1).astype(np.float32)

def gen_geometric_shapes(n_each=150, size=28):
    shapes = []
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size/2, size/2
    for _ in range(n_each):
        r = np.random.uniform(3, size/2.5)
        shapes.append(norm(np.sqrt((xx-cx)**2+(yy-cy)**2) <= r))
        s = np.random.uniform(5, size/1.5)
        shapes.append(norm((np.abs(xx-cx)<=s/2)&(np.abs(yy-cy)<=s/2)))
        s = np.random.uniform(5, size/1.5)
        shapes.append(norm((yy-cy) <= s/2-(s/size)*np.abs(xx-cx)))
        w = np.random.uniform(size/2, size-4); h = np.random.uniform(3, 8)
        shapes.append(norm((np.abs(xx-cx)<=w/2)&(np.abs(yy-cy)<=h/2)))
        w = np.random.uniform(2, 5); h = np.random.uniform(size/2, size-4)
        shapes.append(norm((np.abs(xx-cx)<=w/2)&(np.abs(yy-cy)<=h/2)))
        top = np.random.uniform(size/2, size-4); bot = np.random.uniform(3, top/2)
        hh = np.random.uniform(4, 10)
        halfw = bot/2 + (top-bot)/2 * (yy-(cy-hh/2)) / max(hh, 1)
        shapes.append(norm((yy>=cy-hh/2)&(yy<=cy+hh/2)&(np.abs(xx-cx)<=halfw)))
    return np.array(shapes, np.float32)

# ═══ Otsu 阈值 ═══
def otsu(x):
    hist, _ = np.histogram(x, bins=256, range=(0,1))
    total = x.size; s = (x*255).sum(); sb = 0.0; wb = 0; best = 0.5; mv = -1
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

def to28(x):
    return np.array(Image.fromarray((x*255).astype(np.uint8)).resize((28,28), Image.LANCZOS)).astype(np.float32)/255

# ═══ 1. 颜色定位 + 填平 → 灰度剪影 ═══
print("1. 颜色定位 (R通道, figure-ground)...")
img = np.array(Image.open(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ship.png')).convert('RGBA'))
R = img[:,:,0].astype(np.float32)/255
ship_mask = R > otsu(R)
filled = ndimage.binary_fill_holes(ship_mask)
closed = ndimage.binary_closing(filled, np.ones((3,3)), iterations=2)
labeled, n = ndimage.label(closed)
if n > 1:
    sizes = ndimage.sum(closed, labels=labeled, index=range(1,n+1))
    closed = (labeled == int(np.argmax(sizes))+1)
ys, xs = np.where(closed)
sil = closed[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.float32)
sil28 = to28(sil)
print(f"   剪影: {sil.shape} → 28×28, 船像素 {sil28.mean():.1%}")

# ═══ 2. 几何基类训练 + 剪影 few-shot ═══
print("2. 几何基类训练 + 剪影 few-shot...")
geo = gen_geometric_shapes(n_each=150)
model = ColdEye(eye_specs=[{"type":"global","n":200},{"type":"patch","ps":16,"st":8,"n":100}])
model.init_templates(geo[:200])
model.train(geo, np.zeros(len(geo)), epochs=5, n_train=len(geo), contrast_aug=True)

# 平移/缩放变体当 few-shot 船记忆
def variants(base):
    out = [base]
    for dy in [-2,-1,1,2]: out.append(np.roll(base, dy, axis=0))
    for dx in [-2,-1,1,2]: out.append(np.roll(base, dx, axis=1))
    for sc in [0.9, 1.1]:
        if sc < 1.0:
            ns = int(28*sc)
            r = np.array(Image.fromarray((base*255).astype(np.uint8)).resize((ns,ns), Image.LANCZOS)).astype(np.float32)/255
            c = np.zeros((28,28), np.float32); y0=(28-ns)//2; c[y0:y0+ns, y0:y0+ns]=r
        else:
            ns = int(28/sc); y0=(28-ns)//2; cr=base[y0:y0+ns, y0:y0+ns]
            c = to28(cr)
        out.append(c)
    return out

var = variants(sil28)
for v in var[:5]: model.memory.append((model._activate_one(v), 1))
for img in geo[:5]: model.memory.append((model._activate_one(img), 0))

# ═══ 3. 测试 ═══
print("3. 测试...")
in_mem = [model.predict(v)[0] for v in var[:5]]
out_mem = [model.predict(v)[0] for v in var[5:]]
geo_neg = [model.predict(geo[5+i])[0] for i in range(20)]
gen_acc = sum(p==1 for p in out_mem)/len(out_mem)
disc_acc = sum(p==0 for p in geo_neg)/len(geo_neg)
print(f"   泛化率 {gen_acc:.0%}  判别率 {disc_acc:.0%}")

# ═══ 4. 可视化 ═══
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(2, 4)

ax = fig.add_subplot(gs[0, 0]); ax.imshow(img); ax.set_title('1. 原图 (彩色)'); ax.axis('off')
ax = fig.add_subplot(gs[0, 1]); ax.imshow(R, cmap='gray'); ax.set_title('2. R通道 (找边界)'); ax.axis('off')
ax = fig.add_subplot(gs[0, 2]); ax.imshow(closed, cmap='gray'); ax.set_title('3. 填平剪影'); ax.axis('off')
ax = fig.add_subplot(gs[0, 3]); ax.imshow(sil28, cmap='gray'); ax.set_title('4. 灰度剪影 28×28'); ax.axis('off')

# 识别结果: 4 个测试样本 (绿=正确)
test_imgs = var[5:7] + list(geo[5:7])
test_true = [1,1,0,0]
for i, (im, t) in enumerate(zip(test_imgs, test_true)):
    ax = fig.add_subplot(gs[1, i])
    ax.imshow(im, cmap='gray')
    p, c = model.predict(im)
    color = 'green' if p == t else 'red'
    ax.set_title(f"{'船' if p==1 else '非船'}", color=color)
    for s in ax.spines.values(): s.set_edgecolor(color); s.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle(f'冷眼真实照片识别 demo — 颜色定位+填平+few-shot (泛化 {gen_acc:.0%}, 判别 {disc_acc:.0%})', fontsize=13)
plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__))
plt.savefig(os.path.join(out_dir, 'demo_result.png'), dpi=120)
print(f"\n保存 {os.path.join(out_dir, 'demo_result.png')}")
print("=== DONE ===")
