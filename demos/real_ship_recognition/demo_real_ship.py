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
    """letterbox: 保持长宽比缩放进 28×28 (长条形船不能硬压成方)"""
    h, w = x.shape
    scale = 28 / max(h, w)
    nh, nw = max(1, int(h*scale)), max(1, int(w*scale))
    r = np.array(Image.fromarray((x*255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)).astype(np.float32)/255
    canvas = np.zeros((28, 28), np.float32)
    y0 = (28-nh)//2; x0 = (28-nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = r
    return canvas

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

def shrink(base, frac):
    ns = max(2, int(28*frac))
    r = np.array(Image.fromarray((base*255).astype(np.uint8)).resize((ns,ns), Image.LANCZOS)).astype(np.float32)/255
    c = np.zeros((28,28), np.float32); y0=(28-ns)//2; c[y0:y0+ns, y0:y0+ns]=r
    return c

def occlude(base, bs):
    o = base.copy()
    y = np.random.RandomState(0).randint(0, 28-bs); x = np.random.RandomState(1).randint(0, 28-bs)
    o[y:y+bs, x:x+bs] = 0
    return o

# 平移/缩放/缩小/遮挡/低对比度 变体工具
var = variants(sil28)
robust_tests = [
    ("缩小 0.5", shrink(sil28, 0.5)),
    ("缩小 0.3", shrink(sil28, 0.3)),
    ("低对比度 c=0.1", (sil28*0.1).astype(np.float32)),
    ("低对比度 c=0.01", (sil28*0.01).astype(np.float32)),
    ("遮挡 8×8", occlude(sil28, 8)),
    ("遮挡 14×14", occlude(sil28, 14)),
]

# ═══ 3. 三种记忆配置: 完整权衡 ═══
print("3. 三种记忆配置对比 (鲁棒性 vs 判别力权衡)...")

configs = {
    "满框 (5 变体)": var[:5],
    "轻度扩充 (+0.5缩小 +8遮挡)": var[:5] + [shrink(sil28,0.5), occlude(sil28,8)],
    "重度扩充 (+0.3缩小 +14遮挡)": var[:5] + [shrink(sil28,0.5), shrink(sil28,0.3),
                                            occlude(sil28,8), occlude(sil28,14)],
}

def evaluate(mem_ship):
    model.memory = []
    for v in mem_ship: model.memory.append((model._activate_one(v), 1))
    for img in geo[:5]: model.memory.append((model._activate_one(img), 0))
    robust = [(n, im, model.predict(im)[0]) for n, im in robust_tests]
    robust_acc = sum(p==1 for _,_,p in robust) / len(robust)
    disc = [model.predict(geo[5+i])[0] for i in range(20)]
    disc_acc = sum(p==0 for p in disc) / len(disc)
    return robust, robust_acc, disc_acc

print(f"\n  {'记忆配置':<28s} {'鲁棒性':>8s} {'判别力':>8s}")
print(f"  {'-'*46}")
all_results = {}
for name, mem in configs.items():
    robust, ra, da = evaluate(mem)
    all_results[name] = (robust, ra, da)
    print(f"  {name:<28s} {ra:>7.0%} {da:>7.0%}")

# ═══ 4. 对比度扫描 (centering 不变性) ═══
print("\n4. 对比度扫描 (centering 不变性验证)...")
# 用轻度配置记忆 (平衡点)
model.memory = []
for v in configs["轻度扩充 (+0.5缩小 +8遮挡)"]:
    model.memory.append((model._activate_one(v), 1))
for img in geo[:5]:
    model.memory.append((model._activate_one(img), 0))

contrast_levels = [1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]
contrast_results = []
for c in contrast_levels:
    im = (sil28 * c).astype(np.float32)
    p, conf = model.predict(im)
    contrast_results.append((c, im, p, conf))
    print(f"  c={c:g}: {'船' if p==1 else '非船'} (置信度 {conf:.3f})")

# ═══ 5. 可视化 ═══
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 6)

ax = fig.add_subplot(gs[0, 0:2]); ax.imshow(img); ax.set_title('1. 原图 (彩色)'); ax.axis('off')
ax = fig.add_subplot(gs[0, 2:4]); ax.imshow(R, cmap='gray'); ax.set_title('2. R通道 (找边界)'); ax.axis('off')
ax = fig.add_subplot(gs[0, 4:6]); ax.imshow(closed, cmap='gray'); ax.set_title('3. 填平剪影'); ax.axis('off')

# 对比度扫描 (第1行): 6 个对比度等级
for i, (c, im, p, conf) in enumerate(contrast_results[:6]):
    ax = fig.add_subplot(gs[1, i])
    ax.imshow(im, cmap='gray', vmin=0, vmax=1)
    color = 'green' if p == 1 else 'red'
    ax.set_title(f'c={c:g}\n{"船" if p==1 else "非船"}', color=color, fontsize=8)
    for s in ax.spines.values(): s.set_edgecolor(color); s.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])

# 三种配置的权衡条形图 (下方面板)
names = list(configs.keys())
ra_vals = [all_results[n][1] for n in names]
da_vals = [all_results[n][2] for n in names]
x = np.arange(len(names))
ax_bar = fig.add_subplot(gs[2:4, 0:3])
w = 0.35
ax_bar.bar(x-w/2, ra_vals, w, label='鲁棒性', color='steelblue')
ax_bar.bar(x+w/2, da_vals, w, label='判别力', color='coral')
ax_bar.set_xticks(x); ax_bar.set_xticklabels(['满框','轻度','重度'], fontsize=9)
ax_bar.set_ylim(0, 1.1); ax_bar.set_ylabel('准确率')
ax_bar.legend(); ax_bar.set_title('记忆扩充的权衡')

# 鲁棒性细节 (重度配置): 6 个降质样本
ax_note = fig.add_subplot(gs[2, 3:6])
ax_note.text(0.5, 0.5, '鲁棒性 = 降质样本识别为"船"的比例\n判别力 = 几何形状正确拒绝的比例\n\n满框: 只认完整船, 降质失败\n重度: 免疫所有降质, 但半擦除船太宽松\n  (14×14 遮挡 ≈ 半空形状 ≈ 几何形状)', 
             ha='center', va='center', fontsize=9)
ax_note.axis('off')

# 重度配置的 6 个降质识别结果 (最下面一行)
heavy_robust = all_results["重度扩充 (+0.3缩小 +14遮挡)"][0]
for i, (name, im, p) in enumerate(heavy_robust):
    ax = fig.add_subplot(gs[3, i])
    ax.imshow(im, cmap='gray')
    color = 'green' if p == 1 else 'red'
    label = '船' if p == 1 else '非船'
    ax.set_title(f'{name}\n{label}', color=color, fontsize=7)
    for s in ax.spines.values(): s.set_edgecolor(color); s.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('冷眼真实照片识别 demo — 颜色定位+填平+letterbox+few-shot (对比度扫描 + 权衡)', fontsize=13)
plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__))
plt.savefig(os.path.join(out_dir, 'demo_result.png'), dpi=120)
print(f"\n保存 {os.path.join(out_dir, 'demo_result.png')}")
print("=== DONE ===")
