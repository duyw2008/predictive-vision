#!/usr/bin/env python3
"""
方案2: 真船剪影直接当 few-shot 样本

颜色定位+填平 → 剪影 → few-shot 记忆 → 测泛化(平移/缩放) + 判别(几何非船)
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
from train_multiscale import ColdEye
from benchmark_ship_fewshot import gen_geometric_shapes
from PIL import Image
from scipy import ndimage

np.random.seed(42)

# ── 1. 颜色定位 + 填平 → 剪影 (同 benchmark_ship_colorfill) ──
img = np.array(Image.open('data/ship.png').convert('RGBA'))
R = img[:,:,0].astype(np.float32)/255

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

ship_mask = R > otsu(R)
filled = ndimage.binary_fill_holes(ship_mask)
closed = ndimage.binary_closing(filled, np.ones((3,3)), iterations=2)
labeled, n = ndimage.label(closed)
if n > 1:
    sizes = ndimage.sum(closed, labels=labeled, index=range(1,n+1))
    closed = (labeled == int(np.argmax(sizes))+1)
ys, xs = np.where(closed)
sil = closed[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.float32)

# ── 2. 缩放到 28×28, 生成平移/缩放变体 ──
def to28(x):
    return np.array(Image.fromarray((x*255).astype(np.uint8)).resize((28,28), Image.LANCZOS)).astype(np.float32)/255

base28 = to28(sil)

def variants(base, n_shift=4, n_scale=2):
    """平移 ±2px + 缩放 0.9/1.1 变体"""
    out = [base]
    H, W = base.shape
    for dy in [-2, -1, 1, 2]:
        shifted = np.roll(base, dy, axis=0)
        out.append(shifted)
    for dx in [-2, -1, 1, 2]:
        out.append(np.roll(base, dx, axis=1))
    for sc in [0.9, 1.1]:
        if sc < 1.0:
            # 缩小: resize 到 ns, 居中补 0
            ns = int(28*sc)
            r = np.array(Image.fromarray((base*255).astype(np.uint8)).resize((ns,ns), Image.LANCZOS)).astype(np.float32)/255
            canvas = np.zeros((28,28), np.float32)
            y0 = (28-ns)//2
            canvas[y0:y0+ns, y0:y0+ns] = r
        else:
            # 放大: 裁中心, resize 回 28
            ns = int(28/sc)
            y0 = (28-ns)//2
            crop = base[y0:y0+ns, y0:y0+ns]
            canvas = np.array(Image.fromarray((crop*255).astype(np.uint8)).resize((28,28), Image.LANCZOS)).astype(np.float32)/255
        out.append(canvas)
    return out

variants_list = variants(base28)

# ── 3. few-shot: 用变体当"船"样本 ──
geo, _ = gen_geometric_shapes(n_each=150)
m = ColdEye(eye_specs=[{"type":"global","n":200},{"type":"patch","ps":16,"st":8,"n":100}])
m.init_templates(geo[:200])
m.train(geo, np.zeros(len(geo)), epochs=5, n_train=len(geo), contrast_aug=True)

# 用前 5 个变体当 few-shot 船记忆
for v in variants_list[:5]:
    m.memory.append((m._activate_one(v), 1))   # 船
for img in geo[:5]:
    m.memory.append((m._activate_one(img), 0)) # 非船

# ── 4. 测试 ──
print(f"{'='*60}")
print("  方案2: 真船剪影 few-shot → 泛化测试")
print(f"{'='*60}")
print("\n[记忆内的变体 (应=船)]")
for i, v in enumerate(variants_list[:5]):
    p, c = m.predict(v)
    print(f"  变体{i} (记忆内): {'船' if p==1 else '非船'} ({c:.2f})")

print("\n[记忆外的变体 (泛化测试, 应=船)]")
for i, v in enumerate(variants_list[5:], start=5):
    p, c = m.predict(v)
    print(f"  变体{i} (记忆外): {'船' if p==1 else '非船'} ({c:.2f})")

print("\n[几何形状 (判别测试, 应=非船)]")
tp_neg = 0
for i in range(20):
    p, c = m.predict(geo[5+i])
    tp_neg += (p == 0)
print(f"  几何非船正确拒绝: {tp_neg}/20")

# 泛化率
mem_out = [m.predict(v)[0] for v in variants_list[5:]]
gen_acc = sum(p==1 for p in mem_out) / len(mem_out)
print(f"\n泛化率 (记忆外变体识别为船): {gen_acc:.0%}")
print("\n=== DONE ===")
