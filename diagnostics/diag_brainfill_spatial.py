#!/usr/bin/env python3
"""诊断v2: 空间分桶能否让脑补重建出空间结构?
对比: 全局计数(当前) vs 区域独立计数(2×2, 3×3, 4×4)
训练不变, 只改激活表示 → 线性回归重建 → MSE对比"""
import sys, os, time, numpy as np, gzip
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import load_mnist
from vision import VisionInterface
from graph import VisionGraph

np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
def upscale(X, f=2):
    return np.kron(X, np.ones((f, f), np.float32))
X56 = upscale(X_tr[:12000])
T56 = upscale(X_te[:300])

ps, st, n = 8, 4, 300
g = VisionGraph(n_nodes=n, template_size=ps*ps)
v = VisionInterface(g, patch_size=ps, stride=st)
nids = sorted(g.nodes.keys())

rng = np.random.RandomState(42)
idxs = rng.choice(len(X56), 200, replace=False)
ap = [p.astype(np.float32) for i in idxs for p in v.extractor.extract(X56[i]-X56[i].mean())]
for k, nid in enumerate(nids):
    p = ap[k%len(ap)]; p/=np.linalg.norm(p)+1e-8; g.nodes[nid].template=p

print("训练 (12000×2)...")
t0=time.time()
for ep in range(2):
    for idx in rng.permutation(len(X56)):
        img = X56[idx].copy(); m=img.mean()
        if rng.random()<0.5: img = m+(img-m)*(0.3+rng.random()*0.7)
        img = img-img.mean(); v.set_image(img)
        best = max(nids, key=lambda x: g.nodes[x].activation); node=g.nodes[best]
        t = np.mean(v.node_assignments.get(best, [v.extractor.extract(img)[0]]), axis=0)
        tn = np.linalg.norm(t); t = t/(tn+1e-8) if tn>0 else t
        node.template += 0.1*(t-node.template); node.template /= np.linalg.norm(node.template)+1e-8
print(f"  完成 {time.time()-t0:.0f}s")

templates = np.array([g.nodes[nid].template for nid in nids], np.float32)

def act_spatial(img, n_regions=1):
    """激活表示: n_regions=1 退化为全局计数; >1 时按区域独立计数。
    返回 [n_regions^2 * n_nodes] 维向量"""
    v.set_image(img - img.mean())
    H, W = img.shape
    positions = v.extractor.patch_positions
    patches = v._last_patches
    if patches is None:
        return np.zeros(n_regions*n_regions*n, np.float32)
    ra = np.zeros((n_regions, n_regions, n), np.float32)
    rc = np.zeros((n_regions, n_regions), np.float32)
    for p_i, (y1, x1, y2, x2) in enumerate(positions):
        if p_i >= len(patches): break
        cy = (y1+y2)//2; cx = (x1+x2)//2
        ry = min(int(cy/(H/n_regions)), n_regions-1)
        rx = min(int(cx/(W/n_regions)), n_regions-1)
        scores = templates @ patches[p_i]
        best = int(np.argmax(scores))
        if scores[best] < 0: continue
        ra[ry, rx, best] += 1.0
        rc[ry, rx] += 1.0
    ra = ra.reshape(n_regions*n_regions, n)
    ra = np.minimum(1.0, ra / (rc.reshape(-1,1) + 1e-8) * 3)
    return ra.reshape(-1)

# 拟合三种线性回归
H, W = 56, 56
n_fit = 3000
fit_idx = rng.choice(len(X56), n_fit, replace=False)
F = np.zeros((n_fit, H*W), np.float32)
for i, idx in enumerate(fit_idx):
    F[i] = X56[idx].reshape(-1).astype(np.float32)

def fit_W(dim, act_fn):
    A = np.zeros((n_fit, dim), np.float32)
    for i, idx in enumerate(fit_idx):
        A[i] = act_fn(X56[idx])
    return (np.linalg.pinv(A.T @ A) @ (A.T @ F)).astype(np.float32)

print("拟合线性回归 W (全局/2×2/3×3/4×4)...")
Ws = {}
for nr in [1, 2, 3, 4]:
    t0=time.time()
    Ws[nr] = fit_W(nr*nr*n, lambda img, nr=nr: act_spatial(img, nr))
    print(f"  {nr}×{nr} ({nr*nr*n}d): {time.time()-t0:.0f}s")

def recon(img, nr):
    act = act_spatial(img, nr)
    return (act @ Ws[nr]).reshape(img.shape)

shades = ' .:-=+*#%@'
def render(img):
    return '\n'.join(''.join(shades[min(9,max(0,int(p*10)))] for p in row[::4]) for row in img[::4])

# 对比
print("\n" + "="*60)
print("脑补重建 MSE 对比 (降质 c=0.2, 越低越好)")
print("="*60)
for t_idx in [3, 5, 18, 0]:
    img = T56[t_idx]; d = int(y_te[t_idx])
    m = img.mean(); deg = m+(img-m)*0.2
    line = f"digit={d}: "
    for nr in [1,2,3,4]:
        r = recon(deg, nr)
        line += f"{nr}×{nr}={np.mean((r-img)**2):.4f}  "
    print(line)

# 视觉对比: digit 3, 全局 vs 4×4
img = T56[3]; m=img.mean(); deg=m+(img-m)*0.2
print("\n视觉对比 digit=3 (降质c=0.2):")
print("原图:"); print(render(img))
print("全局(300d)重建:"); print(render(recon(deg,1)))
print("4×4(4800d)重建:"); print(render(recon(deg,4)))
