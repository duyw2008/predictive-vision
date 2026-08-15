#!/usr/bin/env python3
"""诊断v4: 重叠区域 vs 硬分桶 — 修复 digit3 边界断裂.
硬分桶: patch 归到中心所在 1 个区域 → 跨边界笔画割裂 (digit3 MSE 0.094→0.199)
重叠区域: patch 硬归到所有物理覆盖区域 (保持稀疏, 非软权重) → 跨边界两边都有支撑"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import load_mnist
from vision import VisionInterface
from graph import VisionGraph

np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
def upscale(X, f=2):
    return np.kron(X, np.ones((f, f), np.float32))
X56 = upscale(X_tr[:12000]); T56 = upscale(X_te[:300])

ps, st, n = 8, 4, 300
g = VisionGraph(n_nodes=n, template_size=ps*ps)
v = VisionInterface(g, patch_size=ps, stride=st)
nids = sorted(g.nodes.keys())

rng = np.random.RandomState(42)
idxs = rng.choice(len(X56), 200, replace=False)
ap = [p.astype(np.float32) for i in idxs for p in v.extractor.extract(X56[i]-X56[i].mean())]
for k, nid in enumerate(nids):
    p = ap[k%len(ap)]; p/=np.linalg.norm(p)+1e-8; g.nodes[nid].template=p

print("训练 (12000×2)..."); t0=time.time()
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

def _activate(img):
    v.set_image(img - img.mean())
    return v._last_patches, v.extractor.patch_positions

def act_hard(img, nr):
    patches, positions = _activate(img)
    if patches is None: return np.zeros(nr*nr*n, np.float32)
    H, W = img.shape
    ra = np.zeros((nr, nr, n), np.float32); rc = np.zeros((nr, nr), np.float32)
    for p_i, (y1, x1, y2, x2) in enumerate(positions):
        if p_i >= len(patches): break
        cy = (y1+y2)//2; cx = (x1+x2)//2
        ry = min(int(cy/(H/nr)), nr-1); rx = min(int(cx/(W/nr)), nr-1)
        scores = templates @ patches[p_i]; best = int(np.argmax(scores))
        if scores[best] < 0: continue
        ra[ry, rx, best] += 1.0; rc[ry, rx] += 1.0
    ra = ra.reshape(nr*nr, n)
    return np.minimum(1.0, ra/(rc.reshape(-1,1)+1e-8)*3).reshape(-1)

def act_overlap(img, nr):
    """重叠区域: patch 硬归到所有物理覆盖区域 (稀疏, 非软权重)"""
    patches, positions = _activate(img)
    if patches is None: return np.zeros(nr*nr*n, np.float32)
    H, W = img.shape
    cell_h, cell_w = H/nr, W/nr
    ra = np.zeros((nr, nr, n), np.float32); rc = np.zeros((nr, nr), np.float32)
    for p_i, (y1, x1, y2, x2) in enumerate(positions):
        if p_i >= len(patches): break
        scores = templates @ patches[p_i]; best = int(np.argmax(scores))
        if scores[best] < 0: continue
        ry1 = max(0, int(y1//cell_h)); ry2 = min(nr-1, int((y2-1)//cell_h))
        rx1 = max(0, int(x1//cell_w)); rx2 = min(nr-1, int((x2-1)//cell_w))
        for ry in range(ry1, ry2+1):
            for rx in range(rx1, rx2+1):
                ra[ry, rx, best] += 1.0; rc[ry, rx] += 1.0
    ra = ra.reshape(nr*nr, n)
    return np.minimum(1.0, ra/(rc.reshape(-1,1)+1e-8)*3).reshape(-1)

H, W = 56, 56
n_fit = 3000
fit_idx = rng.choice(len(X56), n_fit, replace=False)
F = np.zeros((n_fit, H*W), np.float32)
for i, idx in enumerate(fit_idx): F[i] = X56[idx].reshape(-1).astype(np.float32)

def fit_W(dim, act_fn):
    A = np.zeros((n_fit, dim), np.float32)
    for i, idx in enumerate(fit_idx): A[i] = act_fn(X56[idx])
    return (np.linalg.pinv(A.T @ A) @ (A.T @ F)).astype(np.float32)

print("拟合 W (硬4×4 / 重叠4×4 / 全局 / 融合全局+硬)...")
Ws = {
    'global': fit_W(n, lambda img: act_hard(img, 1)),
    'hard_4x4': fit_W(16*n, lambda img: act_hard(img, 4)),
    'overlap_4x4': fit_W(16*n, lambda img: act_overlap(img, 4)),
    'fusion': fit_W(17*n, lambda img: np.concatenate([act_hard(img,1), act_hard(img,4)])),
}

def recon(img, key, act_fn, nr):
    return (act_fn(img, nr) @ Ws[key]).reshape(img.shape)

def recon_fusion(img):
    act = np.concatenate([act_hard(img,1), act_hard(img,4)])
    return (act @ Ws['fusion']).reshape(img.shape)

print("\n" + "="*70)
print("脑补 MSE 对比 (56×56 降质 c=0.2): 全局/硬分桶/重叠/融合")
print("="*70)
for t_idx in [3, 5, 18, 0, 1]:
    img = T56[t_idx]; d = int(y_te[t_idx])
    m = img.mean(); deg = m+(img-m)*0.2
    r_g = recon(deg, 'global', act_hard, 1)
    r_h = recon(deg, 'hard_4x4', act_hard, 4)
    r_o = recon(deg, 'overlap_4x4', act_overlap, 4)
    r_f = recon_fusion(deg)
    print(f"digit={d}: 全局={np.mean((r_g-img)**2):.4f}  硬={np.mean((r_h-img)**2):.4f}  重叠={np.mean((r_o-img)**2):.4f}  融合={np.mean((r_f-img)**2):.4f}")

shades = ' .:-=+*#%@'
def render(img):
    return '\n'.join(''.join(shades[min(9,max(0,int(p*10)))] for p in row[::4]) for row in img[::4])
img = T56[3]; m=img.mean(); deg=m+(img-m)*0.2
print("\n视觉对比 digit=3:")
print("原图:"); print(render(img))
print("硬4×4:"); print(render(recon(deg,'hard_4x4',act_hard,4)))
print("融合(全局+硬):"); print(render(recon_fusion(deg)))
