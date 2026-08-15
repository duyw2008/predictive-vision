#!/usr/bin/env python3
"""诊断v3: soft高斯空间编码 vs 硬分桶
硬分桶边界断裂 → soft高斯权重平滑过渡"""
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

# 预计算: patch位置 → 各区域的高斯权重 (每图固定)
H, W = 56, 56

def make_weights(n_regions, sigma_frac):
    """[n_patches, n_regions^2] 高斯权重矩阵"""
    positions = v.extractor.patch_positions
    n_patches = len(positions)
    sigma = (H / n_regions) * sigma_frac
    centers = [(i+0.5)*H/n_regions for i in range(n_regions)]
    Wt = np.zeros((n_patches, n_regions*n_regions), np.float32)
    for p_i, (y1, x1, y2, x2) in enumerate(positions):
        cy = (y1+y2)/2.0; cx = (x1+x2)/2.0
        k = 0
        for ry in range(n_regions):
            for rx in range(n_regions):
                d2 = (cy-centers[ry])**2 + (cx-centers[rx])**2
                Wt[p_i, k] = np.exp(-d2/(2*sigma**2))
                k += 1
    return Wt

def act_encoded(img, n_regions, Wt):
    """编码激活: 硬分桶(n_regions>0) 或 soft高斯(Wt)"""
    v.set_image(img - img.mean())
    positions = v.extractor.patch_positions
    patches = v._last_patches
    if patches is None:
        return np.zeros(n_regions*n_regions*n, np.float32)
    ra = np.zeros((n_regions*n_regions, n), np.float32)
    rc = np.zeros(n_regions*n_regions, np.float32)
    for p_i, patch in enumerate(patches):
        scores = templates @ patch
        best = int(np.argmax(scores))
        if scores[best] < 0: continue
        if Wt is None:
            # 硬分桶: one-hot 权重
            y1, x1, y2, x2 = positions[p_i]
            cy = (y1+y2)/2.0; cx = (x1+x2)/2.0
            ry = min(int(cy/(H/n_regions)), n_regions-1)
            rx = min(int(cx/(W/n_regions)), n_regions-1)
            ra[ry*n_regions+rx, best] += 1.0
            rc[ry*n_regions+rx] += 1.0
        else:
            w = Wt[p_i]
            ra[:, best] += w
            rc += w
    ra = np.minimum(1.0, ra / (rc.reshape(-1,1) + 1e-8) * 3)
    return ra.reshape(-1)

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

# 对比: 硬4×4 vs soft4×4(sigma 0.3/0.6/1.0)
configs = []
for nr in [4]:
    configs.append((f"硬{nr}×{nr}", nr*nr*n, lambda img, nr=nr: act_encoded(img, nr, None)))
    for sf in [0.3, 0.6, 1.0]:
        Wt = make_weights(nr, sf)
        configs.append((f"soft{nr}×{nr}σ{sf}", nr*nr*n, lambda img, nr=nr, Wt=Wt: act_encoded(img, nr, Wt)))

print("拟合 W...")
Ws = {}
for name, dim, fn in configs:
    t0=time.time()
    Ws[name] = fit_W(dim, fn)
    print(f"  {name} ({dim}d): {time.time()-t0:.0f}s")

def recon(img, name):
    fn = next(fn for nm, _dim, fn in configs if nm == name)
    return (fn(img) @ Ws[name]).reshape(img.shape)

shades = ' .:-=+*#%@'
def render(img):
    return '\n'.join(''.join(shades[min(9,max(0,int(p*10)))] for p in row[::4]) for row in img[::4])

print("\n" + "="*66)
print("MSE 对比 (降质 c=0.2): 硬分桶 vs soft高斯")
print("="*66)
names = [c[0] for c in configs]
for t_idx in [3, 0, 5, 7]:
    img = T56[t_idx]; d = int(y_te[t_idx])
    m = img.mean(); deg = m+(img-m)*0.2
    line = f"digit={d}: "
    for nm in names:
        line += f"{nm}={np.mean((recon(deg,nm)-img)**2):.4f}  "
    print(line)

# 视觉: digit 3 硬 vs soft
img = T56[3]; m=img.mean(); deg=m+(img-m)*0.2
print("\n视觉 digit=3:")
print("原图:"); print(render(img))
print("硬4×4:"); print(render(recon(deg,"硬4×4")))
print("soft4×4σ0.6:"); print(render(recon(deg,"soft4×4σ0.6")))
