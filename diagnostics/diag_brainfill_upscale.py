#!/usr/bin/env python3
"""诊断: 脑补瓶颈 = 分辨率 or 重建方式?
对比: 线性回归(当前) vs 带空间定位的模板叠加
在放大2×的MNIST(56×56)上验证"""
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

# ── PatchEye (ps=8, st=4, n=300) — 支持任意尺寸 ──
ps, st, n = 8, 4, 300
g = VisionGraph(n_nodes=n, template_size=ps*ps)
v = VisionInterface(g, patch_size=ps, stride=st)
nids = sorted(g.nodes.keys())

# init templates
rng = np.random.RandomState(42)
idxs = rng.choice(len(X56), min(200, len(X56)), replace=False)
ap = [p.astype(np.float32) for i in idxs for p in v.extractor.extract(X56[i] - X56[i].mean())]
for k, nid in enumerate(nids):
    p = ap[k % len(ap)]; p /= np.linalg.norm(p) + 1e-8
    g.nodes[nid].template = p

# train Hebbian
lr = 0.1
print("训练中 (12000×2 epochs)...")
t0 = time.time()
for ep in range(2):
    for idx in rng.permutation(len(X56)):
        img = X56[idx].copy()
        m = img.mean()
        if rng.random() < 0.5:
            img = m + (img - m) * (0.3 + rng.random() * 0.7)
        img = img - img.mean()
        v.set_image(img)
        best_nid = max(nids, key=lambda x: g.nodes[x].activation)
        node = g.nodes[best_nid]
        t = np.mean(v.node_assignments.get(best_nid, [v.extractor.extract(img)[0]]), axis=0)
        tn = np.linalg.norm(t); t = t/(tn+1e-8) if tn > 0 else t
        node.template += lr*(t-node.template)
        node.template /= np.linalg.norm(node.template)+1e-8
print(f"  训练完成 {time.time()-t0:.0f}s")

# ── 重建函数 ──
def get_acts(img):
    v.set_image(img - img.mean())
    return np.array([g.nodes[nid].activation for nid in nids], np.float32)

def template_overlay(img):
    """带空间定位的模板叠加: 每个patch位置放回winner模板×激活"""
    v.set_image(img - img.mean())
    H, W = img.shape
    canvas = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    positions = v.extractor.patch_positions
    patches = v._last_patches
    templates = np.array([g.nodes[nid].template for nid in nids], np.float32)
    node_acts = np.array([g.nodes[nid].activation for nid in nids], np.float32)
    if patches is None:
        return canvas
    for p_i, (y1, x1, y2, x2) in enumerate(positions):
        if p_i >= len(patches): break
        scores = templates @ patches[p_i]
        best = int(np.argmax(scores))
        if scores[best] < 0: continue
        tmpl = templates[best].reshape(ps, ps)
        act = node_acts[best]
        canvas[y1:y2, x1:x2] += tmpl * act
        cnt[y1:y2, x1:x2] += 1
    canvas = canvas / (cnt + 1e-8)
    return canvas

# ── 拟合线性回归 W (act→原图) ──
print("拟合线性回归 W...")
n_fit = 3000
fit_idx = rng.choice(len(X56), n_fit, replace=False)
H, W = 56, 56
A = np.zeros((n_fit, n), np.float32)
F = np.zeros((n_fit, H*W), np.float32)
for i, idx in enumerate(fit_idx):
    A[i] = get_acts(X56[idx])
    F[i] = X56[idx].reshape(-1).astype(np.float32)
W = (np.linalg.pinv(A.T @ A) @ (A.T @ F)).astype(np.float32)

def linear_recon(img):
    act = get_acts(img)
    return (act @ W).reshape(img.shape)

# ── 对比 ──
shades = ' .:-=+*#%@'
def ascii_render(img):
    lines = []
    for row in img[::4]:
        lines.append(''.join(shades[min(9, max(0, int(p*10)))] for p in row[::4]))
    return '\n'.join(lines)

for t_idx in [3, 5, 18]:
    img = T56[t_idx]; d = int(y_te[t_idx])
    m = img.mean(); deg = m + (img - m) * 0.2
    lin = linear_recon(deg)
    tpl = template_overlay(deg)
    mse_lin = np.mean((lin - img)**2)
    mse_tpl = np.mean((tpl - img)**2)
    print(f"\n{'='*50}\ndigit={d} (56×56, 降质c=0.2)")
    print(f"  MSE 线性回归: {mse_lin:.4f}  |  模板叠加: {mse_tpl:.4f}")
    print("  原图:"); print(ascii_render(img))
    print("  线性回归重建:"); print(ascii_render(lin))
    print("  模板叠加重建:"); print(ascii_render(tpl))
