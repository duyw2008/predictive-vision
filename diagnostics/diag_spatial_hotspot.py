#!/usr/bin/env python3
"""诊断: 节点能否涌现稳定空间热点? (费曼脑设计思路的第一块基石)
训练时节点每赢一个 patch 就累积位置 → 看节点是否专精于特定空间区域
(而不是随机散布 = 无空间拓扑)"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import load_mnist
from vision import VisionInterface
from graph import VisionGraph

np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()
def upscale(X, f=2):
    return np.kron(X, np.ones((f, f), np.float32))
X56 = upscale(X_tr[:12000])

ps, st, n = 8, 4, 300
g = VisionGraph(n_nodes=n, template_size=ps*ps)
v = VisionInterface(g, patch_size=ps, stride=st)
nids = sorted(g.nodes.keys())

rng = np.random.RandomState(42)
idxs = rng.choice(len(X56), 200, replace=False)
ap = [p.astype(np.float32) for i in idxs for p in v.extractor.extract(X56[i]-X56[i].mean())]
for k, nid in enumerate(nids):
    p = ap[k%len(ap)]; p/=np.linalg.norm(p)+1e-8; g.nodes[nid].template=p

# 空间热点累积: cy_sum/cx_sum/count (每个节点赢的 patch 位置加权平均)
cy_sum = {nid: 0.0 for nid in nids}; cx_sum = {nid: 0.0 for nid in nids}
cnt = {nid: 0 for nid in nids}

print("训练 + 累积空间热点 (12000×2)..."); t0=time.time()
for ep in range(2):
    for idx in rng.permutation(len(X56)):
        img = X56[idx].copy(); m=img.mean()
        if rng.random()<0.5: img = m+(img-m)*(0.3+rng.random()*0.7)
        img = img-img.mean(); v.set_image(img)
        # 记录每个 patch 的 winner 节点 + patch 位置
        patches = v._last_patches
        positions = v.extractor.patch_positions
        templates = np.array([g.nodes[nid].template for nid in nids], np.float32)
        for p_i, (y1, x1, y2, x2) in enumerate(positions):
            if p_i >= len(patches): break
            scores = templates @ patches[p_i]
            best = int(np.argmax(scores))
            if scores[best] < 0: continue
            cy = (y1+y2)/2; cx = (x1+x2)/2
            nid = nids[best]
            cy_sum[nid] += cy; cx_sum[nid] += cx; cnt[nid] += 1
        # Hebbian 更新 (同诊断脚本)
        best = max(nids, key=lambda x: g.nodes[x].activation); node=g.nodes[best]
        t = np.mean(v.node_assignments.get(best, [v.extractor.extract(img)[0]]), axis=0)
        tn = np.linalg.norm(t); t = t/(tn+1e-8) if tn>0 else t
        node.template += 0.1*(t-node.template); node.template /= np.linalg.norm(node.template)+1e-8
print(f"  完成 {time.time()-t0:.0f}s")

# 分析空间热点分布
cy = np.array([cy_sum[nid]/max(cnt[nid],1) for nid in nids])
cx = np.array([cx_sum[nid]/max(cnt[nid],1) for nid in nids])
cnts = np.array([cnt[nid] for nid in nids])

# 1. 热点是否分化 (标准差 vs 随机均匀分布的标准差)
# 随机均匀: cy 在 [0,56] 均匀, std = 56/sqrt(12) ≈ 16.2
uniform_std = 56/np.sqrt(12)
print(f"\n节点空间热点分析 (n={n}, 56×56):")
print(f"  cy 范围: [{cy.min():.1f}, {cy.max():.1f}]  均值 {cy.mean():.1f}")
print(f"  cx 范围: [{cx.min():.1f}, {cx.max():.1f}]  均值 {cx.mean():.1f}")
print(f"  cy std = {cy.std():.1f}  (随机均匀 std ≈ {uniform_std:.1f})")

# 2. 高胜率节点的热点 (top 10 活跃节点)
top = np.argsort(-cnts)[:10]
print(f"\n  top 10 活跃节点 (胜率最高):")
print(f"  {'node':6s} {'count':>6s} {'cy':>6s} {'cx':>6s}")
for i in top:
    print(f"  {nids[i]:6s} {cnts[i]:6d} {cy[i]:6.1f} {cx[i]:6.1f}")

# 3. 热点集中度: 每个节点热点到图像中心的距离分布
dist_center = np.sqrt((cy-28)**2 + (cx-28)**2)
print(f"\n  热点到图像中心距离: 均值 {dist_center.mean():.1f} (随机≈{np.sqrt(2)*56/np.sqrt(12)*0.5:.0f})")
print(f"  热点是否均匀覆盖: cy/cx 相关系数 = {np.corrcoef(cy,cx)[0,1]:.3f} (≈0 表示独立散布)")

# 4. 关键判断: 节点热点是否"分化" (有空间专精) vs "混杂" (都在中心)
spread_ratio = np.sqrt(cy.var()+cx.var()) / uniform_std
print(f"\n  ★ 空间分化比 = {spread_ratio:.2f}  (>1 分化明显, <0.5 热点挤在中心)")
if spread_ratio > 0.8:
    print(f"  → 节点形成了空间热点分化, 空间拓扑有望从结构中涌现")
else:
    print(f"  → 节点热点挤在中心 (MNIST 数字居中导致), 需要先做位置归一化或去中心化")

# ═══ 第二步: 用空间热点重建, 对比全局计数的线性回归 ═══
print("\n" + "="*60)
print("空间热点重建 vs 全局计数线性回归")
print("="*60)

H, W = 56, 56
T56 = upscale(X_te[:300])
n_fit = 3000
fit_idx = rng.choice(len(X56), n_fit, replace=False)
F = np.zeros((n_fit, H*W), np.float32)
for i, idx in enumerate(fit_idx): F[i] = X56[idx].reshape(-1).astype(np.float32)

templates = np.array([g.nodes[nid].template for nid in nids], np.float32)
hot_cy = np.array([cy_sum[nid]/max(cnt[nid],1) for nid in nids])
hot_cx = np.array([cx_sum[nid]/max(cnt[nid],1) for nid in nids])

def act_global(img):
    v.set_image(img - img.mean())
    return np.array([g.nodes[nid].activation for nid in nids], np.float32)

def act_hotspot(img):
    """空间热点激活: 每个 patch 的 winner 节点, 按热点位置软投影到网格"""
    v.set_image(img - img.mean())
    patches = v._last_patches
    positions = v.extractor.patch_positions
    if patches is None: return np.zeros(n*16, np.float32)
    # 4×4 网格, 但用节点热点位置投影 (而非 patch 中心硬归)
    ra = np.zeros((4, 4, n), np.float32)
    for p_i, (y1, x1, y2, x2) in enumerate(positions):
        if p_i >= len(patches): break
        scores = templates @ patches[p_i]; best = int(np.argmax(scores))
        if scores[best] < 0: continue
        # 用 winner 节点的热点位置 (而非 patch 位置) 归到网格
        hcy = hot_cy[best]; hcx = hot_cx[best]
        ry = min(int(hcy/(H/4)), 3); rx = min(int(hcx/(W/4)), 3)
        ra[ry, rx, best] += 1.0
    ra = ra.reshape(16, n)
    return np.minimum(1.0, ra/(np.maximum(ra.sum(axis=1, keepdims=True),1e-8))*3).reshape(-1)

def fit_W(dim, act_fn):
    A = np.zeros((n_fit, dim), np.float32)
    for i, idx in enumerate(fit_idx): A[i] = act_fn(X56[idx])
    return (np.linalg.pinv(A.T @ A) @ (A.T @ F)).astype(np.float32)

print("拟合 W (全局300d / 热点4×4 4800d)...")
W_global = fit_W(n, act_global)
W_hot = fit_W(16*n, act_hotspot)

def recon_global(img): return (act_global(img) @ W_global).reshape(img.shape)
def recon_hot(img): return (act_hotspot(img) @ W_hot).reshape(img.shape)

print("\n脑补 MSE 对比 (降质 c=0.2):")
for t_idx in [3, 5, 18, 0, 1]:
    img = T56[t_idx]; d = int(y_te[t_idx])
    m = img.mean(); deg = m+(img-m)*0.2
    rg = recon_global(deg); rh = recon_hot(deg)
    print(f"  digit={d}: 全局={np.mean((rg-img)**2):.4f}  热点重建={np.mean((rh-img)**2):.4f}")

shades = ' .:-=+*#%@'
def render(img):
    return '\n'.join(''.join(shades[min(9,max(0,int(p*10)))] for p in row[::4]) for row in img[::4])
img = T56[3]; m=img.mean(); deg=m+(img-m)*0.2
print("\n视觉对比 digit=3:")
print("原图:"); print(render(img))
print("全局重建:"); print(render(recon_global(deg)))
print("热点重建:"); print(render(recon_hot(deg)))
