#!/usr/bin/env python3
"""
brainfill_test.py — 用当前最优模型做脑补重建

测试: c=0.1 / 0.2 / 0.5 下 hint-guided 传播能否重建数字轮廓
"""

import gzip, numpy as np, time
np.random.seed(42)

def load_mnist():
    import os
    base = '/home/duyw/predictive-vision'
    for kind in ['train','t10k']:
        for suf in ['images-idx3-ubyte.gz','labels-idx1-ubyte.gz']:
            fpath = os.path.join(base, f'{kind}-{suf}')
            if not os.path.exists(fpath):
                import urllib.request
                urllib.request.urlretrieve(f"https://storage.googleapis.com/cvdf-datasets/mnist/{kind}-{suf}", fpath)
    with gzip.open(os.path.join(base,'train-images-idx3-ubyte.gz'),'rb') as f:
        Xt=np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255.0
    with gzip.open(os.path.join(base,'train-labels-idx1-ubyte.gz'),'rb') as f:
        yt=np.frombuffer(f.read(),np.uint8,offset=8)
    with gzip.open(os.path.join(base,'t10k-images-idx3-ubyte.gz'),'rb') as f:
        Xv=np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255.0
    with gzip.open(os.path.join(base,'t10k-labels-idx1-ubyte.gz'),'rb') as f:
        yv=np.frombuffer(f.read(),np.uint8,offset=8)
    return Xt, yt, Xv, yv

def low_contrast(X, c):
    Xc = X.copy(); m = Xc.mean(axis=(1,2), keepdims=True); return m + (Xc - m) * c

Xt, yt, Xv, yv = load_mnist()

# ═══ 训练 (60K × 3ep, 快速) ═══
from graph import VisionGraph
from vision import VisionInterface

cfgs = [{'ps':4,'st':4,'n':100},{'ps':8,'st':4,'n':100},{'ps':16,'st':8,'n':50}]
eyes = []
for cfg in cfgs:
    ts = cfg['ps']**2
    g = VisionGraph(n_nodes=cfg['n'], template_size=ts)
    v = VisionInterface(g, patch_size=cfg['ps'], stride=cfg['st'])
    eyes.append((g, v))

print("Training..."); t0 = time.time()
for g, v in eyes:
    ex = v.extractor; nids = sorted(g.nodes.keys())
    idxs = np.random.choice(500, 500, replace=False)
    ap = [p for i in idxs for p in ex.extract(Xt[i%500])]
    for k, nid in enumerate(nids):
        p = ap[k%len(ap)]; g.nodes[nid].template = p.astype(np.float32)
        g.nodes[nid].template /= np.linalg.norm(g.nodes[nid].template) + 1e-8

lr = 0.1
for ep in range(3):
    for idx in np.random.permutation(60000):
        img = Xt[idx].copy()
        if np.random.random() < 0.5:
            m = img.mean(); c = 0.3 + np.random.random() * 0.7; img = m + (img - m) * c
        for g, v in eyes:
            v.set_image(img)
            for nid, aps in v.node_assignments.items():
                t = np.mean(aps, axis=0); n = np.linalg.norm(t)
                if n > 0: t /= n
                g.nodes[nid].template += lr*(t - g.nodes[nid].template)
                g.nodes[nid].template /= np.linalg.norm(g.nodes[nid].template) + 1e-8
print(f"  done ({time.time()-t0:.0f}s)")

# ═══ 脑补 ═══
print("\n=== Brain Fill Test ===")

for test_idx in [3, 5, 18, 42]:  # 不同数字
    digit = int(yv[test_idx])
    print(f"\n--- test[{test_idx}] digit={digit} ---")
    
    for c_level in [1.0, 0.2, 0.1]:
        img = Xv[test_idx] if c_level == 1.0 else low_contrast(Xv[test_idx:test_idx+1], c_level)[0]
        
        # 路由
        all_act = []
        for g, v in eyes:
            v.set_image(img)
            nids = sorted(g.nodes.keys())
            all_act.extend([g.nodes[nid].activation for nid in nids])
        
        act = np.array(all_act)
        n_active = int(np.sum(act > 0.01))
        top = np.argsort(-act)[:10]
        top_vals = act[top]
        
        # 热点重建
        canvas = np.zeros((28, 28), dtype=np.float32)
        off = 0
        for g, v in eyes:
            nids = sorted(g.nodes.keys())
            nn = len(nids)
            for j in range(nn):
                idx = off + j
                if act[idx] > 0.01:
                    node = g.nodes[nids[j]]
                    cy, cx = 14, 14  # fallback center
                    yy, xx = np.mgrid[0:28, 0:28]
                    sigma = 6
                    gauss = np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*sigma**2))
                    canvas += gauss * act[idx]
            off += nn
        
        # 热图归一化
        canvas = canvas / max(canvas.max(), 0.001)
        
        # ASCII art
        shades = ' .-=+*#%@'
        print(f"  c={c_level:.1f}  n_active={n_active}  max_act={act.max():.3f}")
        print(f"  top5: values={[f'{top_vals[i]:.2f}' for i in range(5)]}")
        # 只打印非零区域
        print("  reconstructed:")
        for row_idx in range(0, 28, 3):
            row_vals = [canvas[row_idx, col] for col in range(0, 28, 3)]
            row_str = ''.join(shades[min(9, int(v*10))] for v in row_vals)
            print(f"  {row_str}")
        print()

print("=== DONE ===")
