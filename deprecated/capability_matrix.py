#!/usr/bin/env python3
"""
capability_matrix.py — 冷眼能力矩阵: 每个组件拆开看贡献

组件叠加:
  L0: KNN (raw pixels) — 传统基线
  L1: 竞争路由 (100 nodes, 8×8 only) — 最简冷眼
  L2: MultiScale (3 scales, no shapes) — 多尺度激活向量
  L3: MultiScale + K-means shapes — 中层形状节点 (当前基线)
  L4: + Colony (tier权重增强) — 细胞行走形成的共识特征
  L5: + FB (误差校正) — 预测反馈闭环

每个级别测 4 个对比度, 看增量贡献。
"""

import sys, os, time, gzip, numpy as np
np.random.seed(42)
import random; random.seed(42)

def load_mnist():
    base = os.path.dirname(__file__) or '.'
    for kind in ['train', 't10k']:
        for suf in ['images-idx3-ubyte.gz', 'labels-idx1-ubyte.gz']:
            fpath = os.path.join(base, f'{kind}-{suf}')
            if not os.path.exists(fpath):
                import urllib.request
                urllib.request.urlretrieve(
                    f"https://storage.googleapis.com/cvdf-datasets/mnist/{kind}-{suf}", fpath)
    with gzip.open(os.path.join(base, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        Xt = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)/255.0
    with gzip.open(os.path.join(base, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        yt = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(os.path.join(base, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        Xv = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)/255.0
    with gzip.open(os.path.join(base, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        yv = np.frombuffer(f.read(), np.uint8, offset=8)
    return Xt, yt, Xv, yv

def low_contrast(X, c):
    Xc = X.copy()
    m = Xc.mean(axis=(1,2), keepdims=True)
    return m + (Xc - m) * c

class SimpleKNN:
    def __init__(self, k=5): self.k = k
    def fit(self, X, y): self.X, self.y = X, y
    def score(self, X, y):
        c = 0
        for i in range(len(X)):
            d = np.sum((self.X - X[i])**2, axis=1)
            nn = np.argpartition(d, self.k)[:self.k]
            if np.bincount(self.y[nn].astype(int)).argmax() == y[i]: c += 1
        return c/len(X)

# ═══ L1: 竞争路由 ═══
def build_l1(Xt, yt, n_nodes=100):
    from graph import VisionGraph
    from vision import VisionInterface
    g = VisionGraph(n_nodes=n_nodes, template_size=64)
    v = VisionInterface(g, patch_size=8, stride=4)
    ex = v.extractor
    idxs = np.random.choice(min(200,len(Xt)), min(200,len(Xt)), replace=False)
    ap = [p for i in idxs for p in ex.extract(Xt[i])]
    nids = sorted(g.nodes.keys())
    for k, nid in enumerate(nids):
        p = ap[k%len(ap)]
        g.nodes[nid].template = p.astype(np.float32)
        g.nodes[nid].template /= np.linalg.norm(g.nodes[nid].template)+1e-8

    # Hebbian 训练
    lr = 0.1
    for ep in range(3):
        for idx in np.random.permutation(min(3000,len(Xt))):
            v.set_image(Xt[idx].copy())
            for nid, aps in v.node_assignments.items():
                t = np.mean(aps, axis=0); n = np.linalg.norm(t)
                if n>0: t/=n
                g.nodes[nid].template += lr*(t-g.nodes[nid].template)
                g.nodes[nid].template /= np.linalg.norm(g.nodes[nid].template)+1e-8

    def feat(img):
        v.set_image(img)
        nids = sorted(g.nodes.keys())
        return np.array([g.nodes[nid].activation for nid in nids], dtype=np.float32)
    return feat

# ═══ L2: MultiScale ═══
def build_l2(Xt, yt, scales=None):
    from graph import VisionGraph
    from vision import VisionInterface
    if scales is None:
        scales = [
            {"ps":4,"st":4,"n":100},
            {"ps":8,"st":4,"n":100},
            {"ps":16,"st":8,"n":50},
        ]
    eyes = []
    for s in scales:
        ts = s["ps"]**2
        g = VisionGraph(n_nodes=s["n"], template_size=ts)
        v = VisionInterface(g, patch_size=s["ps"], stride=s["st"])
        eyes.append((g,v,s["ps"],s["st"]))

    for g,v,ps,st in eyes:
        ex=v.extractor
        nids=sorted(g.nodes.keys())
        idxs=np.random.choice(min(200,len(Xt)),min(200,len(Xt)),replace=False)
        ap=[p for i in idxs for p in ex.extract(Xt[i])]
        for k,nid in enumerate(nids):
            p=ap[k%len(ap)]
            g.nodes[nid].template=p.astype(np.float32)
            g.nodes[nid].template/=np.linalg.norm(g.nodes[nid].template)+1e-8

    lr=0.1
    for ep in range(3):
        for idx in np.random.permutation(min(5000,len(Xt))):
            img=Xt[idx].copy()
            if np.random.random()<0.5:
                m=img.mean(); c=0.3+np.random.random()*0.7; img=m+(img-m)*c
            for g,v,ps,st in eyes:
                v.set_image(img)
                for nid,aps in v.node_assignments.items():
                    t=np.mean(aps,axis=0); n=np.linalg.norm(t)
                    if n>0: t/=n
                    g.nodes[nid].template+=lr*(t-g.nodes[nid].template)
                    g.nodes[nid].template/=np.linalg.norm(g.nodes[nid].template)+1e-8

    def feat(img):
        parts=[]
        for g,v,ps,st in eyes:
            v.set_image(img)
            nids=sorted(g.nodes.keys())
            parts.append(np.array([g.nodes[nid].activation for nid in nids],dtype=np.float32))
        return np.concatenate(parts)
    return feat, eyes

# ═══ L3: +K-means shapes ═══
def build_l3(feat_fn, eyes, Xt, yt, k=100):
    n=min(5000,len(Xt))
    idxs=np.random.choice(len(Xt),n,replace=False)
    X=np.array([feat_fn(Xt[i]) for i in idxs],dtype=np.float32)

    rng=np.random.RandomState(42)
    cidx=rng.choice(len(X),k,replace=False)
    centers=X[cidx].copy()
    for it in range(50):
        dists=np.sum((X[:,None,:]-centers[None,:,:])**2,axis=2)
        km=np.argmin(dists,axis=1)
        newc=np.zeros_like(centers)
        for cc in range(k):
            m=km==cc
            newc[cc]=X[m].mean(axis=0) if m.sum()>0 else X[rng.choice(len(X))]
        if np.sum((centers-newc)**2)<1e-6: break
        centers=newc

    dists=np.sum((X[:,None,:]-centers[None,:,:])**2,axis=2)
    km=np.argmin(dists,axis=1)
    lbls=np.array([yt[i] for i in idxs])
    shape_dist={}
    for cc in range(k):
        m=km==cc
        if m.sum()>0: shape_dist[cc]=np.bincount(lbls[m],minlength=10)
    return centers.astype(np.float32), shape_dist

def predict_l3(images, feat_fn, centers, shape_dist):
    preds=[]
    for img in images:
        v=feat_fn(img)
        d=np.sum((centers-v)**2,axis=1)
        sid=int(np.argmin(d))
        preds.append(int(np.argmax(shape_dist.get(sid, np.ones(10)))) if sid in shape_dist and shape_dist[sid].sum()>=5 else 0)
    return np.array(preds)

# ═══ L4: +Colony ═══
def build_l4(eyes, Xt, yt, n_train=2000, gens=80):
    from synapse import SynapticLayer
    from vision_colony import VisionColony

    colonies = []
    for g,v,ps,st in eyes:
        g.synapse = SynapticLayer()
        coactive = {}
        for i in range(min(n_train, len(Xt))):
            v.set_image(Xt[i].copy())
            active = [nid for nid, n in g.nodes.items() if n.activation > 0.05]
            for j, src in enumerate(active):
                for dst in active[j+1:]:
                    coactive[(src,dst)] = coactive.get((src,dst),0)+1
        for (src,dst), cnt in coactive.items():
            if cnt >= 3:
                g.adjacency[src][dst] = min(1.0, cnt/20.0)
                g.adjacency[dst][src] = min(1.0, cnt/20.0)

        colony = VisionColony(g)
        colony.seed_cells(n_per_node=2, max_cells=200)
        colony.breathe(n_generations=gens, verbose=False)
        colonies.append(colony)
    return colonies

def extract_l4_feat(eyes, colonies, images, alpha=0.0):
    """alpha=0: 只用 tier 权重; alpha>0: 加误差校正"""
    feats = []
    for img in images:
        parts = []
        for (g,v,ps,st), col in zip(eyes, colonies):
            v.set_image(img)
            nids = sorted(g.nodes.keys())
            act = np.array([g.nodes[nid].activation for nid in nids], dtype=np.float32)
            if alpha > 0:
                error = v.predictive_boost(col, strength=0.3)
                act = np.clip(act + error*alpha, 0, 1)
            # tier 权重
            tw = np.zeros(len(nids), dtype=np.float32)
            for j, nid in enumerate(nids):
                c2=c3=0
                for (s,d), tier in col.synapse.tiers.items():
                    if s==nid or d==nid:
                        if tier==2: c2+=1
                        elif tier==3: c3+=1
                tw[j] = c3*0.3 + c2*0.1
            parts.append(np.concatenate([act, tw]))
        feats.append(np.concatenate(parts))
    return np.array(feats, dtype=np.float32)

# ═══ 主测试 ═══
def main():
    print("=" * 65)
    print("冷眼 能力矩阵 — 组件增量分析")
    print("=" * 65)

    Xt, yt, Xv, yv = load_mnist()
    n_train = 2000
    n_test = 500
    test_imgs = {c: (Xv[:n_test] if c==1.0 else low_contrast(Xv[:n_test], c)) for c in [1.0,0.5,0.3,0.2]}
    test_labels = yv[:n_test]
    contrasts = [1.0, 0.5, 0.3, 0.2]

    results = {}

    # ── L0: KNN raw ──
    print("\n── L0: KNN (raw pixels) ──")
    t0 = time.time()
    knn = SimpleKNN(k=5)
    knn.fit(Xt[:n_train].reshape(n_train,-1), yt[:n_train])
    r = {}
    for c in contrasts:
        r[c] = knn.score(test_imgs[c].reshape(n_test,-1), test_labels)
        print(f"  c={c:.1f}: {r[c]*100:.1f}%")
    results['L0: KNN raw'] = r
    print(f"  ({time.time()-t0:.1f}s)")

    # ── L1: 竞争路由 ──
    print("\n── L1: 竞争路由 (100 nodes, 8×8) ──")
    t0 = time.time()
    l1_feat = build_l1(Xt, yt, n_nodes=100)
    X1 = np.array([l1_feat(Xt[i]) for i in range(n_train)], dtype=np.float32)
    clf1 = SimpleKNN(k=5); clf1.fit(X1, yt[:n_train])
    r = {}
    for c in contrasts:
        Xt_c = np.array([l1_feat(test_imgs[c][i]) for i in range(n_test)], dtype=np.float32)
        r[c] = clf1.score(Xt_c, test_labels)
        print(f"  c={c:.1f}: {r[c]*100:.1f}%")
    results['L1: CompRouting'] = r
    print(f"  ({time.time()-t0:.1f}s)")

    # ── L2: MultiScale ──
    print("\n── L2: MultiScale (250 nodes, 3 scales) ──")
    t0 = time.time()
    l2_feat, eyes = build_l2(Xt, yt)
    X2 = np.array([l2_feat(Xt[i]) for i in range(n_train)], dtype=np.float32)
    clf2 = SimpleKNN(k=5); clf2.fit(X2, yt[:n_train])
    r = {}
    for c in contrasts:
        Xt_c = np.array([l2_feat(test_imgs[c][i]) for i in range(n_test)], dtype=np.float32)
        r[c] = clf2.score(Xt_c, test_labels)
        print(f"  c={c:.1f}: {r[c]*100:.1f}%")
    results['L2: MultiScale'] = r
    print(f"  ({time.time()-t0:.1f}s)")

    # ── L3: +Shapes ──
    print("\n── L3: +K-means shapes ──")
    t0 = time.time()
    centers, shape_dist = build_l3(l2_feat, eyes, Xt, yt, k=100)
    r = {}
    for c in contrasts:
        preds = predict_l3(test_imgs[c], l2_feat, centers, shape_dist)
        r[c] = (preds == test_labels).mean()
        print(f"  c={c:.1f}: {r[c]*100:.1f}%")
    results['L3: +Shapes'] = r
    print(f"  ({time.time()-t0:.1f}s)")

    # ── L4: +Colony ──
    print("\n── L4: +Colony (tier weights, no FB) ──")
    t0 = time.time()
    colonies = build_l4(eyes, Xt, yt, n_train=min(3000,len(Xt)), gens=80)
    X4 = extract_l4_feat(eyes, colonies, [Xt[i] for i in range(n_train)], alpha=0)
    clf4 = SimpleKNN(k=5); clf4.fit(X4, yt[:n_train])
    r = {}
    for c in contrasts:
        Xt_c = extract_l4_feat(eyes, colonies, [test_imgs[c][i] for i in range(n_test)], alpha=0)
        r[c] = clf4.score(Xt_c, test_labels)
        print(f"  c={c:.1f}: {r[c]*100:.1f}%")
    results['L4: +Colony'] = r
    print(f"  ({time.time()-t0:.1f}s)")

    # ── L5: +FB ──
    print("\n── L5: +FB (error-corrected activation) ──")
    t0 = time.time()
    X5 = extract_l4_feat(eyes, colonies, [Xt[i] for i in range(n_train)], alpha=0.3)
    clf5 = SimpleKNN(k=5); clf5.fit(X5, yt[:n_train])
    r = {}
    for c in contrasts:
        Xt_c = extract_l4_feat(eyes, colonies, [test_imgs[c][i] for i in range(n_test)], alpha=0.3)
        r[c] = clf5.score(Xt_c, test_labels)
        print(f"  c={c:.1f}: {r[c]*100:.1f}%")
    results['L5: +FB'] = r
    print(f"  ({time.time()-t0:.1f}s)")

    # ═══ 矩阵 ═══
    print("\n" + "=" * 75)
    print("CAPABILITY MATRIX")
    print("=" * 75)
    print(f"{'Level':<22s} {'c=1.0':>7s} {'c=0.5':>7s} {'c=0.3':>7s} {'c=0.2':>7s}  {'Δ0.2':>7s}")
    print("-" * 62)
    prev = None
    for name in ['L0: KNN raw','L1: CompRouting','L2: MultiScale','L3: +Shapes','L4: +Colony','L5: +FB']:
        r = results.get(name, {})
        row = f"{name:<22s}"
        vals = []
        for cl in contrasts:
            v = r.get(cl, 0)
            row += f" {v*100:>6.1f}%"
            vals.append(v)
        drop = vals[0] - vals[-1] if vals else 0
        row += f" {drop*100:>6.1f}%"
        print(row)

        # 增量
        if prev:
            incs = [vals[i] - prev[i] for i in range(len(vals))]
            lst = ' '.join([f'{x*100:>+6.1f}%' for x in incs])
            print(f"  {'':>20s}  Δ     {lst}")
        prev = vals

    # 衰减率对比
    print(f"\n{'Contrast resilience (c0.2/c1.0):':<22s}")
    for name in ['L0: KNN raw','L1: CompRouting','L2: MultiScale','L3: +Shapes','L4: +Colony','L5: +FB']:
        r = results.get(name, {})
        ratio = r.get(0.2,0)/max(0.01, r.get(1.0,0))
        print(f"  {name:<20s} {(1-ratio)*100:>5.1f}% decay  (keep {ratio*100:.0f}%)")

    print("\n=== DONE ===")

if __name__ == '__main__':
    main()
