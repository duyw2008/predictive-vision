#!/usr/bin/env python3
"""
sweep_alpha.py — FB α 参数扫 4 档, 直接出对比表
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

print("=" * 55)
print("FB α parameter sweep: 0.3 / 0.5 / 0.8 / 1.0")
print("=" * 55)

Xt, yt, Xv, yv = load_mnist()

# ═══ 训练 MultiScaleEye (一次, 共享) ═══
from graph import VisionGraph
from vision import VisionInterface

configs = [{"ps":4,"st":4,"n":100},{"ps":8,"st":4,"n":100},{"ps":16,"st":8,"n":50}]
eyes = []
for cfg in configs:
    ts = cfg["ps"]**2
    g = VisionGraph(n_nodes=cfg["n"], template_size=ts)
    v = VisionInterface(g, patch_size=cfg["ps"], stride=cfg["st"])
    eyes.append((g,v,cfg["ps"],cfg["st"]))

# init templates
for g,v,ps,st in eyes:
    ex=v.extractor; nids=sorted(g.nodes.keys())
    idxs=np.random.choice(min(200,len(Xt)),min(200,len(Xt)),replace=False)
    ap=[p for i in idxs for p in ex.extract(Xt[i])]
    for k,nid in enumerate(nids):
        p=ap[k%len(ap)]; g.nodes[nid].template=p.astype(np.float32)
        g.nodes[nid].template/=np.linalg.norm(g.nodes[nid].template)+1e-8

# Hebbian
print("\n[1] Training MultiScaleEye (250n, 10K, 3ep)...")
t0=time.time()
lr=0.1
for ep in range(3):
    for idx in np.random.permutation(10000):
        img=Xt[idx].copy()
        if np.random.random()<0.5: m=img.mean(); c=0.3+np.random.random()*0.7; img=m+(img-m)*c
        for g,v,ps,st in eyes:
            v.set_image(img)
            for nid,aps in v.node_assignments.items():
                t=np.mean(aps,axis=0); n=np.linalg.norm(t)
                if n>0: t/=n
                g.nodes[nid].template+=lr*(t-g.nodes[nid].template)
                g.nodes[nid].template/=np.linalg.norm(g.nodes[nid].template)+1e-8
print(f"  done ({time.time()-t0:.1f}s)")

# ═══ Colony (一次) ═══
from synapse import SynapticLayer
from vision_colony import VisionColony

print("\n[2] Building KG + Colony on each eye...")
t0=time.time()
colonies = []
for g,v,ps,st in eyes:
    g.synapse = SynapticLayer()
    coactive={}
    for i in range(2000):
        v.set_image(Xt[i].copy())
        active=[nid for nid,n in g.nodes.items() if n.activation>0.05]
        for j,src in enumerate(active):
            for dst in active[j+1:]:
                coactive[(src,dst)]=coactive.get((src,dst),0)+1
    for (src,dst),cnt in coactive.items():
        if cnt>=3: g.adjacency[src][dst]=min(1.0,cnt/20.0); g.adjacency[dst][src]=min(1.0,cnt/20.0)
    colony=VisionColony(g); colony.seed_cells(n_per_node=2,max_cells=200)
    colony.breathe(n_generations=80,verbose=False)
    colonies.append(colony)
print(f"  done ({time.time()-t0:.1f}s)")

# Precompute tier weights (static, not per-image)
tier_weights = []
for (g,v,ps,st),col in zip(eyes, colonies):
    nids = sorted(g.nodes.keys())
    tw = np.zeros(len(nids), dtype=np.float32)
    for j, nid in enumerate(nids):
        c2=c3=0
        for (s,d),tier in col.synapse.tiers.items():
            if s==nid or d==nid:
                if tier==2: c2+=1
                elif tier==3: c3+=1
        tw[j] = c3*0.3 + c2*0.1
    tier_weights.append((nids, tw))

def extract_fast(eyes, colonies, images, alpha, tier_weights):
    feats=[]
    for img in images:
        parts=[]
        for (g,v,ps,st), col, (nids, tw) in zip(eyes, colonies, tier_weights):
            v.set_image(img)
            act = np.array([g.nodes[nid].activation for nid in nids], dtype=np.float32)
            if alpha > 0:
                error = v.predictive_boost(col, strength=0.3)
                act = np.clip(act + error*alpha, 0, 1)
            parts.append(np.concatenate([act, tw]))
        feats.append(np.concatenate(parts))
    return np.array(feats, dtype=np.float32)

print(f"\n[3] Precomputing tier weights...")
t0=time.time()
print(f"    done ({time.time()-t0:.1f}s)")

n_train=2000; n_test=500
contrasts=[1.0,0.5,0.3,0.2]

print(f"\n[4] Sweep α = 0.0 / 0.3 / 0.5 / 0.8 / 1.0")
print(f"    ({n_train} train, {n_test} test per α)")
t0=time.time()

results={}
for alpha in [0.0, 0.3, 0.5, 0.8, 1.0]:
    Xtr=extract_fast(eyes, colonies, [Xt[i] for i in range(n_train)], alpha, tier_weights)
    knn=SimpleKNN(k=5); knn.fit(Xtr, yt[:n_train])
    r={}
    for c in contrasts:
        Xc = Xv[:n_test] if c==1.0 else low_contrast(Xv[:n_test], c)
        Xte=extract_fast(eyes, colonies, Xc, alpha, tier_weights)
        r[c]=knn.score(Xte, yv[:n_test])
    results[alpha]=r
    label="baseline (α=0)" if alpha==0 else f"α={alpha}"
    print(f"  {label:18s}  {r[1.0]*100:5.1f}%  {r[0.5]*100:5.1f}%  {r[0.3]*100:5.1f}%  {r[0.2]*100:5.1f}%")

print(f"\n  total: {time.time()-t0:.1f}s")

print("\n" + "=" * 55)
print("RESULTS: FB alpha sweep")
print("=" * 55)
print(f"{'α':<8s}  {'c=1.0':>7s}  {'c=0.5':>7s}  {'c=0.3':>7s}  {'c=0.2':>7s}")
print("-" * 45)
baseline = results[0.0]
for alpha in [0.0, 0.3, 0.5, 0.8, 1.0]:
    r = results[alpha]
    row = f"{'baseline' if alpha==0 else f'α={alpha}':<8s}"
    for cl in contrasts:
        row += f"  {r[cl]*100:>5.1f}%"
        if alpha > 0:
            delta = r[cl] - baseline[cl]
            if delta != 0: row += f"({delta*100:+.1f})"
    print(row)

# ═══ 找最优 α ═══
print("\nOptimal α per contrast:")
for cl in contrasts:
    best=0.0; best_acc=baseline[cl]
    for a in [0.3,0.5,0.8,1.0]:
        if results[a][cl] > best_acc: best=a; best_acc=results[a][cl]
    print(f"  c={cl:.1f}: α={best} ({best_acc*100:.1f}%, Δ={best_acc-baseline[cl]:+.3%})")

print("\n=== DONE ===")
