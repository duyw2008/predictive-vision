#!/usr/bin/env python3
"""
adaptive_fb.py — 自适应 α: 根据激活水平自动调 FB 强度

α = clip(1.0 - mean_activation × 3.0, 0.0, 0.8)
  满对比度 (act ≈ 0.3-0.4) → α ≈ 0.0-0.1
  低对比度 (act ≈ 0.08-0.12) → α ≈ 0.6-0.8

对比: baseline (α=0) vs fixed α=0.8 vs adaptive α
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
                import urllib.request; urllib.request.urlretrieve(
                    f"https://storage.googleapis.com/cvdf-datasets/mnist/{kind}-{suf}", fpath)
    with gzip.open(os.path.join(base, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        Xt=np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255.0
    with gzip.open(os.path.join(base, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        yt=np.frombuffer(f.read(),np.uint8,offset=8)
    with gzip.open(os.path.join(base, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        Xv=np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255.0
    with gzip.open(os.path.join(base, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        yv=np.frombuffer(f.read(),np.uint8,offset=8)
    return Xt,yt,Xv,yv

def low_contrast(X,c):
    Xc=X.copy(); m=Xc.mean(axis=(1,2),keepdims=True); return m+(Xc-m)*c

class SimpleKNN:
    def __init__(self,k=5): self.k=k
    def fit(self,X,y): self.X,self.y=X,y
    def score(self,X,y): 
        c=0
        for i in range(len(X)):
            d=np.sum((self.X-X[i])**2,axis=1); nn=np.argpartition(d,self.k)[:self.k]
            if np.bincount(self.y[nn].astype(int)).argmax()==y[i]: c+=1
        return c/len(X)

print("="*55)
print("Adaptive FB α — 对比度自适应")
print("="*55)

Xt,yt,Xv,yv=load_mnist()

# ═══ 训练 MultiScaleEye ═══
from graph import VisionGraph
from vision import VisionInterface

configs=[{"ps":4,"st":4,"n":100},{"ps":8,"st":4,"n":100},{"ps":16,"st":8,"n":50}]
eyes=[]
for cfg in configs:
    ts=cfg["ps"]**2; g=VisionGraph(n_nodes=cfg["n"],template_size=ts)
    v=VisionInterface(g,patch_size=cfg["ps"],stride=cfg["st"])
    eyes.append((g,v,cfg["ps"],cfg["st"]))

for g,v,ps,st in eyes:
    ex=v.extractor; nids=sorted(g.nodes.keys())
    idxs=np.random.choice(min(200,len(Xt)),min(200,len(Xt)),replace=False)
    ap=[p for i in idxs for p in ex.extract(Xt[i])]
    for k,nid in enumerate(nids): p=ap[k%len(ap)]; g.nodes[nid].template=p.astype(np.float32); g.nodes[nid].template/=np.linalg.norm(g.nodes[nid].template)+1e-8

print("\n[1] Training MultiScaleEye...")
t0=time.time(); lr=0.1
for ep in range(3):
    for idx in np.random.permutation(10000):
        img=Xt[idx].copy()
        if np.random.random()<0.5: m=img.mean(); c=0.3+np.random.random()*0.7; img=m+(img-m)*c
        for g,v,ps,st in eyes:
            v.set_image(img)
            for nid,aps in v.node_assignments.items():
                t=np.mean(aps,axis=0); n=np.linalg.norm(t)
                if n>0: t/=n
                g.nodes[nid].template+=lr*(t-g.nodes[nid].template); g.nodes[nid].template/=np.linalg.norm(g.nodes[nid].template)+1e-8
print(f"  done ({time.time()-t0:.1f}s)")

# ═══ Colony ═══
from synapse import SynapticLayer
from vision_colony import VisionColony

print("\n[2] Building KG + Colony...")
t0=time.time()
colonies=[]
for g,v,ps,st in eyes:
    g.synapse=SynapticLayer()
    coactive={}
    for i in range(2000):
        v.set_image(Xt[i].copy()); active=[nid for nid,n in g.nodes.items() if n.activation>0.05]
        for j,src in enumerate(active):
            for dst in active[j+1:]: coactive[(src,dst)]=coactive.get((src,dst),0)+1
    for (src,dst),cnt in coactive.items():
        if cnt>=3: g.adjacency[src][dst]=min(1.0,cnt/20.0); g.adjacency[dst][src]=min(1.0,cnt/20.0)
    colony=VisionColony(g); colony.seed_cells(n_per_node=2,max_cells=200)
    colony.breathe(n_generations=80,verbose=False); colonies.append(colony)
print(f"  done ({time.time()-t0:.1f}s)")

# ═══ Precompute tier weights ═══
tier_weights=[]
for (g,v,ps,st),col in zip(eyes,colonies):
    nids=sorted(g.nodes.keys()); tw=np.zeros(len(nids),dtype=np.float32)
    for j,nid in enumerate(nids):
        c2=c3=0
        for (s,d),tier in col.synapse.tiers.items():
            if s==nid or d==nid:
                if tier==2: c2+=1
                elif tier==3: c3+=1
        tw[j]=c3*0.3+c2*0.1
    tier_weights.append((nids,tw))

# ═══ Extract with adaptive α ═══
def extract_adaptive(eyes, colonies, images, alpha_mode, tier_weights):
    alphas_used = []
    feats=[]
    for img in images:
        # 先跑所有眼, 收集全图激活
        all_act = []
        eye_data = []
        for (g,v,ps,st), col, (nids, tw) in zip(eyes, colonies, tier_weights):
            v.set_image(img)
            act = np.array([g.nodes[nid].activation for nid in nids], dtype=np.float32)
            eye_data.append((g,v,ps,st,col,nids,tw,act))
            all_act.append(act)
        
        # 统一算 α (全 250 维激活)
        full_act = np.concatenate(all_act)
        if alpha_mode == 'baseline':
            alpha = 0.0
        elif alpha_mode == 'fixed':
            alpha = 0.8
        elif alpha_mode == 'adaptive':
            active_ratio = float(np.mean(full_act > 0.02))
            alpha = 0.0 if active_ratio > 0.15 else 0.8
        else:
            alpha = 0.0
        
        alphas_used.append(alpha)
        
        # 用统一 α 处理每个眼
        parts = []
        for g,v,ps,st,col,nids,tw,act in eye_data:
            if alpha > 0:
                error = v.predictive_boost(col, strength=0.3)
                act = np.clip(act + error * alpha, 0, 1)
            parts.append(np.concatenate([act, tw]))
        feats.append(np.concatenate(parts))
    return np.array(feats, dtype=np.float32), np.array(alphas_used)

n_train=2000; n_test=500
contrasts=[1.0,0.5,0.3,0.2]

print(f"\n[3] Testing: baseline / fixed α=0.8 / adaptive α")
print(f"    adaptive: α = clip((1 - active_ratio) × 1.5, 0, 0.8)")
t0=time.time()

results={}
alpha_stats={}
for mode, label in [('baseline','α=0'), ('fixed','α=0.8'), ('adaptive','adaptive')]:
    Xtr, a_tr = extract_adaptive(eyes, colonies, [Xt[i] for i in range(n_train)], mode, tier_weights)
    knn=SimpleKNN(k=5); knn.fit(Xtr, yt[:n_train])
    r={}; a_vals={}
    for c in contrasts:
        Xc = Xv[:n_test] if c==1.0 else low_contrast(Xv[:n_test], c)
        Xte, a_te = extract_adaptive(eyes, colonies, Xc, mode, tier_weights)
        r[c]=knn.score(Xte, yv[:n_test])
        a_vals[c]=float(np.mean(a_te))
    results[label]=r; alpha_stats[label]=a_vals
    print(f"  {label:12s}  {r[1.0]*100:5.1f}%  {r[0.5]*100:5.1f}%  {r[0.3]*100:5.1f}%  {r[0.2]*100:5.1f}%"
          f"  (α: {a_vals[1.0]:.2f}/{a_vals[0.5]:.2f}/{a_vals[0.3]:.2f}/{a_vals[0.2]:.2f})")

print(f"\n  total: {time.time()-t0:.1f}s")

# ═══ Results ═══
print("\n" + "="*65)
print("RESULTS: Adaptive FB α")
print("="*65)
print(f"{'Mode':<12s}  {'c=1.0':>7s}  {'c=0.5':>7s}  {'c=0.3':>7s}  {'c=0.2':>7s}")
print("-"*48)
base = results['α=0']
best = {}
for cl in contrasts:
    best[cl] = base[cl]
for label in ['α=0','α=0.8','adaptive']:
    r = results[label]; a = alpha_stats[label]
    row = f"{label:<12s}"
    for cl in contrasts:
        row += f"  {r[cl]*100:>5.1f}%"
    print(row + f"  α:{a[1.0]:.2f}/{a[0.5]:.2f}/{a[0.3]:.2f}/{a[0.2]:.2f}")
    for cl in contrasts:
        if r[cl] > best[cl]: best[cl] = r[cl]

print(f"\n{'Best per contrast':<12s}", end="")
for cl in contrasts:
    print(f"  {best[cl]*100:>5.1f}%", end="")
print()

# Gain over baseline
print("\nGain over α=0:")
for label in ['α=0.8','adaptive']:
    r = results[label]
    gains = [r[cl]-base[cl] for cl in contrasts]
    print(f"  {label:12s}  " + "  ".join([f"{g*100:>+5.1f}%" for g in gains]))

print("\n=== DONE ===")
