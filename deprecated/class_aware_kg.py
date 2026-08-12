#!/usr/bin/env python3
"""
class_aware_kg.py — tier 边带类别上下文

改动: 建 KG 时记录每对共激活的类别分布, 边权重 × 类别纯度
  
对比: baseline (无类别) vs class-aware (带纯度)

纯边: 只在 1-2 个数字里共现 → purity 高 → 权重高 → 细胞偏好走
杂边: 均匀分布在所有数字 → purity 低 → 权重低 → 细胞少走
"""

import sys, os, time, gzip, numpy as np
from collections import Counter
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
print("Class-Aware KG — 边带类别纯度")
print("="*55)

Xt,yt,Xv,yv=load_mnist()

# ═══ MultiScaleEye ═══
from graph import VisionGraph
from vision import VisionInterface

configs=[{"ps":4,"st":4,"n":100},{"ps":8,"st":4,"n":100},{"ps":16,"st":8,"n":50}]
eyes=[]
for cfg in configs:
    ts=cfg["ps"]**2; g=VisionGraph(n_nodes=cfg["n"],template_size=ts)
    v=VisionInterface(g,patch_size=cfg["ps"],stride=cfg["st"])
    eyes.append((g,v,cfg["ps"],cfg["st"]))

print("\n[1] Training MultiScaleEye...")
t0=time.time()
for g,v,ps,st in eyes:
    ex=v.extractor; nids=sorted(g.nodes.keys())
    idxs=np.random.choice(min(200,len(Xt)),min(200,len(Xt)),replace=False)
    ap=[p for i in idxs for p in ex.extract(Xt[i])]
    for k,nid in enumerate(nids): p=ap[k%len(ap)]; g.nodes[nid].template=p.astype(np.float32); g.nodes[nid].template/=np.linalg.norm(g.nodes[nid].template)+1e-8

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
                g.nodes[nid].template+=lr*(t-g.nodes[nid].template); g.nodes[nid].template/=np.linalg.norm(g.nodes[nid].template)+1e-8
print(f"  done ({time.time()-t0:.1f}s)")

# ═══ Build KG: baseline vs class-aware ═══
from synapse import SynapticLayer
from vision_colony import VisionColony

def build_kg_baseline(eyes, Xt, yt, n=2000):
    """基线: 共激活 count → 边权重"""
    for g,v,ps,st in eyes:
        g.synapse = SynapticLayer()
        coactive = {}
        for i in range(n):
            v.set_image(Xt[i].copy())
            active = [nid for nid,nd in g.nodes.items() if nd.activation>0.05]
            for j,src in enumerate(active):
                for dst in active[j+1:]:
                    coactive[(src,dst)] = coactive.get((src,dst),0)+1
        for (src,dst),cnt in coactive.items():
            if cnt>=3:
                g.adjacency[src][dst]=min(1.0, cnt/20.0)
                g.adjacency[dst][src]=min(1.0, cnt/20.0)

def build_kg_class_aware(eyes, Xt, yt, n=2000):
    """类别感知: 共激活 count × 类别纯度"""
    purities = []
    for g,v,ps,st in eyes:
        g.synapse = SynapticLayer()
        coactive = {}  # {(src,dst): [total, {digit: count}]}
        for i in range(n):
            v.set_image(Xt[i].copy())
            active = [nid for nid,nd in g.nodes.items() if nd.activation>0.05]
            digit = int(yt[i])
            for j,src in enumerate(active):
                for dst in active[j+1:]:
                    key = (src,dst)
                    if key not in coactive:
                        coactive[key] = [0, {}]
                    coactive[key][0] += 1
                    coactive[key][1][digit] = coactive[key][1].get(digit, 0) + 1
        
        for (src,dst), (cnt, cls_dist) in coactive.items():
            if cnt >= 3:
                purity = max(cls_dist.values()) / cnt
                purities.append(purity)
                weight = min(1.0, cnt/20.0) * purity
                g.adjacency[src][dst] = max(0.02, weight)
                g.adjacency[dst][src] = max(0.02, weight)
    
    avg_purity = np.mean(purities) if purities else 0
    return avg_purity

# ═══ Extract features ═══
def extract_features(eyes, colonies, images, node_ids_cache, tier_weights_cache, alpha=0):
    feats = []
    for img in images:
        parts = []
        for (g,v,ps,st), col, nids, tw in zip(eyes, colonies, node_ids_cache, tier_weights_cache):
            v.set_image(img)
            act = np.array([g.nodes[nid].activation for nid in nids], dtype=np.float32)
            if alpha > 0:
                error = v.predictive_boost(col, strength=0.3)
                act = np.clip(act + error*alpha, 0, 1)
            parts.append(np.concatenate([act, tw]))
        feats.append(np.concatenate(parts))
    return np.array(feats, dtype=np.float32)

# ═══ Run both ═══
for label, build_fn in [("baseline", build_kg_baseline), ("class-aware", build_kg_class_aware)]:
    # Fresh eyes copy (templates already trained, just reset adj/synapse)
    for g,v,ps,st in eyes:
        g.adjacency.clear()
    
    print(f"\n[2a] Building KG ({label})...")
    t0=time.time()
    if label == "class-aware":
        avg_purity = build_fn(eyes, Xt, yt)
        print(f"      avg purity: {avg_purity:.3f}")
    else:
        build_fn(eyes, Xt, yt)
    kg_edges = sum(len(v) for v in [g.adjacency for g,v,ps,st in eyes])
    print(f"      KG edges: {kg_edges} ({time.time()-t0:.1f}s)")
    
    print(f"[2b] Colony ({label})...")
    t0=time.time()
    colonies = []
    for g,v,ps,st in eyes:
        colony = VisionColony(g); colony.seed_cells(n_per_node=2, max_cells=200)
        colony.breathe(n_generations=80, verbose=False)
        colonies.append(colony)
    print(f"      done ({time.time()-t0:.1f}s)")
    
    # tier 权重
    nids_list = [sorted(g.nodes.keys()) for g,v,ps,st in eyes]
    tier_weights = []
    for (g,v,ps,st), col, nids in zip(eyes, colonies, nids_list):
        tw = np.zeros(len(nids), dtype=np.float32)
        for j, nid in enumerate(nids):
            c2=c3=0
            for (s,d),tier in col.synapse.tiers.items():
                if s==nid or d==nid:
                    if tier==2: c2+=1
                    elif tier==3: c3+=1
            tw[j] = c3*0.3 + c2*0.1
        tier_weights.append(tw)
    
    # Test
    n_train=2000; n_test=500
    Xtr = extract_features(eyes, colonies, [Xt[i] for i in range(n_train)], nids_list, tier_weights, alpha=0)
    knn = SimpleKNN(k=5); knn.fit(Xtr, yt[:n_train])
    
    print(f"      {label}:", end="")
    for c in [1.0, 0.5, 0.3, 0.2]:
        Xc = Xv[:n_test] if c==1.0 else low_contrast(Xv[:n_test], c)
        Xte = extract_features(eyes, colonies, Xc, nids_list, tier_weights, alpha=0)
        acc = knn.score(Xte, yv[:n_test])
        print(f"  {acc*100:5.1f}%", end="")
    print()

print("\n=== DONE ===")
