#!/usr/bin/env python3
import sys,os,gzip,numpy as np,time
np.random.seed(42)
base='.'
with gzip.open(os.path.join(base,'train-images-idx3-ubyte.gz'),'rb') as f: Xt=np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255.0
with gzip.open(os.path.join(base,'train-labels-idx1-ubyte.gz'),'rb') as f: yt=np.frombuffer(f.read(),np.uint8,offset=8)
with gzip.open(os.path.join(base,'t10k-images-idx3-ubyte.gz'),'rb') as f: Xv=np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255.0
with gzip.open(os.path.join(base,'t10k-labels-idx1-ubyte.gz'),'rb') as f: yv=np.frombuffer(f.read(),np.uint8,offset=8)
def lc(X,c): Xc=X.copy(); m=Xc.mean(axis=(1,2),keepdims=True); return m+(Xc-m)*c
from graph import VisionGraph; from vision import VisionInterface
from synapse import SynapticLayer; from vision_colony import VisionColony
cfgs=[{'ps':4,'st':4,'n':100},{'ps':8,'st':4,'n':100},{'ps':16,'st':8,'n':50}]
eyes=[]
for cfg in cfgs:
    ts=cfg['ps']**2; g=VisionGraph(n_nodes=cfg['n'],template_size=ts)
    v=VisionInterface(g,patch_size=cfg['ps'],stride=cfg['st']); eyes.append((g,v))
for g,v in eyes:
    ex=v.extractor; nids=sorted(g.nodes.keys())
    idxs=np.random.choice(min(200,len(Xt)),min(200,len(Xt)),replace=False)
    ap=[p for i in idxs for p in ex.extract(Xt[i])]
    for k,nid in enumerate(nids): p=ap[k%len(ap)]; g.nodes[nid].template=p.astype(np.float32); g.nodes[nid].template/=np.linalg.norm(g.nodes[nid].template)+1e-8
lr=0.1
for ep in range(3):
    for idx in np.random.permutation(10000):
        img=Xt[idx].copy()
        if np.random.random()<0.5: m=img.mean(); c=0.3+np.random.random()*0.7; img=m+(img-m)*c
        for g,v in eyes:
            v.set_image(img)
            for nid,aps in v.node_assignments.items():
                t=np.mean(aps,axis=0); n=np.linalg.norm(t)
                if n>0: t/=n
                g.nodes[nid].template+=lr*(t-g.nodes[nid].template); g.nodes[nid].template/=np.linalg.norm(g.nodes[nid].template)+1e-8

for g,v in eyes:
    g.synapse=SynapticLayer()
    coactive={}
    for i in range(2000):
        v.set_image(Xt[i].copy()); active=[nid for nid,n in g.nodes.items() if n.activation>0.05]
        for j,src in enumerate(active):
            for dst in active[j+1:]: coactive[(src,dst)]=coactive.get((src,dst),0)+1
    for (src,dst),cnt in coactive.items():
        if cnt>=3: g.adjacency[src][dst]=min(1.0,cnt/20.0); g.adjacency[dst][src]=min(1.0,cnt/20.0)
    g._colony=VisionColony(g); g._colony.seed_cells(n_per_node=2,max_cells=200)
    g._colony.breathe(n_generations=80,verbose=False)

# tier weights
tws=[]
for g,v in eyes:
    nids=sorted(g.nodes.keys()); tw=np.zeros(len(nids),dtype=np.float32)
    for j,nid in enumerate(nids):
        c2=0; c3=0
        for (s,d),tier in g._colony.synapse.tiers.items():
            if s==nid or d==nid:
                if tier==2: c2+=1
                elif tier==3: c3+=1
        tw[j]=c3*0.3+c2*0.1
    tws.append((nids,tw))

def feat(images,alpha=0):
    F=[]
    for img in images:
        parts=[]
        for (g,v,*_), (nids,tw) in zip(eyes,tws):
            v.set_image(img)
            act=np.array([g.nodes[nid].activation for nid in nids],dtype=np.float32)
            if alpha>0:
                error=v.predictive_boost(g._colony,strength=0.3)
                act=np.clip(act+error*alpha,0,1)
            parts.append(np.concatenate([act,tw]))
        F.append(np.concatenate(parts))
    return np.array(F,dtype=np.float32)

# train KNN
n_tr=2000; Xtr=feat([Xt[i] for i in range(n_tr)])
class KNN:
    def __init__(s,k=5): s.k=k
    def fit(s,X,y): s.X,s.y=X,y
    def score(s,X,y):
        c=0
        for i in range(len(X)):
            d=np.sum((s.X-X[i])**2,axis=1); nn=np.argpartition(d,s.k)[:s.k]
            if np.bincount(s.y[nn].astype(int)).argmax()==y[i]: c+=1
        return c/len(X)
knn=KNN(); knn.fit(Xtr,yt[:n_tr])

n_t=300
print(f"{'c':>6s}  {'α=0':>7s}  {'α=0.8':>7s}  {'Δ':>7s}")
print("-"*33)
for c in [1.0,0.5,0.3,0.2,0.15,0.1,0.07]:
    Xc=Xv[:n_t] if c==1.0 else lc(Xv[:n_t],c)
    Xte=feat(Xc,alpha=0)
    acc0=knn.score(Xte,yv[:n_t])
    Xte_fb=feat(Xc,alpha=0.8)
    acc_fb=knn.score(Xte_fb,yv[:n_t])
    print(f"{c:5.2f}  {acc0*100:6.1f}%  {acc_fb*100:6.1f}%  {acc_fb-acc0:+6.1%}")
print("\n  (CNN reference at c=0.2: ~45%, ColdEye c=0.2: ~53%)")
