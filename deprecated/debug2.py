#!/usr/bin/env python3
"""Quick debug: check what reconstruct actually returns"""
import sys, os, numpy as np, gzip, urllib.request
from collections import defaultdict
sys.path.insert(0, '/home/duyw/predictive-vision')
from graph import VisionGraph
from vision import VisionInterface

class HintEye:
    def __init__(self, n=500):
        self.graph = VisionGraph(n_nodes=n)
        self.vision = VisionInterface(self.graph)
        self.node_votes = defaultdict(lambda: defaultdict(int))
    def init_templates(self, images):
        ex = self.vision.extractor
        nids = sorted(self.graph.nodes.keys())
        idxs = np.random.choice(len(images), min(50, len(images)), replace=False)
        all_p = []
        for i in idxs: all_p.extend(ex.extract(images[i]))
        if not all_p: return
        for i, nid in enumerate(nids):
            p = all_p[i % len(all_p)]
            self.graph.nodes[nid].template = p.astype(np.float32) + np.random.randn(len(p)).astype(np.float32)*0.01
            self.graph.nodes[nid].template /= np.linalg.norm(self.graph.nodes[nid].template)+1e-8
    def train(self, imgs, lbls, epochs=3):
        lr=0.1
        for ep in range(epochs):
            perm = np.random.permutation(len(imgs))
            for idx in perm:
                img,label = imgs[idx],lbls[idx]
                self.vision.set_image(img)
                for nid, patches in self.vision.node_assignments.items():
                    node = self.graph.nodes[nid]
                    target = np.mean(patches, axis=0)
                    norm=np.linalg.norm(target)
                    if norm>0: target=target/norm
                    node.template += lr*(target-node.template)
                    node.template /= np.linalg.norm(node.template)+1e-8
                    self.node_votes[nid][label] += 1
        for nid, votes in self.node_votes.items():
            if votes: self.graph.nodes[nid].domain_tag = str(max(votes, key=votes.get))

def load_mnist(d="data"):
    fs = {"tr_i":"train-images-idx3-ubyte.gz","tr_l":"train-labels-idx1-ubyte.gz","te_i":"t10k-images-idx3-ubyte.gz","te_l":"t10k-labels-idx1-ubyte.gz"}
    os.makedirs(d,exist_ok=True)
    url="https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
    for f in fs.values():
        p=os.path.join(d,f)
        if not os.path.exists(p): urllib.request.urlretrieve(url+f,p)
    def li(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255
    def ll(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(),np.uint8,offset=8).astype(np.int64)
    return li(os.path.join(d,fs["tr_i"])),ll(os.path.join(d,fs["tr_l"])),li(os.path.join(d,fs["te_i"])),ll(os.path.join(d,fs["te_l"]))

X_train,y_train,X_test,y_test = load_mnist()
model = HintEye(500)
model.init_templates(X_train[:1000])
model.train(X_train[:10000], y_train[:10000], 3)

sample = X_test[0]
true_label = int(y_test[0])
print(f"True label: {true_label}")

mean = sample.mean()
img_low = mean + (sample-mean)*0.12

# Without hint - baseline activation for reconstruction
model.vision.set_image(img_low)
recon_base = model.vision.reconstruct()
print(f"Base recon: min={recon_base.min():.4f} max={recon_base.max():.4f} nonzero={(recon_base>0).sum()}")

# With hint
model.vision.set_image_with_hint(img_low, true_label, contrast=1.0, boost=10.0)
hinted = [nid for nid,n in model.graph.nodes.items() if n.domain_tag and int(n.domain_tag)==true_label]
acts = {nid: model.graph.nodes[nid].activation for nid in hinted if model.graph.nodes[nid].activation>0}
print(f"Hinted with activation>0: {len(acts)}, values={sorted(set(acts.values()))[:10]}")

# All nodes with activation>0
all_acts = {nid: n.activation for nid,n in model.graph.nodes.items() if n.activation>0}
print(f"All nodes with act>0: {len(all_acts)}")

recon = model.vision.reconstruct(hint_label=true_label, boost=10.0)
print(f"Hint recon: min={recon.min():.4f} max={recon.max():.4f} nonzero={(recon>0).sum()}")
print(f"recon unique values: {sorted(set(recon.flatten().round(3)))[:20]}")

# Check _last_patches
print(f"_last_patches shape: {model.vision._last_patches.shape}")
print(f"_last_patches norm sample: {[np.linalg.norm(model.vision._last_patches[i]) for i in range(min(5, len(model.vision._last_patches)))]}")

# Check templates match with patches
templates = np.array([model.graph.nodes[nid].template for nid in sorted(model.graph.nodes.keys())])
patches = model.vision._last_patches
scores = templates @ patches[0]
print(f"Scores for patch 0: min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.4f}")
print(f"Top 5 scores: {sorted(scores, reverse=True)[:5]}")
