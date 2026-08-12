#!/usr/bin/env python3
"""输出一张 c=0.1 的 MNIST 图像对比"""
import numpy as np, gzip, os
from PIL import Image

def load_mnist(d="data"):
    with gzip.open(os.path.join(d,'t10k-images-idx3-ubyte.gz'),'rb') as f:
        X = np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255
    with gzip.open(os.path.join(d,'t10k-labels-idx1-ubyte.gz'),'rb') as f:
        y = np.frombuffer(f.read(),np.uint8,offset=8)
    return X, y

def low_contrast(img, c):
    m = img.mean()
    return m + (img - m) * c

X, y = load_mnist()
idx = 0  # first test image
orig = X[idx]
lc = low_contrast(orig.copy(), 0.1)

# side by side: original | c=0.1
composite = np.hstack([orig, lc])
img = Image.fromarray((composite * 255).astype(np.uint8), 'L')
img = img.resize((28*8, 14*8), Image.NEAREST)  # 4x scale
img.save("data/c01_sample.png")
print(f"Saved data/c01_sample.png — label={y[idx]}, mean={orig.mean():.4f}, std(orig)={orig.std():.4f}, std(c01)={lc.std():.4f}")
