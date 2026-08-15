#!/usr/bin/env python3
"""诊断: RGB per-channel centering 能否保持对比度不变性 + 提升 CIFAR-10 准确率.
关键数学: per-channel 减均值+L2 → c 约掉 (每个通道独立, 不混整体范数)"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye, load_cifar10, load_mnist, low_contrast
from train_multiscale import GlobalEye, PatchEye

np.random.seed(42)

# 1. 数学验证: per-channel centering 让 c 约掉
print("=== 数学验证: per-channel centering 对比度不变性 ===")
for c in [1.0, 0.5, 0.1, 0.01]:
    rgb = np.random.rand(32, 32, 3).astype(np.float32)
    m = rgb.mean(axis=(0,1), keepdims=True)  # per-channel mean
    low = m + (rgb - m) * c  # per-channel low contrast
    # per-channel centering + L2
    flat = low.reshape(-1, 3)
    flat = flat - flat.mean(axis=0, keepdims=True)
    flat /= np.linalg.norm(flat, axis=0, keepdims=True) + 1e-8
    if c == 1.0:
        ref = flat.copy()
    else:
        cos = np.dot(ref.reshape(-1), flat.reshape(-1)) / (np.linalg.norm(ref)*np.linalg.norm(flat))
        print(f"  c={c:.2f}: 与 c=1.0 的余弦 = {cos:.6f}  (应≈1.0)")

# 2. 实验: RGB GlobalEye vs 灰度 GlobalEye (CIFAR-10 小样本)
print("\n=== 实验: RGB per-channel centering 的 GlobalEye ===")
X_tr, y_tr, X_te, y_te = load_cifar10(n_train_per_class=1000, n_test_per_class=300)
# 灰度版 (当前)
X_tr_g = X_tr  # 已经是灰度了 (load_cifar10 返回灰度)
# RGB 版: 重新加载 RGB
# (这里先手动构造一个 RGB 版 GlobalEye 验证, 完整 load_cifar10_rgb 稍后固化)

# 用 PIL 直接加载 RGB 验证
from PIL import Image
import glob
def load_rgb_subset(n_per_class=200):
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    X, y = [], []
    for ci, cname in enumerate(classes):
        d = os.path.join('data/cifar10/train', cname)
        for f in sorted(os.listdir(d))[:n_per_class]:
            img = np.array(Image.open(os.path.join(d, f)).convert('RGB'), dtype=np.float32)/255.0
            X.append(img); y.append(ci)
    return np.array(X, np.float32), np.array(y, np.int64)

X_rgb, y_rgb = load_rgb_subset(300)
X_rgb_te, y_rgb_te = load_rgb_subset(100)  # 简化: 用训练子集当测试

# RGB GlobalEye (per-channel centering)
class RGBGlobalEye(GlobalEye):
    def _center(self, img):
        h, w, c = img.shape
        flat = img.reshape(-1, c).astype(np.float32)
        flat = flat - flat.mean(axis=0, keepdims=True)
        flat /= np.linalg.norm(flat, axis=0, keepdims=True) + 1e-8
        return flat.reshape(-1)
    def init_templates(self, images):
        idxs = self.rng.choice(len(images), min(200, len(images)), replace=False)
        patches = np.array([self._center(images[i]) for i in idxs], np.float32)
        self.templates = np.empty((self.n, patches.shape[1]), np.float32)
        for i in range(self.n):
            self.templates[i] = patches[i % len(patches)]
            self.templates[i] /= np.linalg.norm(self.templates[i]) + 1e-8
    def train(self, images, epochs=3, n_train=None, **kw):
        if n_train is None: n_train = len(images)
        lr = 0.1
        for ep in range(epochs):
            for idx in self.rng.permutation(min(n_train, len(images))):
                flat = self._center(images[idx])
                scores = self.templates @ flat
                best = int(np.argmax(scores))
                self.templates[best] += lr*(flat - self.templates[best])
                self.templates[best] /= np.linalg.norm(self.templates[best]) + 1e-8
    def activate(self, images):
        return np.array([self.activate_one(img) for img in images], np.float32)
    def activate_one(self, image):
        flat = self._center(image)
        return np.clip(self.templates @ flat, 0, 1).astype(np.float32)

print(f"RGB 子集: {X_rgb.shape} (300张), 训练 RGB GlobalEye...")
ge = RGBGlobalEye(n_nodes=100, seed=42)
ge.init_templates(X_rgb)
ge.train(X_rgb, epochs=5, n_train=300)
# KNN 用模板激活分类
def knn_classify(X, y, X_te):
    # 用 GlobalEye 激活做 KNN
    acts_tr = ge.activate(X)
    acts_te = ge.activate(X_te)
    correct = 0
    for i, a in enumerate(acts_te):
        sims = acts_tr @ a / (np.linalg.norm(acts_tr, axis=1)*np.linalg.norm(a)+1e-8)
        if y[np.argmax(sims)] == y_rgb_te[i]: correct += 1
    return correct/len(X_te)

acc = knn_classify(X_rgb, y_rgb, X_rgb_te)
print(f"RGB GlobalEye (100d, per-channel centering) KNN: {acc:.1%}")

# 灰度 GlobalEye 对比
X_rgb_g = np.array([0.299*i[...,0]+0.587*i[...,1]+0.114*i[...,2] for i in X_rgb], np.float32)
X_rgb_te_g = np.array([0.299*i[...,0]+0.587*i[...,1]+0.114*i[...,2] for i in X_rgb_te], np.float32)
ge_g = GlobalEye(n_nodes=100, seed=42)
ge_g.init_templates(X_rgb_g)
ge_g.train(X_rgb_g, epochs=5, n_train=300)
acc_g = knn_classify(X_rgb_g, y_rgb, X_rgb_te_g)
print(f"灰度 GlobalEye (100d) KNN: {acc_g:.1%}")
print(f"\nRGB vs 灰度 提升: {acc-acc_g:+.1%}")
