#!/usr/bin/env python3
"""
CIFAR-10 灰度多尺度实验 — 纹理 (小 patch) 能否救动物类
A: global(200) + patch16(100)         [baseline 23.1%]
B: global(200) + patch4/8/16 多尺度   [加纹理]
C: B + 全量数据 (5000/类)
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye, load_cifar10

CIFAR_CLASSES = ['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck']
animal_idx = [2,3,4,5,6,7]  # bird,cat,deer,dog,frog,horse

np.random.seed(42)

def eval_model(eye_specs, n_per_class, label):
    X_tr, y_tr, X_te, y_te = load_cifar10(gray=True, n_train_per_class=n_per_class, n_test_per_class=200)
    model = ColdEye(eye_specs=eye_specs)
    model.init_templates(X_tr[:3000])
    model.train(X_tr, y_tr, epochs=5, n_train=len(X_tr), contrast_aug=True)
    model.build_memory(X_tr, y_tr, size=5000)

    n_te = len(X_te)
    preds = np.array([model.predict(X_te[i])[0] for i in range(n_te)])
    acc = (preds == y_te).mean()
    animal_mask = np.isin(y_te, animal_idx)
    animal_acc = (preds[animal_mask] == y_te[animal_mask]).mean()
    print(f"  {label}: 整体 {acc:.1%}  动物 {animal_acc:.1%}  ({model.dim}d)")
    return acc, animal_acc

print("CIFAR-10 灰度多尺度实验\n")

# A: baseline
eval_model([
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
], 1000, "A: global+patch16 (baseline)")

# B: 多尺度纹理
eval_model([
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 4,  "st": 4, "n": 100},   # 纹理 (4×4)
    {"type": "patch", "ps": 8,  "st": 4, "n": 100},   # 中纹理 (8×8)
    {"type": "patch", "ps": 16, "st": 8, "n": 100},   # 形状 (16×16)
], 1000, "B: 多尺度 (patch4/8/16)")

# C: 多尺度 + 全量数据
eval_model([
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 4,  "st": 4, "n": 100},
    {"type": "patch", "ps": 8,  "st": 4, "n": 100},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
], 5000, "C: 多尺度 + 50K 数据")

print("\n=== DONE ===")
