#!/usr/bin/env python3
"""CIFAR-10 灰度 per-class 诊断 — 22.2% 卡在哪"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye, load_cifar10

CIFAR_CLASSES = ['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck']

np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_cifar10(gray=True, n_train_per_class=1000, n_test_per_class=200)
print(f"加载: train={X_tr.shape}, test={X_te.shape}")

model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},
    {"type": "patch", "ps": 16, "st": 8, "n": 100},
])
model.init_templates(X_tr[:3000])
model.train(X_tr, y_tr, epochs=5, n_train=len(X_tr), contrast_aug=True)
model.build_memory(X_tr, y_tr, size=5000)

# per-class 准确率 + 混淆
n_te = len(X_te)
preds = np.array([model.predict(X_te[i])[0] for i in range(n_te)])
acc = (preds == y_te).mean()
print(f"\n整体: {acc:.1%}  (随机 10%)\n")

print(f"  {'class':>10s}  {'acc':>6s}   {'最常误判为':>12s}")
print(f"  {'-'*40}")
confusion = np.zeros((10, 10), dtype=int)
for i in range(n_te):
    confusion[y_te[i], preds[i]] += 1

for c in range(10):
    cls_acc = (preds[y_te == c] == c).mean() if (y_te == c).sum() > 0 else 0
    # 最常误判为 (排除自己)
    row = confusion[c].copy(); row[c] = 0
    top_wrong = CIFAR_CLASSES[np.argmax(row)] if row.max() > 0 else '-'
    print(f"  {CIFAR_CLASSES[c]:>10s}  {cls_acc:5.1%}   {top_wrong:>12s}")

# 动物 vs 交通分组
animal = ['bird','cat','deer','dog','frog','horse']
vehicle = ['airplane','auto','ship','truck']
animal_idx = [CIFAR_CLASSES.index(a) for a in animal]
vehicle_idx = [CIFAR_CLASSES.index(v) for v in vehicle]

animal_mask = np.isin(y_te, animal_idx)
animal_acc = (preds[animal_mask] == y_te[animal_mask]).mean()
vehicle_mask = np.isin(y_te, vehicle_idx)
vehicle_acc = (preds[vehicle_mask] == y_te[vehicle_mask]).mean()
print(f"\n  动物类 (bird/cat/deer/dog/frog/horse): {animal_acc:.1%}")
print(f"  交通类 (airplane/auto/ship/truck):   {vehicle_acc:.1%}")

print("\n=== DONE ===")
