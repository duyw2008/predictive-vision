"""快速测 c=0.05 及以下"""
import sys, os, numpy as np, gzip
sys.path.insert(0, os.path.dirname(__file__))
from train_multiscale import ColdEye, load_mnist, low_contrast

np.random.seed(42)
X_tr, y_tr, X_te, y_te = load_mnist()

model = ColdEye()
model.init_templates(X_tr[:5000])
model.train(X_tr, y_tr, epochs=5, n_train=60000, contrast_aug=True)
model.build_memory(X_tr[:15000], y_tr[:15000], size=5000)

n_test = 500
for c in [0.07, 0.05, 0.03, 0.02, 0.01]:
    test_batch = low_contrast(X_te[:n_test], c)
    acc = model.evaluate(test_batch, y_te[:n_test])
    print(f"  c={c:.2f}  {acc*100:.1f}%")
