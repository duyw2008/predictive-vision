# ColdEye v3 — 冷眼预测反馈视觉系统

竞争路由 + per-patch centering = **完美对比度不变性** (c=0.01 == c=1.0)

## 架构

```
Image → GlobalEye(28×28, 100n) + PatchEye(16×16, 50n)
      → Competitive routing (hard assignment, cosine similarity)
      → 150-dim activation vector
      → Direct KNN (k=5)
```

**核心创新: per-patch centering** — 匹配前 `patch -= patch.mean()` 然后 L2 归一化。
对比度 c 从数学上完全约掉。不是"鲁棒"，是"无关"。

## 能力

| 能力 | 结果 |
|------|------|
| 对比度不变性 | MNIST 84.4% flat (c=1.0 → c=0.01) |
| 跨域泛化 | Fashion-MNIST 84.0% flat, 零代码改动 |
| 小样本学习 | 1-shot 30%, 10-shot 60%, 零对比度退化 |
| 闭环推理 | 遮挡 14×14: 48.5% → 59.5% (+11%) |
| 良心机制 | β=0.5: +2.4% via 节点生态位分化 |
| 脑补重建 | paired decoder: 降质激活→干净图, 对比度不变 |
| 全局部学习 | Hebbian, 无反向传播, 1000样本/1秒 CPU |

## 快速开始

```python
from train_multiscale import ColdEye, load_mnist, low_contrast

# 训练
X_tr, y_tr, X_te, y_te = load_mnist()
model = ColdEye()                    # 默认 150d
model.init_templates(X_tr[:5000])
model.train(X_tr, y_tr, epochs=5, n_train=60000)
model.build_memory(X_tr, y_tr, size=5000)

# 测试 (所有对比度一致)
for c in [1.0, 0.5, 0.1, 0.05, 0.01]:
    test = X_te[:500] if c == 1.0 else low_contrast(X_te[:500], c)
    print(f"c={c:.2f}: {model.evaluate(test, y_te[:500]):.1%}")

# 良心机制
model = ColdEye()
model.train(X_tr, y_tr, conscience_beta=0.5)

# 配对脑补 + 闭环推理
model.build_paired_decoder(X_tr, degrade_fn=occlude_fn)
model.predict_brainfill(occluded_image, alpha=0.5)  # 迭代到收敛
```

## 自定义架构

```python
model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},           # 全局眼
    {"type": "patch", "ps": 16, "st": 8, "n": 100},  # 粗尺度
    {"type": "patch", "ps": 8,  "st": 4, "n": 50},   # 中尺度 (可选)
])
```

## 对比 CNN

| 指标 | ColdEye v3 | CNN (2-conv) |
|------|-----------|-------------|
| c=1.0 | 84.4% | ~98% |
| c=0.1 | 84.4% | ~25% |
| c=0.01 | 84.4% | ~10% |
| 衰减 | **0%** | >90% |
| 训练 | Hebbian, CPU, 1s/1K | BP, GPU, 分钟级 |
| 少样本 | 1-shot 30% | 需要全量重训 |

## 目录结构

```
src/            — 核心源码
  train_multiscale.py  — 主文件: ColdEye, GlobalEye, PatchEye
  graph.py             — VisionGraph + VisionNode
  vision.py            — VisionInterface (竞争路由)
  synapse.py           — SynapticLayer (Colony用, v3不用)
  vision_colony.py     — VisionColony (MNIST无效, 保留)

benchmarks/     — 活跃 benchmark
  benchmark_fashion.py            — Fashion-MNIST 跨域验证
  benchmark_conscience.py         — 良心机制 sweep
  benchmark_brainfill_paired.py   — 配对脑补 + 闭环推理
  benchmark_fewshot_capacity.py   — 少样本 + 容量 scaling
  benchmark_occlusion.py          — 遮挡 + FB 测试
  benchmark_global_v2.py          — v1→v2→v3 发现过程
  benchmark_v3_full.py            — v3 全配验证
  benchmark_20class.py            — 20类混合验证
  benchmark_two_step.py           — 两步推理验证
  benchmark_retina_multiscale.py  — retina vs multi-scale

diagnostics/    — 诊断工具
  diag_centering.py     — **突破验证**: centering vs raw
  diag_scale_profile.py — 三尺度独立 profiling

demos/          — 演示
  demo_feedback_brainfill.py — FB + 脑补 demo

deprecated/     — 废弃脚本 (Colony, retina, brainfill v1, 旧 FB)
```

## 架构演化

| 版本 | 架构 | 维度 | c=0.1 | 突破 |
|------|------|------|-------|------|
| v1 | fine+mid+coarse | 250d | 10.6% | — |
| v2 | global+coarse | 150d | 24.2% | fine 4×4是垃圾 |
| v3 | v2 + centering | 150d | **84.4%** | c≤0.1崩塌是归一化bug |

## 原理: per-patch centering

```
图像:    img = mean + (pattern - mean) × c
centering: img - mean(img) = (pattern - mean) × c
L2归一化: (pattern - mean) / |pattern - mean|     ← c 约掉
```

c=1.0 和 c=0.01 在数学上是同一个方向向量。余弦相似度完全取决于形状方向，与对比度幅度无关。
