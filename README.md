# ColdEye v3 — 冷眼预测反馈视觉系统

竞争路由 + per-patch centering = **完美对比度不变性** (c=0.01 == c=1.0)

> 📐 算法与模型的完整技术参考见 [ALGORITHMS.md](ALGORITHMS.md)。

## 架构

```
Image → GlobalEye(28×28, 100n) + PatchEye(16×16, 50n)
      → Competitive routing (hard assignment, cosine similarity)
      → 150-dim activation vector
      → Direct KNN (k=5)
```

**核心创新: per-patch centering** — 匹配前 `patch -= patch.mean()` 然后 L2 归一化。
对比度 c 从数学上完全约掉。不是"鲁棒"，是"无关"。

## 能力矩阵

| 能力 | 结果 |
|------|------|
| 对比度不变性 | MNIST 84.4% flat (c=1.0 → c=0.01) |
| 跨域泛化 | Fashion-MNIST 84.0% flat, 零代码改动 |
| 小样本学习 | 1-shot 30%, 10-shot 60%, 零对比度退化 |
| 闭环脑补 | 遮挡 14×14: 48.5% → 59.5% (+11%) |
| 良心机制 | β=0.5: +2.4% via 节点生态位分化 |
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
```

## 自定义架构

```python
model = ColdEye(eye_specs=[
    {"type": "global", "n": 200},           # 全局眼 (整图路由)
    {"type": "patch", "ps": 16, "st": 8, "n": 100},  # 粗尺度
    {"type": "patch", "ps": 8,  "st": 4, "n": 50},   # 中尺度 (可选)
])
```

## 持续学习 (纵向)

```python
# 增量记忆 — 不重训模板, 不碰旧记忆
model.add_samples(new_images, new_labels)

# 记忆淘汰 — 每类保留最新 N 条
model.prune_memory(max_per_class=200)

# 适应新数据 — 低学习率微调模板 (分布漂移)
model.adapt(new_images, new_labels, epochs=1)

# 睡眠巩固 — 用近期记忆微调模板
model.consolidate(n_recent=500, lr=0.01)
```

## 自优化 (横向)

```python
# 动态扩容 — 不确定样本初始化新节点
model.spawn_nodes(uncertain_images, n_new=10)

# 回收死节点 — 低胜率模板重初始化
model.recycle_nodes(uncertain_images, max_recycle=20)

# 模板健康度 — 胜率/空转率监控
health = model.template_health()  # {active, stale, dead_nodes}

# 不确定采样 — 主动学习 (置信度 < threshold)
uncertain = model.uncertain_samples(images, labels, threshold=0.8)

# 元参数自调 — 扫 conscience_beta
best_beta, acc = model.auto_tune(eval_imgs, eval_labels)

# 共激活图 — 常共激活节点连边
model.build_graph(images, edge_thresh=0.5)

# 图增强预测 — 激活沿边传播一步
model.predict_with_graph(image, alpha=0.3)

# 层级路由 — 粗尺度筛候选 → 全局精细匹配
model.route_hierarchical(image, top_k_coarse=5)
```

## 脑补 (降质 → 干净重建)

```python
# 配对脑补解码器: act(降质图) → clean_image
model.build_paired_decoder(clean_images, degrade_fn=occlude_fn)

# 闭环推理: 降质 → 激活 → 重建 → 重路由 → 融合 → KNN (迭代到收敛)
model.predict_brainfill(occluded_image, alpha=0.5)

# 重建
recon = model.reconstruct_paired(image)
```

**脑补关键结论:**
- 退化越重，脑补越有用 — 40% 擦除 +9.2%, 轻退化 -71.9% (信息没丢没活干)
- 线性解码器适合"独特组合"数据 (合成场景), 记忆式适合"重复模式"数据 (MNIST 同类)
- 线性解码是脑补的诚实天花板, 非线性重建需要背离 Hebbian

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
  generate_synthetic.py           — 合成复杂场景生成器
  benchmark_synthetic_brainfill.py   — 合成数据脑补
  benchmark_memory_brainfill.py      — 记忆式 vs 线性脑补
  benchmark_severe_brainfill.py      — 重度退化脑补

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

## 核心原理

### per-patch centering

```
图像:    img = mean + (pattern - mean) × c
centering: img - mean(img) = (pattern - mean) × c
L2归一化: (pattern - mean) / |pattern - mean|     ← c 约掉
```

c=1.0 和 c=0.01 在数学上是同一个方向向量。余弦相似度完全取决于形状方向，与对比度幅度无关。

### 竞争路由 + Hebbian

每个 patch 通过余弦相似度找最匹配的模板 (hard assignment)，
赢者模板向输入靠近一步 (Hebbian)。纯局部学习，无梯度传播。

### 脑补闭环

```
降质图 → 激活 → 重建干净图 → 重路由 → 融合 → 分类
        ↑___________________________↓
            (迭代直到收敛或置信度高)
```

## 路线图

```
✓ MNIST (84.4% flat)
✓ Fashion-MNIST (84% flat, 跨域泛化)
✓ 少样本 (1-shot 30%, 零退化)
✓ 合成复杂场景 (脑补重度退化 +9.2%)
✓ 持续学习 (增量/剪枝/适应/巩固/回收/扩容)
→ CIFAR-10 / 自然图像 (图路由/自组织在此发挥)
→ 非线性脑补 (不靠 BP 的非线性重建)
```
