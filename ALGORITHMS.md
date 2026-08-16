# 冷眼 ColdEye v3 — 算法与模型文档

本文档是 ColdEye 的算法/模型技术参考。架构概览、能力矩阵、快速开始见 [README.md](README.md)。

---

## 1. 模型架构总览

```
输入图像 (H×W, 灰度或 RGB)
   │
   ├── GlobalEye   整图竞争路由 → 全局激活 [n_global]
   ├── PatchEye    分块竞争路由 → 局部激活 [n_patch × 尺度数]
   │
   └── 拼接 → 激活向量 act ∈ R^D
                │
                ├── KNN 分类 (默认)
                ├── 脑补重建 (act → 图像)
                └── 闭环推理 (降质 → 重建 → 重路由 → 分类)
```

三个核心组件：

| 组件 | 职责 | 关键机制 |
|------|------|---------|
| `GlobalEye` | 整图特征提取 | 全局竞争路由 + per-patch centering |
| `PatchEye` | 局部特征提取 | patch 竞争路由 + 空间热点累积 |
| `ColdEye` | 编排 + 推理 | 多 eye 拼接、KNN、脑补、持续学习 |

`eye_specs` 配置示例：

```python
[
    {"type": "global", "n": 200},            # 整图 200 节点
    {"type": "patch", "ps": 16, "st": 8, "n": 100},  # 16×16 patch, stride 8
    {"type": "patch", "ps": 32, "st": 16, "n": 50},  # 32×32 patch, stride 16
]
```

---

## 2. 核心算法

### 2.1 per-patch centering（对比度不变性的数学基础）

**这是冷眼的核心创新。** 匹配前对每个 patch 做均值减法 + L2 归一化。

```
图像:       img = mean + (pattern - mean) × c
centering:  img - mean(img) = (pattern - mean) × c
L2 归一化:  (pattern - mean) × c / |(pattern - mean) × c|
          = (pattern - mean) / |pattern - mean|      ← c 完全约掉
```

**结论**：c=0.01 和 c=1.0 在数学上是同一个方向向量。余弦相似度只取决于形状方向，与对比度幅度无关。这是"衰减 0%"的数学保证，不是经验鲁棒性。

实现（`GlobalEye._center`）：

```python
def _center(self, img):
    if img.ndim == 3:  # RGB: per-channel centering (关键! 整体会破坏不变性)
        flat = img.reshape(-1, img.shape[2])
        flat = flat - flat.mean(axis=0, keepdims=True)
        flat /= np.linalg.norm(flat, axis=0, keepdims=True) + 1e-8
        return flat.reshape(-1)
    flat = img.reshape(-1)  # 灰度: 整体 centering
    flat = flat - flat.mean()
    flat /= np.linalg.norm(flat) + 1e-8
    return flat
```

> ⚠ RGB 必须 per-channel（每通道独立减均值 + L2）。整体 centering（flatten 后减整体均值）会破坏对比度不变性（c=0.1 时余弦降到 0.99）。

### 2.2 竞争路由（competitive routing）

每个 patch 通过余弦相似度找最匹配的模板，硬分配（winner-take-all）。

```
对每个 patch p:
    scores = templates @ p          # 所有节点的余弦相似度
    winner = argmax(scores)         # 硬分配, top_k=1
    node[winner].activation += 1     # 累计
最终激活: act[node] = min(1.0, count / n_patches × 3)
```

**关键参数**：`top_k=1`（硬分配）。top_k>1 会导致激活饱和（所有节点激活值 = 1.0，无区分度）。

**非线性来源**：`argmax` 是硬阈值，这使 `activate()` 是非线性映射——但同时也是**有损瓶颈**（只保留 winner，丢弃其他匹配信息）。

### 2.3 Hebbian 学习

赢者模板向输入靠近一步（纯局部，无反向传播）。

```
winner.template += lr × (input - winner.template)
winner.template /= |winner.template|        # L2 保持单位范数
```

- `lr = 0.1`（训练），`lr = 0.01`（睡眠巩固 consolidate）
- 无需链式求导，CPU 上 1000 样本 1 epoch 约 1 秒

### 2.4 良心机制（conscience，节点生态位分化）

常胜节点被临时抑制，其他节点获得训练机会。

```
freq = zeros(n_nodes)
对每个输入:
    scores = templates @ input - beta × freq   # 惩罚常胜者
    winner = argmax(scores)
    freq[winner] += 1.0
    freq *= 0.999                              # 缓慢衰减
```

- β=0.5 在 MNIST 单域 +2.4%
- β=1.0 过度惩罚（-x%）
- 适用：20+ 类场景需要节点专精时

### 2.5 KNN 分类

激活向量上的最近邻（余弦相似度）。

```
sim(vec, query) = vec · query / (|vec| × |query|)
pred = argmax over memory 的 top-k 多数投票
```

**向量化优化**：预归一化 memory 矩阵 → `queries @ mem.T` 一次 matmul（5000 张 114s → 1.1s）。

---

## 3. 脑补 / 重建算法

### 3.0 脑补的机理（双向联想记忆）

脑补的本质是**双向联想记忆**（类似 Hopfield 网络 / 自编码器）：

```
正向 (编码):  图像 → 竞争路由 → 激活向量 act (300d)
反向 (解码):  act → W → 重建图像
```

**激活向量 = 形状基元直方图。** 竞争路由做 hard assignment，每个 patch 匹配最像的模板节点，激活 = 每个节点赢了多少次。act 记录"哪些形状基元出现了、出现多频繁"（边/曲线/纹理方向），但**不记录位置**——这就是"有损摘要"。

**脑补能补的三步机理：**

1. **基元鲁棒**：centering 让基元对对比度/亮度不变。遮挡掉一块，剩余部分照样激活同样的基元——数字 7 被遮上半截，下半截竖线还是激活"竖线"节点。

2. **联想记忆**：解码器 W 训练时见过海量完整图像，记住了"竖线基元 + 横线基元 = 完整数字 7 的像素"。这是 learned correlation，不是规则。

3. **反向补全**：推理时，残缺图 → 部分基元激活 → 解码器按"这些基元通常对应什么完整图"→ 输出完整图。被遮部分从"基元关联"里猜出来。

```
看残缺 → 识别"这是 7 的竖线" → 联想"完整 7 长这样" → 补出来
```

**为什么模糊（诚实的边界）**：act 是 300d 直方图丢了位置信息，解码器只能重建"和这些基元一致的**平均**图像"——模糊轮廓而非锐利笔画。同一组"竖线+横线"激活可能对应 7 也可能对应 1，解码器输出的是平均。

### 3.1 线性解码器（最简脑补）

```
act → W → 图像
W = pinv(ATA) @ ATI
其中 ATA = acts.T @ acts, ATI = acts.T @ flat_images
```

- **数学上最优的线性重建**（最小二乘），别换
- 这是脑补的**诚实天花板**：只能重建激活向量线性跨度内的内容
- 输出完全对比度不变（因为 act 对比度不变）

### 3.2 配对解码器（降质 → 干净）

```
W_paired: act(降质图) → clean_image
```

训练：`(降质图, 干净图)` 配对 → 最小二乘。用于遮挡/噪声/擦除修复。

### 3.3 闭环推理（predict_brainfill）

```
降质图 → act → 重建干净图 → 重路由 → 融合 → 分类
                ↑________________↓
            (迭代直到收敛或置信度高)
```

- 至少迭代一次（置信度门控只决定"是否继续"，不决定"是否启动"）
- 首次迭代 α 减半（保守试探），后续全 α
- 收敛条件：激活变化 < tol

### 3.4 空间热点（spatial hotspot）

节点累积自己"赢"的 patch 位置，形成热点。重建时按热点位置归网格（而非 patch 物理位置），避免边界断裂。

```
VisionNode.accumulate_spatial(cy, cx):   # 路由时累积
    spatial_cy += cy; spatial_cx += cx; spatial_count += 1

spatial_hotspot = (cy/count, cx/count)   # 平均位置

hotspot_activation(image, n_regions):    # 重建时
    每个 patch 的 winner → 其热点位置 → 归 n_regions×n_regions 网格
    → [n_regions² × n_nodes] 激活
```

**适用范围**：单物体、位置可预测的场景（56×56 单数字 digit3 0.094→0.089）。**复杂多物体场景失效**（热点激活极度稀疏，欠定系统最小范数解摊薄能量）。

---

## 4. 小样本学习

### 4.0 机理（两阶段分离 + 形状基元类无关）

**两阶段架构：**

```
阶段1 (系统发生): 模板学习 — 从基类学形状基元 (边/曲线/环/线)
阶段2 (个体发生): 记忆构建 — 新类几发样本路由 → 存 (激活, 标签)
```

**关键洞察：形状基元是类无关的。** 数字 7 = 斜线 + 横线，数字 9 = 环 + 竖线。这些基元在基类里全都有（环在 0/6/8，竖线在 1/4，斜线在 2/4，曲线在 2/3/5）。所以基类训出的模板，已经掌握新类需要的所有基元——**新数字不是"新特征"，只是"已有基元的新组合"。**

**少样本 = 只加记忆，不重训：**

```
学新类 = 新样本 → 冻结模板路由 → 激活向量 → 存进 memory
```

难的活（特征提取）冻结模板已经干完，剩下的只是"记住这个激活模式 = 数字 7"。激活空间已被基元结构化（形状相似 → 激活相似），KNN 几个参考点就能锚定新类。

**shift augment 的作用（1-shot 30%→45%）：** 1-shot 每类只有 1 条记忆，KNN k>1 会拉到别类。shift（±1px 平移）把 1 发变 5 条，同类有足够邻居支撑多数投票。这是记忆密度问题，不是特征问题。

**零对比度退化：** centering 让激活对比度不变 → 记忆条目在所有对比度下都有效 → 少样本识别的退化 = 0。

**一句话：小样本学习 = 形状基元的"组合复用"，模板负责基元（一次学好），记忆负责组合（几发就够）。**

### 4.1 实验数据

| shots | no aug | +shift | 说明 |
|-------|--------|--------|------|
| 1-shot | 26.8% | 45.4% | random=20% |
| 3-shot | 48.2% | 52.6% | |
| 5-shot | 50.6% | 63.0% | |

零对比度退化（c=1.0/0.5/0.2 完全一致）。

### 4.2 最根本的边界：冷眼 vs 人脑的小样本

**人脑的小样本是"激活"，冷眼的小样本是"重组"，前提不同：**

```
人脑: 进化几千万年 → 拓扑结构预装 (V1 边缘/V4 颜色形状/IT 物体识别)
      小样本 = 几张照片激活已有结构 (结构本来就在)

冷眼: 基类几轮 Hebbian → 基元从基类学出来
      小样本 = 重组学到的基元 (基元是"学"的, 不是"预装"的)
```

**关键区别：基元的来源。** 人脑基元是**基因预装**（新生儿已有 V1 方向柱、LGN 中心-环绕）；冷眼基元是**从基类学**（模板从 0-4 学到环/线/曲线）。

**因此冷眼小样本的上限 = 基类的基元覆盖范围：**

- MNIST 0-4 → 5-9 好使：数字共享笔划（横竖斜环），基类全覆盖
- 一旦新图像需要基类没见过的基元（自然图像的纹理/3D 结构/颜色），小样本立即失效——模板提取不出它没学过的特征

**证据：CIFAR-10。** 简单形状基元迁移不到自然图像，动物类 22%。不是调参能救，是基元覆盖不足。

**结论：**

1. 冷眼小样本 = **域内小样本**（新类需基类基元覆盖），不是人脑的**跨域小样本**
2. 逼近人脑的等价物是"进化"——用海量多样基类（ImageNet 级）预训练出通用基元，小样本才跨域好使
3. 当前基类（MNIST 数字）太窄，基元太窄，小样本只在窄域内有效

**这是冷眼架构最诚实的一条边界：人脑的预装结构 = 进化给容量，冷眼缺这个容量，只能用窄基类当弱代理。**

---

## 5. 持续学习 / 自优化

### 5.1 纵向（模型不腐）

| 方法 | 算法 | 用途 |
|------|------|------|
| `add_samples` | 新样本 → 激活 → 追加记忆 | 增量学习新类 |
| `prune_memory` | 每类保留最新 N 条 | 控制记忆规模 |
| `adapt` | 低 lr Hebbian 微调模板 | 分布漂移 |
| `consolidate` | 近期记忆低 lr Hebbian（睡眠巩固） | 强化近期经验 |
| `uncertain_samples` | 置信度 < threshold 的样本 | 主动采样 |

### 5.2 横向（拉高天花板）

| 方法 | 算法 | 用途 |
|------|------|------|
| `template_health` | 每节点胜率统计 | 暴露死节点 |
| `spawn_nodes` | 不确定样本初始化新节点 | 动态扩容 |
| `recycle_nodes` | 最低胜率模板重初始化 | 回收死节点 |
| `build_graph` | 共激活率 → 稀疏边 | 特征关系图 |
| `predict_with_graph` | 激活沿边传播一步 | 图增强预测 |
| `route_hierarchical` | 粗尺度筛候选 → 全局精匹配 | 层级路由 |
| `auto_tune` | 扫 conscience_beta | 元参数自调 |

---

## 6. 已验证的能力与边界

### 能力

| 能力 | 结果 | 算法来源 |
|------|------|---------|
| 对比度不变性 | 84.4% flat (c=1.0→0.01) | centering |
| 跨域泛化 | Fashion 84% flat | 形状基元 |
| 少样本 | 1-shot 45% (+shift), 10-shot 63% | 记忆 + shift augment |
| 闭环脑补 | 遮挡 14×14 +11%, 64×64 +23.4% | 配对解码 + 闭环 |
| 持续学习 | 加 5-9 类零遗忘 71.2% | add_samples + adapt |

### 已证伪的路线（负结果，别重试）

| 路线 | 结果 | 根因 |
|------|------|------|
| 多尺度金字塔脑补 | -51.8% | activate() 是有损瓶颈，级联只丢更多 |
| Laplacian 残差 | 与单层等价 | 仍是纯线性 |
| 空间热点（复杂场景） | -17.1% | 热点激活稀疏，欠定最小范数解 |
| 全局投影 (Hadamard) | 比 raw 差 | 降维丢信息 |
| Colony/FB (MNIST) | 无效 | 图拓扑不适配图像特征 |
| Retina (多尺度) | -6% | 多尺度已含对比度不变性 |

### 诚实边界

1. **脑补的瓶颈是激活向量信息量**（300d 有损摘要），不是解码器。线性解码 `W=pinv(ATA)@ATI` 已近最优。
2. **打破天花板只剩两条路**：丰富激活（更高分辨率 + 容量，已验证 +23.4%）或真非线性解码器（需 BP，背离 Hebbian 哲学）。
3. **峰值精度 84% vs CNN 98%**——用峰值换稳定性，这是定位不是缺陷。

---

## 7. 关键 Pitfalls（实现细节）

| # | Pitfall | Fix |
|---|---------|-----|
| 1 | cosine 前未 centering → c≤0.1 崩塌 | `patch -= patch.mean()` |
| 2 | fine 4×4 是噪声 (c=1.0 仅 32%) | 全局 28×28 替代 |
| 3 | top_k>1 → 激活饱和 | top_k=1 硬分配 |
| 4 | RGB 整体 centering 破坏不变性 | per-channel centering |
| 5 | conscience `best_score=-1` 阈值过严 | `-float('inf')` |
| 6 | 迭代脑补 conf_thresh 太低 → 零迭代 | `if it>0 and sim>=thresh: break` |
| 7 | spawn 后模板/记忆维度不匹配 | 取 min 维度切片 |
| 8 | benchmark import path | `sys.path.insert(0, '..', 'src')` |
| 9 | evaluate 双循环慢 100x | 向量化 matmul |
| 10 | Fashion/MNIST 文件名冲突 | 独立目录 |
| 11 | uncertain_samples 阈值 | 0.8+（cosine 错判也 0.7+） |
| 12 | pinv 大矩阵慢 (9600²) | lstsq 或降 n_regions |
