# 冷眼真实图像离线处理操作指南

本指南记录「冷眼识别一张真实照片」的完整离线处理流程，以 ship.png（邮轮顶视图）为例。

---

## 0. 处理流程总览

```
真实照片
  │
  ├─ 第1步 图像检查    判断物体类型/视角/颜色特征 → 决定后续每个选择
  ├─ 第2步 颜色定位    选对通道分离「物体 vs 背景」(figure-ground)
  ├─ 第3步 填平内部    去掉内部复杂结构 → 实心剪影
  ├─ 第4步 letterbox  保长宽比缩放 (非方形物体绝不能硬压成方)
  ├─ 第5步 few-shot    几何基类 + 剪影记忆 → 识别
  └─ 第6步 测试权衡    鲁棒性 vs 判别力 旋钮
```

**核心原则**：颜色只用于「找边界」（figure-ground），识别仍走灰度。
颜色不是语义捷径（不是「红=船」），是定位信号（如视网膜色拮抗细胞）。

---

## 1. 图像检查（决定一切的第0步）

处理前必须先回答三个问题，否则后面步步错：

| 问题 | ship.png 答案 | 影响 |
|------|--------------|------|
| 物体是什么视角？ | **顶视图**（长条形船体） | 决定剪影的期望形状 |
| 物体是什么颜色？ | 红棕色船体 vs 蓝色海 | 决定用哪个通道分离 |
| 物体长宽比？ | 85×443 = 5.2:1 | 决定必须 letterbox |

**教训**：ship.png 是邮轮**顶视图**——没有桅杆、没有窗户，就是一条长条形船（尖船头圆船尾）。如果误判成「侧视图有桅杆」，后面的剪影形状全错。

---

## 2. 颜色定位（选对通道分离物体和背景）

**先看各通道的物体/背景对比度**，选对比度最大的通道：

```python
img = np.array(Image.open('ship.png').convert('RGBA')).astype(np.float32)
R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

# 船 vs 海 各通道均值:
#   R: 船 52 vs 海 1    → 差 51 倍  ← 选 R 通道
#   G: 船 73 vs 海 83   → 差 -10
#   B: 船 66 vs 海 79   → 差 -13
# 灰度: 船 0.26 vs 海 0.23 → 只差 0.03 (信息论上分不开)
```

**结论**：这艘红棕船在蓝色海面上，灰度下船海亮度几乎一样（0.26 vs 0.23），**只有 R 通道有 51 倍对比度**。

**阈值选择**：看通道直方图找「物体」和「背景」两个峰之间的谷。注意 **Otsu 在多峰分布上会切错**（ship.png 的 R 通道有「海/船体/白色高光」三峰，Otsu 切在 0.361，会偏向抓白色高光而非船体本体 R=0.204）。

两种做法都行（letterbox 修好后都能出船形）：
```python
# 做法A (demo 用): Otsu 自动阈值
ship_mask = R_norm > otsu(R_norm)

# 做法B (更精确): 按物体像素值手动设 (海 R=0.004, 船 R=0.204)
ship_mask = R_norm > 0.08
```

**关键**：无论哪种，都要**人工核对剪影是否包住了物体本体**（不是只抓高光）。

---

## 3. 填平内部（去复杂结构 → 实心剪影）

```python
from scipy import ndimage
filled = ndimage.binary_fill_holes(ship_mask)       # 填内部洞
closed  = ndimage.binary_closing(filled, np.ones((3,3)), iterations=2)  # 平滑
# 保留最大连通域 (去掉散落的浪花反光)
labeled, n = ndimage.label(closed)
sizes = ndimage.sum(closed, labels=labeled, index=range(1, n+1))
closed = (labeled == int(np.argmax(sizes)) + 1)
# 裁剪到物体 bounding box
ys, xs = np.where(closed)
sil = closed[ys.min():ys.max()+1, xs.min():xs.max()+1]
```

**注意**：如果物体本身就是实心的（如这艘邮轮顶视图），fill_holes 没有内部洞可填，反而会把桅杆/细节糊进船体。顶视图无桅杆无窗户时，可以只做「阈值 + 最大连通域」，不 fill 不 closing。

---

## 4. letterbox（非方形物体保形缩放）

**这是最容易踩的坑**：冷眼输入是 28×28 方形（MNIST 训练出来的），但真实物体常非方。

```python
def letterbox(x, target=28):
    """保持长宽比缩放进 target×target, 短边补 0"""
    h, w = x.shape
    scale = target / max(h, w)
    nh, nw = max(1, int(h*scale)), max(1, int(w*scale))
    r = np.array(Image.fromarray((x*255).astype(np.uint8))
                 .resize((nw, nh), Image.LANCZOS)).astype(np.float32)/255
    canvas = np.zeros((target, target), np.float32)
    y0, x0 = (target-nh)//2, (target-nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = r
    return canvas
```

**错误做法**：`resize((28,28))` 把 5.2:1 的长船硬压成方 → 变成「大白斑」方块，形状全毁。

**正确做法**：letterbox 保形 → 长条船（尖头圆尾）完整保留。

---

## 5. few-shot（几何基类 + 剪影记忆）

```python
# 几何基类训练模板 (形状基元: 圆/方/三角/梯形/矩形/线/环)
geo = gen_geometric_shapes(n_each=150)
model = ColdEye(eye_specs=[{"type":"global","n":200},
                           {"type":"patch","ps":16,"st":8,"n":100}])
model.init_templates(geo[:200])
model.train(geo, np.zeros(len(geo)), epochs=5, n_train=len(geo))

# 剪影 + 变体 当 few-shot 船记忆
for v in ship_variants:
    model.memory.append((model._activate_one(v), 1))   # 船
for g in geo[:5]:
    model.memory.append((model._activate_one(g), 0))   # 非船
```

**记忆扩充的权衡**（鲁棒性 vs 判别力）：

| 记忆配置 | 鲁棒性 | 判别力 | 说明 |
|---------|--------|--------|------|
| 满框(5变体) | 67% | 80% | 只认完整船 |
| 轻度(0.5缩小+8遮挡) | 100% | 80% | **平衡点** |
| 重度(0.3缩小+14遮挡) | 100% | 75% | 半擦除船太宽松 |

权衡是单调的：记忆扩充越多 → 鲁棒性越高、判别力越低。**14×14 半擦除的船 ≈ 半空形状 ≈ 几何形状**，会误吸几何形状。

---

## 6. 测试（鲁棒性三件套 + 判别力）

```python
robust_tests = [
    ("缩小 0.5", shrink(sil28, 0.5)),
    ("低对比度 c=0.01", (sil28*0.01).astype(np.float32)),   # centering 免疫
    ("遮挡 14×14", occlude(sil28, 14)),
]
# 鲁棒性 = 降质样本识别为"船"的比例
# 判别力 = 几何形状正确拒绝的比例
```

**低对比度永远免疫**（centering 数学保证 c=0.01 == c=1.0），这是冷眼的招牌能力，任何真实物体都该验证这一点。

---

## 7. 关键 Pitfalls（按踩坑顺序）

| # | Pitfall | 后果 | 修法 |
|---|---------|------|------|
| 1 | 误判视角（顶视当侧视） | 剪影形状全错 | 第1步先看图 |
| 2 | 用灰度分离颜色依赖的物体 | 船海只差0.03分不开 | 选对比度最大的通道 |
| 3 | Otsu 在多峰分布切错 | 漏掉物体本体 | 按物体像素值设阈值 |
| 4 | 非方形物体硬压成方 | 长船变"大白斑" | letterbox 保形 |
| 5 | 物体实心却还 fill_holes | 细节糊进本体 | 实心物体只阈值+最大连通域 |
| 6 | 记忆加重度降质变体 | 判别力崩(60%) | 只加轻度变体取平衡 |

---

## 8. 完整可运行脚本

见同目录 `demo_real_ship.py`，自包含（几何基类 + Otsu + letterbox + few-shot + 三配置权衡），运行：

```bash
cd /home/duyw/predictive-vision
python3 demos/real_ship_recognition/demo_real_ship.py
```

输出 `demo_result.png`（管线三图 + 权衡条形图 + 降质识别结果）。
