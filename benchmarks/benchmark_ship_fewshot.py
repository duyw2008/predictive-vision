#!/usr/bin/env python3
"""
邮轮小样本适应实验 — 验证"小样本上限 = 基类基元覆盖"

基类: 几何形状库 (圆/方/三角/梯形/长矩形/细线/圆环)
目标: 邮轮 (船体梯形 + 上层建筑矩形 + 烟囱 + 桅杆)
对照: MNIST 数字基类 (笔划基元, 应无法覆盖邮轮)

假设:
  几何基类 → 邮轮几发 → 好使 (梯形/矩形/线覆盖船体/建筑/桅杆)
  MNIST 基类 → 邮轮几发 → 失效 (笔划基元 ≠ 船体/机翼)
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_multiscale import ColdEye, load_mnist

def norm(x): return np.clip(x, 0, 1).astype(np.float32)

# ═══ 几何基类 (含邮轮所需基元) ═══
def gen_geometric_shapes(n_each=150, size=28):
    shapes, labels = [], []
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size/2, size/2
    for _ in range(n_each):
        # 圆
        r = np.random.uniform(3, size/2.5)
        shapes.append(norm(np.sqrt((xx-cx)**2+(yy-cy)**2) <= r)); labels.append('circle')
        # 方
        s = np.random.uniform(5, size/1.5)
        shapes.append(norm((np.abs(xx-cx)<=s/2)&(np.abs(yy-cy)<=s/2))); labels.append('square')
        # 三角
        s = np.random.uniform(5, size/1.5)
        shapes.append(norm((yy-cy) <= s/2-(s/size)*np.abs(xx-cx))); labels.append('triangle')
        # 长矩形 (横 — 船体)
        w = np.random.uniform(size/2, size-4); h = np.random.uniform(3, 8)
        shapes.append(norm((np.abs(xx-cx)<=w/2)&(np.abs(yy-cy)<=h/2))); labels.append('hrect')
        # 长矩形 (竖 — 桅杆/塔)
        w = np.random.uniform(2, 5); h = np.random.uniform(size/2, size-4)
        shapes.append(norm((np.abs(xx-cx)<=w/2)&(np.abs(yy-cy)<=h/2))); labels.append('vrect')
        # 梯形 (船体轮廓 — 上宽下窄)
        top = np.random.uniform(size/2, size-4); bot = np.random.uniform(3, top/2)
        h = np.random.uniform(4, 10)
        halfw = bot/2 + (top-bot)/2 * (yy-(cy-h/2)) / max(h, 1)
        shapes.append(norm((yy>=cy-h/2)&(yy<=cy+h/2)&(np.abs(xx-cx)<=halfw))); labels.append('trapezoid')
        # 细线 (桅杆)
        shapes.append(norm(np.abs(xx-cx)<=1.5)); labels.append('vline')
        # 圆环
        ro = np.random.uniform(4, size/2.2); ri = ro*np.random.uniform(0.3,0.7)
        d = np.sqrt((xx-cx)**2+(yy-cy)**2)
        shapes.append(norm((d<=ro)&(d>=ri))); labels.append('ring')
    return np.array(shapes, np.float32), labels

# ═══ 合成邮轮 ═══
def gen_cruise_ship(size=28, n=100, rng=None):
    rng = rng or np.random
    ships = []
    for _ in range(n):
        img = np.zeros((size, size), np.float32)
        # 船体 (下部长梯形, 上宽下窄)
        hull_top = rng.uniform(size*0.6, size-4)
        hull_bot = rng.uniform(3, hull_top*0.4)
        hull_h = rng.uniform(6, 10)
        hull_y = size - rng.uniform(4, 8)  # 船体靠下
        yy_, xx_ = np.mgrid[0:size, 0:size]
        # 梯形船体: 从 hull_y 到 hull_y+hull_h, 顶部宽 hull_top, 底部窄 hull_bot
        for y in range(size):
            t = (y - hull_y) / max(hull_h, 1)
            if 0 <= t <= 1:
                halfw_y = hull_bot/2 + (hull_top-hull_bot)/2 * (1-t)
                img[y, int(size/2-halfw_y):int(size/2+halfw_y)] = 0.8
        # 上层建筑 (船体上方叠 2-3 个矩形, 逐层变窄)
        super_x = size/2
        super_w = hull_top * rng.uniform(0.5, 0.7)
        super_y = hull_y - 2
        for deck in range(rng.randint(2, 4)):
            deck_h = rng.uniform(2, 4)
            img[int(super_y-deck_h):int(super_y), int(super_x-super_w/2):int(super_x+super_w/2)] = 0.7
            super_y -= deck_h
            super_w *= rng.uniform(0.6, 0.8)  # 上层变窄
        # 烟囱 (顶部小矩形)
        fun_w = rng.uniform(3, 5)
        img[int(super_y-3):int(super_y), int(super_x-fun_w/2):int(super_x+fun_w/2)] = 0.9
        # 桅杆 (细竖线)
        mast_x = int(super_x + rng.uniform(-4, 4))
        img[2:int(super_y-2), mast_x:mast_x+2] = 0.6
        ships.append(norm(img))
    return np.array(ships, np.float32)

# ═══ 两阶段小样本 ═══
def two_stage_fewshot(base_imgs, target_imgs, n_shot, seed=42):
    np.random.seed(seed)
    model = ColdEye(eye_specs=[
        {"type": "global", "n": 200},
        {"type": "patch", "ps": 16, "st": 8, "n": 100},
    ])
    model.init_templates(base_imgs[:2000])
    model.train(base_imgs, np.zeros(len(base_imgs)), epochs=5, n_train=len(base_imgs), contrast_aug=True)
    # 几发建记忆: 邮轮 = 类1, 随机几何 = 类0
    pos = target_imgs[:n_shot]
    neg = base_imgs[:n_shot]  # 用基类当"非邮轮"干扰
    for img in pos: model.memory.append((model._activate_one(img), 1))
    for img in neg: model.memory.append((model._activate_one(img), 0))
    # 测试: 邮轮 vs 几何
    test_pos = target_imgs[n_shot:n_shot+100]
    test_neg = base_imgs[n_shot:n_shot+100]
    tp = sum(1 for i in range(100) if model.predict(test_pos[i])[0] == 1)
    tn = sum(1 for i in range(100) if model.predict(test_neg[i])[0] == 0)
    return (tp + tn) / 200

if __name__ == "__main__":
    np.random.seed(42)
    size = 28
    print("生成几何基类 (含梯形/长矩形/细线)...")
    geo, geo_labels = gen_geometric_shapes(n_each=150)
    print(f"  几何基类: {len(geo)} 张")

    print("合成邮轮 (船体+建筑+烟囱+桅杆)...")
    ships = gen_cruise_ship(size=28, n=200, rng=np.random.RandomState(0))
    print(f"  邮轮: {len(ships)} 张")

    print("加载 MNIST (对照基类)...")
    Xm = load_mnist()[0]

    print(f"\n{'='*60}")
    print("  邮轮小样本适应: 几何基类 vs MNIST 基类")
    print(f"{'='*60}")

    for n_shot in [1, 3, 5, 10]:
        geo_acc = two_stage_fewshot(geo, ships, n_shot)
        mnist_acc = two_stage_fewshot(Xm[:2000], ships, n_shot)
        print(f"  {n_shot:2d}-shot: 几何基类 {geo_acc:.1%}   MNIST基类 {mnist_acc:.1%}   (随机 50%)")

    print("\n=== DONE ===")
