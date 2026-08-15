#!/usr/bin/env python3
"""
合成复杂灰度数据生成器 — 加强脑补能力

场景构成:
  1-3 个物体 (MNIST数字 / Fashion物品 / 几何形状) 随机摆放旋转缩放, 可重叠
退化库:
  块遮挡 / 椒盐噪声 / 高斯噪声 / 高斯模糊 / 线条划痕 / 随机擦除 / 降采样
输出: (降质, 干净) 配对
"""
import sys, os, numpy as np, gzip, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_mnist():
    from train_multiscale import load_mnist
    return load_mnist()

def load_fashion(d="data/fashion"):
    base = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    files = {"ti":"train-images-idx3-ubyte.gz","tl":"train-labels-idx1-ubyte.gz"}
    os.makedirs(d, exist_ok=True)
    for f in files.values():
        p = os.path.join(d, f)
        if not os.path.exists(p): urllib.request.urlretrieve(base + f, p)
    def li(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)/255
    def ll(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return li(os.path.join(d, files["ti"])), ll(os.path.join(d, files["tl"]))


# ═══ 几何形状库 ═══

def gen_shapes(n_each=200, size=28):
    """生成几何形状: 圆/方/三角/十字/斜线/圆环/半圆/点"""
    shapes = []
    labels = []
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size/2, size/2

    def norm(x): return np.clip(x, 0, 1)

    for _ in range(n_each):
        # 圆 (实心)
        r = np.random.uniform(3, size/2.5)
        c = norm(np.sqrt((xx-cx)**2 + (yy-cy)**2) <= r)
        shapes.append(c); labels.append(0)

        # 方 (实心)
        s = np.random.uniform(5, size/1.5)
        c = norm((np.abs(xx-cx) <= s/2) & (np.abs(yy-cy) <= s/2))
        shapes.append(c); labels.append(1)

        # 三角
        s = np.random.uniform(5, size/1.5)
        c = norm((yy - cy) <= s/2 - (s/size)*np.abs(xx-cx))
        shapes.append(c); labels.append(2)

        # 十字
        w = np.random.uniform(2, size/4)
        c = norm((np.abs(xx-cx) <= w/2) | (np.abs(yy-cy) <= w/2))
        shapes.append(c); labels.append(3)

        # 斜线
        d = np.random.uniform(0, size)
        c = norm(np.abs((xx+yy) - d) <= 1.5)
        shapes.append(c); labels.append(4)

        # 圆环
        r_out = np.random.uniform(4, size/2.2)
        r_in = r_out * np.random.uniform(0.3, 0.7)
        dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
        c = norm((dist <= r_out) & (dist >= r_in))
        shapes.append(c); labels.append(5)

    return np.array(shapes, dtype=np.float32), np.array(labels)


# ═══ 场景合成 ═══

def compose_scene(objs, size=28, n_objects=None, rng=None):
    """把 1-3 个物体随机摆放/缩放/旋转进一张图"""
    rng = rng or np.random
    scene = np.zeros((size, size), dtype=np.float32)
    n = n_objects or rng.randint(1, 4)
    chosen = rng.choice(len(objs), n, replace=True)
    for obj_idx in chosen:
        obj = objs[obj_idx]
        # 缩放
        scale = rng.uniform(0.5, 1.5)
        # 旋转 (90度倍数最简, 避免插值)
        rot = rng.choice([0, 1, 2, 3])
        o = np.rot90(obj, rot) if rot else obj
        # 缩放到目标尺寸
        oh, ow = o.shape
        new_h = max(4, int(oh * scale))
        new_w = max(4, int(ow * scale))
        # 简单 resize (最近邻 via 分块)
        o_resized = o[:new_h, :new_w] if new_h <= oh and new_w <= ow else \
            np.kron(o, np.ones((max(1,new_h//oh+1), max(1,new_w//ow+1))))[:new_h, :new_w]
        # 随机位置
        max_y = max(0, size - new_h)
        max_x = max(0, size - new_w)
        y0 = rng.randint(0, max_y + 1) if max_y > 0 else 0
        x0 = rng.randint(0, max_x + 1) if max_x > 0 else 0
        # 叠加 (不覆盖, 直接加 → 重叠处更亮, 模拟遮挡是后处理)
        patch = o_resized[:size-y0, :size-x0]
        scene[y0:y0+patch.shape[0], x0:x0+patch.shape[1]] = np.maximum(
            scene[y0:y0+patch.shape[0], x0:x0+patch.shape[1]], patch)
    return scene


# ═══ 退化库 ═══

def degrade(images, mode=None, rng=None):
    """单种退化"""
    rng = rng or np.random
    out = images.copy()
    N = len(images)
    if mode == 'occlude':
        bs = rng.randint(3, 10)
        for i in range(N):
            y = rng.randint(0, images.shape[1]-bs); x = rng.randint(0, images.shape[2]-bs)
            out[i, y:y+bs, x:x+bs] = images[i].mean()
    elif mode == 'salt_pepper':
        for i in range(N):
            mask = rng.random(images[i].shape) < 0.1
            out[i][mask] = rng.choice([0.0, 1.0], size=mask.sum())
    elif mode == 'gaussian':
        out = np.clip(out + rng.randn(*out.shape).astype(np.float32) * 0.15, 0, 1)
    elif mode == 'blur':
        # 3x3 box blur
        k = np.ones((3,3))/9
        for i in range(N):
            pad = np.pad(out[i], 1, mode='reflect')
            for dy in range(3):
                for dx in range(3):
                    out[i] += k[dy,dx] * pad[dy:dy+images.shape[1], dx:dx+images.shape[2]]
            out[i] = out[i] / 2.0  # 修正累加
    elif mode == 'scratch':
        for i in range(N):
            n_lines = rng.randint(1, 4)
            for _ in range(n_lines):
                if rng.random() < 0.5:  # 水平线
                    y = rng.randint(0, images.shape[1])
                    x0 = rng.randint(0, images.shape[2]-5)
                    out[i, y, x0:x0+5] = 1.0
                else:  # 垂直线
                    x = rng.randint(0, images.shape[2])
                    y0 = rng.randint(0, images.shape[1]-5)
                    out[i, y0:y0+5, x] = 1.0
    elif mode == 'erase':
        for i in range(N):
            h = rng.randint(3, 8); w = rng.randint(3, 8)
            y = rng.randint(0, images.shape[1]-h); x = rng.randint(0, images.shape[2]-w)
            out[i, y:y+h, x:x+w] = 0.0
    elif mode == 'downsample':
        for i in range(N):
            small = out[i][::2, ::2]
            out[i] = np.kron(small, np.ones((2,2)))[:images.shape[1], :images.shape[2]]
    return out


def mixed_degrade(images, rng=None):
    """随机选 1-3 种退化叠加"""
    rng = rng or np.random
    out = images.copy()
    n_degrade = rng.randint(1, 4)
    modes = ['occlude', 'salt_pepper', 'gaussian', 'blur', 'scratch', 'erase', 'downsample']
    chosen = rng.choice(modes, n_degrade, replace=False)
    for m in chosen:
        out = degrade(out, mode=m, rng=rng)
    return out


# ═══ 主生成 ═══

if __name__ == "__main__":
    np.random.seed(42)
    print("加载 MNIST + Fashion...")
    Xm, ym, _, _ = load_mnist()
    Xf, yf = load_fashion()

    print("生成几何形状...")
    Xs, ys = gen_shapes(n_each=200)

    # 物体库: MNIST + Fashion + 几何
    objs = np.concatenate([Xm[:5000], Xf[:5000], Xs])
    print(f"物体库: {len(objs)} 个 (MNIST 5K + Fashion 5K + 几何 1.2K)")

    # 生成配对数据: 干净场景 → 退化
    n_scenes = 10000
    print(f"生成 {n_scenes} 个场景 + 退化配对...")
    clean = np.zeros((n_scenes, 28, 28), dtype=np.float32)
    for i in range(n_scenes):
        clean[i] = compose_scene(objs, size=28)
    degraded = mixed_degrade(clean)

    os.makedirs("data/synthetic", exist_ok=True)
    np.savez_compressed("data/synthetic/scenes.npz", clean=clean, degraded=degraded)
    print(f"保存 data/synthetic/scenes.npz: clean={clean.shape}, degraded={degraded.shape}")
    print("=== DONE ===")
