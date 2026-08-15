#!/usr/bin/env python3
"""
64×64 合成复杂灰度场景生成器 — 空间热点脑补的高分辨率验证
物体库: MNIST + Fashion (上采样到 64) + 几何形状 (64 原生)
"""
import sys, os, numpy as np, gzip, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_mnist():
    from train_multiscale import load_mnist
    return load_mnist()

def load_fashion(d="data/fashion"):
    base = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    files = {"ti":"train-images-idx3-ubyte.gz"}
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, files["ti"])
    if not os.path.exists(p): urllib.request.urlretrieve(base + files["ti"], p)
    with gzip.open(p) as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)/255

def upscale(img, size=64):
    """块上采样 28→64"""
    h, w = img.shape
    return np.kron(img, np.ones((max(1,size//h+1), max(1,size//w+1))))[:size, :size]

def gen_shapes(n_each=100, size=64):
    """几何形状 64×64 原生"""
    shapes = []
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size/2, size/2
    def norm(x): return np.clip(x, 0, 1).astype(np.float32)
    for _ in range(n_each):
        r = np.random.uniform(8, size/2.2)
        shapes.append(norm(np.sqrt((xx-cx)**2 + (yy-cy)**2) <= r))
        s = np.random.uniform(10, size/1.4)
        shapes.append(norm((np.abs(xx-cx) <= s/2) & (np.abs(yy-cy) <= s/2)))
        s = np.random.uniform(10, size/1.4)
        shapes.append(norm((yy - cy) <= s/2 - (s/size)*np.abs(xx-cx)))
        w = np.random.uniform(3, size/3)
        shapes.append(norm((np.abs(xx-cx) <= w/2) | (np.abs(yy-cy) <= w/2)))
        d = np.random.uniform(0, size*1.4)
        shapes.append(norm(np.abs((xx+yy) - d) <= 2.5))
        r_out = np.random.uniform(8, size/2.0); r_in = r_out * np.random.uniform(0.3, 0.7)
        dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
        shapes.append(norm((dist <= r_out) & (dist >= r_in)))
    return np.array(shapes, dtype=np.float32)

def compose(objs, size=64, n_objects=None, rng=None):
    rng = rng or np.random
    scene = np.zeros((size, size), dtype=np.float32)
    n = n_objects or rng.randint(2, 5)
    chosen = rng.choice(len(objs), n, replace=True)
    for oi in chosen:
        obj = objs[oi]
        rot = rng.choice([0, 1, 2, 3])
        o = np.rot90(obj, rot) if rot else obj
        scale = rng.uniform(0.4, 1.2)
        oh, ow = o.shape
        nh = max(6, int(oh*scale)); nw = max(6, int(ow*scale))
        # 缩放 (块下采样或上采样)
        if nh <= oh:
            o2 = o[:nh, :nw]
        else:
            o2 = np.kron(o, np.ones((max(1,nh//oh+1), max(1,nw//ow+1))))[:nh, :nw]
        y0 = rng.randint(0, max(1, size-nh)); x0 = rng.randint(0, max(1, size-nw))
        patch = o2[:size-y0, :size-x0]
        scene[y0:y0+patch.shape[0], x0:x0+patch.shape[1]] = np.maximum(
            scene[y0:y0+patch.shape[0], x0:x0+patch.shape[1]], patch)
    return scene

def severe_erase(images, frac=0.4, rng=None):
    rng = rng or np.random
    out = images.copy()
    H, W = images.shape[1], images.shape[2]
    for i in range(len(images)):
        bh = int(H*rng.uniform(0.3,0.5)); bw = int(W*rng.uniform(0.3,0.5))
        y = rng.randint(0,H-bh+1); x = rng.randint(0,W-bw+1)
        out[i,y:y+bh,x:x+bw]=0
        for _ in range(3):
            sh=rng.randint(5,14); sw=rng.randint(5,14)
            y2=rng.randint(0,H-sh); x2=rng.randint(0,W-sw)
            out[i,y2:y2+sh,x2:x2+sw]=0
    return out

if __name__ == "__main__":
    np.random.seed(42)
    print("加载 MNIST + Fashion...")
    Xm = load_mnist()[0]
    Xf = load_fashion()

    # 上采样到 64×64
    print("上采样物体到 64×64...")
    Xm64 = np.array([upscale(x) for x in Xm[:3000]], dtype=np.float32)
    Xf64 = np.array([upscale(x) for x in Xf[:3000]], dtype=np.float32)
    print("生成几何形状 (64×64)...")
    Xs64 = gen_shapes(n_each=100)

    objs = np.concatenate([Xm64, Xf64, Xs64])
    print(f"物体库: {len(objs)} (MNIST 3K + Fashion 3K + 几何 600)")

    n_scenes = 6000
    print(f"生成 {n_scenes} 个 64×64 场景 + 重度擦除...")
    clean = np.zeros((n_scenes, 64, 64), dtype=np.float32)
    for i in range(n_scenes):
        clean[i] = compose(objs, size=64)
    degraded = severe_erase(clean, rng=np.random.RandomState(1))

    os.makedirs("data/synthetic", exist_ok=True)
    np.savez_compressed("data/synthetic/scenes64.npz", clean=clean, degraded=degraded)
    print(f"保存 data/synthetic/scenes64.npz: clean={clean.shape}")
    print("=== DONE ===")
