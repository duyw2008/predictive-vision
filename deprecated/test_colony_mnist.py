#!/usr/bin/env python3
"""
test_colony_mnist.py — 费曼脑架构移植到冷眼的验证测试

流程:
  1. 加载 MNIST → 训练 VisionGraph (竞争路由 + Hebbian)
  2. 创建 VisionColony → 细胞在特征图上行走
  3. 运行 200 代 → 观察突触 tier 分布、髓鞘分化
  4. 输出 top 边 (高共识特征对)
"""

import sys, os, time, gzip, numpy as np

# 加载 MNIST
def load_mnist(path='.', kind='train', limit=5000):
    url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    fname = f'{kind}-images-idx3-ubyte.gz'
    fpath = os.path.join(path, fname)
    if not os.path.exists(fpath):
        import urllib.request
        os.makedirs(path, exist_ok=True)
        urllib.request.urlretrieve(url + fname, fpath)

    with gzip.open(fpath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28).astype(np.float32) / 255.0

    label_fname = f'{kind}-labels-idx1-ubyte.gz'
    label_path = os.path.join(path, label_fname)
    if not os.path.exists(label_path):
        import urllib.request
        urllib.request.urlretrieve(url + label_fname, label_path)
    with gzip.open(label_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return data[:limit], labels[:limit]


def train_vision_graph(images, labels, n_nodes=200, n_epochs=3, n_train=None, contrast=1.0):
    """训练 VisionGraph (Stage 1: 竞争路由 + 构建共激活边)"""
    from graph import VisionGraph
    from vision import VisionInterface

    ts = 8 * 8  # 8×8 patches
    graph = VisionGraph(n_nodes=n_nodes, template_size=ts)
    vision = VisionInterface(graph, patch_size=8, stride=4)

    if n_train is None:
        n_train = len(images)

    # 用真实图像 patch 初始化模板 (替代随机初始化)
    print("  initializing templates from image patches...")
    patch_samples = []
    for i in range(min(1000, n_train)):
        patches = vision.extractor.extract(images[i])
        patch_samples.append(patches[np.random.randint(len(patches))])
    for nid in list(graph.nodes.keys()):
        idx = np.random.randint(len(patch_samples))
        graph.nodes[nid].template = patch_samples[idx].copy().astype(np.float32)
        graph.nodes[nid].template /= np.linalg.norm(graph.nodes[nid].template) + 1e-8

    t0 = time.time()

    # 建立共激活边 (SynapticLayer)
    print("  building co-activation edges...")
    coactive = {}
    sample_images = images[:min(2000, n_train)]
    sample_labels = labels[:min(2000, n_train)]

    for i, img in enumerate(sample_images):
        vision.set_image(img.copy(), contrast=contrast)
        active_nodes = [nid for nid, n in graph.nodes.items() if n.activation > 0.05]
        for j, src in enumerate(active_nodes):
            for dst in active_nodes[j+1:]:
                key = (src, dst)
                coactive[key] = coactive.get(key, 0) + 1

    # 初始化 SynapticLayer
    from synapse import SynapticLayer
    synapse = SynapticLayer()
    graph.synapse = synapse

    edge_count = 0
    for (src, dst), count in coactive.items():
        if count >= 5:  # 至少 5 张图中同时激活
            synapse.fire(src, dst, 0.3, sample_id="init", generation=0)
            synapse.fire(dst, src, 0.3, sample_id="init", generation=0)  # 对称
            edge_count += 2

    elapsed = time.time() - t0
    print(f"  training: {n_train} imgs, {edge_count} co-activation edges ({elapsed:.1f}s)")

    return graph, vision


def main():
    print("=== 冷眼费曼脑移植测试 ===\n")

    # 1. 加载数据
    print("[1] Loading MNIST...")
    images, labels = load_mnist(limit=5000)
    print(f"  {len(images)} images, {len(np.unique(labels))} classes")

    # 2. 训练特征图 (Stage 1)
    print("[2] Training VisionGraph...")
    graph, vision = train_vision_graph(images, labels, n_nodes=50, n_epochs=0)

    # 3. 创建群落 (Stage 2)
    print("\n[3] Creating VisionColony...")
    from vision_colony import VisionColony
    colony = VisionColony(graph)

    # 播种细胞
    colony.seed_cells(n_per_node=3, max_cells=150)
    print(f"  initial stats: {colony.stats()}")

    # 4. 运行呼吸 (细胞在特征图上行走)
    print("\n[4] Running colony (breathe + sleep)...")
    colony.breathe(n_generations=30, verbose=False)
    colony.print_stats()

    # 阶段检验
    print("\n--- gen 30 ---")
    print(f"  cells: {len(colony.cells)}")
    print(f"  synapses: {colony.stats()['synapses']}")
    print(f"  t3 edges: {colony.stats()['t3']}")

    # 髓鞘分布
    myelin_counts = {}
    for cell in colony.cells:
        for k, v in cell.myelin.items():
            if v > 0.5:
                myelin_counts[k] = myelin_counts.get(k, 0) + 1
    high_myelin = [(k, c) for k, c in myelin_counts.items() if c >= 3]
    high_myelin.sort(key=lambda x: -x[1])
    print(f"  high-myelin edges (≥3 cells): {len(high_myelin)}")
    for k, c in high_myelin[:5]:
        print(f"    {k[0][:12]} → {k[1][:12]}: {c} cells myelinated")

    # 继续跑
    print("\n[5] Running more generations...")
    colony.run(n_generations=70, verbose=False)
    colony.print_stats()

    # Top 边
    print("\n[6] Top edges (by s-value):")
    for src, dst, s, n, tier in colony.top_edges(15):
        src_name = src[:20]
        dst_name = dst[:20]
        print(f"  {src_name:20s} → {dst_name:20s}  s={s:.2f}  n={n:3d}  t{tier}")

    # 轴突/髓鞘统计
    total_axons = sum(len(c.axons) for c in colony.cells)
    total_myelin = sum(len(c.myelin) for c in colony.cells)
    print(f"\n[7] Cell stats:")
    print(f"  total axons: {total_axons}")
    print(f"  total myelin edges: {total_myelin}")
    print(f"  avg axons/cell: {total_axons/max(1,len(colony.cells)):.1f}")
    print(f"  avg myelin/cell: {total_myelin/max(1,len(colony.cells)):.1f}")

    print("\n=== DONE ===")


if __name__ == '__main__':
    main()
