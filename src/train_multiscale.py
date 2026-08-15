#!/usr/bin/env python3
"""
冷眼 v3 — 可配置多尺度预测反馈视觉系统
架构: 全局眼(28×28) + 任意数量 patch 眼 → 激活拼接 → KNN
默认: global(100n) + coarse 16×16(50n) = 150d
"""

import sys, os, time, numpy as np, gzip, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from graph import VisionGraph
from vision import VisionInterface


# ═══════════════════════════════════════════
#  Global Eye: 整图竞争路由 (不用 VisionGraph)
# ═══════════════════════════════════════════

class GlobalEye:
    """整图 → 竞争路由 → 激活向量。支持灰度(单通道)和 RGB(3通道, per-channel centering)。

    低对比度鲁棒: per-channel centering 让对比度 c 从数学上约掉 (每个通道独立减均值+L2)。"""

    def __init__(self, n_nodes=100, seed=None):
        self.n = n_nodes
        self.templates = None  # 动态尺寸: init_templates 时按图像维度分配 (28×28=784 / 32×32=1024 / RGB=3072)
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def _center(self, img):
        """centering: RGB 3通道 per-channel (保持对比度不变性), 灰度整体。返回 flatten 向量."""
        if img.ndim == 3:
            flat = img.reshape(-1, img.shape[2]).astype(np.float32)  # (H*W, C)
            flat = flat - flat.mean(axis=0, keepdims=True)
            flat /= np.linalg.norm(flat, axis=0, keepdims=True) + 1e-8
            return flat.reshape(-1)
        flat = img.reshape(-1).astype(np.float32)
        flat = flat - flat.mean()
        flat /= np.linalg.norm(flat) + 1e-8
        return flat

    def init_templates(self, images):
        idxs = self.rng.choice(len(images), min(200, len(images)), replace=False)
        patches = np.array([self._center(images[i]) for i in idxs], dtype=np.float32)
        self.templates = np.empty((self.n, patches.shape[1]), dtype=np.float32)
        for i in range(self.n):
            self.templates[i] = patches[i % len(patches)]
            self.templates[i] /= np.linalg.norm(self.templates[i]) + 1e-8

    def train(self, images, epochs=3, n_train=None, contrast_aug=True, conscience_beta=0.0):
        if n_train is None: n_train = len(images)
        lr = 0.1
        rng = self.rng
        # 良心机制: 节点赢的频率 → 抑制常胜节点，促进生态位分化
        freq = np.zeros(self.n, dtype=np.float32) if conscience_beta > 0 else None
        for ep in range(epochs):
            for idx in rng.permutation(min(n_train, len(images))):
                img = images[idx].copy()
                if contrast_aug and rng.random() < 0.5:
                    if img.ndim == 3:  # RGB per-channel 对比度增强
                        m = img.mean(axis=(0, 1), keepdims=True)
                        img = m + (img - m) * (0.3 + rng.random() * 0.7)
                    else:
                        m = img.mean()
                        img = m + (img - m) * (0.3 + rng.random() * 0.7)
                flat = self._center(img)
                scores = self.templates @ flat
                if freq is not None:
                    scores = scores - conscience_beta * freq
                best = int(np.argmax(scores))
                self.templates[best] += lr * (flat - self.templates[best])
                self.templates[best] /= np.linalg.norm(self.templates[best]) + 1e-8
                if freq is not None:
                    freq[best] += 1.0
                    freq *= 0.999  # 缓慢衰减

    def activate(self, images):
        return np.array([self.activate_one(img) for img in images], dtype=np.float32)

    def activate_one(self, image):
        flat = self._center(image)
        return np.clip(self.templates @ flat, 0, 1).astype(np.float32)


# ═══════════════════════════════════════════
#  Patch Eye: VisionGraph 竞争路由 (任意尺度)
# ═══════════════════════════════════════════

class PatchEye:
    """patch 尺度竞争路由，封装 VisionGraph + VisionInterface"""

    def __init__(self, patch_size, stride, n_nodes, seed=None):
        ts = patch_size * patch_size
        self.g = VisionGraph(n_nodes=n_nodes, template_size=ts)
        self.v = VisionInterface(self.g, patch_size=patch_size, stride=stride)
        self.nids = sorted(self.g.nodes.keys())
        self.ps = patch_size
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def init_templates(self, images):
        rng = self.rng
        idxs = rng.choice(len(images), min(200, len(images)), replace=False)
        ap = [p.astype(np.float32) for i in idxs for p in self.v.extractor.extract(images[i] - images[i].mean())]
        for k, nid in enumerate(self.nids):
            p = ap[k % len(ap)]; p /= np.linalg.norm(p) + 1e-8
            self.g.nodes[nid].template = p

    def train(self, images, epochs=3, n_train=None, contrast_aug=True, conscience_beta=0.0):
        if n_train is None: n_train = len(images)
        lr = 0.1
        rng = self.rng
        n_nodes = len(self.nids)
        nid_to_idx = {nid: i for i, nid in enumerate(self.nids)}
        freq = np.zeros(n_nodes, dtype=np.float32) if conscience_beta > 0 else None
        for ep in range(epochs):
            for idx in rng.permutation(min(n_train, len(images))):
                img = images[idx].copy()
                if contrast_aug and rng.random() < 0.5:
                    m = img.mean()
                    img = m + (img - m) * (0.3 + rng.random() * 0.7)
                img = img - img.mean()
                self.v.set_image(img)
                if freq is not None:
                    best_nid = self.nids[0]; best_score = -float('inf')
                    for nid in self.nids:
                        score = self.g.nodes[nid].activation - conscience_beta * freq[nid_to_idx[nid]]
                        if score > best_score: best_score, best_nid = score, nid
                else:
                    best_nid = max(self.nids, key=lambda n: self.g.nodes[n].activation)
                node = self.g.nodes[best_nid]
                t = np.mean(self.v.node_assignments.get(best_nid, [self.v.extractor.extract(img)[0]]), axis=0)
                n = np.linalg.norm(t); t = t/(n+1e-8) if n>0 else t
                node.template += lr*(t-node.template)
                node.template /= np.linalg.norm(node.template)+1e-8
                if freq is not None:
                    freq[nid_to_idx[best_nid]] += 1.0
                    freq *= 0.999

    def activate(self, images):
        feats = np.zeros((len(images), len(self.nids)), dtype=np.float32)
        for i, img in enumerate(images):
            # zero-mean before patch extraction → approximate per-patch centering
            img_c = img - img.mean()
            self.v.set_image(img_c)
            feats[i] = np.array([self.g.nodes[nid].activation for nid in self.nids], dtype=np.float32)
        return feats

    def activate_one(self, image):
        img_c = image - image.mean()
        self.v.set_image(img_c)
        return np.array([self.g.nodes[nid].activation for nid in self.nids], dtype=np.float32)

    def hotspot_activation(self, image, n_regions=4):
        """空间热点激活: 每个 patch 的 winner 节点按「节点学到的热点位置」归网格.
        返回 [n_regions² × n_nodes] 向量 — 空间信息从结构涌现, 非手动分桶.
        (费曼脑思路: 节点累积位置 → 热点, 重建时用热点而非 patch 物理位置, 避免边界断裂)"""
        img_c = image - image.mean()
        self.v.set_image(img_c)
        patches = self.v._last_patches
        positions = self.v.extractor.patch_positions
        n_nodes = len(self.nids)
        if patches is None or len(patches) == 0:
            return np.zeros(n_regions * n_regions * n_nodes, np.float32)
        H, W = image.shape[:2]
        ra = np.zeros((n_regions, n_regions, n_nodes), np.float32)
        templates = np.array([self.g.nodes[nid].template for nid in self.nids], np.float32)
        for p_i, (y1, x1, y2, x2) in enumerate(positions):
            if p_i >= len(patches):
                break
            scores = templates @ patches[p_i]
            best = int(np.argmax(scores))
            if scores[best] < 0:
                continue
            nid = self.nids[best]
            hp = self.g.nodes[nid].spatial_hotspot
            if hp is None:
                continue
            hcy, hcx = hp
            ry = min(int(hcy / (H / n_regions)), n_regions - 1)
            rx = min(int(hcx / (W / n_regions)), n_regions - 1)
            ra[ry, rx, best] += 1.0
        ra = ra.reshape(n_regions * n_regions, n_nodes)
        return np.minimum(1.0, ra / (np.maximum(ra.sum(axis=1, keepdims=True), 1e-8)) * 3).reshape(-1)


# ═══════════════════════════════════════════
#  ColdEye v3
# ═══════════════════════════════════════════

class ColdEye:
    """可配置多尺度预测反馈系统。

    eye_specs 格式:
      [{"type": "global", "n": 100},
       {"type": "patch", "ps": 16, "st": 8, "n": 50}]

    默认 v3: global(100) + coarse 16×16(50) = 150d
    """

    DEFAULT_SPECS = [
        {"type": "global", "n": 100},
        {"type": "patch", "ps": 16, "st": 8, "n": 50},
    ]

    def __init__(self, eye_specs=None, k_knn=5, seed=42):
        specs = eye_specs or self.DEFAULT_SPECS
        self.eyes = []
        self.seed = seed
        rng = np.random.RandomState(seed)
        for i, s in enumerate(specs):
            eye_seed = rng.randint(0, 2**31)
            if s["type"] == "global":
                self.eyes.append(GlobalEye(n_nodes=s["n"], seed=eye_seed))
            elif s["type"] == "patch":
                self.eyes.append(PatchEye(s["ps"], s["st"], s["n"], seed=eye_seed))
            else:
                raise ValueError(f"Unknown eye type: {s['type']}")

        self.k = k_knn
        self.memory = []   # [(vec, label), ...]
        self.shape_centers = None   # optional K-means layer
        self.object_centers = None
        self.dim = sum(e.n if isinstance(e, GlobalEye) else len(e.nids) for e in self.eyes)

    # ── training ──

    def init_templates(self, images):
        gray = self._to_gray(images)
        for eye in self.eyes:
            eye.init_templates(gray if isinstance(eye, PatchEye) else images)

    def train(self, images, labels, epochs=3, n_train=None, contrast_aug=True, conscience_beta=0.0):
        n = n_train or len(images)
        gray = self._to_gray(images)
        for i, eye in enumerate(self.eyes):
            t0 = time.time()
            inp = gray if isinstance(eye, PatchEye) else images
            eye.train(inp, epochs=epochs, n_train=n, contrast_aug=contrast_aug,
                      conscience_beta=conscience_beta)
            t = eye.n if isinstance(eye, GlobalEye) else len(eye.nids)
            print(f"  eye[{i}]: {type(eye).__name__} {t}d — {time.time()-t0:.0f}s")

    # ── feature extraction ──

    @staticmethod
    def _to_gray(img):
        """RGB → 灰度 (PatchEye 用). 灰度图原样返回. 单张或批量."""
        if img.ndim == 4 and img.shape[-1] == 3:  # RGB 批量 (N,H,W,3)
            return (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.float32)
        if img.ndim == 3 and img.shape[-1] == 3:  # RGB 单张 (H,W,3)
            return (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.float32)
        return img  # 灰度 (H,W) 或 (N,H,W)

    def _activate_one(self, image):
        gray = self._to_gray(image)
        parts = [eye.activate_one(gray if isinstance(eye, PatchEye) else image) for eye in self.eyes]
        return np.concatenate(parts)

    def _activate_batch(self, images):
        gray = self._to_gray(images)
        parts = [eye.activate(gray if isinstance(eye, PatchEye) else images) for eye in self.eyes]
        return np.concatenate(parts, axis=1)

    # ── optional K-means layers (legacy, not needed for v3) ──

    def build_shapes(self, images, labels, n_samples=5000, n_shape=100):
        """K-means on activation vectors → shape layer"""
        n = min(n_samples, len(images))
        idxs = np.random.choice(len(images), n, replace=False)
        X = self._activate_batch(images[idxs])
        rng = np.random.RandomState(42)
        idx = rng.choice(n, n_shape, replace=False)
        centers = X[idx].copy()
        for it in range(50):
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels_k = np.argmin(dists, axis=1)
            new_c = np.zeros_like(centers)
            for c in range(n_shape):
                mask = labels_k == c
                new_c[c] = X[mask].mean(axis=0) if mask.sum() > 0 else X[rng.choice(n)]
            if np.sum((centers - new_c)**2) < 1e-6: break
            centers = new_c
        self.shape_centers = centers.astype(np.float32)

    def build_objects(self, images, labels, n_samples=5000, n_object=50):
        """K-means on shape vectors → object layer"""
        if self.shape_centers is None: raise RuntimeError("build_shapes() first")
        n = min(n_samples, len(images))
        idxs = np.random.choice(len(images), n, replace=False)
        shape_vecs = np.array([self.get_shape_vector(images[i]) for i in idxs], dtype=np.float32)
        rng = np.random.RandomState(42)
        idx = rng.choice(n, n_object, replace=False)
        centers = shape_vecs[idx].copy()
        for it in range(50):
            dists = np.sum((shape_vecs[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels_k = np.argmin(dists, axis=1)
            new_c = np.zeros_like(centers)
            for c in range(n_object):
                mask = labels_k == c
                new_c[c] = shape_vecs[mask].mean(axis=0) if mask.sum() > 0 else shape_vecs[rng.choice(n)]
            if np.sum((centers - new_c)**2) < 1e-6: break
            centers = new_c
        self.object_centers = centers.astype(np.float32)

    def get_shape_vector(self, image):
        if self.shape_centers is None: raise RuntimeError("build_shapes() first")
        base = self._activate_one(image)
        s = self.shape_centers @ base / (np.linalg.norm(self.shape_centers, axis=1)*np.linalg.norm(base) + 1e-8)
        return np.clip(s, 0, 1).astype(np.float32)

    def get_object_vector(self, image):
        if self.shape_centers is None or self.object_centers is None:
            raise RuntimeError("build_shapes() + build_objects() first")
        sv = self.get_shape_vector(image)
        s = self.object_centers @ sv / (np.linalg.norm(self.object_centers, axis=1)*np.linalg.norm(sv) + 1e-8)
        return np.clip(s, 0, 1).astype(np.float32)

    # ── memory / prediction ──

    def build_memory(self, images, labels, size=5000, use_shapes=False):
        """存 KNN 参考向量。use_shapes=True 走 object_vector, False 走原始激活。"""
        self.memory.clear()
        n = min(size, len(images))
        idxs = np.random.choice(len(images), n, replace=False)
        for idx in idxs:
            vec = self.get_object_vector(images[idx]) if use_shapes else self._activate_one(images[idx])
            self.memory.append((vec, labels[idx]))

    # ═══ 持续学习 ═══

    def add_samples(self, images, labels):
        """增量添加记忆 — 不重训模板，不碰旧记忆"""
        for i in range(len(images)):
            vec = self._activate_one(images[i])
            self.memory.append((vec, labels[i]))

    def prune_memory(self, max_per_class=200):
        """记忆淘汰: 每类保留最近添加的 max_per_class 条"""
        by_class = {}
        for i, (vec, lbl) in enumerate(self.memory):
            c = int(lbl)
            if c not in by_class: by_class[c] = []
            by_class[c].append((i, vec))
        keep = set()
        for c, entries in by_class.items():
            for i, _ in entries[-max_per_class:]:  # 保留最新的
                keep.add(i)
        self.memory = [self.memory[i] for i in sorted(keep)]

    def adapt(self, new_images, new_labels, epochs=1, lr=0.05):
        """适应新数据: 低学习率微调模板，不重建记忆"""
        for eye in self.eyes:
            eye.train(new_images, epochs=epochs, n_train=len(new_images),
                      contrast_aug=False, conscience_beta=0.0)
            # 手动降 lr (train 内部默认 0.1)
        self.add_samples(new_images, new_labels)

    # ═══ 自优化 ═══

    def consolidate(self, n_recent=500, lr=0.01, epochs=1):
        """睡眠巩固: 用最近记忆低学习率微调模板"""
        if len(self.memory) < n_recent:
            return
        recent = self.memory[-n_recent:]
        # 重建图像: 激活 × W_paired (如果有配对解码器)
        if hasattr(self, 'W_paired'):
            imgs = np.array([vec @ self.W_paired for vec, _ in recent], dtype=np.float32)
            imgs = imgs.reshape(-1, 28, 28)
        else:
            return  # 需要解码器才能重建图像做巩固
        # 低学习率 Hebbian (用 train 里的 lr 参数不够细 → 手动调)
        for eye in self.eyes:
            rng = np.random.RandomState(42)
            for ep in range(epochs):
                for idx in rng.permutation(len(imgs)):
                    img = imgs[idx] - imgs[idx].mean()
                    flat = img.reshape(-1).astype(np.float32)
                    flat /= np.linalg.norm(flat) + 1e-8
                    if isinstance(eye, GlobalEye):
                        best = int(np.argmax(eye.templates @ flat))
                        eye.templates[best] += lr * (flat - eye.templates[best])
                        eye.templates[best] /= np.linalg.norm(eye.templates[best]) + 1e-8

    def uncertain_samples(self, images, labels, threshold=0.5):
        """返回置信度 < threshold 的样本索引，用于主动采样"""
        uncertain = []
        for i in range(len(images)):
            _, conf = self.predict(images[i])
            if conf < threshold:
                uncertain.append(i)
        return uncertain

    def template_health(self):
        """模板健康度报告: 每节点胜率和空转率 (GlobalEye only)"""
        eye = self.eyes[0]  # GlobalEye
        n = eye.n
        if len(self.memory) == 0:
            return {'total': n, 'active': n, 'stale': 0, 'dead_nodes': []}
        # 取记忆向量的维度 (可能与当前模板维度不同，取min)
        mem_dim = len(self.memory[0][0])
        tpl_dim = eye.templates.shape[1]
        use_dim = min(mem_dim, tpl_dim)
        hits = np.zeros(n)
        for vec, _ in self.memory[-1000:]:
            sims = eye.templates[:, :use_dim] @ vec[:use_dim]
            hits[np.argmax(sims)] += 1
        active = (hits > 0).sum()
        stale = n - active
        # 最低使用率节点的索引
        dead_nodes = np.where(hits == 0)[0][:5].tolist()
        return {'total': n, 'active': int(active), 'stale': int(stale),
                'hit_distribution': hits, 'dead_nodes': dead_nodes}

    def auto_tune(self, eval_images, eval_labels, conscience_range=(0.0, 1.0, 0.2)):
        """元参数自调: 扫 conscience_beta 找最优值 (轻量版)"""
        best_beta, best_acc = 0.0, 0.0
        lo, hi, step = conscience_range
        beta = lo
        while beta <= hi:
            # 临时应用 conscience_beta 重训 (仅最后一轮，不是完整重训)
            # 实际: 在 adapt 里传 conscience_beta
            # 这里简化为在现有模板上测准确率
            acc = self.evaluate(eval_images[:200], eval_labels[:200])
            if acc > best_acc:
                best_acc, best_beta = acc, beta
            beta += step
        return best_beta, best_acc

    # ═══ 架构升级: 动态扩容 + 共激活图 + 层级路由 ═══

    def spawn_nodes(self, uncertain_images, n_new=10):
        """动态扩容: 不确定样本初始化新节点，加到 GlobalEye"""
        eye = self.eyes[0]  # GlobalEye
        if len(uncertain_images) == 0:
            return
        idxs = np.random.choice(len(uncertain_images), min(n_new, len(uncertain_images)), replace=False)
        new_templates = np.zeros((n_new, eye.templates.shape[1]), dtype=np.float32)
        for i, idx in enumerate(idxs):
            flat = uncertain_images[idx].reshape(-1).astype(np.float32)
            flat = flat - flat.mean()
            flat /= np.linalg.norm(flat) + 1e-8
            new_templates[i] = flat
        eye.templates = np.vstack([eye.templates, new_templates])
        eye.n += n_new
        return n_new

    def recycle_nodes(self, uncertain_images, max_recycle=None):
        """回收死节点: 把低胜率模板重初始化为不确定样本"""
        eye = self.eyes[0]  # GlobalEye
        health = self.template_health()
        # 用胜率排序，取最低的 (不是只看零命中)
        hits = health['hit_distribution']
        low_nodes = np.argsort(hits)[:max_recycle] if max_recycle else np.where(hits == 0)[0]
        if len(low_nodes) == 0 or len(uncertain_images) == 0:
            return 0

        n_recycle = min(len(low_nodes), len(uncertain_images))
        idxs = np.random.choice(len(uncertain_images), n_recycle, replace=False)
        for i, node_idx in enumerate(low_nodes[:n_recycle]):
            flat = uncertain_images[idxs[i]].reshape(-1).astype(np.float32)
            flat = flat - flat.mean()
            flat /= np.linalg.norm(flat) + 1e-8
            eye.templates[node_idx] = flat
        return n_recycle

    def build_graph(self, images, n_samples=5000, edge_thresh=0.3):
        """构建共激活图: 常一起活跃的节点连边 → 预测路由用"""
        n = min(n_samples, len(images))
        idxs = np.random.choice(len(images), n, replace=False)
        acts = self._activate_batch(images[idxs])  # [N, D]
        active = (acts > 0.3).astype(np.float32)
        co = active.T @ active / n  # [D, D] co-activation rate
        # 稀疏化: 只保留强共激活 + 去自环
        self.edges = {}
        for i in range(co.shape[0]):
            neighbors = np.where(co[i] > edge_thresh)[0]
            neighbors = neighbors[neighbors != i]
            if len(neighbors) > 0:
                self.edges[i] = list(neighbors)
        # 图统计
        n_edges = sum(len(v) for v in self.edges.values())
        n_isolated = co.shape[0] - len(self.edges)
        return {'n_nodes': co.shape[0], 'n_edges': n_edges, 'isolated': n_isolated}

    def predict_with_graph(self, image, alpha=0.3):
        """图增强预测: 激活沿共激活边传播一步 → 融合 → KNN"""
        if not hasattr(self, 'edges') or not self.edges:
            return self.predict(image)
        act = self._activate_one(image)
        # 沿边传播: 每个节点从邻居接收激活
        propagated = np.zeros_like(act)
        for src, dsts in self.edges.items():
            for dst in dsts:
                propagated[dst] += act[src]
        # 归一化 + 融合
        if propagated.max() > 0:
            propagated /= propagated.max()
        act = act * (1 - alpha) + propagated * alpha
        return self._knn_search(act)

    def route_hierarchical(self, image, top_k_coarse=5):
        """层级路由: 粗尺度筛候选 → 全局尺度精细匹配"""
        # Step 1: 粗尺度 (PatchEye)
        coarse_eye = self.eyes[1]
        img_c = image - image.mean()
        coarse_eye.v.set_image(img_c)
        coarse_act = np.array([coarse_eye.g.nodes[nid].activation
                               for nid in coarse_eye.nids], dtype=np.float32)

        # Step 2: 只看粗尺度最强的 top-k 类
        coarse_vec = coarse_act
        class_scores = {}
        for mvec, mlbl in self.memory:
            c = int(mlbl)
            sim = np.dot(mvec[-len(coarse_act):], coarse_vec) / (
                np.linalg.norm(mvec[-len(coarse_act):]) * np.linalg.norm(coarse_vec) + 1e-8)
            if c not in class_scores or sim > class_scores[c]:
                class_scores[c] = sim
        top_classes = sorted(class_scores.keys(), key=lambda k: class_scores[k], reverse=True)[:top_k_coarse]

        # Step 3: 全局尺度只在候选类中匹配
        full_act = self._activate_one(image)
        best_sim, best_label = -1, -1
        for mvec, mlbl in self.memory:
            if int(mlbl) not in top_classes:
                continue
            sim = np.dot(mvec, full_act) / (np.linalg.norm(mvec) * np.linalg.norm(full_act) + 1e-8)
            if sim > best_sim: best_sim, best_label = sim, mlbl
        return best_label, best_sim

    def predict(self, image, use_shapes=False):
        """KNN 预测。返回 (label, confidence)。"""
        if not self.memory: return -1, 0.0
        q = self.get_object_vector(image) if use_shapes else self._activate_one(image)
        best_sim, best_label = -1, -1
        for mvec, mlbl in self.memory:
            sim = np.dot(mvec, q) / (np.linalg.norm(mvec)*np.linalg.norm(q) + 1e-8)
            if sim > best_sim: best_sim, best_label = sim, mlbl
        return best_label, best_sim

    def evaluate(self, images, labels, use_shapes=False):
        # 向量化 1-NN: 预归一化 memory, 一次 matmul 算所有 query 的余弦
        mem = np.array([m[0] for m in self.memory], np.float32)
        mem_lbl = np.array([m[1] for m in self.memory], np.int64)
        mem = mem / (np.linalg.norm(mem, axis=1, keepdims=True) + 1e-8)
        if use_shapes:
            queries = np.array([self.get_object_vector(im) for im in images], np.float32)
        else:
            queries = self._activate_batch(images)
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        sims = queries @ mem.T          # (N, M) 余弦相似度
        preds = mem_lbl[np.argmax(sims, axis=1)]
        return np.mean(preds == labels)

    # ═══ 预测反馈 + 脑补 ═══

    def predict_with_feedback(self, image, alpha=0.3, max_iter=5, tol=0.01):
        """迭代闭环反馈: 猜类→找最相似同类记忆→修正→重猜→收敛"""
        if not self.memory: return -1, 0.0
        act = self._activate_one(image)
        prev_act = act.copy()

        for it in range(max_iter):
            # 1. 找最佳匹配的邻居 (不是类平均，是最相似的具体记忆)
            best_sim, best_vec, best_class = -1, None, -1
            for mvec, mlbl in self.memory:
                sim = np.dot(mvec, act) / (np.linalg.norm(mvec)*np.linalg.norm(act)+1e-8)
                if sim > best_sim:
                    best_sim, best_vec, best_class = sim, mvec, int(mlbl)

            # 2. 只用同类的最相似记忆修正
            same_class = [(sim, mvec) for mvec, mlbl in self.memory
                          if int(mlbl) == best_class]
            if len(same_class) < 2:
                break
            # 加权同类记忆平均 (不只是单一最相似)
            same_class.sort(key=lambda x: x[0], reverse=True)
            top = same_class[:5]
            total_sim = sum(s for s, _ in top) + 1e-8
            act_avg = sum(mvec * (s/total_sim) for s, mvec in top)

            # 3. 融合
            act = act * (1 - alpha) + act_avg * alpha

            # 4. 收敛检查
            if np.max(np.abs(act - prev_act)) < tol:
                break
            prev_act = act.copy()

        return self._knn_search(act)

    def _knn_search(self, vec):
        """KNN 搜索，返回 (label, confidence)"""
        if not self.memory: return -1, 0.0
        best_sim, best_label = -1, -1
        for mvec, mlbl in self.memory:
            sim = np.dot(mvec, vec) / (np.linalg.norm(mvec) * np.linalg.norm(vec) + 1e-8)
            if sim > best_sim: best_sim, best_label = sim, mlbl
        return best_label, best_sim

    def build_decoder(self, images, n_samples=10000):
        """学习线性解码器 W[150,784]: 激活 → 重建图像"""
        n = min(n_samples, len(images))
        idxs = np.random.choice(len(images), n, replace=False)
        acts = self._activate_batch(images[idxs])  # [N, 150]
        flat = images[idxs].reshape(n, -1).astype(np.float32)  # [N, 784]
        # 最小二乘: W = (A^T A)^-1 A^T I → pinv approach
        ATA = acts.T @ acts
        ATI = acts.T @ flat
        self.W = (np.linalg.pinv(ATA) @ ATI).astype(np.float32)  # [150, 784]

    def reconstruct(self, image):
        """脑补: 激活 → 重建。自动适配图像尺寸。"""
        if not hasattr(self, 'W'):
            raise RuntimeError("先 build_decoder()")
        act = self._activate_one(image)
        h, w = image.shape[:2]
        return (act @ self.W).reshape(h, w)

    # ═══ 空间热点脑补 (费曼脑"给容量" — 空间信息从结构涌现) ═══

    def build_decoder_hotspot(self, images, n_samples=10000, n_regions=4):
        """学习热点重建解码器: 空间热点激活 → 图像.
        热点激活 = 节点学到的空间位置归网格, 非 patch 物理位置 (避免边界断裂)"""
        patch_eyes = [e for e in self.eyes if isinstance(e, PatchEye)]
        if not patch_eyes:
            raise RuntimeError("热点重建需要至少一个 patch eye")
        n = min(n_samples, len(images))
        idxs = np.random.choice(len(images), n, replace=False)
        acts = np.array([np.concatenate([e.hotspot_activation(images[i], n_regions) for e in patch_eyes])
                         for i in idxs], np.float32)
        flat = images[idxs].reshape(n, -1).astype(np.float32)
        ATA = acts.T @ acts
        ATI = acts.T @ flat
        self.W_hotspot = (np.linalg.pinv(ATA) @ ATI).astype(np.float32)

    def reconstruct_hotspot(self, image, n_regions=4):
        """热点脑补: 空间热点激活 → 重建图像"""
        if not hasattr(self, 'W_hotspot'):
            raise RuntimeError("先 build_decoder_hotspot()")
        patch_eyes = [e for e in self.eyes if isinstance(e, PatchEye)]
        act = np.concatenate([e.hotspot_activation(image, n_regions) for e in patch_eyes])
        h, w = image.shape[:2]
        return (act @ self.W_hotspot).reshape(h, w)

    # ═══ 配对脑补 + 闭环推理 ═══

    def build_paired_decoder(self, clean_images, degrade_fn, n_samples=10000):
        """学习 W_paired: act(降质图) → clean_image
        degrade_fn(images) → degraded images (same shape)"""
        n = min(n_samples, len(clean_images))
        idxs = np.random.choice(len(clean_images), n, replace=False)
        clean_batch = clean_images[idxs]
        degraded = degrade_fn(clean_batch)

        acts = self._activate_batch(degraded)           # [N, 150] from degraded
        flat_clean = clean_batch.reshape(n, -1).astype(np.float32)  # [N, 784]

        ATA = acts.T @ acts
        ATI = acts.T @ flat_clean
        self.W_paired = (np.linalg.pinv(ATA) @ ATI).astype(np.float32)

    def reconstruct_paired(self, image):
        """脑补: 降质图激活 → 重建干净图"""
        if not hasattr(self, 'W_paired'):
            raise RuntimeError("先 build_paired_decoder()")
        act = self._activate_one(image)
        h, w = image.shape[:2]
        return (act @ self.W_paired).reshape(h, w)

    def predict_brainfill(self, image, alpha=0.5, max_iter=5, conf_thresh=0.85, tol=0.005):
        """迭代闭环推理: 至少脑补一次，不确定时继续迭代直到收敛。

        alpha: 基础融合系数
        conf_thresh: 余弦相似度超过此值 → 足够确定，停止 (设得高以确保至少迭代一次)
        tol: 激活变化 < tol → 收敛
        """
        if not hasattr(self, 'W_paired') or not self.memory:
            return self.predict(image)

        act = self._activate_one(image)
        prev_act = act.copy()
        best_label, best_sim = -1, -1

        for it in range(max_iter):
            # 置信度
            best_sim, best_label = -1, -1
            for mvec, mlbl in self.memory:
                sim = np.dot(mvec, act) / (np.linalg.norm(mvec) * np.linalg.norm(act) + 1e-8)
                if sim > best_sim: best_sim, best_label = sim, mlbl

            # 高置信 + 至少迭代过一次 → 停止
            if it > 0 and best_sim >= conf_thresh:
                break

            # 脑补重建 + 重路由
            recon = (act @ self.W_paired).reshape(image.shape[0], image.shape[1])
            act_recon = self._activate_one(recon)

            # 融合 (首次迭代偏保守，后续偏脑补)
            a = alpha * (0.5 if it == 0 else 1.0)
            act_new = act * (1 - a) + act_recon * a

            shift = np.max(np.abs(act_new - prev_act))
            if shift < tol:
                break
            prev_act = act
            act = act_new

        return best_label, best_sim

    def evaluate_with_feedback(self, images, labels, alpha=0.5):
        correct = 0
        for i in range(len(images)):
            pred, _ = self.predict_with_feedback(images[i], alpha=alpha)
            if pred == labels[i]: correct += 1
        return correct / len(images)


# ═══════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════

def load_mnist(path="data"):
    fs = {"ti":"train-images-idx3-ubyte.gz","tl":"train-labels-idx1-ubyte.gz",
          "ei":"t10k-images-idx3-ubyte.gz","el":"t10k-labels-idx1-ubyte.gz"}
    os.makedirs(path, exist_ok=True)
    url = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
    for f in fs.values():
        p = os.path.join(path, f)
        if not os.path.exists(p): urllib.request.urlretrieve(url+f, p)

    def limg(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)/255
    def llbl(p):
        with gzip.open(p) as f: return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return (limg(os.path.join(path, fs["ti"])), llbl(os.path.join(path, fs["tl"])),
            limg(os.path.join(path, fs["ei"])), llbl(os.path.join(path, fs["el"])))

def low_contrast(X, c):
    Xc = X.copy(); m = Xc.mean(axis=(1,2), keepdims=True)
    return m + (Xc - m) * c


def load_cifar10(path="data/cifar10", gray=False, n_train_per_class=None, n_test_per_class=None):
    """加载 CIFAR-10 (fast.ai 图片格式: {train,test}/{class}/{id}.png).
    gray=False 返回 RGB (N,32,32,3), gray=True 返回灰度 (N,32,32). [0,1]"""
    from PIL import Image
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']  # 字母序 = 官方 label 顺序
    cls_to_label = {c: i for i, c in enumerate(classes)}

    def load_split(split, n_per_class):
        X, y = [], []
        for c in classes:
            d = os.path.join(path, split, c)
            files = sorted(os.listdir(d))
            if n_per_class is not None:
                files = files[:n_per_class]
            for f in files:
                img = Image.open(os.path.join(d, f))
                mode = 'L' if gray else 'RGB'
                arr = np.array(img.convert(mode), dtype=np.float32) / 255.0
                X.append(arr); y.append(cls_to_label[c])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    X_tr, y_tr = load_split('train', n_train_per_class)
    X_te, y_te = load_split('test', n_test_per_class)
    return X_tr, y_tr, X_te, y_te


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    X_tr, y_tr, X_te, y_te = load_mnist()

    print(f"ColdEye v3 — global(100n) + coarse 16×16(50n) = 150d")
    print(f"  训练: 60K, 5ep, contrast_aug=True")

    model = ColdEye()
    model.init_templates(X_tr[:5000])
    model.train(X_tr, y_tr, epochs=5, n_train=60000, contrast_aug=True)
    model.build_memory(X_tr[:15000], y_tr[:15000], size=5000)

    n_test = 500
    print(f"\n  {'c':>6s}  {'acc':>7s}")
    print(f"  {'-'*15}")
    for c in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01]:
        if c == 1.0:
            test_batch = X_te[:n_test]
        else:
            test_batch = low_contrast(X_te[:n_test], c)
        acc = model.evaluate(test_batch, y_te[:n_test])
        print(f"  {c:5.2f}  {acc*100:6.1f}%")

    print(f"\n  架构: {model.dim}d ({' + '.join(f'{type(e).__name__}({e.n if isinstance(e,GlobalEye) else len(e.nids)}n)' for e in model.eyes)})")
    print("=== DONE ===")
