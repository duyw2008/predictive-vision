#!/usr/bin/env python3
"""
冷眼 — 预测反馈视觉系统
train.py: 训练循环

跟费曼脑的 breathe() 一样 — 逐样本迭代，不搞 batch
学习: Hebbian (共同激活→强化模板), 非 BP
"""

import sys
import os
import json
import time
import numpy as np
from collections import defaultdict
from typing import Optional, Callable, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from graph import VisionGraph, VisionNode
from synapse import SynapticLayer
from vision import VisionInterface, PatchExtractor


class ColdEye:
    """冷眼 — 预测反馈视觉系统

    核心机制:
      - 对比学习: 同类 attract / 异类 repel (生态位分化)
      - 锚点涌现: coincidence → tier 晋升 → 自动识别类锚点
    """

    def __init__(self, n_nodes: int = 200):
        self.graph = VisionGraph(n_nodes=n_nodes)
        self.synapse = SynapticLayer()
        self.graph.synapse = self.synapse
        self.synapse.update_context(n_nodes, self.graph)

        self.vision = VisionInterface(self.graph, top_k=1)  # hard assignment

        # 训练状态
        self.generation = 0
        self.total_samples = 0

        # 分类: 涌现的类锚点 {label: node_id}
        self.class_anchors: dict = {}

        # 对比学习
        self._class_recent_nodes: dict = defaultdict(set)
        self._recent_samples: list = []
        self._node_class_votes: dict = defaultdict(lambda: defaultdict(int))

    def init_templates_from_data(self, images: np.ndarray):
        """用真实图像 patches 初始化节点模板 (不用随机)"""
        extractor = self.vision.extractor
        node_ids = sorted(self.graph.nodes.keys())

        # 随机选一些图像, 提取 patches 分配给节点
        n_samples = min(len(images), 50)
        sample_indices = np.random.choice(len(images), n_samples, replace=False)

        all_patches = []
        for idx in sample_indices:
            patches = extractor.extract(images[idx])
            if len(patches) > 0:
                all_patches.extend(patches)

        if not all_patches:
            return

        for i, nid in enumerate(node_ids):
            idx = i % len(all_patches)
            patch = all_patches[idx]
            norm = np.linalg.norm(patch)
            if norm > 0:
                self.graph.nodes[nid].template = patch.astype(np.float32) / norm
            # 加一点噪声避免完全一样
            self.graph.nodes[nid].template += np.random.randn(
                len(self.graph.nodes[nid].template)
            ).astype(np.float32) * 0.01
            self.graph.nodes[nid].template /= np.linalg.norm(
                self.graph.nodes[nid].template
            ) + 1e-8

    def augment(self, image: np.ndarray, n_variants: int = 4,
                min_contrast: float = 0.3) -> list:
        """数据增强: 每个样本生成多个变体 (对比度 + 平移)

        返回: [(augmented_image, contrast_level), ...]
        """
        variants = [(image.copy(), 1.0)]

        H, W = image.shape
        for _ in range(n_variants - 1):
            contrast = np.random.uniform(min_contrast, 1.0)
            img = image.copy()
            mean = img.mean()
            img = mean + (img - mean) * contrast

            # 小幅度随机平移
            dx, dy = np.random.randint(-2, 3, size=2)
            shifted = np.zeros_like(img)
            shifted[max(0, dy):min(H, H+dy), max(0, dx):min(W, W+dx)] = \
                img[max(0, -dy):min(H, H-dy), max(0, -dx):min(W, W-dx)]
            variants.append((shifted, contrast))

        return variants

    def train_step(self, image: np.ndarray, label: int, contrast: float = 1.0):
        """单样本训练 — 对比学习 + 预测填充

        1. 数据增强 → 多变体输入 (contrast 控制增强对比度下限)
        2. 同类 attract: 同类样本 → 强化共享节点 & 边
        3. 异类 repel: 随机抽异类样本 → 抑制共享节点
        4. 预测传播 + Hebbian 模板学习
        5. 锚点晋升
        """
        self.generation += 1
        self.total_samples += 1

        # 存入缓冲池
        self._recent_samples.append((image.copy(), label))
        if len(self._recent_samples) > 500:
            self._recent_samples.pop(0)

        # 数据增强: 生成变体 (contrast 控制最低对比度)
        variants = self.augment(image, n_variants=3, min_contrast=max(0.1, contrast))

        # ── 1. 同类 attract (在所有变体上) ──
        for aug_img, aug_contrast in variants:
            # aug_img 已经降了对比度, 直接如实输入
            self._process_image(aug_img, label, 1.0,
                                attract=True, repel=False)

        # ── 2. 异类 repel (随机抽一个异类样本) ──
        if len(self._recent_samples) > 10:
            other = self._recent_samples[
                np.random.randint(len(self._recent_samples))]
            other_img, other_label = other
            if other_label != label:
                self._process_image(other_img, other_label, 1.0,
                                    attract=False, repel=True)

        # ── 3. 更新节点 tier ──
        self._promote_nodes()

        # ── 4. 周期性维护 ──
        if self.generation % 10 == 0:
            n_elim = self.synapse.decay(self.generation)
            self.synapse.promote_coincident_edges(self.graph, self.generation)
            self._update_class_anchors()

    def _process_image(self, image: np.ndarray, label: int,
                       contrast: float, attract: bool, repel: bool):
        """处理单张图像: 竞争路由激活 → 学习"""
        self.vision.set_image(image, contrast=contrast)

        # Hebbian 学习 (带对比信号)
        self._contrastive_hebbian(image, label, attract, repel)

        # 清理
        for node in self.graph.nodes.values():
            node.predicted_by.clear()
            node.predicts_to.clear()

    def _contrastive_hebbian(self, image: np.ndarray, label: int,
                              attract: bool, repel: bool):
        """对比 Hebbian 学习 (竞争路由版)

        attract: 节点模板 → 向其匹配的 patches 移动 (K-means 风格)
        repel:   异类样本 → 轻微推远被误激活的节点
        """
        node_ids = sorted(self.graph.nodes.keys())

        if attract:
            lr = 0.1  # 提高学习率 (竞争路由下更稳定)

            for nid, patches in self.vision.node_assignments.items():
                node = self.graph.nodes[nid]
                if node.frozen or not patches:
                    continue

                # 模板 → 匹配 patches 的平均方向
                target = np.mean(patches, axis=0)
                norm = np.linalg.norm(target)
                if norm > 0:
                    target = target / norm

                # 移动模板
                node.template += lr * (target - node.template)
                node.template /= np.linalg.norm(node.template) + 1e-8

                # 累积 reward (被匹配 = 生态位有效)
                node.reward += 0.02
                node.last_active_gen = self.generation
                node.generation = self.generation

                # 记录类投票
                self._node_class_votes[nid][label] += 1

            # 建立预测边: 只连最 co-activated 的 top-5 对
            active_nodes = [nid for nid in node_ids
                           if self.graph.nodes[nid].activation > 0.2]
            pairs = []
            for i, nid_src in enumerate(active_nodes):
                src_act = self.graph.nodes[nid_src].activation
                for nid_dst in active_nodes[i+1:]:
                    pairs.append((src_act * self.graph.nodes[nid_dst].activation,
                                 nid_src, nid_dst))
            pairs.sort(reverse=True)
            for strength, nid_src, nid_dst in pairs[:5]:  # 只 top 5
                if strength > 0.1:
                    self.synapse.fire(
                        nid_src, nid_dst,
                        strength=strength * 0.1,
                        sample_id=f"cls{label}:{self.generation}",
                        generation=self.generation,
                    )

        if repel:
            # 异类样本: 被激活但属于其他类的节点 → 轻微惩罚
            for nid in node_ids:
                node = self.graph.nodes[nid]
                if node.frozen or node.activation < 0.3:
                    continue
                if node.domain_tag and node.domain_tag != str(label):
                    node.reward -= 0.01

        # 更新 adjacency
        for (src, dst), edge in self.synapse.activations.items():
            if src in self.graph.adjacency:
                self.graph.adjacency[src][dst] = edge['s']

    def _promote_nodes(self):
        """节点晋升: reward 累积 → tier 下降 (冻结节点不变)"""
        for node in self.graph.nodes.values():
            if node.frozen:
                continue
            if node.reward >= 50 and node.tier > 0:
                node.tier = 0  # 核心锚点
            elif node.reward >= 25 and node.tier > 1:
                node.tier = 1
            elif node.reward >= 12 and node.tier > 2:
                node.tier = 2
            elif node.reward >= 5 and node.tier > 3:
                node.tier = 3

        # 每 50 代: 用多数投票更新 domain_tag
        if self.generation % 50 == 0:
            for nid, votes in self._node_class_votes.items():
                if not votes:
                    continue
                best_label = max(votes, key=votes.get)
                best_count = votes[best_label]
                total = sum(votes.values())
                # 相对多数即可 (不要求绝对多数, 只要有足够样本)
                if total >= 5:
                    self.graph.nodes[nid].domain_tag = str(best_label)

    def _update_class_anchors(self):
        """锚点涌现: 每类找到 tier≤1 + 最高 reward 的节点作为锚点"""
        class_best: dict = {}  # {label: (nid, reward)}

        for nid, node in self.graph.nodes.items():
            if node.tier > 1:
                continue
            tag = node.domain_tag
            if tag is None:
                continue
            try:
                label = int(tag)
            except ValueError:
                continue
            if label not in class_best or class_best[label][1] < node.reward:
                class_best[label] = (nid, node.reward)

        for label, (nid, reward) in class_best.items():
            if label not in self.class_anchors:
                self.class_anchors[label] = nid
            else:
                old = self.class_anchors[label]
                old_node = self.graph.nodes.get(old)
                if old_node is None or old_node.reward < reward:
                    self.class_anchors[label] = nid

    def _get_best_prediction(self) -> Tuple[int, float]:
        """加权投票: 每个节点投票权重 = 匹配分值 × 类专一度

        通用节点(匹配所有类)投票权低, 专有节点(只匹配某类)投票权高
        """
        # 收集所有有 domain_tag 且激活的节点
        votes = defaultdict(float)
        total_weight = 0.0

        for nid, node in self.graph.nodes.items():
            if node.activation < 0.05 or not node.domain_tag:
                continue
            try:
                label = int(node.domain_tag)
            except ValueError:
                continue

            # 类专一度: 该类票数 / 总票数
            class_votes = self._node_class_votes.get(nid, {})
            total_votes = sum(class_votes.values())
            if total_votes == 0:
                purity = 0.5  # 未知
            else:
                purity = class_votes.get(label, 0) / total_votes

            # 权重 = 激活值 × 专一度
            weight = node.activation * purity
            votes[label] += weight
            total_weight += weight

        if total_weight > 0 and votes:
            best_label = max(votes, key=votes.get)
            return best_label, votes[best_label] / total_weight

        # 回退: class_anchors
        if self.class_anchors:
            best_label = max(self.class_anchors.keys(),
                           key=lambda l: self.graph.nodes[self.class_anchors[l]].activation)
            return best_label, self.graph.nodes[self.class_anchors[best_label]].activation

        return -1, 0.0

    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """推理: 返回预测类别和置信度"""
        self.vision.set_image(image, contrast=1.0)

        # 迭代推理
        for _ in range(20):
            errors = self.graph.propagate_predictions()
            max_err = max(errors.values()) if errors else 0
            if max_err < 0.001:
                break

        return self._get_best_prediction()

    def stats(self) -> dict:
        """返回当前状态"""
        tier_dist = self.synapse.tier_distribution
        node_tiers = defaultdict(int)
        for n in self.graph.nodes.values():
            node_tiers[n.tier] += 1

        frozen_count = sum(1 for n in self.graph.nodes.values() if n.frozen)

        return {
            "generation": self.generation,
            "nodes": len(self.graph.nodes),
            "synapse_edges": self.synapse.total_edges,
            "graph_edges": self.graph.edge_count,
            "tier_dist": tier_dist,
            "node_tiers": dict(node_tiers),
            "class_anchors": len(self.class_anchors),
            "frozen": frozen_count,
        }

    # ═══════════════════════════════════════════════════════════════
    #  两阶段训练
    # ═══════════════════════════════════════════════════════════════

    def freeze_low_tier(self, max_tier: int = 1, min_reward: float = 30):
        """冻结可靠底层节点: 预训练形成的特征不再修改"""
        n = 0
        for node in self.graph.nodes.values():
            if node.tier <= max_tier and node.reward >= min_reward:
                node.frozen = True
                n += 1
        print(f"  🧊 冻结 {n} 个底层节点 (tier≤{max_tier}, reward≥{min_reward})")
        return n

    def unfreeze_all(self):
        """解冻所有节点 (用于重新训练)"""
        for node in self.graph.nodes.values():
            node.frozen = False

    def pretrain_visual_cortex(
        self, images: np.ndarray, labels: np.ndarray,
        n_epochs: int = 5, contrast_schedule=None,
    ):
        """阶段1: 系统发生 — 全量数据形成底层拓扑

        大量数据 → 节点生态位分化 → 涌现边缘/角/纹理检测器
        完成后冻结低tier节点作为视觉皮层
        """
        print(f"\n{'='*50}")
        print(f"🧠 阶段1: 视觉皮层预训练")
        print(f"   {len(images)} 样本, {n_epochs} epochs")
        print(f"{'='*50}")

        train_mnist(self, images, labels, n_epochs=n_epochs,
                    contrast_schedule=contrast_schedule)

        # 冻结底层
        n_frozen = self.freeze_low_tier(max_tier=1, min_reward=30)
        st = self.stats()
        print(f"\n📊 预训练后: 节点tier分布={st['node_tiers']}, "
              f"边tier分布={st['tier_dist']}, 冻结={n_frozen}")
        return n_frozen

    def few_shot_adapt(
        self, images: np.ndarray, labels: np.ndarray,
        n_epochs: int = 10, contrast_schedule=None,
    ):
        """阶段2: 个体发生 — 小样本只在冻结底层上建新锚点

        底层节点已冻结 → 只学习高层锚点 + 新连接
        """
        n_samples = len(images)
        n_classes = len(set(labels))
        print(f"\n{'='*50}")
        print(f"🎯 阶段2: 小样本适应")
        print(f"   {n_samples} 样本 ({n_classes} 类), {n_epochs} epochs")
        print(f"   底层已冻结, 只建高层锚点")
        print(f"{'='*50}")

        # 重置 class_anchors (新任务，新锚点)
        self.class_anchors.clear()
        self._recent_samples.clear()

        # 解冻高层节点 (tier > 1)
        for node in self.graph.nodes.values():
            if node.tier > 1:
                node.frozen = False

        train_mnist(self, images, labels, n_epochs=n_epochs,
                    contrast_schedule=contrast_schedule)

        st = self.stats()
        print(f"\n📊 适应后: 锚点={self.class_anchors}, "
              f"节点tier分布={st['node_tiers']}")
        return len(self.class_anchors)


def train_mnist(
    model: ColdEye,
    images: np.ndarray,
    labels: np.ndarray,
    n_epochs: int = 5,
    contrast_schedule: Optional[Callable[[int], float]] = None,
    on_batch: Optional[Callable] = None,
):
    """训练循环

    images: (N, H, W) 灰度图
    labels: (N,) 整数标签
    contrast_schedule: gen → contrast, None 表示全对比度
    """
    n_samples = len(images)
    print(f"🧊 冷眼训练: {n_samples} 样本, {model.graph.n_nodes} 节点")
    print(f"   初始边: {model.graph.edge_count}, 突触: {model.synapse.total_edges}")

    start_time = time.time()

    for epoch in range(n_epochs):
        # 随机打乱
        perm = np.random.permutation(n_samples)
        epoch_correct = 0

        for i, idx in enumerate(perm):
            image = images[idx]
            label = labels[idx]

            contrast = contrast_schedule(model.generation) if contrast_schedule else 1.0

            model.train_step(image, label, contrast=contrast)

            # 每100步打印状态
            if model.generation % 100 == 0:
                st = model.stats()
                elapsed = time.time() - start_time
                print(
                    f"  gen={st['generation']:5d} | "
                    f"边: {st['synapse_edges']:5d} | "
                    f"t4={st['tier_dist'].get(4,0):4d} "
                    f"t3={st['tier_dist'].get(3,0):4d} "
                    f"t2={st['tier_dist'].get(2,0):4d} "
                    f"t1={st['tier_dist'].get(1,0):4d} | "
                    f"anchors={st['class_anchors']} | "
                    f"{elapsed:.0f}s"
                )
                if on_batch:
                    on_batch(model)

    elapsed = time.time() - start_time
    st = model.stats()
    print(f"\n✅ 训练完成: {elapsed:.0f}s, gen={st['generation']}, "
          f"突触={st['synapse_edges']}, "
          f"锚点={st['class_anchors']}")
    print(f"   tier分布: {st['tier_dist']}")


# ── 下载 MNIST ──
def load_mnist(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """下载/加载 MNIST"""
    import gzip
    import urllib.request

    os.makedirs(data_dir, exist_ok=True)

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    base_url = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"

    for name, fname in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  下载 {fname}...")
            urllib.request.urlretrieve(base_url + fname, path)

    def load_images(path):
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28).astype(np.float32) / 255.0

    def load_labels(path):
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data.astype(np.int64)

    X_train = load_images(os.path.join(data_dir, files["train_images"]))
    y_train = load_labels(os.path.join(data_dir, files["train_labels"]))
    X_test = load_images(os.path.join(data_dir, files["test_images"]))
    y_test = load_labels(os.path.join(data_dir, files["test_labels"]))

    return X_train, y_train, X_test, y_test


def curriculum_contrast(gen: int) -> float:
    """课程对比度: 逐步降低"""
    if gen < 300:
        return 1.0
    elif gen < 600:
        return 0.5
    elif gen < 1000:
        return 0.25
    else:
        return 0.1


def evaluate_contrasts(model, X_test, y_test, n_test=500):
    """多对比度评估"""
    results = {}
    for contrast in [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]:
        correct = 0
        n = min(n_test, len(X_test))
        for i in range(n):
            img = X_test[i].copy()
            mean = img.mean()
            img_low = mean + (img - mean) * contrast
            pred, conf = model.predict(img_low)
            if pred == y_test[i]:
                correct += 1
        results[contrast] = correct / n
    return results


if __name__ == "__main__":
    print("🧊 冷眼 — 两阶段训练")
    print("   阶段1: 系统发生 (全量 → 底层拓扑)")
    print("   阶段2: 个体发生 (小样本 → 锚点适应)")
    print("=" * 50)

    # ── 准备数据 ──
    model = ColdEye(n_nodes=500)
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"MNIST: train={len(X_train)}, test={len(X_test)}")

    # 用真实图像初始化模板 (不用随机)
    model.init_templates_from_data(X_train)
    print(f"节点模板已从真实图像初始化")

    # 预训练对比度课程
    def pretrain_contrast(gen):
        if gen < 500:   return 1.0
        elif gen < 1500: return 0.5
        elif gen < 3000: return 0.3
        else:            return 0.2

    # ── 阶段1: 全量预训练 ──
    n_pretrain = 10000
    n_frozen = model.pretrain_visual_cortex(
        X_train[:n_pretrain], y_train[:n_pretrain],
        n_epochs=5,
        contrast_schedule=pretrain_contrast,
    )
    print(f"\n🧊 底层拓扑: {n_frozen} 节点冻结")

    # 调试: 打印节点 domain_tag 分布
    from collections import Counter
    tag_counts = Counter(
        n.domain_tag for n in model.graph.nodes.values() if n.domain_tag
    )
    print(f"   domain_tag 分布 (top 10): {tag_counts.most_common(10)}")

    # 调试: 对几个样本看预测
    print(f"   预测调试 (前5个测试样本):")
    for i in range(5):
        img = X_test[i]
        true_label = y_test[i]
        model.vision.set_image(img, contrast=1.0)
        active = sorted(
            [(nid, n.activation, n.domain_tag)
             for nid, n in model.graph.nodes.items() if n.activation > 0.1],
            key=lambda x: -x[1]
        )[:5]
        pred, conf = model.predict(img)
        print(f"    样本{i} 真实={true_label} 预测={pred} conf={conf:.2f} "
              f"active={[(a[0], f'{a[1]:.2f}', a[2]) for a in active[:3]]}")

    # ── 基线: 预训练后全对比度识别率 ──
    print(f"\n📊 预训练后基线:")
    results_base = evaluate_contrasts(model, X_test, y_test)
    for c, acc in sorted(results_base.items()):
        print(f"  contrast={c:.2f}: {acc:.1%}")

    # ── 阶段2: 每类仅 5 样本适应 (从预训练未见过的新数据取) ──
    n_per_class = 5
    few_shot_idx = []
    for c in range(10):
        c_idx = np.where(y_train[n_pretrain:] == c)[0]
        if len(c_idx) >= n_per_class:
            c_idx = c_idx[:n_per_class] + n_pretrain  # 偏移回原始索引
            few_shot_idx.extend(c_idx.tolist())
    few_shot_idx = np.array(few_shot_idx, dtype=np.int64)
    np.random.shuffle(few_shot_idx)

    # 小样本对比度课程 (更激进 — 底层已稳定)
    def fewshot_contrast(gen):
        if gen < 100:   return 0.5
        elif gen < 300: return 0.25
        else:           return 0.1

    n_anchors = model.few_shot_adapt(
        X_train[few_shot_idx], y_train[few_shot_idx],
        n_epochs=20,
        contrast_schedule=fewshot_contrast,
    )

    # ── 最终评估 ──
    print(f"\n📊 小样本适应后 ({n_per_class}样本/类):")
    results_adapt = evaluate_contrasts(model, X_test, y_test)
    for c, acc in sorted(results_adapt.items()):
        delta = acc - results_base.get(c, 0)
        sign = "+" if delta >= 0 else ""
        print(f"  contrast={c:.2f}: {acc:.1%} ({sign}{delta:+.1%})")

    # 汇总
    print(f"\n{'='*50}")
    print(f"总结:")
    print(f"  基底: {n_frozen} 冻结节点 (底层拓扑)")
    print(f"  锚点: {n_anchors} 个类")
    print(f"  全对比度: {results_adapt.get(1.0, 0):.1%}")
    print(f"  对比度 0.1: {results_adapt.get(0.1, 0):.1%}")
    print(f"  对比度 0.05: {results_adapt.get(0.05, 0):.1%}")
