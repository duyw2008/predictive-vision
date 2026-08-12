#!/usr/bin/env python3
"""
冷眼 — 预测反馈视觉系统
graph.py: 平图节点引擎

核心理念: 不给层级, 给容量 → 结构从预测误差中涌现
每个节点 = 一个可学习的视觉特征模板 (patch)
边 = 预测关系, 由 synapse 层管理
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set


class VisionNode:
    """视觉特征节点 — 生态位由预测-惊讶-学习自然分化"""

    __slots__ = (
        "node_id", "template", "tier", "reward", "generation",
        "activation", "prediction_error", "predicted_by", "predicts_to",
        "domain_tag", "last_active_gen", "frozen",
        "spatial_cy", "spatial_cx", "spatial_count",  # 空间热点
    )

    def __init__(self, node_id: str, template_size: int = 64):
        self.node_id = node_id
        # 可学习的特征模板 (初始随机, 训练中 Hebbian 更新)
        self.template = np.random.randn(template_size).astype(np.float32) * 0.1
        self.template /= np.linalg.norm(self.template) + 1e-8

        self.tier = 4  # 所有节点从 tier 4 (假说) 开始
        self.reward = 0.0
        self.generation = 0

        # 运行时状态
        self.activation = 0.0  # 当前激活值 [0,1]
        self.prediction_error = 0.0
        self.predicted_by: Set[str] = set()
        self.predicts_to: Set[str] = set()

        self.domain_tag: Optional[str] = None  # 生态位标签 (涌现后赋予)
        self.last_active_gen = 0
        self.frozen = False  # 冻结后不更新模板 (预训练好的底层特征)

    def to_dict(self) -> dict:
        return {
            "id": self.node_id,
            "template": self.template.tolist(),
            "tier": self.tier,
            "reward": self.reward,
            "generation": self.generation,
            "domain_tag": self.domain_tag,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VisionNode":
        n = cls(d["id"], template_size=len(d["template"]))
        n.template = np.array(d["template"], dtype=np.float32)
        n.tier = d.get("tier", 4)
        n.reward = d.get("reward", 0.0)
        n.generation = d.get("generation", 0)
        n.domain_tag = d.get("domain_tag")
        return n


class VisionGraph:
    """平图: 所有节点平等, 不预设层级 — 结构从预测关系涌现"""

    PATCH_SIZE = 8  # 默认 8×8

    def __init__(self, n_nodes: int = 200, template_size: int = None):
        self.nodes: Dict[str, VisionNode] = {}
        self.synapse = None
        ts = template_size if template_size else self.PATCH_SIZE ** 2
        self._template_size = ts

        self.generation = 0
        self.n_nodes = n_nodes

        # 图结构: {node_id: {node_id: s_value}} — 简化的邻接
        self.adjacency: Dict[str, Dict[str, float]] = defaultdict(dict)

        # 生态位标签 (涌现)
        self.domain_tags: Dict[str, str] = {}

        # 混沌初始化: 密集随机连接
        self._chaos_init()

    def _chaos_init(self):
        """先密后疏: 初始密集随机连接, 由用进废退自然修剪"""
        node_ids = [f"n{i:04d}" for i in range(self.n_nodes)]
        for nid in node_ids:
            self.nodes[nid] = VisionNode(nid, template_size=self._template_size)

        # 随机密集连接 (~20% 密度)
        for nid in node_ids:
            targets = np.random.choice(
                node_ids,
                size=max(1, self.n_nodes // 5),
                replace=False,
            )
            for t in targets:
                if t != nid:
                    self.adjacency[nid][t] = np.random.uniform(0.05, 0.3)

    def activate(self, node_id: str, match_score: float, gen: int):
        """节点被图像激活: bottom-up 信号"""
        node = self.nodes[node_id]
        node.activation = match_score
        node.last_active_gen = gen

    def propagate_predictions(self) -> Dict[str, float]:
        """一轮预测传播: 高激活节点沿边向下游节点发送预测信号

        返回: {node_id: prediction_error} 用于学习
        """
        errors: Dict[str, float] = {}

        for src_id, src_node in self.nodes.items():
            if src_node.activation < 0.1:
                continue

            for dst_id, s_val in self.adjacency.get(src_id, {}).items():
                dst_node = self.nodes.get(dst_id)
                if dst_node is None:
                    continue

                # 预测信号: activation * s — 高s=强预测
                predicted_activation = src_node.activation * s_val
                actual_activation = dst_node.activation

                error = actual_activation - predicted_activation
                dst_node.prediction_error = error
                dst_node.predicted_by.add(src_id)
                src_node.predicts_to.add(dst_id)

                errors[dst_id] = errors.get(dst_id, 0.0) + abs(error)

        return errors

    def settle(self, image_patches: np.ndarray, max_iter: int = 20, lr: float = 0.1):
        """迭代推理: 预测信号循环传播直至收敛

        低对比度时 bottom-up 激活弱 → 高层预测向下填充
        这是 '脑补' 的计算实现
        """
        # 初始 bottom-up 激活
        for i, nid in enumerate(sorted(self.nodes.keys())):
            if i < len(image_patches):
                patch = image_patches[i]
                match = self._match_score(self.nodes[nid].template, patch.flatten())
                self.activate(nid, match, self.generation)

        # 迭代预测传播
        for _ in range(max_iter):
            errors = self.propagate_predictions()

            # 每个节点根据预测误差更新激活值
            max_change = 0.0
            for nid, node in self.nodes.items():
                if nid in errors and node.predicted_by:
                    # 误差大 → 激活向预测方向移动
                    old_act = node.activation
                    node.activation += lr * (sum(
                        self.nodes[p].activation * self.adjacency.get(p, {}).get(nid, 0)
                        for p in node.predicted_by
                    ) / len(node.predicted_by) - node.activation)
                    max_change = max(max_change, abs(node.activation - old_act))

            # 清空本轮预测关系
            for node in self.nodes.values():
                node.predicted_by.clear()
                node.predicts_to.clear()

            if max_change < 0.001:
                break

    def _match_score(self, template: np.ndarray, patch: np.ndarray) -> float:
        """余弦相似度 → [0, 1]"""
        dot = np.dot(template, patch)
        norm_t = np.linalg.norm(template)
        norm_p = np.linalg.norm(patch) + 1e-8
        return max(0.0, (dot / (norm_t * norm_p) + 1) / 2)

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self.adjacency.values())

    @property
    def tier_distribution(self) -> Dict[int, int]:
        dist = defaultdict(int)
        for n in self.nodes.values():
            dist[n.tier] += 1
        return dict(dist)

    def get_high_tier_predictions(self, n: int = 5) -> List[Tuple[str, float]]:
        """获取 tier ≤ 2 节点(可靠特征)的激活值和预测"""
        results = []
        for nid, node in self.nodes.items():
            if node.tier <= 2 and node.activation > 0.3:
                results.append((nid, node.activation, node.tier, node.domain_tag))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]
