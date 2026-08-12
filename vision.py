#!/usr/bin/env python3
"""
冷眼 — 预测反馈视觉系统
vision.py: 竞争路由 + 提示引导
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class PatchExtractor:
    def __init__(self, patch_size: int = 8, stride: int = 4):
        self.patch_size = patch_size
        self.stride = stride
        self.patch_positions: List[Tuple[int, int, int, int]] = []

    def extract(self, image: np.ndarray) -> np.ndarray:
        H, W = image.shape
        patches = []
        self.patch_positions = []
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                patch = image[y:y + self.patch_size, x:x + self.patch_size].flatten()
                norm = np.linalg.norm(patch)
                if norm > 0:
                    patch = patch / norm
                patches.append(patch)
                self.patch_positions.append((y, x, y + self.patch_size, x + self.patch_size))
        return np.array(patches, dtype=np.float32)


class VisionInterface:
    """竞争路由 + 提示引导"""

    def __init__(self, graph, patch_size: int = 8, stride: int = 4, top_k: int = 1):
        self.graph = graph
        self.extractor = PatchExtractor(patch_size, stride)
        self.top_k = top_k
        self.image_shape: Optional[Tuple[int, int]] = None
        self.node_assignments: Dict[str, List[np.ndarray]] = {}
        self._last_patches: Optional[np.ndarray] = None  # 存下来给 reconstruct 用

    def set_image(self, image: np.ndarray, contrast: float = 1.0):
        self._route(image, contrast, hint_label=None, boost=1.0)

    def set_image_with_hint(self, image: np.ndarray, hint_label: int,
                            contrast: float = 1.0, boost: float = 2.0):
        return self._route(image, contrast, hint_label=hint_label, boost=boost)

    def _route(self, image: np.ndarray, contrast: float,
               hint_label=None, boost: float = 1.0):
        """竞争路由: hint_label 非 None 时对应节点 score×boost"""
        self.image_shape = image.shape
        if contrast < 1.0:
            mean = image.mean()
            image = image.copy()
            image = mean + (image - mean) * contrast

        patches = self.extractor.extract(image)
        self._last_patches = patches  # 存下来给 reconstruct
        node_ids = sorted(self.graph.nodes.keys())
        n_patches = len(patches)

        # 找到 hinted 节点
        hinted_nodes = set()
        if hint_label is not None:
            for nid, node in self.graph.nodes.items():
                try:
                    if node.domain_tag and int(node.domain_tag) == hint_label:
                        hinted_nodes.add(nid)
                except ValueError:
                    pass

        templates = np.array([
            self.graph.nodes[nid].template for nid in node_ids
        ], dtype=np.float32)

        boost_vec = np.array([
            boost if nid in hinted_nodes else 1.0
            for nid in node_ids
        ], dtype=np.float32)

        self.node_assignments.clear()
        for node in self.graph.nodes.values():
            node.activation = 0.0

        match_counts = {}
        match_patches = {}

        for patch in patches:
            scores = templates @ patch * boost_vec
            best_idx = int(np.argmax(scores))
            if float(scores[best_idx]) < 0:
                continue
            nid = node_ids[best_idx]
            match_counts[nid] = match_counts.get(nid, 0) + 1
            if nid not in match_patches:
                match_patches[nid] = []
            match_patches[nid].append(patch)

        for nid, count in match_counts.items():
            self.graph.nodes[nid].activation = min(1.0, count / n_patches * 3)
            self.node_assignments[nid] = match_patches[nid]

        return len(hinted_nodes)

    def predictive_boost(self, colony, strength: float = 0.5, n_iters: int = 3):
        """预测反馈 v2: 预测误差作为特征 (不修改激活值)

        费曼脑的预测: 细胞走 A→B 后检查 B 是否真的激活了
        → 是的 → 多巴胺 → 加固
        → 不是 → 预测误差 → 这个差本身就是有用信号

        这里实现: 对每个节点, 用高 tier 入边预测其应有激活,
        返回预测误差向量 (预测值 - 实际值), 不修改原始激活。
        """
        synapse = colony.synapse
        node_ids = sorted(self.graph.nodes.keys())
        n_nodes = len(node_ids)
        nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # 实际激活
        actual = np.array([self.graph.nodes[nid].activation for nid in node_ids], dtype=np.float32)

        # 预测激活: 每个节点 = 所有入边 (源激活 × s × tier权重) 的加权和
        predicted = np.zeros(n_nodes, dtype=np.float32)
        incoming_weights = np.zeros(n_nodes, dtype=np.float32)

        for (src, dst), tier in synapse.tiers.items():
            if tier > 3:
                continue
            edge = synapse.activations.get((src, dst), {})
            s = edge.get('s', 0.3)
            if s < 0.1:
                continue
            src_idx = nid_to_idx.get(src)
            dst_idx = nid_to_idx.get(dst)
            if src_idx is None or dst_idx is None:
                continue

            tier_w = {1: 1.0, 2: 0.8, 3: 0.4}.get(tier, 0.2) * strength
            weight = actual[src_idx] * s * tier_w
            predicted[dst_idx] += weight
            incoming_weights[dst_idx] += 1.0

        # 归一化: 除以入边数
        mask = incoming_weights > 0
        predicted[mask] /= incoming_weights[mask]

        # 预测误差 = 预测值 - 实际值 (正=比预期更强, 负=比预期更弱)
        error = predicted - actual

        # 返回误差向量作为第二特征 (不修改节点激活)
        return error

    def reconstruct(self, hint_label=None, boost=1.0) -> np.ndarray:
        """重建: 每个位置 = 最佳模板匹配分 × 节点激活 (脑补增强生效)"""
        if self.image_shape is None or self._last_patches is None:
            raise ValueError("先调用 set_image()")

        H, W = self.image_shape
        patches = self._last_patches
        node_ids = sorted(self.graph.nodes.keys())
        templates = np.array([
            self.graph.nodes[nid].template for nid in node_ids
        ], dtype=np.float32)

        # 节点激活值 (脑补后会被增强)
        node_acts = np.array([
            self.graph.nodes[nid].activation for nid in node_ids
        ], dtype=np.float32)

        canvas = np.zeros((H, W), dtype=np.float32)

        for p_i, (y1, x1, y2, x2) in enumerate(self.extractor.patch_positions):
            if p_i >= len(patches):
                break
            # 模板匹配分 × 节点激活 → 脑补增强的节点获得更高权重
            scores = (templates @ patches[p_i]) * node_acts
            best_score = float(np.max(scores))
            val = max(0.0, best_score)
            canvas[y1:y2, x1:x2] = max(canvas[y1:y2, x1:x2].max(), val)

        return canvas
