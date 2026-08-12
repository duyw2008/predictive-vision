#!/usr/bin/env python3
"""
冷眼 — 预测反馈视觉系统
synapse.py: s值三位一体 + 四机制衰减 + 晋升逻辑

s值 = 预测置信度 + 惊讶 + 学习率 (三位一体)
每条边存: {s, c (coincidence), g (generation), tier, neurons}

四机制垃圾免疫:
  (A) tier 代谢成本 — 高层(tier低)更贵
  (B) 相干场 — 孤立边加速衰减
  (C) 用进废退 — 不用的边自然消失
  (D) 晋升窗口期 — tier 4 超时未晋升 → 消除
"""

from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional


class SynapticLayer:
    """预测边突触层 — 管理所有节点间的预测关系"""

    # 晋升阈值
    LTP_THRESHOLD = 8       # t4 → t3 需要 >8 个不同图像样本验证
    CONSOLIDATION_THRESHOLD = 30  # t3 → t2 需要 >30 个样本

    # 衰减参数
    LTD_WINDOW = 100         # 失活窗口 (代)
    TIER4_WINDOW = 200       # t4 晋升窗口期
    MAX_NEURONS_PER_EDGE = 50

    # tier 代谢成本 (越高越贵)
    TIER_COST = {0: 0.3, 2: 0.7, 3: 2.5, 4: 3.0}

    def __init__(self):
        # {(src, dst): {s, c, g, n (set of sample_ids), tier}}
        self.activations: Dict[Tuple[str, str], dict] = {}

        # 最后激活代
        self._last_fired: Dict[Tuple[str, str], int] = {}

        # tier 记录
        self.tiers: Dict[Tuple[str, str], int] = {}

        # 上下文
        self._cell_count: int = 0
        self._graph_ref = None

    def update_context(self, n_nodes: int, graph):
        self._cell_count = n_nodes
        self._graph_ref = graph

    def fire(self, src: str, dst: str, strength: float, sample_id: str,
             generation: int):
        """边被激活: 预测信号沿 src→dst 传播

        strength: 预测强度 (匹配度 × s值)
        """
        key = (src, dst)
        edge = self.activations.get(key)

        if edge is None:
            edge = {
                'n': set(),
                'g': generation,
                's': 0.05,  # 初始低s, 通过验证提升
                'c': 0,
                't4_birth': generation,
            }
            self.activations[key] = edge
            self.tiers[key] = 4

        # 神经元ID (样本去重)
        if len(edge['n']) < self.MAX_NEURONS_PER_EDGE:
            edge['n'].add(sample_id)

        edge['g'] = generation

        # 🏋️ 马太效应: s越高, 强化越多
        edge['s'] += strength * (1.0 + edge['s'] * 0.5)
        edge['s'] = min(1.0, edge['s'])  # cap at 1.0
        edge['c'] += 1

        self._last_fired[key] = generation

        # 检查晋升
        self._check_promotion(key, generation)

    def _check_promotion(self, key: Tuple[str, str], generation: int):
        """三道闸门: t4 → t3 晋升"""
        current = self.tiers.get(key, 4)
        edge = self.activations[key]
        unique = len(edge['n'])

        if current == 4:
            # 闸门1: 需要足够的独立样本
            threshold = max(10, self._cell_count // 8) if self._cell_count > 0 else 10
            if unique < threshold:
                return

            src, dst = key

            # 闸门2: 两端节点必须存在
            if self._graph_ref is not None:
                if src not in self._graph_ref.nodes or dst not in self._graph_ref.nodes:
                    return

            # 闸门3: 方向合理 (预测边应该从简单→复杂)
            src_node = self._graph_ref.nodes.get(src) if self._graph_ref else None
            dst_node = self._graph_ref.nodes.get(dst) if self._graph_ref else None
            if src_node and dst_node:
                # 低tier(可靠特征)不应该预测高tier(假说)
                if src_node.tier > dst_node.tier:
                    return

            self.tiers[key] = 3
            edge.pop('t4_birth', None)  # 撤销死刑计时器

        # t3 → t2 巩固
        if current == 3 and unique >= self.CONSOLIDATION_THRESHOLD:
            self.tiers[key] = 2

        # t2 → t1
        if current == 2 and unique >= 80:
            self.tiers[key] = 1

    def decay(self, generation: int) -> List[Tuple[str, str]]:
        """四机制衰减: 返回被消除的边"""
        eliminated = []

        # (D) 晋升窗口期: t4 超时 → 消除
        for key, edge in list(self.activations.items()):
            if self.tiers.get(key, 4) != 4:
                continue
            t4_birth = edge.get('t4_birth')
            if t4_birth and (generation - t4_birth) > self.TIER4_WINDOW:
                eliminated.append(key)

        # (B) 相干节点: tier ≤ 2 的节点
        coherent_nodes: Set[str] = set()
        for (s, d), t in self.tiers.items():
            if t <= 2:
                coherent_nodes.add(s)
                coherent_nodes.add(d)

        # (A, B, C) 衰减
        for key, last_gen in list(self._last_fired.items()):
            age = generation - last_gen
            tier = self.tiers.get(key, 4)
            effective_window = self.LTD_WINDOW
            if tier == 4:
                effective_window = self.LTD_WINDOW // 3  # t4 三倍速衰减

            if age <= effective_window:
                continue

            edge = self.activations.get(key)
            if edge is None:
                continue

            base_decay = 0.5 ** (age / effective_window)
            tier_mult = self.TIER_COST.get(tier, 1.0)
            effective_decay = base_decay ** tier_mult

            # (B) 孤立节点: 不在相干场 → 加速衰减
            src, dst = key
            if tier >= 3 and src not in coherent_nodes and dst not in coherent_nodes:
                effective_decay = effective_decay ** 1.5

            # (C) t4 额外加速
            if tier == 4 and age > self.LTD_WINDOW // 2:
                effective_decay = effective_decay ** 1.5

            edge['c'] = max(1, int(edge['c'] * effective_decay))
            edge['s'] *= effective_decay

            # 降级: t3 → t4
            unique = len(edge['n'])
            dyn_threshold = max(10, self._cell_count // 8) if self._cell_count > 0 else 10
            if tier == 3 and unique < dyn_threshold:
                self.tiers[key] = 4
                edge['t4_birth'] = generation

            # 消除条件
            if edge['s'] < 0.001 or (tier == 4 and edge['s'] < 0.01 and edge['c'] < 2):
                eliminated.append(key)

        # 清理从未点火的边
        fired_keys = set(self._last_fired.keys())
        for key, edge in list(self.activations.items()):
            if key in fired_keys:
                continue
            creation_gen = edge.get('g', generation)
            age = generation - creation_gen
            if age < self.LTD_WINDOW // 2:
                continue
            base_decay = 0.5 ** (age / (self.LTD_WINDOW // 4))
            edge['s'] *= base_decay
            if edge['s'] < 0.001:
                eliminated.append(key)

        for key in set(eliminated):
            self.activations.pop(key, None)
            self._last_fired.pop(key, None)
            self.tiers.pop(key, None)

        return eliminated

    def promote_coincident_edges(self, graph, generation: int):
        """晋升常共激活的边对

        如果 A→B 和 B→C 经常同时激活, 建立 A→C (三元闭包)
        """
        # 收集近期高频激活的边
        active_pairs = []
        for key in self._last_fired:
            tier = self.tiers.get(key, 4)
            if tier > 3:
                continue
            edge = self.activations.get(key)
            if edge and edge['s'] > 0.3:
                active_pairs.append(key)

        # 查找可组合的边
        from collections import Counter as Ctr
        mid_nodes = Ctr()
        src_dst_map: Dict[str, List[str]] = defaultdict(list)

        for src, dst in active_pairs:
            mid_nodes[dst] += 1
            src_dst_map[src].append(dst)

        new_edges = 0
        for src, dsts in src_dst_map.items():
            for mid in dsts:
                if mid not in src_dst_map:
                    continue
                for far in src_dst_map[mid]:
                    if far == src:
                        continue
                    key = (src, far)
                    if key not in self.activations:
                        # 创建组合边
                        self.fire(src, far, 0.15, f"compose:{generation}", generation)
                        new_edges += 1

        return new_edges

    @property
    def tier_distribution(self) -> Dict[int, int]:
        dist = defaultdict(int)
        for t in self.tiers.values():
            dist[t] += 1
        return dict(dist)

    @property
    def total_edges(self) -> int:
        return len(self.activations)

    def top_edges(self, n: int = 10) -> List[dict]:
        """返回最强的边"""
        items = []
        for key, edge in self.activations.items():
            items.append({
                'src': key[0],
                'dst': key[1],
                's': edge['s'],
                'c': edge['c'],
                'tier': self.tiers.get(key, 4),
                'unique': len(edge['n']),
            })
        items.sort(key=lambda x: x['s'], reverse=True)
        return items[:n]
