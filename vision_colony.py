#!/usr/bin/env python3
"""
冷眼 — 费曼脑架构移植
vision_colony.py: 细胞行走 + 突触投票 + 睡眠巩固

从费曼脑移植到视觉特征图:
- 知识图谱 → 视觉特征图 (VisionGraph 模板节点 + 共激活边)
- 细胞 → VisionCell (精简版 EvolvableCell, 去 sympy/derive)
- 突触 → 复用 VisionGraph.synapse (SynapticLayer, 已支持 s/n/tier)
- 睡眠 → 巩固 + 密度竞争 + tier 审计

生物映射:
- 树突 (dendrites) — 细胞监听哪些特征节点
- 轴突 (axons)     — 细胞投射了哪些特征边
- 髓鞘 (myelin)     — 频繁走的边形成高速公路
- 突触 (synapse)    — s/n/tier 共识投票
"""

import random
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional


class VisionCell:
    """视觉特征图上的可进化细胞 — 移植自 EvolvableCell (精简: 去 sympy/derive)"""

    MAX_AGE = 200

    def __init__(self, node_id: str, graph, synapse, genome=None):
        self.node = node_id          # 当前所在特征节点
        self.graph = graph           # VisionGraph 引用
        self.synapse = synapse       # SynapticLayer 引用
        self.age = 0
        self.total_reward = 0.0

        # 基因组: 行动倾向权重
        if genome is None:
            self.genome = {
                "step_forward":  0.60,   # 沿共激活边走
                "step_backward": 0.15,   # 回溯
                "rest":          0.05,   # 不动
                "jump":          0.05,   # 随机跳到任意特征
                "echo":          0.05,   # 在当前节点徘徊
                "curiosity":     1.0,    # 探索/深耕倾向 (>1 探索, <1 深耕)
                "reinforce_rate": 0.12,  # 突触增强敏感度
                "decay_rate":     0.005, # 遗忘速度
                "mutation_rate":  0.10,  # 变异幅度
            }
        else:
            self.genome = dict(genome)
            self.genome.setdefault("curiosity", 1.0)
            self.genome.setdefault("reinforce_rate", 0.12)
            self.genome.setdefault("decay_rate", 0.005)
            self.genome.setdefault("mutation_rate", 0.10)

        # 树突: 我监听哪些特征节点
        self.dendrites: set = set()
        # 轴突: 我投射了哪些突触边 (src, dst)
        self.axons: set = set()
        # 髓鞘: 每条边的传导速度 (频繁走→变厚→更快)
        self.myelin: Dict[tuple, float] = {}
        # 私有权重: target_node → 累积权重
        self.weights: Dict[str, float] = {}
        # 预测模型: 当前节点 → {下级节点: 计数}
        self.prediction_model: Dict[str, Counter] = {}
        # 路径记忆
        self.walk_memory: List[List[tuple]] = []
        self.current_walk: List[tuple] = []

    # ─── 邻接边查询 ───

    def get_outgoing(self) -> List[Tuple[str, str, float]]:
        """从知识图谱 (VisionGraph.adjacency) 获取邻接边, 而非突触层"""
        neighbors = []
        adj = self.graph.adjacency.get(self.node, {})
        for dst, weight in adj.items():
            if dst != self.node and weight > 0.01:
                neighbors.append((dst, "coactive", weight))
        return neighbors

    def get_incoming(self) -> List[Tuple[str, str, float]]:
        """获取指向当前节点的所有突触边"""
        neighbors = []
        for (src, dst), edge in self.synapse.activations.items():
            if dst == self.node:
                s = edge.get('s', 0.05)
                neighbors.append((src, f"coactive", s))
        return neighbors

    # ─── 行走 ───

    def step_forward(self, epsilon: float = 0.15) -> Optional[Tuple[str, str, float]]:
        """
        沿共激活边走到目标特征节点。
        选边权重 = s × (1 + myelin × 2) × curiosity 调制
        epsilon: 随机探索概率
        """
        neighbors = self.get_outgoing()
        if not neighbors:
            return None

        # ε-贪心: 随机探索
        if random.random() < epsilon:
            dst, law, s = random.choice(neighbors)
            return (dst, law, s * 0.3)  # 探索降低强度

        # 加权选择
        curiosity = self.genome.get("curiosity", 1.0)
        weighted = []
        for dst, law, s in neighbors:
            mye = self.myelin.get((self.node, dst), 0)
            w = max(0.01, s) * (1.0 + mye * 2.0)  # 髓鞘加成
            if curiosity != 1.0:
                w = w ** curiosity            # 好奇调制
            weighted.append((dst, law, s, w))

        total = sum(w for _, _, _, w in weighted)
        if total <= 0:
            return None

        r = random.random() * total
        cum = 0
        for dst, law, s, w in weighted:
            cum += w
            if r <= cum:
                return (dst, law, s)
        return (weighted[-1][0], weighted[-1][1], weighted[-1][2])

    def walk(self) -> Optional[Tuple[str, str, float]]:
        """执行一步行走: 返回 (src, dst, strength) 或 None"""
        result = self.step_forward()
        if result is None:
            return None

        dst, law, strength = result
        old = self.node
        self.node = dst

        step = (old, dst, strength)
        self.current_walk.append(step)

        # 树突: 记录监听
        self.dendrites.add(dst)

        return step

    # ─── 衰减 ───

    def apply_decay(self):
        """权重衰减 (每代)"""
        for k in list(self.weights.keys()):
            self.weights[k] *= (1.0 - self.genome["decay_rate"])
            if abs(self.weights[k]) < 0.001:
                del self.weights[k]

    def myelin_decay(self):
        """髓鞘衰退: 不用的边慢慢消失 (-0.5%/代)"""
        for k in list(self.myelin.keys()):
            self.myelin[k] *= 0.995
            if self.myelin[k] < 0.01:
                del self.myelin[k]


class VisionColony:
    """视觉特征图上的细胞群落 — 移植自 EvoColony 的呼吸/睡眠循环"""

    MAX_SNAPSHOTS = 10

    def __init__(self, vision_graph):
        self.graph = vision_graph       # VisionGraph
        self.synapse = vision_graph.synapse  # SynapticLayer
        self.generation = 0
        self.cells: List[VisionCell] = []
        self.total_rewards = 0.0
        self._walk_buffer: List[Tuple[List, int, int]] = []

        # 睡眠参数
        self.SLEEP_EVERY_N = 10

    # ─── 播种 ───

    def seed_cells(self, n_per_node: int = 3, max_cells: int = 1000):
        """在每个特征节点上播种细胞"""
        self.cells = []
        node_ids = list(self.graph.nodes.keys())
        total = 0

        for node_id in node_ids:
            if total >= max_cells:
                break
            for _ in range(n_per_node):
                if total >= max_cells:
                    break
                cell = VisionCell(node_id, self.graph, self.synapse)
                # 初始树突: 监听知识图谱中的邻接特征
                adj = self.graph.adjacency.get(node_id, {})
                for dst in adj:
                    cell.dendrites.add(dst)
                self.cells.append(cell)
                total += 1

        print(f"[SEED] {len(self.cells)} cells on {len(node_ids)} feature nodes, "
              f"{sum(len(v) for v in self.graph.adjacency.values())} KG edges")

    # ─── 呼吸 ───

    def _step_cells(self, n_steps: int = 3):
        """所有细胞各走 n_steps 步"""
        for cell in self.cells:
            for _ in range(n_steps):
                step = cell.walk()
                if step is None:
                    continue
                src, dst, strength = step

                # 突触增强 — fire() 自动管理 n/s/c/神经元集合
                self.synapse.fire(
                    src, dst, strength,
                    sample_id=str(id(cell) % 10000),
                    generation=self.generation,
                )

                # 轴突投射
                cell.axons.add((src, dst))

                # 髓鞘化: 频繁走→传导加速
                k = (src, dst)
                cell.myelin[k] = min(3.0, cell.myelin.get(k, 0) + 0.1)

    def _digest_walk_buffer(self):
        """消化行走缓冲 (云端细胞重放时)"""
        if not self._walk_buffer:
            return
        for walk, gen, cell_id in self._walk_buffer:
            for step in walk:
                if len(step) >= 3:
                    self.synapse.fire(
                        step[0], step[2], 0.1,
                        sample_id=str(cell_id),
                        generation=gen,
                    )
        print(f"  [DIGEST] {len(self._walk_buffer)} walks → synapses")

    # ─── 睡眠 ───

    def _sleep_consolidate(self):
        """睡眠巩固: 高 s 边重放加固"""
        replayed = 0
        for (src, dst), edge in list(self.synapse.activations.items()):
            s = edge.get('s', 0)
            n = len(edge.get('n', set()))
            if s > 0.5 and n >= 2:
                self.synapse.fire(src, dst, 0.3,
                                 sample_id="consolidate",
                                 generation=self.generation)
                replayed += 1
        if replayed:
            print(f"  [CONSOLIDATE] {replayed} high-s edges replayed")

    def _sleep_prune(self):
        """修剪弱边 + 噪声"""
        to_del = []
        for key, edge in list(self.synapse.activations.items()):
            s = edge.get('s', 0)
            n = len(edge.get('n', set()))
            # 弱边 + 单神经元 → 删除
            if s < 0.02 and n <= 1:
                to_del.append(key)
            # 自环 → 删除
            if key[0] == key[1]:
                to_del.append(key)

        for key in set(to_del):
            self.synapse.activations.pop(key, None)
            self.synapse.tiers.pop(key, None)
            self.synapse._last_fired.pop(key, None)

        if to_del:
            print(f"  [PRUNE] {len(set(to_del))} weak/self-loop edges removed")

    def _sleep_tier_audit(self):
        """tier 晋升审计"""
        for key, edge in list(self.synapse.activations.items()):
            s = edge.get('s', 0)
            n = len(edge.get('n', set()))
            current_tier = self.synapse.tiers.get(key, 4)

            # t4 → t3: s>1.0 + n>=3
            if current_tier == 4 and s > 1.0 and n >= 3:
                self.synapse.tiers[key] = 3
            # t3 → t2: s>3.0 + n>=10
            elif current_tier == 3 and s > 3.0 and n >= 10:
                self.synapse.tiers[key] = 2

    # ─── 主循环 ───

    def breathe(self, n_generations: int = 1, verbose: bool = True):
        """呼吸循环: 细胞行走 + 周期性睡眠"""
        for g in range(n_generations):
            # 1. 细胞行走
            self._step_cells(n_steps=3)

            # 2. 消化行走缓冲
            self._digest_walk_buffer()
            self._walk_buffer = []

            # 3. 每代衰减
            for cell in self.cells:
                cell.age += 1
                cell.apply_decay()
                cell.myelin_decay()
                # 年龄上限
                if cell.age > VisionCell.MAX_AGE:
                    cell.age = 0
                    cell.total_reward = 0

            self.generation += 1

            # 4. 睡眠 (每 SLEEP_EVERY_N 代)
            if self.generation % self.SLEEP_EVERY_N == 0:
                if verbose:
                    print(f"  [SLEEP] gen={self.generation}")
                self._sleep_consolidate()
                self._sleep_prune()
                self._sleep_tier_audit()

            if verbose and self.generation % 50 == 0:
                stats = self.stats()
                print(f"  gen {self.generation}: cells={len(self.cells)} "
                      f"synapses={stats['synapses']} "
                      f"t3={stats['t3']} t2={stats['t2']}")

    def run(self, n_generations: int = 100, verbose: bool = True):
        """运行指定代数"""
        t0 = time.time()
        self.breathe(n_generations, verbose=verbose)
        elapsed = time.time() - t0
        if verbose:
            print(f"[DONE] {n_generations} generations in {elapsed:.1f}s")
            self.print_stats()

    # ─── 统计 ───

    def stats(self) -> dict:
        """突触统计"""
        tc = Counter()
        for k, v in self.synapse.tiers.items():
            tc[v] += 1
        edges = self.synapse.activations
        strong = sum(1 for e in edges.values() if e.get('s', 0) > 1.0)
        multi = sum(1 for e in edges.values() if len(e.get('n', set())) >= 2)
        total = len(edges)
        return {
            "synapses": total,
            "t1": tc.get(1, 0), "t2": tc.get(2, 0),
            "t3": tc.get(3, 0), "t4": tc.get(4, 0),
            "strong": strong,
            "multi_neuron": multi,
            "strong_pct": strong / max(1, total) * 100,
            "multi_pct": multi / max(1, total) * 100,
        }

    def print_stats(self):
        """打印统计"""
        s = self.stats()
        print(f"  synapses: {s['synapses']}")
        print(f"  tiers: t1={s['t1']} t2={s['t2']} t3={s['t3']} t4={s['t4']}")
        print(f"  strong(s>1): {s['strong']} ({s['strong_pct']:.1f}%)")
        print(f"  multi-neuron(n>=2): {s['multi_neuron']} ({s['multi_pct']:.1f}%)")

    def top_edges(self, n: int = 15) -> List[Tuple]:
        """返回 top n 高 s 边"""
        edges = []
        for (src, dst), edge in self.synapse.activations.items():
            s = edge.get('s', 0)
            n_neurons = len(edge.get('n', set()))
            tier = self.synapse.tiers.get((src, dst), 4)
            edges.append((src, dst, s, n_neurons, tier))
        edges.sort(key=lambda x: -x[2])
        return edges[:n]
