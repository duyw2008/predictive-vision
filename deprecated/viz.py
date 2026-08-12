#!/usr/bin/env python3
"""
冷眼 — 预测反馈视觉系统
viz.py: 可视化 — 图结构, tier分布, 预测填充效果
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from train import ColdEye, load_mnist


def plot_tier_distribution(stats_history: list, save_path: str = "data/tier_evolution.png"):
    """绘 tier 分布随时间变化"""
    fig, ax = plt.subplots(figsize=(10, 5))

    gens = [s["generation"] for s in stats_history]
    tiers = [1, 2, 3, 4]
    colors = {1: "#22c55e", 2: "#3b82f6", 3: "#f59e0b", 4: "#ef4444"}

    for tier in tiers:
        counts = [s["tier_dist"].get(tier, 0) for s in stats_history]
        ax.plot(gens, counts, label=f"Tier {tier}", color=colors[tier], linewidth=2)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Edge Count")
    ax.set_title("冷眼 — 突触 Tier 分布演化")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → {save_path}")


def plot_contrast_curve(results: dict, save_path: str = "data/contrast_curve.png"):
    """绘对比度-准确率曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    contrasts = sorted(results.keys())
    accs = [results[c] for c in contrasts]

    ax.plot(contrasts, accs, 'o-', color="#3b82f6", linewidth=2, markersize=8)
    ax.fill_between(contrasts, accs, alpha=0.1, color="#3b82f6")

    ax.set_xlabel("Contrast")
    ax.set_ylabel("Accuracy")
    ax.set_title("冷眼 — 对比度 vs 识别率")
    ax.set_xlim(0, max(contrasts) * 1.1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 标注每个点
    for c, acc in zip(contrasts, accs):
        ax.annotate(f"{acc:.1%}", (c, acc), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → {save_path}")


def plot_prediction_fill(
    model: ColdEye,
    image: np.ndarray,
    contrast: float = 0.1,
    save_path: str = "data/prediction_fill.png",
):
    """可视化预测填充效果: 原始图 → 低对比度 → 预测填充后"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 原始
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    # 低对比度
    mean = image.mean()
    img_low = mean + (image - mean) * contrast
    axes[1].imshow(img_low, cmap="gray")
    axes[1].set_title(f"Low Contrast ({contrast:.0%})", fontsize=12)
    axes[1].axis("off")

    # 推理: bottom-up 激活 (无预测)
    model.vision.set_image(img_low, contrast=1.0)
    enhanced_bottomup = model.vision.get_enhanced_image()
    axes[2].imshow(enhanced_bottomup, cmap="gray")
    axes[2].set_title("Bottom-up Only", fontsize=12)
    axes[2].axis("off")

    # 推理: 预测填充后
    for _ in range(20):
        errors = model.graph.propagate_predictions()
        if not errors or max(errors.values()) < 0.001:
            break
    enhanced_pred = model.vision.get_enhanced_image()
    axes[3].imshow(enhanced_pred, cmap="gray")
    axes[3].set_title("With Prediction Fill", fontsize=12)
    axes[3].axis("off")

    plt.suptitle(f"冷眼 — 预测填充 (contrast={contrast:.0%})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


def plot_graph_structure(model: ColdEye, save_path: str = "data/graph_structure.png"):
    """可视化图结构: 节点按 tier 着色, 边按 s值 粗细"""
    fig, ax = plt.subplots(figsize=(12, 10))

    nodes = model.graph.nodes
    node_ids = sorted(nodes.keys())
    n = len(node_ids)

    # 布局: 圆环 + 中心聚团
    positions = {}
    for i, nid in enumerate(node_ids):
        node = nodes[nid]
        tier = node.tier
        # 低tier靠近中心
        radius = 0.1 + tier * 0.2 + np.random.uniform(-0.05, 0.05)
        angle = (i / n) * 2 * np.pi + np.random.uniform(-0.1, 0.1)
        positions[nid] = (radius * np.cos(angle), radius * np.sin(angle))

    # 边
    for src in node_ids:
        for dst, s_val in model.graph.adjacency.get(src, {}).items():
            if src not in positions or dst not in positions:
                continue
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            alpha = min(1.0, s_val * 2)
            lw = max(0.1, s_val * 3)
            color = "#3b82f6" if s_val > 0.3 else "#94a3b8"
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                    linewidth=lw, zorder=1)

    # 节点
    tier_colors = {1: "#22c55e", 2: "#3b82f6", 3: "#f59e0b", 4: "#ef4444"}
    for nid, (x, y) in positions.items():
        tier = nodes[nid].tier
        size = 30 + (4 - tier) * 20  # 低tier更大
        ax.scatter(x, y, s=size, c=tier_colors.get(tier, "#gray"),
                   alpha=0.8, edgecolors="white", linewidth=0.5, zorder=2)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=tier_colors[t], label=f"Tier {t}")
        for t in [1, 2, 3, 4]
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_title(f"冷眼 — 图结构 (gen={model.generation}, 边={model.graph.edge_count})")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


def plot_node_templates(model: ColdEye, n_examples: int = 16,
                        save_path: str = "data/node_templates.png"):
    """可视化节点模板: 每个节点学到了什么特征"""
    nodes = sorted(model.graph.nodes.values(),
                   key=lambda n: n.tier * 100 - n.reward)  # 低tier高reward优先

    n = min(n_examples, len(nodes))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if rows > 1 else [axes]

    for i in range(n):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break

        node = nodes[i]
        template = node.template.reshape(8, 8)
        ax.imshow(template, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax.set_title(f"{node.node_id}\nt{node.tier} r={node.reward:.1f}",
                     fontsize=7)
        ax.axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("冷眼 — 节点特征模板 (低tier=可靠特征)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


def main():
    print("🧊 冷眼 — 可视化")
    print("=" * 40)

    # 加载预训练模型 (或创建新模型快速训练)
    X_train, y_train, X_test, y_test = load_mnist()

    model = ColdEye(n_nodes=100)
    stats_history = []

    from train import train_mnist

    def contrast_schedule(gen):
        if gen < 300:
            return 1.0
        elif gen < 800:
            return 0.5
        else:
            return 0.2

    def on_batch_callback(model):
        if model.generation % 100 == 0:
            stats_history.append(model.stats())

    # 快速训练
    train_mnist(
        model,
        X_train[:2000],
        y_train[:2000],
        n_epochs=3,
        contrast_schedule=contrast_schedule,
    )

    os.makedirs("data", exist_ok=True)

    # 生成可视化
    print("\n生成图表...")
    plot_tier_distribution(stats_history)
    plot_node_templates(model)
    plot_graph_structure(model)

    # 预测填充示例
    sample_img = X_test[0]
    plot_prediction_fill(model, sample_img, contrast=0.1,
                         save_path="data/prediction_fill_010.png")
    plot_prediction_fill(model, sample_img, contrast=0.3,
                         save_path="data/prediction_fill_030.png")

    print("\n✅ 全部可视化完成 → data/")


if __name__ == "__main__":
    main()
