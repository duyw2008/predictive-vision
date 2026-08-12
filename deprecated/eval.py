#!/usr/bin/env python3
"""
冷眼 — 预测反馈视觉系统
eval.py: 评估 — 对比度 vs 识别率曲线
"""

import sys
import os
import json
import time
import numpy as np
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from train import ColdEye, load_mnist, train_mnist


def evaluate_contrast_curve(
    model: ColdEye,
    X_test: np.ndarray,
    y_test: np.ndarray,
    contrasts: List[float] = None,
    n_samples: int = 500,
) -> dict:
    """评估不同对比度下的识别率

    返回: {contrast: accuracy}
    """
    if contrasts is None:
        contrasts = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05]

    results = {}
    n = min(n_samples, len(X_test))

    for contrast in contrasts:
        correct = 0
        total = 0

        for i in range(n):
            # 降低对比度
            img = X_test[i].copy()
            mean = img.mean()
            img_low = mean + (img - mean) * contrast

            pred, conf = model.predict(img_low)
            if pred == y_test[i]:
                correct += 1
            total += 1

        acc = correct / total
        results[contrast] = acc
        print(f"  contrast={contrast:.2f}: {acc:.1%} ({correct}/{total})")

    return results


def compare_with_baseline(
    model: ColdEye,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples: int = 500,
) -> dict:
    """对比: 有预测传播 vs 无预测传播 (纯 bottom-up)

    无预测传播 = 只做一次 patch 匹配, 不做迭代 settle
    """
    model.graph.settle = model.graph.settle  # save ref

    contrasts = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
    n = min(n_samples, len(X_test))

    results = {"with_prediction": {}, "without_prediction": {}}

    for contrast in contrasts:
        # 有预测
        correct_with = 0
        correct_without = 0

        for i in range(n):
            img = X_test[i].copy()
            mean = img.mean()
            img_low = mean + (img - mean) * contrast

            # 有预测传播
            model.vision.set_image(img_low, contrast=1.0)
            for _ in range(20):
                errors = model.graph.propagate_predictions()
                if not errors or max(errors.values()) < 0.001:
                    break
            pred, _ = model._get_best_prediction()
            if pred == y_test[i]:
                correct_with += 1

            # 无预测传播 (只看初始 bottom-up 匹配)
            model.vision.set_image(img_low, contrast=1.0)
            pred, _ = model._get_best_prediction()
            if pred == y_test[i]:
                correct_without += 1

        results["with_prediction"][contrast] = correct_with / n
        results["without_prediction"][contrast] = correct_without / n

        print(f"  contrast={contrast:.2f}: with={correct_with/n:.1%} "
              f"without={correct_without/n:.1%} "
              f"Δ={abs(correct_with-correct_without)/n:.1%}")

    return results


def main():
    print("🧊 冷眼 — 评估")
    print("=" * 40)

    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist()

    # 训练模型
    model = ColdEye(n_nodes=100)

    def contrast_schedule(gen):
        if gen < 300:
            return 1.0
        elif gen < 800:
            return 0.5
        else:
            return 0.2

    train_mnist(
        model,
        X_train[:3000],
        y_train[:3000],
        n_epochs=5,
        contrast_schedule=contrast_schedule,
    )

    # 评估
    print("\n📊 对比度曲线:")
    results = evaluate_contrast_curve(model, X_test, y_test, n_samples=500)

    # 保存
    os.makedirs("data", exist_ok=True)
    with open("data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果保存到 data/eval_results.json")

    # 打印总结
    print(f"\n   对比度   | 准确率")
    print(f"  ---------|------")
    for c, acc in sorted(results.items()):
        bar = "█" * int(acc * 30)
        print(f"   {c:.2f}     | {acc:.1%} {bar}")


if __name__ == "__main__":
    main()
