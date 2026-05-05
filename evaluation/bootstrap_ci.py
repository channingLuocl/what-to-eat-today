"""
Bootstrap 置信区间

对评估结果做有放回采样，输出每个指标的 95% 置信区间。
用于判断"指标变化是否显著"——如果两次评估的 CI 重叠，那说明
差异可能只是采样噪声，不是真的提升/下降。

使用示例:
    from evaluation.bootstrap_ci import bootstrap_ci

    ci = bootstrap_ci([0.7, 0.8, 0.6, 0.9, 0.75, ...], n_iter=1000)
    # 返回 {"mean": 0.75, "lower": 0.68, "upper": 0.82}
"""

import random
from typing import Dict, List, Optional, Sequence


def bootstrap_ci(
    values: Sequence[float],
    n_iter: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = 42,
) -> Dict[str, float]:
    """
    对一组数值做 bootstrap 置信区间估计。

    Args:
        values: 原始指标值列表（一条 query 一个值）
        n_iter: 重采样次数，默认 1000
        confidence: 置信度，默认 0.95
        seed: 随机种子，None 则不固定

    Returns:
        dict: {"mean": float, "std": float, "lower": float, "upper": float, "n": int}
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}

    rng = random.Random(seed) if seed is not None else random.Random()
    n = len(values)
    values = list(values)

    means = []
    for _ in range(n_iter):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = (1 - confidence) / 2
    lower_idx = int(alpha * n_iter)
    upper_idx = int((1 - alpha) * n_iter) - 1
    upper_idx = max(0, min(upper_idx, n_iter - 1))

    mean_val = sum(values) / n
    var = sum((v - mean_val) ** 2 for v in values) / n
    std = var ** 0.5

    return {
        "mean": round(mean_val, 4),
        "std": round(std, 4),
        "lower": round(means[lower_idx], 4),
        "upper": round(means[upper_idx], 4),
        "n": n,
    }


def is_significantly_different(
    ci_a: Dict[str, float],
    ci_b: Dict[str, float],
) -> bool:
    """两个置信区间是否不重叠（粗略的显著性判断）"""
    return ci_a["upper"] < ci_b["lower"] or ci_b["upper"] < ci_a["lower"]


def format_ci(ci: Dict[str, float], precision: int = 4) -> str:
    """格式化输出，例如: 0.7500 [0.6800, 0.8200]"""
    fmt = f"{{:.{precision}f}}"
    mean = fmt.format(ci["mean"])
    lo = fmt.format(ci["lower"])
    hi = fmt.format(ci["upper"])
    return f"{mean} [{lo}, {hi}]"
