"""
检索硬指标模块

不依赖 LLM 的确定性指标，用于评估"检索阶段"的效果。
基于 expected_doc_paths（标注的标准答案文档）和 retrieved_doc_paths
（系统实际检索到的文档）计算。

指标：
    Recall@K   - 标准文档中被检索到的比例（漏检率的反面）
    Precision@K - 检索结果中相关文档的比例（噪声率的反面）
    F1@K       - Recall 和 Precision 的调和平均
    MRR        - 第一个相关文档排在第几位的倒数
    NDCG@K     - 考虑排序的归一化打分
    Hit@K      - 至少检索到 1 个相关文档的二元指标

使用示例:
    from evaluation.retrieval_metrics import compute_retrieval_metrics

    metrics = compute_retrieval_metrics(
        retrieved_paths=["a.md", "b.md", "c.md"],
        expected_paths=["a.md", "x.md"],
        k=5,
    )
"""

import math
import os
from typing import Dict, List, Optional


def _normalize_path(path: str) -> str:
    """规范化为 basename 去扩展名，使两端都收敛到菜名。

    这样 expected `data/dishes/.../糖醋鲤鱼.md` 与 retrieved `糖醋鲤鱼`（来自
    metadata.recipe_name）能对齐。RAG 端 Document.metadata 不带 source 路径，
    只带 recipe_name，所以双方收敛到菜名是最务实的对齐方式。
    """
    if not path:
        return ""
    p = os.path.basename(path.strip().lstrip("./").lstrip("/"))
    if p.endswith(".md"):
        p = p[:-3]
    return p


def recall_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    """检索到的相关文档数 / 应该检索到的文档总数。
    分子按 unique 文档计数：同一菜被多个 chunk 命中只算 1 次，避免 recall>1。
    """
    if not expected:
        return 0.0
    expected_set = {_normalize_path(p) for p in expected}
    retrieved_set = {_normalize_path(p) for p in retrieved[:k] if p}
    hits = len(retrieved_set & expected_set)
    return hits / len(expected_set)


def precision_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    """检索到的相关 unique 文档数 / 检索到的 unique 文档总数（截断到 k）。
    分子分母都按 unique 算，避免同一菜的多个 chunk 把 precision 推高过 1。
    """
    if not retrieved or k <= 0:
        return 0.0
    expected_set = {_normalize_path(p) for p in expected}
    retrieved_set = {_normalize_path(p) for p in retrieved[:k] if p}
    if not retrieved_set:
        return 0.0
    hits = len(retrieved_set & expected_set)
    return hits / len(retrieved_set)


def f1_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    """Recall@K 和 Precision@K 的调和平均"""
    r = recall_at_k(retrieved, expected, k)
    p = precision_at_k(retrieved, expected, k)
    if r + p == 0:
        return 0.0
    return 2 * r * p / (r + p)


def mrr(retrieved: List[str], expected: List[str]) -> float:
    """Mean Reciprocal Rank: 第一个相关文档排在第几位的倒数。
    如果一个相关文档都没找到，返回 0。
    """
    expected_set = {_normalize_path(p) for p in expected}
    for i, p in enumerate(retrieved):
        if _normalize_path(p) in expected_set:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    """至少检索到 1 个相关文档则为 1，否则 0"""
    expected_set = {_normalize_path(p) for p in expected}
    retrieved_topk = [_normalize_path(p) for p in retrieved[:k]]
    return 1.0 if any(p in expected_set for p in retrieved_topk) else 0.0


def ndcg_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    """归一化折损累计增益。这里把所有相关文档当作同样相关（rel=1），
    所以 DCG 退化为 sum(1 / log2(i+2))，IDCG 是把所有相关文档放在前面时的 DCG。

    NDCG 也按 unique 文档统计：同一菜的多个 chunk 只在首次出现位置计 gain，
    避免重复计入导致 NDCG > 1。
    """
    expected_set = {_normalize_path(p) for p in expected}
    if not expected_set:
        return 0.0

    # 按首次出现位置去重，保留排序信息
    seen = set()
    retrieved_unique = []
    for p in retrieved[:k]:
        np_p = _normalize_path(p)
        if np_p and np_p not in seen:
            seen.add(np_p)
            retrieved_unique.append(np_p)

    dcg = 0.0
    for i, p in enumerate(retrieved_unique):
        if p in expected_set:
            dcg += 1.0 / math.log2(i + 2)

    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    retrieved_paths: List[str],
    expected_paths: List[str],
    k: Optional[int] = None,
) -> Dict[str, float]:
    """
    一次性计算所有检索指标。

    Args:
        retrieved_paths: 系统实际检索到的文档路径列表（按相关性排序）
        expected_paths: 标注的相关文档路径列表
        k: 截断深度，默认为 len(retrieved_paths)

    Returns:
        dict 含 recall@k / precision@k / f1@k / mrr / hit@k / ndcg@k
    """
    if k is None:
        k = len(retrieved_paths) if retrieved_paths else 5

    return {
        f"recall@{k}": recall_at_k(retrieved_paths, expected_paths, k),
        f"precision@{k}": precision_at_k(retrieved_paths, expected_paths, k),
        f"f1@{k}": f1_at_k(retrieved_paths, expected_paths, k),
        "mrr": mrr(retrieved_paths, expected_paths),
        f"hit@{k}": hit_at_k(retrieved_paths, expected_paths, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_paths, expected_paths, k),
    }


# ----- 拒答与幻觉评估（基于 should_answer 标记） -----

REJECTION_PHRASES = [
    "抱歉",
    "无法回答",
    "不知道",
    "没有相关",
    "没有找到",
    "暂未收录",
    "超出我的范围",
    "无法提供",
    "无相关信息",
    "未找到",
    "no information",
    "i don't know",
    "i cannot",
    "i'm sorry",
]


def is_rejection(response: str) -> bool:
    """启发式判断：响应是否是拒答。
    用关键词匹配，简单但够用。如果想更准，可以让 LLM 判断。
    """
    if not response:
        return True
    low = response.lower().strip()
    return any(p in low for p in REJECTION_PHRASES)


def evaluate_rejection_behavior(
    response: str,
    should_answer: bool,
) -> Dict[str, float]:
    """
    评估拒答行为。

    返回三个 0/1 指标:
        correct_rejection  : 应该拒答时确实拒答了 (true negative)
        false_answer       : 应该拒答时回答了 (false positive / 幻觉)
        false_rejection    : 应该回答时拒答了 (false negative / 过度保守)
    """
    rejected = is_rejection(response)
    return {
        "correct_rejection": float((not should_answer) and rejected),
        "false_answer_hallucination": float((not should_answer) and (not rejected)),
        "false_rejection": float(should_answer and rejected),
    }
