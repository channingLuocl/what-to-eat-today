"""
指标收集和汇总模块

功能：
1. 收集 EvaluationResult，按多维度汇总
2. 按 category / difficulty 分组统计
3. 拒答评估汇总
4. 计算 bootstrap 置信区间
5. 生成 Markdown / JSON / CSV 报告
"""

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bootstrap_ci import bootstrap_ci, format_ci, is_significantly_different
from .ragas_evaluator import EvaluationResult

logger = logging.getLogger(__name__)


# ============== 汇总数据结构 ==============


@dataclass
class MetricStats:
    """单个指标的统计信息"""
    avg: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n: int = 0


@dataclass
class MetricsSummary:
    """完整指标汇总"""
    total_queries: int = 0
    answered_count: int = 0
    rejection_case_count: int = 0

    # RAGAS 软指标
    context_recall: MetricStats = field(default_factory=MetricStats)
    faithfulness: MetricStats = field(default_factory=MetricStats)
    answer_relevancy: MetricStats = field(default_factory=MetricStats)

    # 检索硬指标
    recall_at_k: MetricStats = field(default_factory=MetricStats)
    precision_at_k: MetricStats = field(default_factory=MetricStats)
    f1_at_k: MetricStats = field(default_factory=MetricStats)
    mrr: MetricStats = field(default_factory=MetricStats)
    hit_at_k: MetricStats = field(default_factory=MetricStats)
    ndcg_at_k: MetricStats = field(default_factory=MetricStats)

    # 拒答行为（占应该拒答样本的比例）
    correct_rejection_rate: float = 0.0
    hallucination_rate: float = 0.0
    false_rejection_rate: float = 0.0

    # 按维度
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ============== 收集器 ==============


class MetricsCollector:
    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[EvaluationResult] = []

    # ---- 收集 ----

    def add_result(self, result: EvaluationResult):
        self.results.append(result)

    def add_results(self, results: List[EvaluationResult]):
        self.results.extend(results)

    # ---- 汇总 ----

    def calculate_summary(self, with_ci: bool = True) -> MetricsSummary:
        if not self.results:
            return MetricsSummary()

        answerable = [r for r in self.results if r.should_answer]
        rejection_cases = [r for r in self.results if not r.should_answer]

        # 软指标只在 answerable 上算
        ctx_recall = [r.context_recall for r in answerable]
        faith = [r.faithfulness for r in answerable]
        relev = [r.answer_relevancy for r in answerable]

        # 硬指标也只在 answerable 上算（拒答样本无 expected_doc_paths）
        rk = [r.recall_at_k for r in answerable]
        pk = [r.precision_at_k for r in answerable]
        f1k = [r.f1_at_k for r in answerable]
        mrr_l = [r.mrr for r in answerable]
        hk = [r.hit_at_k for r in answerable]
        ndcgk = [r.ndcg_at_k for r in answerable]

        # 拒答指标
        correct_rej = sum(r.correct_rejection for r in rejection_cases)
        hallu = sum(r.false_answer_hallucination for r in rejection_cases)
        false_rej = sum(r.false_rejection for r in answerable)

        n_rej = max(len(rejection_cases), 1)
        n_ans = max(len(answerable), 1)

        summary = MetricsSummary(
            total_queries=len(self.results),
            answered_count=len(answerable),
            rejection_case_count=len(rejection_cases),
            context_recall=self._stats(ctx_recall, with_ci),
            faithfulness=self._stats(faith, with_ci),
            answer_relevancy=self._stats(relev, with_ci),
            recall_at_k=self._stats(rk, with_ci),
            precision_at_k=self._stats(pk, with_ci),
            f1_at_k=self._stats(f1k, with_ci),
            mrr=self._stats(mrr_l, with_ci),
            hit_at_k=self._stats(hk, with_ci),
            ndcg_at_k=self._stats(ndcgk, with_ci),
            correct_rejection_rate=correct_rej / n_rej,
            hallucination_rate=hallu / n_rej,
            false_rejection_rate=false_rej / n_ans,
            by_category=self._group_metrics(self.results, "category"),
            by_difficulty=self._group_metrics(self.results, "difficulty"),
        )
        return summary

    @staticmethod
    def _stats(values: List[float], with_ci: bool = True) -> MetricStats:
        # 过滤 NaN（部分 RAGAS job 失败时不让单个 NaN 污染整列 mean）
        import math
        values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not values:
            return MetricStats()
        ci = bootstrap_ci(values, n_iter=1000, confidence=0.95) if with_ci else {
            "mean": sum(values) / len(values),
            "std": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "n": len(values),
        }
        return MetricStats(
            avg=round(sum(values) / len(values), 4),
            min=round(min(values), 4),
            max=round(max(values), 4),
            std=round(ci["std"], 4),
            ci_lower=round(ci["lower"], 4),
            ci_upper=round(ci["upper"], 4),
            n=len(values),
        )

    def _group_metrics(self, results: List[EvaluationResult], key: str) -> Dict[str, Dict[str, float]]:
        """按 key（category/difficulty）分组算关键指标的均值"""
        groups: Dict[str, List[EvaluationResult]] = {}
        for r in results:
            k = getattr(r, key, "") or "unknown"
            groups.setdefault(k, []).append(r)

        out: Dict[str, Dict[str, float]] = {}
        for k, items in groups.items():
            ans = [r for r in items if r.should_answer]
            n_ans = max(len(ans), 1)
            n_total = max(len(items), 1)
            out[k] = {
                "n": len(items),
                "answerable_n": len(ans),
                "context_recall_avg": round(sum(r.context_recall for r in ans) / n_ans, 4) if ans else 0.0,
                "faithfulness_avg": round(sum(r.faithfulness for r in ans) / n_ans, 4) if ans else 0.0,
                "answer_relevancy_avg": round(sum(r.answer_relevancy for r in ans) / n_ans, 4) if ans else 0.0,
                "recall_at_k_avg": round(sum(r.recall_at_k for r in ans) / n_ans, 4) if ans else 0.0,
                "mrr_avg": round(sum(r.mrr for r in ans) / n_ans, 4) if ans else 0.0,
                "hit_at_k_avg": round(sum(r.hit_at_k for r in ans) / n_ans, 4) if ans else 0.0,
            }
        return out

    # ---- 报告生成 ----

    def generate_report(self, format: str = "markdown") -> str:
        summary = self.calculate_summary()
        if format == "json":
            return json.dumps(self._summary_to_dict(summary), ensure_ascii=False, indent=2)
        if format == "html":
            md = self._markdown(summary)
            return f"<html><body><pre>{md}</pre></body></html>"
        return self._markdown(summary)

    def _summary_to_dict(self, s: MetricsSummary) -> Dict[str, Any]:
        d = asdict(s)
        return d

    def _markdown(self, s: MetricsSummary) -> str:
        L = []
        L.append("# RAG 评估报告")
        L.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        L.append(f"总查询数: **{s.total_queries}**  (应答 {s.answered_count} / 拒答样本 {s.rejection_case_count})")

        # 总体指标
        L.append("\n## 总体指标 (95% Bootstrap CI)\n")
        L.append("| 指标 | 平均值 [CI 下界, CI 上界] | std | n |")
        L.append("|------|---------------------------|-----|---|")
        for name, stat in [
            ("Context Recall (RAGAS)", s.context_recall),
            ("Faithfulness (RAGAS)", s.faithfulness),
            ("Answer Relevancy (RAGAS)", s.answer_relevancy),
            ("Recall@K (硬)", s.recall_at_k),
            ("Precision@K (硬)", s.precision_at_k),
            ("F1@K (硬)", s.f1_at_k),
            ("MRR (硬)", s.mrr),
            ("Hit@K (硬)", s.hit_at_k),
            ("NDCG@K (硬)", s.ndcg_at_k),
        ]:
            L.append(
                f"| {name} | {stat.avg:.4f} [{stat.ci_lower:.4f}, {stat.ci_upper:.4f}] | "
                f"{stat.std:.4f} | {stat.n} |"
            )

        # 拒答评估
        L.append("\n## 拒答行为评估\n")
        L.append(f"- 正确拒答率 (correct_rejection / 应拒答样本数): **{s.correct_rejection_rate:.2%}**")
        L.append(f"- 幻觉率 (false_answer / 应拒答样本数): **{s.hallucination_rate:.2%}**")
        L.append(f"- 过度拒答率 (false_rejection / 应答样本数): **{s.false_rejection_rate:.2%}**")

        # 按 category
        if s.by_category:
            L.append("\n## 按 category 分组\n")
            L.append("| category | n | ctx_recall | faithfulness | answer_relevancy | recall@k | mrr | hit@k |")
            L.append("|----------|---|------------|--------------|-------------------|----------|-----|-------|")
            for cat, m in sorted(s.by_category.items(), key=lambda x: -x[1]["n"]):
                L.append(
                    f"| {cat} | {int(m['n'])} | {m['context_recall_avg']:.3f} | "
                    f"{m['faithfulness_avg']:.3f} | {m['answer_relevancy_avg']:.3f} | "
                    f"{m['recall_at_k_avg']:.3f} | {m['mrr_avg']:.3f} | {m['hit_at_k_avg']:.3f} |"
                )

        # 按 difficulty
        if s.by_difficulty:
            L.append("\n## 按 difficulty 分组\n")
            L.append("| difficulty | n | ctx_recall | faithfulness | recall@k | mrr |")
            L.append("|------------|---|------------|--------------|----------|-----|")
            for d, m in sorted(s.by_difficulty.items(), key=lambda x: x[0]):
                L.append(
                    f"| {d} | {int(m['n'])} | {m['context_recall_avg']:.3f} | "
                    f"{m['faithfulness_avg']:.3f} | {m['recall_at_k_avg']:.3f} | {m['mrr_avg']:.3f} |"
                )

        # Top worst cases
        L.append("\n## Top 10 最差检索 cases (按 hit@k=0 + 低 recall)\n")
        worst_retrieval = sorted(
            [r for r in self.results if r.should_answer and r.expected_doc_paths],
            key=lambda r: (r.hit_at_k, r.recall_at_k, r.mrr),
        )[:10]
        for r in worst_retrieval:
            L.append(f"\n- **{r.query_id}** (`{r.category}`) **Q**: {r.query}")
            L.append(f"  - expected: `{r.expected_doc_paths[:3]}`")
            L.append(f"  - retrieved: `{r.retrieved_doc_paths[:3]}`")
            L.append(f"  - recall@k={r.recall_at_k:.2f}, mrr={r.mrr:.2f}")

        # Top worst RAGAS faithfulness
        worst_faith = sorted(
            [r for r in self.results if r.should_answer],
            key=lambda r: r.faithfulness,
        )[:5]
        if worst_faith:
            L.append("\n## Top 5 最低 faithfulness (可能幻觉) cases\n")
            for r in worst_faith:
                L.append(f"\n- **{r.query_id}** **Q**: {r.query}")
                L.append(f"  - response: {r.response[:120]}{'...' if len(r.response) > 120 else ''}")
                L.append(f"  - faithfulness={r.faithfulness:.3f}")

        # 出错的 case
        errored = [r for r in self.results if r.error]
        if errored:
            L.append(f"\n## ⚠️  出错样本 ({len(errored)} 条)\n")
            for r in errored[:10]:
                L.append(f"- {r.query_id}: {r.error}")

        return "\n".join(L)

    # ---- 文件导出 ----

    def save_results(self, filename: Optional[str] = None) -> Path:
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_results_{ts}.json"
        path = self.output_dir / filename
        summary = self.calculate_summary()
        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self._summary_to_dict(summary),
            "results": [r.to_dict() for r in self.results],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"评估结果已保存: {path}")
        return path

    def save_csv(self, filename: Optional[str] = None) -> Path:
        """导出每条 query 一行的 CSV，方便人工 review"""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_per_query_{ts}.csv"
        path = self.output_dir / filename

        if not self.results:
            return path

        fieldnames = [
            "query_id", "category", "difficulty", "should_answer",
            "query", "response_preview",
            "context_recall", "faithfulness", "answer_relevancy",
            "recall_at_k", "precision_at_k", "mrr", "hit_at_k", "ndcg_at_k",
            "correct_rejection", "false_answer_hallucination", "false_rejection",
            "expected_count", "retrieved_count",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self.results:
                w.writerow({
                    "query_id": r.query_id,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "should_answer": r.should_answer,
                    "query": r.query,
                    "response_preview": (r.response or "")[:200],
                    "context_recall": round(r.context_recall, 4),
                    "faithfulness": round(r.faithfulness, 4),
                    "answer_relevancy": round(r.answer_relevancy, 4),
                    "recall_at_k": round(r.recall_at_k, 4),
                    "precision_at_k": round(r.precision_at_k, 4),
                    "mrr": round(r.mrr, 4),
                    "hit_at_k": round(r.hit_at_k, 4),
                    "ndcg_at_k": round(r.ndcg_at_k, 4),
                    "correct_rejection": r.correct_rejection,
                    "false_answer_hallucination": r.false_answer_hallucination,
                    "false_rejection": r.false_rejection,
                    "expected_count": len(r.expected_doc_paths),
                    "retrieved_count": len(r.retrieved_doc_paths),
                })
        logger.info(f"per-query CSV 已保存: {path}")
        return path

    def save_markdown(self, filename: Optional[str] = None) -> Path:
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_report_{ts}.md"
        path = self.output_dir / filename
        path.write_text(self.generate_report("markdown"), encoding="utf-8")
        logger.info(f"Markdown 报告已保存: {path}")
        return path

    # ---- 加载历史结果 ----

    def load_results(self, filepath: str) -> List[EvaluationResult]:
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        results = []
        for item in data.get("results", []):
            # 兼容旧格式 (有 metrics 子字典) 和新格式
            if "metrics" in item:
                m = item["metrics"]
                results.append(EvaluationResult(
                    query_id=item.get("query_id", ""),
                    query=item.get("query", ""),
                    response=item.get("response", ""),
                    retrieved_contexts=item.get("retrieved_contexts", []),
                    context_recall=m.get("context_recall", 0),
                    faithfulness=m.get("faithfulness", 0),
                    answer_relevancy=m.get("answer_relevancy", 0),
                ))
            else:
                # 假设是新格式：所有字段在顶层
                results.append(EvaluationResult(**{
                    k: v for k, v in item.items()
                    if k in EvaluationResult.__dataclass_fields__
                }))
        return results


# ============== 对比模式 ==============


def compare_runs(
    old_results_path: str,
    new_results_path: str,
) -> str:
    """对比两次评估结果，输出 Markdown 报告"""
    old_collector = MetricsCollector()
    new_collector = MetricsCollector()
    old_collector.results = old_collector.load_results(old_results_path)
    new_collector.results = new_collector.load_results(new_results_path)

    old_s = old_collector.calculate_summary()
    new_s = new_collector.calculate_summary()

    L = []
    L.append("# 评估结果对比\n")
    L.append(f"- 旧: `{old_results_path}` (n={old_s.total_queries})")
    L.append(f"- 新: `{new_results_path}` (n={new_s.total_queries})\n")

    # 总体指标对比
    L.append("## 指标变化 (含显著性)\n")
    L.append("| 指标 | 旧 [CI] | 新 [CI] | Δ | 显著? |")
    L.append("|------|---------|---------|---|--------|")

    pairs = [
        ("Context Recall", old_s.context_recall, new_s.context_recall),
        ("Faithfulness", old_s.faithfulness, new_s.faithfulness),
        ("Answer Relevancy", old_s.answer_relevancy, new_s.answer_relevancy),
        ("Recall@K", old_s.recall_at_k, new_s.recall_at_k),
        ("MRR", old_s.mrr, new_s.mrr),
        ("Hit@K", old_s.hit_at_k, new_s.hit_at_k),
        ("NDCG@K", old_s.ndcg_at_k, new_s.ndcg_at_k),
    ]
    for name, o, n in pairs:
        delta = n.avg - o.avg
        sig = is_significantly_different(
            {"lower": o.ci_lower, "upper": o.ci_upper, "mean": o.avg},
            {"lower": n.ci_lower, "upper": n.ci_upper, "mean": n.avg},
        )
        sig_str = "✅" if sig else "❌"
        L.append(
            f"| {name} | {o.avg:.4f} [{o.ci_lower:.4f}, {o.ci_upper:.4f}] | "
            f"{n.avg:.4f} [{n.ci_lower:.4f}, {n.ci_upper:.4f}] | "
            f"{delta:+.4f} | {sig_str} |"
        )

    # case 翻转
    old_by_id = {r.query_id: r for r in old_collector.results}
    flipped_better = []
    flipped_worse = []
    for r_new in new_collector.results:
        r_old = old_by_id.get(r_new.query_id)
        if not r_old:
            continue
        if r_old.hit_at_k == 0 and r_new.hit_at_k == 1:
            flipped_better.append(r_new)
        elif r_old.hit_at_k == 1 and r_new.hit_at_k == 0:
            flipped_worse.append(r_new)

    L.append(f"\n## case-level 翻转 (基于 Hit@K)\n")
    L.append(f"- 改善 (旧 miss → 新 hit): **{len(flipped_better)}** 条")
    L.append(f"- 退化 (旧 hit → 新 miss): **{len(flipped_worse)}** 条")
    if flipped_worse:
        L.append("\n### 退化的 case（重点检查）\n")
        for r in flipped_worse[:10]:
            L.append(f"- **{r.query_id}** ({r.category}) {r.query}")

    return "\n".join(L)
