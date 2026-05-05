"""
评估流水线冒烟测试

目的：在不连接 Milvus/Neo4j/RAGAS 的前提下，验证评估代码本身能跑通。
跑这个脚本不会消耗 API 费用。

运行:
    python evaluation/smoke_test.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.bootstrap_ci import bootstrap_ci, format_ci, is_significantly_different
from evaluation.metrics_collector import MetricsCollector
from evaluation.ragas_evaluator import EvaluationResult, RAGASEvaluator
from evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    evaluate_rejection_behavior,
    is_rejection,
)


def test_retrieval_metrics():
    print("【1】检索硬指标")
    m = compute_retrieval_metrics(
        retrieved_paths=["a.md", "b.md", "c.md", "d.md", "e.md"],
        expected_paths=["a.md", "c.md", "x.md"],
        k=5,
    )
    print(f"  {m}")
    assert abs(m["recall@5"] - 2 / 3) < 1e-6, f"recall@5 should be 2/3, got {m['recall@5']}"
    assert abs(m["precision@5"] - 2 / 5) < 1e-6
    assert abs(m["mrr"] - 1.0) < 1e-6  # a.md 排在第一位
    assert m["hit@5"] == 1.0
    print("  ✅ 通过")


def test_rejection():
    print("\n【2】拒答行为评估")
    # 应拒答且确实拒答了 -> correct
    r1 = evaluate_rejection_behavior("抱歉，我无法回答这个问题。", should_answer=False)
    assert r1["correct_rejection"] == 1.0
    assert r1["false_answer_hallucination"] == 0.0
    # 应拒答但回答了 -> 幻觉
    r2 = evaluate_rejection_behavior("今天上海多云转晴，气温20度。", should_answer=False)
    assert r2["correct_rejection"] == 0.0
    assert r2["false_answer_hallucination"] == 1.0
    # 应回答但拒答了 -> 过度保守
    r3 = evaluate_rejection_behavior("抱歉，我不知道。", should_answer=True)
    assert r3["false_rejection"] == 1.0
    print(f"  正确拒答: {r1}")
    print(f"  幻觉:     {r2}")
    print(f"  过度拒答: {r3}")
    print("  ✅ 通过")


def test_bootstrap():
    print("\n【3】Bootstrap CI")
    values = [0.7, 0.8, 0.6, 0.9, 0.75, 0.65, 0.85, 0.72, 0.78, 0.68] * 10
    ci = bootstrap_ci(values, n_iter=500)
    print(f"  {format_ci(ci)} (n={ci['n']})")
    assert ci["lower"] < ci["mean"] < ci["upper"]
    # 不显著差异
    ci_a = bootstrap_ci([0.7] * 50, n_iter=500)
    ci_b = bootstrap_ci([0.71] * 50, n_iter=500)
    print(f"  显著性 0.70 vs 0.71: {is_significantly_different(ci_a, ci_b)}")
    print("  ✅ 通过")


def test_evaluator_no_ragas():
    print("\n【4】Evaluator (无 RAGAS, 关键词匹配兜底)")
    items = [
        {
            "id": "T001",
            "query": "怎么做清蒸鲈鱼？",
            "category": "entity_query",
            "difficulty": "easy",
            "should_answer": True,
            "expected_doc_paths": ["data/dishes/aquatic/清蒸鲈鱼/清蒸鲈鱼.md"],
            "reference_answer": "清蒸鲈鱼蒸10分钟。",
        },
        {
            "id": "T002",
            "query": "今天天气怎么样？",
            "category": "rejection_case",
            "difficulty": "easy",
            "should_answer": False,
            "expected_doc_paths": [],
            "reference_answer": "无法回答。",
        },
    ]
    retrieved_paths = [
        ["data/dishes/aquatic/清蒸鲈鱼/清蒸鲈鱼.md", "data/dishes/aquatic/红烧鱼.md"],
        [],
    ]
    retrieved_ctx = [
        ["[MOCK]鲈鱼蒸10分钟"],
        [],
    ]
    responses = [
        "鲈鱼蒸10分钟即可。",
        "抱歉，我无法回答。",
    ]

    evaluator = RAGASEvaluator(top_k=5)
    results = evaluator.simplified_evaluate(
        items=items,
        retrieved_paths_list=retrieved_paths,
        retrieved_contexts_list=retrieved_ctx,
        responses=responses,
    )
    assert len(results) == 2
    r1 = results[0]
    print(f"  T001 hit@k={r1.hit_at_k}, recall@k={r1.recall_at_k}, mrr={r1.mrr}")
    assert r1.hit_at_k == 1.0
    assert r1.recall_at_k == 1.0
    assert r1.mrr == 1.0

    r2 = results[1]
    print(f"  T002 should_answer={r2.should_answer}, correct_rejection={r2.correct_rejection}")
    assert r2.correct_rejection == 1.0
    print("  ✅ 通过")
    return results


def test_collector(results):
    print("\n【5】MetricsCollector + 报告生成")
    c = MetricsCollector(output_dir="/tmp/eval_smoke_test")
    c.add_results(results)
    s = c.calculate_summary()
    print(f"  总样本: {s.total_queries}, 应答: {s.answered_count}, 拒答: {s.rejection_case_count}")
    print(f"  正确拒答率: {s.correct_rejection_rate:.2%}")
    print(f"  幻觉率:     {s.hallucination_rate:.2%}")
    print(f"  Recall@K:   {s.recall_at_k.avg:.4f} [{s.recall_at_k.ci_lower:.4f}, {s.recall_at_k.ci_upper:.4f}]")
    print(f"  by_category keys: {list(s.by_category.keys())}")

    md = c.generate_report("markdown")
    assert "RAG 评估报告" in md
    assert "Context Recall" in md
    print(f"  Markdown 报告长度: {len(md)} chars")

    json_path = c.save_results("smoke_results.json")
    csv_path = c.save_csv("smoke_per_query.csv")
    md_path = c.save_markdown("smoke_report.md")
    print(f"  写入: {json_path}, {csv_path}, {md_path}")
    print("  ✅ 通过")


def test_dataset_validity():
    print("\n【6】eval_dataset.json 完整性检查")
    ds_path = Path(__file__).parent / "eval_dataset.json"
    if not ds_path.exists():
        print(f"  ⚠️  {ds_path} 不存在，跳过 (请先运行 merge_dataset.py)")
        return
    items = json.loads(ds_path.read_text(encoding="utf-8"))
    print(f"  共 {len(items)} 条")
    # 检查必填字段
    required = ["id", "query", "category", "should_answer"]
    for it in items:
        for f in required:
            assert f in it, f"missing {f} in {it.get('id', '?')}"
    # 检查 ID 唯一
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "ID 有重复"
    # 检查 expected_doc_paths 真实存在（仅抽查应该回答的样本）
    repo_root = Path(__file__).parent.parent
    missing = []
    answered = [it for it in items if it.get("should_answer", True)]
    for it in answered:
        for p in it.get("expected_doc_paths", []):
            full = repo_root / p
            if not full.exists():
                missing.append((it["id"], p))
    if missing:
        print(f"  ⚠️  {len(missing)} 个 expected_doc_paths 引用了不存在的文件，前 5 条:")
        for qid, p in missing[:5]:
            print(f"     {qid}: {p}")
    else:
        print("  所有 expected_doc_paths 都真实存在 ✅")

    cats = {}
    for it in items:
        cats[it["category"]] = cats.get(it["category"], 0) + 1
    print(f"  category 分布: {dict(sorted(cats.items(), key=lambda x: -x[1]))}")
    print("  ✅ 通过")


def main():
    print("=" * 60)
    print("评估流水线冒烟测试")
    print("=" * 60)
    test_retrieval_metrics()
    test_rejection()
    test_bootstrap()
    results = test_evaluator_no_ragas()
    test_collector(results)
    test_dataset_validity()
    print("\n" + "=" * 60)
    print("✅ 全部通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
