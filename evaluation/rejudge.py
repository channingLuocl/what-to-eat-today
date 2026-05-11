"""
只重跑 RAGAS 打分，复用已有检索+生成结果
用法: python evaluation/rejudge.py evaluation_results/v1.0_full/eval_results_xxx.json
"""
import json, sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(results_path: str):
    with open(results_path) as f:
        data = json.load(f)
    results = data["results"]

    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import context_recall, faithfulness, answer_relevancy
    from evaluation.llm_factory import build_ragas_llm, build_ragas_embeddings, inject_into_metrics
    from evaluation.ragas_evaluator import EvaluationResult
    from evaluation.metrics_collector import MetricsCollector
    from evaluation.bootstrap_ci import bootstrap_ci, format_ci

    llm = build_ragas_llm()
    emb = build_ragas_embeddings()

    # 构建 RAGAS samples
    samples = []
    for r in results:
        if not r.get("should_answer"):
            continue
        kw = {
            "user_input": r["query"],
            "response": r["response"],
            "retrieved_contexts": r.get("retrieved_contexts") or [""],
        }
        if r.get("reference_answer"):
            kw["reference"] = r["reference_answer"]
        samples.append((r["query_id"], SingleTurnSample(**kw)))

    logger.info(f"加载 {len(samples)} 条样本，开始 RAGAS 打分...")

    metrics = [context_recall, faithfulness, answer_relevancy]
    inject_into_metrics(metrics, llm=llm, embeddings=emb)

    dataset = EvaluationDataset(samples=[s for _, s in samples])
    from ragas.run_config import RunConfig
    run_config = RunConfig(timeout=120, max_workers=16, max_retries=2)

    eval_result = evaluate(
        dataset=dataset, metrics=metrics,
        llm=llm, embeddings=emb,
        run_config=run_config, show_progress=True,
    )

    # 填回 RAGAS 分数
    df = eval_result.to_pandas()
    id_to_result = {r["query_id"]: r for r in results}
    for idx, (qid, _) in enumerate(samples):
        r = id_to_result.get(qid)
        if r is None:
            continue
        try:
            row = df.iloc[idx]
            r["context_recall"] = float(row.get("context_recall", 0) or 0)
            r["faithfulness"] = float(row.get("faithfulness", 0) or 0)
            r["answer_relevancy"] = float(row.get("answer_relevancy", 0) or 0)
        except Exception as e:
            logger.warning(f"无法解析 {qid}: {e}")

    # 转换为 EvaluationResult 并重新算摘要
    eval_results = []
    for r in results:
        eval_results.append(EvaluationResult(
            query_id=r.get("query_id", ""),
            query=r.get("query", ""),
            category=r.get("category", ""),
            difficulty=r.get("difficulty", ""),
            should_answer=r.get("should_answer", True),
            expected_doc_paths=r.get("expected_doc_paths", []),
            retrieved_doc_paths=r.get("retrieved_doc_paths", []),
            retrieved_contexts=r.get("retrieved_contexts", []),
            response=r.get("response", ""),
            reference_answer=r.get("reference_answer", ""),
            context_recall=r.get("context_recall", 0),
            faithfulness=r.get("faithfulness", 0),
            answer_relevancy=r.get("answer_relevancy", 0),
            recall_at_k=r.get("recall_at_k", 0),
            precision_at_k=r.get("precision_at_k", 0),
            f1_at_k=r.get("f1_at_k", 0),
            mrr=r.get("mrr", 0),
            hit_at_k=r.get("hit_at_k", 0),
            ndcg_at_k=r.get("ndcg_at_k", 0),
            correct_rejection=r.get("correct_rejection", 0),
            false_answer_hallucination=r.get("false_answer_hallucination", 0),
            false_rejection=r.get("false_rejection", 0),
        ))

    out_dir = str(Path(results_path).parent)
    collector = MetricsCollector(output_dir=out_dir)
    collector.results = eval_results
    ts = Path(results_path).stem.replace("eval_results_", "")
    collector.save_results(f"eval_results_{ts}_rejudged.json")
    collector.save_csv(f"eval_per_query_{ts}_rejudged.csv")
    collector.save_markdown(f"eval_report_{ts}_rejudged.md")

    # 打印摘要
    summary = collector._compute_summary(eval_results)
    s = summary

    ci = lambda vals: format_ci(*bootstrap_ci(vals))

    rag_vals = {
        "Context Recall": [r.context_recall for r in eval_results if r.should_answer and r.context_recall > 0],
        "Faithfulness": [r.faithfulness for r in eval_results if r.should_answer and r.faithfulness > 0],
        "Answer Relevancy": [r.answer_relevancy for r in eval_results if r.should_answer and r.answer_relevancy > 0],
    }
    hard_vals = {
        "Recall@K": [r.recall_at_k for r in eval_results if r.should_answer],
        "Precision@K": [r.precision_at_k for r in eval_results if r.should_answer],
        "MRR": [r.mrr for r in eval_results if r.should_answer],
        "Hit@K": [r.hit_at_k for r in eval_results if r.should_answer],
        "NDCG@K": [r.ndcg_at_k for r in eval_results if r.should_answer],
    }

    total = len(eval_results)
    ans = sum(1 for r in eval_results if r.should_answer)
    rej = total - ans
    print(f"\n{'='*60}")
    print(f"RAG 评估结果摘要 (rejudged)")
    print(f"{'='*60}")
    print(f"总样本数: {total}  (应答 {ans} / 拒答 {rej})")
    print(f"{'-'*60}")
    for name, vals in rag_vals.items():
        if vals:
            print(f"{name:25s} {sum(vals)/len(vals):.4f}  {ci(vals)}  n={len(vals)}")
        else:
            print(f"{name:25s} N/A (no valid scores)")
    print(f"{'-'*60}")
    for name, vals in hard_vals.items():
        if vals:
            print(f"{name:25s} {sum(vals)/len(vals):.4f}  {ci(vals)}  n={len(vals)}")
    print(f"{'-'*60}")
    # rejection stats
    rejection_cases = [r for r in eval_results if not r.should_answer]
    answered_cases = [r for r in eval_results if r.should_answer]
    if rejection_cases:
        correct_rej = sum(1 for r in rejection_cases if r.correct_rejection > 0.5) / len(rejection_cases)
        print(f"正确拒答率:    {correct_rej*100:.2f}%")
    if rejection_cases:
        halluc = sum(1 for r in rejection_cases if r.false_answer_hallucination > 0.5) / len(rejection_cases)
        print(f"幻觉率:        {halluc*100:.2f}%")
    if answered_cases:
        false_rej = sum(1 for r in answered_cases if r.false_rejection > 0.5) / len(answered_cases)
        print(f"过度拒答率:    {false_rej*100:.2f}%")
    print(f"{'='*60}")

    logger.info("RAGAS 重打分完成")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python evaluation/rejudge.py <eval_results_xxx.json>")
        sys.exit(1)
    main(sys.argv[1])
