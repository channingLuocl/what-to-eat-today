"""
RAGAS 评估器封装

支持评估维度：
1. RAGAS 软指标 (LLM-as-judge):
   - context_recall   : 检索内容是否覆盖了 reference_answer 中的信息
   - faithfulness     : 回答是否完全基于检索内容（无幻觉）
   - answer_relevancy : 回答与问题的相关性
2. 检索硬指标 (确定性):
   - recall@k / precision@k / mrr / ndcg@k 等
3. 拒答评估:
   - correct_rejection / false_answer / false_rejection
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from .retrieval_metrics import (
    compute_retrieval_metrics,
    evaluate_rejection_behavior,
)

logger = logging.getLogger(__name__)


# ============== 数据结构 ==============


@dataclass
class EvaluationResult:
    """单条评估结果"""

    # ---- 标识与元信息 ----
    query_id: str
    query: str
    category: str = ""
    difficulty: str = ""
    should_answer: bool = True

    # ---- 输入输出 ----
    expected_doc_paths: List[str] = field(default_factory=list)
    retrieved_doc_paths: List[str] = field(default_factory=list)
    retrieved_contexts: List[str] = field(default_factory=list)
    response: str = ""
    reference_answer: str = ""

    # ---- RAGAS 软指标 ----
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0

    # ---- 检索硬指标 ----
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    f1_at_k: float = 0.0
    mrr: float = 0.0
    hit_at_k: float = 0.0
    ndcg_at_k: float = 0.0

    # ---- 拒答相关 ----
    correct_rejection: float = 0.0
    false_answer_hallucination: float = 0.0
    false_rejection: float = 0.0

    # ---- 其他 ----
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============== 评估器 ==============


class RAGASEvaluator:
    """RAGAS 评估器封装"""

    def __init__(
        self,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        top_k: int = 5,
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.api_base = api_base
        self.api_key = api_key
        self.top_k = top_k

        # lazy init
        self._ragas_llm = None
        self._ragas_emb = None

    def _get_ragas_components(self):
        """懒加载 LLM 和 Embedding（只构造一次，避免重复加载 BGE 模型）"""
        if self._ragas_llm is None or self._ragas_emb is None:
            from .llm_factory import (
                build_ragas_llm,
                build_ragas_embeddings,
            )

            self._ragas_llm = build_ragas_llm(
                model=self.llm_model,
                base_url=self.api_base,
                api_key=self.api_key,
            )
            self._ragas_emb = build_ragas_embeddings(model=self.embedding_model)
        return self._ragas_llm, self._ragas_emb

    def evaluate(
        self,
        items: List[Dict[str, Any]],
        retrieved_paths_list: List[List[str]],
        retrieved_contexts_list: List[List[str]],
        responses: List[str],
        metrics: Optional[List[str]] = None,
        run_ragas: bool = True,
    ) -> List[EvaluationResult]:
        """
        执行评估。

        Args:
            items: 原始评测数据列表，每项是 eval_dataset.json 中的一条
                   （含 query_id / query / category / expected_doc_paths /
                    reference_answer / should_answer 等字段）
            retrieved_paths_list: 每条 query 对应的检索文档路径列表
            retrieved_contexts_list: 每条 query 对应的检索上下文文本列表
            responses: 每条 query 的最终生成回答
            metrics: 要跑的 RAGAS 指标，默认全跑
            run_ragas: 是否调用 RAGAS（False 时只算硬指标 + 拒答评估）

        Returns:
            EvaluationResult 列表
        """
        if metrics is None:
            metrics = ["context_recall", "faithfulness", "answer_relevancy"]

        # 第一步：算硬指标和拒答评估（不依赖 LLM，先填好）
        results: List[EvaluationResult] = []
        for i, item in enumerate(items):
            retrieved_paths = retrieved_paths_list[i] if i < len(retrieved_paths_list) else []
            retrieved_ctx = retrieved_contexts_list[i] if i < len(retrieved_contexts_list) else []
            response = responses[i] if i < len(responses) else ""
            expected_paths = item.get("expected_doc_paths", []) or []
            should_answer = item.get("should_answer", True)

            # 检索硬指标
            ret_m = compute_retrieval_metrics(
                retrieved_paths=retrieved_paths,
                expected_paths=expected_paths,
                k=self.top_k,
            )

            # 拒答评估
            rej_m = evaluate_rejection_behavior(response, should_answer)

            res = EvaluationResult(
                query_id=item.get("id", f"Q{i:04d}"),
                query=item.get("query", ""),
                category=item.get("category", ""),
                difficulty=item.get("difficulty", ""),
                should_answer=should_answer,
                expected_doc_paths=expected_paths,
                retrieved_doc_paths=retrieved_paths,
                retrieved_contexts=retrieved_ctx,
                response=response,
                reference_answer=item.get("reference_answer", ""),
                # 硬指标
                recall_at_k=ret_m.get(f"recall@{self.top_k}", 0.0),
                precision_at_k=ret_m.get(f"precision@{self.top_k}", 0.0),
                f1_at_k=ret_m.get(f"f1@{self.top_k}", 0.0),
                mrr=ret_m.get("mrr", 0.0),
                hit_at_k=ret_m.get(f"hit@{self.top_k}", 0.0),
                ndcg_at_k=ret_m.get(f"ndcg@{self.top_k}", 0.0),
                # 拒答
                correct_rejection=rej_m["correct_rejection"],
                false_answer_hallucination=rej_m["false_answer_hallucination"],
                false_rejection=rej_m["false_rejection"],
            )
            results.append(res)

        # 第二步：跑 RAGAS（可选）
        if run_ragas:
            try:
                self._run_ragas_and_fill(results, metrics)
            except Exception as e:
                logger.error(f"RAGAS 评估失败，所有 RAGAS 软指标置 0: {e}", exc_info=True)
                for r in results:
                    r.error = (r.error or "") + f"; RAGAS_ERROR: {e}"
        else:
            logger.warning(
                "⚠️  RAGAS 软指标已跳过 (run_ragas=False)。"
                "context_recall / faithfulness / answer_relevancy 都是 0。"
            )

        return results

    def _run_ragas_and_fill(self, results: List[EvaluationResult], metrics: List[str]):
        """实际调用 RAGAS 并把分数填回 results"""
        try:
            from ragas import evaluate, EvaluationDataset
            from ragas.metrics import (
                context_recall,
                faithfulness,
                answer_relevancy,
            )
            from ragas.dataset_schema import SingleTurnSample
        except ImportError:
            raise ImportError(
                "需要安装 ragas: pip install 'ragas>=0.2,<0.3'"
            )

        from .llm_factory import inject_into_metrics

        llm, emb = self._get_ragas_components()

        # 构建 SingleTurnSample 列表
        # RAGAS 0.2 要求 reference 字段（不是 reference_answer / reference_context）
        samples = []
        for r in results:
            # 只跑那些应该回答且有 reference 的样本（拒答样本 RAGAS 算不动）
            if not r.should_answer:
                continue

            sample_kwargs = {
                "user_input": r.query,
                "response": r.response,
                "retrieved_contexts": r.retrieved_contexts or [""],
            }

            if r.reference_answer:
                sample_kwargs["reference"] = r.reference_answer

            samples.append((r.query_id, SingleTurnSample(**sample_kwargs)))

        if not samples:
            logger.warning("没有可供 RAGAS 评估的样本（全都是拒答样本？）")
            return

        # 构建 metric 列表 + 注入 LLM
        metric_objs = []
        if "context_recall" in metrics:
            metric_objs.append(context_recall)
        if "faithfulness" in metrics:
            metric_objs.append(faithfulness)
        if "answer_relevancy" in metrics:
            metric_objs.append(answer_relevancy)

        inject_into_metrics(metric_objs, llm=llm, embeddings=emb)

        # 跑评估
        sample_only = [s for _, s in samples]
        dataset = EvaluationDataset(samples=sample_only)
        logger.info(f"开始 RAGAS 评估 ({len(sample_only)} 条样本, {len(metric_objs)} 个指标)...")

        # MiniMax 并发能力差且响应慢，降并发 + 拉长超时，避免大量 TimeoutError
        try:
            from ragas.run_config import RunConfig
            run_config = RunConfig(timeout=240, max_workers=2, max_retries=3)
        except ImportError:
            run_config = None

        eval_kwargs = dict(
            dataset=dataset,
            metrics=metric_objs,
            llm=llm,
            embeddings=emb,
            show_progress=True,
        )
        if run_config is not None:
            eval_kwargs["run_config"] = run_config
        eval_result = evaluate(**eval_kwargs)

        # 把结果转成 DataFrame，按 query_id 对齐填回
        try:
            df = eval_result.to_pandas()
        except Exception:
            # 兜底：直接当 list 处理
            df = None
            logger.warning("无法 to_pandas()，尝试用迭代方式取值")

        # 建立 query_id -> EvaluationResult 的索引
        id_to_result = {r.query_id: r for r in results}

        for idx, (query_id, _) in enumerate(samples):
            r = id_to_result.get(query_id)
            if r is None:
                continue
            try:
                if df is not None:
                    row = df.iloc[idx]
                    r.context_recall = float(row.get("context_recall", 0.0) or 0.0)
                    r.faithfulness = float(row.get("faithfulness", 0.0) or 0.0)
                    r.answer_relevancy = float(row.get("answer_relevancy", 0.0) or 0.0)
                else:
                    # eval_result 可迭代，取第 idx 项
                    item = eval_result[idx]
                    if isinstance(item, dict):
                        r.context_recall = float(item.get("context_recall", 0.0) or 0.0)
                        r.faithfulness = float(item.get("faithfulness", 0.0) or 0.0)
                        r.answer_relevancy = float(item.get("answer_relevancy", 0.0) or 0.0)
            except Exception as e:
                logger.warning(f"无法解析 query_id={query_id} 的 RAGAS 结果: {e}")

        logger.info("RAGAS 评估完成")

    # ============== 兜底简化评估（仅用于 RAGAS 不可用时） ==============

    def simplified_evaluate(
        self,
        items: List[Dict[str, Any]],
        retrieved_paths_list: List[List[str]],
        retrieved_contexts_list: List[List[str]],
        responses: List[str],
    ) -> List[EvaluationResult]:
        """关键词匹配版评估，用于 RAGAS 不可用时。
        ⚠️ 这个分数可信度很低，只能用来做 smoke test，不能用来做版本对比。
        """
        logger.warning("=" * 60)
        logger.warning("⚠️  正在使用简化关键词匹配评估！")
        logger.warning("⚠️  此分数仅供 debug 用途，不可信。")
        logger.warning("⚠️  请安装 ragas 后重新评估: pip install 'ragas>=0.2,<0.3'")
        logger.warning("=" * 60)

        results = self.evaluate(
            items=items,
            retrieved_paths_list=retrieved_paths_list,
            retrieved_contexts_list=retrieved_contexts_list,
            responses=responses,
            run_ragas=False,
        )

        # 给 RAGAS 字段填上简化版的关键词匹配分数
        for i, r in enumerate(results):
            ctx_text = " ".join(r.retrieved_contexts or [])
            r.context_recall = self._kw_overlap(r.reference_answer, ctx_text)
            r.faithfulness = self._kw_overlap(r.response, ctx_text)
            r.answer_relevancy = self._kw_overlap(r.query, r.response)

        return results

    @staticmethod
    def _kw_overlap(a: str, b: str) -> float:
        """超粗略的关键词重叠率，仅供兜底"""
        if not a or not b:
            return 0.0
        a_words = {w for w in a.lower().split() if len(w) > 1}
        b_words = {w for w in b.lower().split() if len(w) > 1}
        if not a_words:
            return 0.0
        return len(a_words & b_words) / len(a_words)


# ============== 兼容旧接口 ==============


def calculate_retrieval_metrics(
    query: str,
    retrieved_docs: List[Any],
    relevant_doc_ids: List[str],
) -> Dict[str, float]:
    """旧接口保留，转调新模块。retrieved_docs 期望是 langchain Document 类似对象"""
    from .retrieval_metrics import compute_retrieval_metrics

    retrieved_ids = []
    for doc in retrieved_docs:
        md = getattr(doc, "metadata", {}) or {}
        retrieved_ids.append(md.get("source") or md.get("node_id", ""))
    return compute_retrieval_metrics(retrieved_ids, relevant_doc_ids)
