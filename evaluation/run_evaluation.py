"""
RAG 评估入口脚本

用法:
    # 完整评估（200 条样本，含 RAGAS 软指标 + 检索硬指标 + 拒答评估）
    python evaluation/run_evaluation.py

    # 只测检索（不调 LLM，速度快）
    python evaluation/run_evaluation.py --mode retrieval

    # 只测生成
    python evaluation/run_evaluation.py --mode generation

    # 自定义评测数据集和输出目录
    python evaluation/run_evaluation.py --dataset evaluation/eval_dataset.json --output ./eval_out

    # 跳过 RAGAS（开发期/快速 smoke test）
    python evaluation/run_evaluation.py --no-ragas

    # 用 mock RAG（不连 Milvus/Neo4j，仅冒烟测试评估流水线本身）
    python evaluation/run_evaluation.py --mock-rag --limit 5

    # 对比两次评估结果
    python evaluation/run_evaluation.py --compare old.json new.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 加载 .env（必须在导入任何业务模块之前）
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from evaluation.metrics_collector import MetricsCollector, compare_runs
from evaluation.ragas_evaluator import EvaluationResult, RAGASEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============== Mock RAG（用于本地冒烟测试） ==============


class MockRAG:
    """没有真实知识库时用的假 RAG，按 query 字面匹配 expected_doc_paths 里第一条返回。
    仅用于验证评估流水线是否打通。
    """

    def retrieve_and_generate(self, query: str, item: Dict[str, Any], top_k: int = 5) -> Tuple[List[str], List[str], str]:
        # 假装命中前 2 个 expected
        expected = item.get("expected_doc_paths", [])
        retrieved_paths = expected[:2] if expected else []
        retrieved_contexts = [
            f"[MOCK]这是 {p} 的假上下文内容" for p in retrieved_paths
        ]
        if not item.get("should_answer", True):
            response = "抱歉，我是菜谱推荐助手，无法回答这个问题。"
        elif retrieved_contexts:
            response = f"根据检索到的内容：{item.get('reference_answer', '')}"
        else:
            response = "抱歉，没有找到相关内容。"
        return retrieved_paths, retrieved_contexts, response


# ============== 真实 RAG 适配 ==============


def _doc_to_path(doc) -> str:
    """从 langchain Document 中提取文档路径，按多个常见 metadata key 尝试"""
    md = getattr(doc, "metadata", {}) or {}
    for key in ("source", "file_path", "path", "doc_path", "recipe_name", "node_id"):
        if md.get(key):
            return str(md[key])
    return ""


class RealRAG:
    """实际项目中的 RAG pipeline 包装"""

    def __init__(self):
        from config import GraphRAGConfig
        from rag_modules import (
            GraphDataPreparationModule,
            MilvusIndexConstructionModule,
            GenerationIntegrationModule,
        )
        from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
        from rag_modules.hybrid_retrieval import HybridRetrievalModule
        from rag_modules.intelligent_query_router import IntelligentQueryRouter

        self.config = GraphRAGConfig()
        logger.info("初始化检索组件...")

        # 1. 数据准备模块
        self.data_module = GraphDataPreparationModule(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password,
            database=self.config.neo4j_database,
        )

        # 2. 向量索引模块
        self.index_module = MilvusIndexConstructionModule(
            host=self.config.milvus_host,
            port=self.config.milvus_port,
            collection_name=self.config.milvus_collection_name,
            dimension=self.config.milvus_dimension,
            model_name=self.config.embedding_model,
        )

        # 3. 生成模块（同时提供 llm_client）
        self.gen = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        llm_client = self.gen.client

        # 4. 传统混合检索（需要 milvus_module / data_module / llm_client）
        self.hybrid = HybridRetrievalModule(
            config=self.config,
            milvus_module=self.index_module,
            data_module=self.data_module,
            llm_client=llm_client,
        )

        # 5. 图 RAG 检索
        self.graph = GraphRAGRetrieval(
            config=self.config,
            llm_client=llm_client,
        )

        # 6. 智能路由器（参数名与实际签名对齐）
        self.router = IntelligentQueryRouter(
            traditional_retrieval=self.hybrid,
            graph_rag_retrieval=self.graph,
            llm_client=llm_client,
            config=self.config,
        )

        # 7. 加载知识库并初始化检索器（需在路由器创建后执行）
        logger.info("加载知识库数据...")
        if self.index_module.has_collection() and self.index_module.load_collection():
            logger.info("已有向量集合，直接加载")
        else:
            logger.info("未找到向量集合，重新构建索引")
        self.data_module.load_graph_data()
        self.data_module.build_recipe_documents()
        chunks = self.data_module.chunk_documents(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        if not self.index_module.has_collection():
            self.index_module.build_vector_index(chunks)
        self.hybrid.initialize(chunks)
        self.graph.initialize()

        logger.info("检索组件初始化完成")

    def retrieve_and_generate(self, query: str, item: Dict[str, Any], top_k: int = 5) -> Tuple[List[str], List[str], str]:
        try:
            docs, _ = self.router.route_query(query, top_k=top_k)
        except Exception as e:
            logger.error(f"检索失败 query={query!r}: {e}")
            return [], [], f"检索出错: {e}"

        retrieved_paths = [_doc_to_path(d) for d in docs]
        retrieved_paths = [p for p in retrieved_paths if p]

        retrieved_contexts = []
        for d in docs:
            content = (getattr(d, "page_content", "") or "").strip()
            level = (getattr(d, "metadata", {}) or {}).get("retrieval_level", "")
            if level:
                retrieved_contexts.append(f"[{str(level).upper()}] {content}")
            else:
                retrieved_contexts.append(content)

        if not docs:
            return [], [], "抱歉，没有找到相关信息。"

        try:
            answer = self.gen.generate_adaptive_answer(query, docs)
        except Exception as e:
            logger.error(f"生成失败: {e}")
            answer = f"生成出错: {e}"

        return retrieved_paths, retrieved_contexts, answer


# ============== 主流程 ==============


def load_dataset(path: str) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_evaluation(
    dataset_path: str,
    output_dir: str,
    mode: str = "full",
    use_mock_rag: bool = False,
    no_ragas: bool = False,
    limit: Optional[int] = None,
) -> List[EvaluationResult]:
    items = load_dataset(dataset_path)
    if limit:
        items = items[:limit]
    logger.info(f"加载 {len(items)} 条评测样本，模式: {mode}")

    # 初始化 RAG
    rag = MockRAG() if use_mock_rag else RealRAG()

    # 运行 RAG pipeline
    retrieved_paths_list: List[List[str]] = []
    retrieved_contexts_list: List[List[str]] = []
    responses: List[str] = []

    for i, item in enumerate(items):
        logger.info(f"  处理 {i+1}/{len(items)}: {item.get('query', '')[:30]}")
        rp, rc, resp = rag.retrieve_and_generate(
            item.get("query", ""),
            item,
            top_k=5,
        )
        retrieved_paths_list.append(rp)
        retrieved_contexts_list.append(rc)
        responses.append(resp)

    # 选择评估指标
    metrics = ["context_recall", "faithfulness", "answer_relevancy"]
    if mode == "retrieval":
        metrics = ["context_recall"]
    elif mode == "generation":
        metrics = ["faithfulness", "answer_relevancy"]

    # 评估
    evaluator = RAGASEvaluator(top_k=5)
    if no_ragas:
        results = evaluator.simplified_evaluate(
            items=items,
            retrieved_paths_list=retrieved_paths_list,
            retrieved_contexts_list=retrieved_contexts_list,
            responses=responses,
        )
    else:
        results = evaluator.evaluate(
            items=items,
            retrieved_paths_list=retrieved_paths_list,
            retrieved_contexts_list=retrieved_contexts_list,
            responses=responses,
            metrics=metrics,
            run_ragas=True,
        )

    # 收集与导出
    collector = MetricsCollector(output_dir=output_dir)
    collector.add_results(results)

    json_path = collector.save_results()
    csv_path = collector.save_csv()
    md_path = collector.save_markdown()

    # 终端摘要
    print_summary(collector)

    print(f"\n📂 完整结果文件:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV : {csv_path}")
    print(f"  - MD  : {md_path}")

    return results


def print_summary(collector: MetricsCollector):
    s = collector.calculate_summary()
    print("\n" + "=" * 60)
    print("RAG 评估结果摘要 (95% Bootstrap CI)")
    print("=" * 60)
    print(f"总样本数: {s.total_queries}  (应答 {s.answered_count} / 拒答 {s.rejection_case_count})")
    print("-" * 60)

    def line(name, st):
        print(f"{name:25s} {st.avg:.4f}  [{st.ci_lower:.4f}, {st.ci_upper:.4f}]  n={st.n}")

    line("Context Recall", s.context_recall)
    line("Faithfulness", s.faithfulness)
    line("Answer Relevancy", s.answer_relevancy)
    print("-" * 60)
    line("Recall@K", s.recall_at_k)
    line("Precision@K", s.precision_at_k)
    line("MRR", s.mrr)
    line("Hit@K", s.hit_at_k)
    line("NDCG@K", s.ndcg_at_k)
    print("-" * 60)
    print(f"正确拒答率:    {s.correct_rejection_rate:.2%}")
    print(f"幻觉率:        {s.hallucination_rate:.2%}")
    print(f"过度拒答率:    {s.false_rejection_rate:.2%}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="evaluation/eval_dataset.json")
    p.add_argument("--output", default="./evaluation_results")
    p.add_argument("--mode", choices=["full", "retrieval", "generation"], default="full")
    p.add_argument("--mock-rag", action="store_true", help="用 MockRAG，不连 Milvus/Neo4j")
    p.add_argument("--no-ragas", action="store_true", help="跳过 RAGAS（关键词匹配兜底）")
    p.add_argument("--limit", type=int, default=None, help="限制评测样本数（debug 用）")
    p.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"), help="对比两次评估结果")

    args = p.parse_args()

    # 对比模式
    if args.compare:
        old_path, new_path = args.compare
        report = compare_runs(old_path, new_path)
        out = Path(args.output) / "compare_report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(report)
        print(f"\n对比报告已保存: {out}")
        return

    # 评估模式
    ds_path = args.dataset
    if not os.path.isabs(ds_path):
        ds_path = str(Path(__file__).parent.parent / ds_path)
    if not Path(ds_path).exists():
        logger.error(f"数据集不存在: {ds_path}")
        sys.exit(1)

    try:
        run_evaluation(
            dataset_path=ds_path,
            output_dir=args.output,
            mode=args.mode,
            use_mock_rag=args.mock_rag,
            no_ragas=args.no_ragas,
            limit=args.limit,
        )
        logger.info("评估完成")
    except Exception as e:
        logger.error(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
