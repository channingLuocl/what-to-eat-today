"""
LLM 和 Embedding 工厂
为 RAGAS 评估提供统一的 LLM/Embedding 包装器

核心目的：让 RAGAS 用上项目自己配置的 MiniMax LLM + BGE embedding，
而不是 RAGAS 默认的 OpenAI（默认会调 OpenAI API）。

使用示例:
    from evaluation.llm_factory import build_ragas_llm, build_ragas_embeddings

    llm = build_ragas_llm()
    emb = build_ragas_embeddings()
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_ragas_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = 240,
):
    """
    构造一个供 RAGAS 评估使用的 LLM 包装器。

    参数缺省时从环境变量读取：
        LLM_MODEL          (默认 "MiniMax-M2.7")
        OPENAI_BASE_URL    (默认 "https://api.minimaxi.com/v1")
        OPENAI_API_KEY     (必填)

    返回:
        LangchainLLMWrapper 实例，可直接赋值给 RAGAS metric 的 .llm 属性
    """
    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as e:
        raise ImportError(
            "需要安装依赖: pip install langchain-openai ragas"
        ) from e

    model = model or os.getenv("LLM_MODEL", "MiniMax-M2.7")
    base_url = base_url or os.getenv(
        "OPENAI_BASE_URL", "https://api.minimaxi.com/v1"
    )
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY 未设置。请在 .env 中配置或通过环境变量传入。"
        )

    logger.info(
        f"Building RAGAS LLM: model={model}, base_url={base_url}"
    )

    chat = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=2,
    )

    return LangchainLLMWrapper(chat)


def build_ragas_embeddings(
    model: Optional[str] = None,
    device: str = "cpu",
):
    """
    构造一个供 RAGAS 评估使用的 Embedding 包装器。

    参数缺省时从环境变量读取：
        EMBEDDING_MODEL    (默认 "BAAI/bge-small-zh-v1.5")

    返回:
        LangchainEmbeddingsWrapper 实例
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as e:
        raise ImportError(
            "需要安装依赖: pip install langchain-huggingface ragas sentence-transformers"
        ) from e

    model = model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    logger.info(f"Building RAGAS Embeddings: model={model}")

    hf_emb = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    return LangchainEmbeddingsWrapper(hf_emb)


def inject_into_metrics(metrics: list, llm=None, embeddings=None) -> None:
    """
    把 LLM 和 Embedding 注入到一组 RAGAS metric 对象中。

    RAGAS 的 metric 对象（如 context_recall）默认 .llm = None，
    会在 evaluate() 调用时 fallback 到全局默认 LLM（OpenAI）。
    我们必须在 evaluate 之前手动注入，否则就用错模型了。

    Args:
        metrics: RAGAS metric 对象列表
        llm: build_ragas_llm() 返回的对象（可选，默认现场构造）
        embeddings: build_ragas_embeddings() 返回的对象（可选，默认现场构造）
    """
    if llm is None:
        llm = build_ragas_llm()
    if embeddings is None:
        embeddings = build_ragas_embeddings()

    for m in metrics:
        # 几乎所有 RAGAS metric 都有 .llm
        if hasattr(m, "llm"):
            m.llm = llm
        # 部分指标（如 answer_relevancy）需要 embedding
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings
        # MiniMax 不支持 n>1（self-consistency 采样会触发 400）。
        # 把 strictness 强制为 1，让 metric 每次只采一个候选。
        if hasattr(m, "strictness"):
            m.strictness = 1

    logger.info(
        f"Injected LLM + Embeddings into {len(metrics)} RAGAS metrics"
    )
