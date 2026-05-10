"""
查询分析器 (Query Analyzer)

纯规则驱动的多维度查询理解，替代 LLM 路由：
1. 意图分类 — 实体查询 / 模糊推荐 / 关系推理 / 拒答
2. 图增强触发 — 判断是否需要知识图谱补充
3. 查询特征 — 复杂度、实体数量等（兼容旧接口)

零 LLM 开销，毫秒级响应。
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    ENTITY_LOOKUP = "entity_lookup"         # "宫保鸡丁怎么做"
    FUZZY_RECOMMENDATION = "fuzzy_rec"      # "推荐几个下饭菜"
    RELATIONAL_QUERY = "relational"          # "鸡肉配什么蔬菜"
    REJECTION = "rejection"                  # 非烹饪问题


@dataclass
class QueryAnalysis:
    """查询分析结果（兼容旧 IntelligentQueryRouter 接口）"""
    intent: QueryIntent
    query_complexity: float          # 0-1
    relationship_intensity: float    # 0-1
    reasoning_required: bool
    entity_count: int
    need_graph_enrichment: bool      # 是否需要图增强
    confidence: float                # 分类置信度
    reasoning: str                   # 分析说明

    # 兼容旧的 SearchStrategy 接口
    @property
    def recommended_strategy(self):
        # 始终返回 hybrid_traditional，实际不再路由
        class _Strategy:
            value = "hybrid_traditional"
        return _Strategy()


# ---- 意图分类规则 ----

# 非烹饪关键词 → 拒答
_REJECTION_PATTERNS = [
    r"(天气|股票|基金|房价|新闻|政治|游戏|电影|综艺|音乐|球赛|NBA|足球|篮球)",
    r"(帮我写|翻译|代码|编程|python|java|sql|http)",
    r"(你是谁|你叫什么|你的名字)",
]

# 关系推理关键词 → 图增强
_RELATIONAL_PATTERNS = [
    r"(配什么|搭配|组合|和.*一起|相克|不能和.*吃)",
    r"(类似|相似|像.*一样|同类|相关|替代|代替)",
    r"(为什么|原因|区别|比较|对比|哪个好|怎么选)",
    r"(关系|联系|关联|影响)",
]

# 实体查询指示词 → 直接检索
_ENTITY_INDICATORS = [
    r"(怎么做|做法|步骤|烹饪方法|食谱|教程|制作|如何做)",
    r"(是什么|什么是|啥是|介绍|简介)",
    r"(需要什么|用什么|食材|调料|配料)",
]


class QueryAnalyzer:
    """规则驱动的查询分析器"""

    def analyze(self, query: str) -> QueryAnalysis:
        query = query.strip()
        logger.info(f"查询分析: {query[:50]}...")

        # 1. 拒答检测
        if self._is_rejection(query):
            return QueryAnalysis(
                intent=QueryIntent.REJECTION,
                query_complexity=0.0,
                relationship_intensity=0.0,
                reasoning_required=False,
                entity_count=0,
                need_graph_enrichment=False,
                confidence=0.95,
                reasoning="非烹饪领域问题，建议拒答",
            )

        # 2. 关系推理检测
        relational_score = self._match_score(query, _RELATIONAL_PATTERNS)
        if relational_score > 0:
            return QueryAnalysis(
                intent=QueryIntent.RELATIONAL_QUERY,
                query_complexity=min(0.5 + relational_score * 0.2, 1.0),
                relationship_intensity=min(0.6 + relational_score * 0.2, 1.0),
                reasoning_required=True,
                entity_count=self._count_entities(query),
                need_graph_enrichment=True,
                confidence=0.85,
                reasoning="检测到关系/推理关键词，建议启用图增强",
            )

        # 3. 实体查询检测
        entity_score = self._match_score(query, _ENTITY_INDICATORS)
        if entity_score > 0 or self._has_specific_dish(query):
            return QueryAnalysis(
                intent=QueryIntent.ENTITY_LOOKUP,
                query_complexity=0.2 + entity_score * 0.2,
                relationship_intensity=0.1,
                reasoning_required=False,
                entity_count=max(self._count_entities(query), 1),
                need_graph_enrichment=False,
                confidence=0.9,
                reasoning="实体查询，直接检索即可",
            )

        # 4. 默认：模糊推荐
        return QueryAnalysis(
            intent=QueryIntent.FUZZY_RECOMMENDATION,
            query_complexity=0.3,
            relationship_intensity=0.2,
            reasoning_required=False,
            entity_count=self._count_entities(query),
            need_graph_enrichment=False,
            confidence=0.75,
            reasoning="模糊推荐/偏好查询，启用 HyDE 语义改写",
        )

    # ---- Private helpers ----

    def _is_rejection(self, query: str) -> bool:
        for pattern in _REJECTION_PATTERNS:
            if re.search(pattern, query):
                return True
        return False

    def _match_score(self, query: str, patterns: List[str]) -> int:
        """匹配命中数"""
        score = 0
        for pattern in patterns:
            if re.search(pattern, query):
                score += 1
        return score

    def _count_entities(self, query: str) -> int:
        """粗略估计查询中的实体数量（基于分词）"""
        # 简单策略：中文按常见分隔和关键词长度估算
        cleaned = re.sub(r"[？?！!，,。、\s]+", " ", query)
        parts = cleaned.split()
        # 2-4 字的短片段可能是食材/菜名
        entity_like = [p for p in parts if 2 <= len(p) <= 6]
        return max(len(entity_like), 1)

    def _has_specific_dish(self, query: str) -> bool:
        """检查是否包含具体菜名特征（无指示词的情况）"""
        # 去掉标点后剩余 2-10 字 → 可能是具体菜名
        cleaned = re.sub(r"[？！。，,?\s]+", "", query)
        return 2 <= len(cleaned) <= 10
