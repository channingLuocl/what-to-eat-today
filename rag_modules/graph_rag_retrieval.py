"""
图增强模块 (Graph Enricher)

从独立检索路径降级为上下文增强器：
- 不再提供 graph_rag_search() 作为检索入口
- 提供 enrich_retrieval_results() 为 hybrid 检索结果补充图结构信息
- 保留 Neo4j 多跳遍历和子图提取能力
"""

import logging
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional

from langchain_core.documents import Document
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class GraphRAGRetrieval:
    """
    图增强模块

    为 hybrid 检索返回的菜谱补充图结构信息：
    1. 食材搭配 — "这道菜常用什么食材？还能配什么？"
    2. 同类菜谱 — "有哪些口味/做法类似的菜？"
    3. 关系网络 — "这个食材还用在哪些菜里？"
    """

    def __init__(self, config, llm_client=None):
        self.config = config
        self.llm_client = llm_client  # 保留兼容，不再调用
        self.driver = None
        self.entity_cache: Dict[str, Dict] = {}
        self.relation_cache: Dict[str, int] = {}

    def initialize(self):
        """连接 Neo4j 并构建索引"""
        logger.info("初始化图增强模块...")
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j 连接成功")
        except Exception as e:
            logger.error(f"Neo4j 连接失败: {e}")
            return
        self._build_graph_index()

    def _build_graph_index(self):
        """构建实体和关系索引"""
        logger.info("构建图索引...")
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    WHERE n.nodeId IS NOT NULL
                    WITH n, COUNT { (n)--() } as degree
                    RETURN labels(n) as node_labels, n.nodeId as node_id,
                           n.name as name, n.category as category, degree
                    ORDER BY degree DESC LIMIT 2000
                """)
                for record in result:
                    self.entity_cache[record["node_id"]] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "category": record["category"],
                        "degree": record["degree"],
                    }

                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(r) as frequency
                    ORDER BY frequency DESC
                """)
                for record in result:
                    self.relation_cache[record["rel_type"]] = record["frequency"]

                logger.info(
                    f"图索引完成: {len(self.entity_cache)} 实体, "
                    f"{len(self.relation_cache)} 关系类型"
                )
        except Exception as e:
            logger.error(f"图索引构建失败: {e}")

    # ========== 公开接口 ==========

    def enrich_retrieval_results(
        self, query: str, retrieved_docs: List[Document]
    ) -> Optional[str]:
        """
        为检索结果补充图结构上下文。

        Returns:
            结构化的图增强文本，可直接注入生成 prompt。
            失败或无结果时返回 None。
        """
        if not self.driver or not retrieved_docs:
            return None

        recipe_names = []
        recipe_node_ids = []
        for doc in retrieved_docs:
            name = doc.metadata.get("recipe_name", "")
            nid = doc.metadata.get("node_id", "")
            if name and nid:
                recipe_names.append(name)
                recipe_node_ids.append(nid)

        if not recipe_node_ids:
            return None

        logger.info(f"图增强: 为 {len(recipe_node_ids)} 个菜谱补充图信息")

        parts = []

        # 1. 食材搭配统计
        pairings = self._get_ingredient_pairings(recipe_node_ids)
        if pairings:
            parts.append("## 食材搭配参考（基于知识图谱）")
            for p in pairings:
                parts.append(f"- {p}")

        # 2. 同类/相似菜谱
        similar = self._get_similar_recipes(recipe_node_ids, recipe_names)
        if similar:
            parts.append("\n## 类似菜谱推荐")
            for s in similar:
                parts.append(f"- {s}")

        if not parts:
            return None

        return "\n".join(parts)

    # ========== 图查询 ==========

    def _get_ingredient_pairings(
        self, recipe_node_ids: List[str], limit: int = 8
    ) -> List[str]:
        """查询菜谱的食材搭配关系"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    UNWIND $recipe_ids as rid
                    MATCH (r:Recipe {nodeId: rid})-[rel:REQUIRES]->(i:Ingredient)
                    WITH i.name as ing, collect(r.name)[0..3] as recipes, count(*) as cnt
                    ORDER BY cnt DESC
                    LIMIT $limit
                    RETURN ing, recipes, cnt
                    """,
                    {"recipe_ids": recipe_node_ids, "limit": limit},
                )
                pairings = []
                for record in result:
                    recipe_list = "、".join(record["recipes"])
                    pairings.append(
                        f"**{record['ing']}** — 用于 {recipe_list}"
                        f"（共 {record['cnt']} 道菜使用）"
                    )
                return pairings
        except Exception as e:
            logger.warning(f"食材搭配查询失败: {e}")
            return []

    def _get_similar_recipes(
        self, recipe_node_ids: List[str], recipe_names: List[str], limit: int = 5
    ) -> List[str]:
        """查找相似菜谱（同菜系/同分类/共享食材）"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    UNWIND $recipe_ids as rid
                    MATCH (r:Recipe {nodeId: rid})
                    WITH collect(DISTINCT r.category) as cats,
                         collect(DISTINCT r.cuisineType) as cuisines
                    UNWIND cats + cuisines as shared_attr
                    WITH shared_attr WHERE shared_attr IS NOT NULL AND shared_attr <> ''
                    MATCH (similar:Recipe)
                    WHERE (similar.category = shared_attr OR similar.cuisineType = shared_attr)
                      AND NOT similar.nodeId IN $recipe_ids
                    RETURN similar.name as name, similar.category as cat,
                           similar.cuisineType as cuisine, similar.difficulty as diff
                    LIMIT $limit
                    """,
                    {
                        "recipe_ids": recipe_node_ids,
                        "limit": limit,
                    },
                )
                recipes = []
                for record in result:
                    recipes.append(
                        f"{record['name']}（{record.get('cuisine', '')} {record.get('cat', '')} "
                        f"难度 {record.get('diff', '?')}/5）"
                    )
                return recipes
        except Exception as e:
            logger.warning(f"相似菜谱查询失败: {e}")
            return []

    # ========== 兼容旧接口（评估脚本引用） ==========

    def graph_rag_search(
        self, query: str, top_k: int = 5
    ) -> List[Document]:
        """
        旧接口保留，已不再作为独立检索路径。
        返回空列表，调用方应使用 hybrid_search + enrich_retrieval_results。
        """
        logger.warning("graph_rag_search() 已废弃，请使用 hybrid_search + enrich_retrieval_results")
        return []

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("图增强模块已关闭")
