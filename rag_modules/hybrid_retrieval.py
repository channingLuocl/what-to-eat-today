"""
混合检索模块
基于双层检索范式：实体级 + 主题级检索
结合图结构检索和向量检索，使用Round-robin轮询策略
"""

import json
import logging
import re
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from neo4j import GraphDatabase
from .graph_indexing import GraphIndexingModule

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> str:
    """从 LLM 响应中提取 JSON，兼容思维链 <think> 标签和 Markdown 代码块"""
    if not text or not text.strip():
        raise ValueError("LLM 返回了空响应")
    # 1. 剥掉 <think>...</think> 推理块（MiniMax-M2.7 思维链输出）
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    if not text:
        raise ValueError("LLM 响应中 <think> 块之外没有内容（token 被截断）")
    # 2. 剥除 ```json ... ``` 或 ``` ... ``` 围栏
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        extracted = match.group(1).strip()
        if not extracted:
            raise ValueError("LLM 返回了空代码块")
        return extracted
    return text.strip()

@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str  # 'low' or 'high'
    metadata: Dict[str, Any]

class HybridRetrievalModule:
    """
    混合检索模块
    核心特点：
    1. 双层检索范式（实体级 + 主题级）
    2. 关键词提取和匹配
    3. 图结构+向量检索结合
    4. 一跳邻居扩展
    5. Round-robin轮询合并策略
    """
    
    def __init__(self, config, milvus_module, data_module, llm_client):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.driver = None
        self.bm25_retriever = None

        # 图索引模块
        self.graph_indexing = GraphIndexingModule(config, llm_client)
        self.graph_indexed = False

        # 延迟加载的 Cross-encoder reranker
        self._reranker = None

        # 元数据过滤规则表
        self._filter_rules = {
            "difficulty_easy": {
                "keywords": ["简单", "快手", "新手", "好做", "容易", "快速", "懒人", "不费事"],
                "expr": "difficulty <= 2",
            },
            "difficulty_hard": {
                "keywords": ["复杂", "大菜", "硬菜", "难做", "功夫", "考验", "高难度"],
                "expr": "difficulty >= 4",
            },
            "cuisine": {
                "川菜": ["川菜", "麻辣", "四川", "成都", "重庆", "花椒", "红油"],
                "粤菜": ["粤菜", "广东", "清淡", "广州", "煲汤", "白切"],
                "湘菜": ["湘菜", "湖南", "香辣", "剁椒"],
                "鲁菜": ["鲁菜", "山东", "酱香"],
                "苏菜": ["苏菜", "江苏", "淮扬", "甜口"],
                "闽菜": ["闽菜", "福建", "福州", "佛跳墙"],
                "浙菜": ["浙菜", "浙江", "杭州", "西湖"],
                "徽菜": ["徽菜", "安徽", "黄山"],
                "东北菜": ["东北", "锅包肉", "炖菜", "酸菜"],
                "西北菜": ["西北", "兰州", "西安", "拉面", "羊肉"],
            },
            "category": {
                "家常菜": ["家常", "下饭", "日常", "普通"],
                "汤羹": ["汤", "羹", "煲", "暖身"],
                "凉菜": ["凉拌", "凉菜", "冷菜", "沙拉"],
                "主食": ["主食", "米饭", "面", "馒头", "饺子", "馄饨"],
                "小吃": ["小吃", "零食", "点心", "夜宵"],
                "甜品": ["甜品", "甜点", "蛋糕", "冰淇淋", "糖水"],
                "早餐": ["早餐", "早饭", "早点", "早上"],
                "快手菜": ["快手菜", "快速", "几分钟", "省时"],
            },
            "diet": {
                "减肥": ["减肥", "低卡", "轻食", "低脂", "瘦身", "减脂", "热量低"],
                "高蛋白": ["高蛋白", "增肌", "健身餐"],
                "素食": ["素食", "素菜", "纯素", "吃素"],
            },
        }

    def _should_hyde(self, query: str) -> bool:
        """
        判断是否需要 HyDE 改写。

        跳过场景：
        - 查询包含明确菜名（如"宫保鸡丁怎么做"）
        - 查询过短（<4字）
        - 纯闲聊/问候语

        触发场景：
        - 模糊推荐查询（如"推荐几个下饭菜"）
        - 偏好查询（如"不想吃辣的有什么"）
        - 场景查询（如"早餐吃什么好"）
        """
        query = query.strip()
        if len(query) < 4:
            return False

        # 闲聊/问候跳过
        greetings = {"你好", "谢谢", "再见", "hello", "hi", "thanks"}
        if any(g in query.lower() for g in greetings):
            return False

        # 包含明确菜名的简单指示词跳过
        entity_indicators = ["怎么做", "做法", "步骤", "烹饪方法", "的食谱", "教程"]
        has_entity_indicator = any(ind in query for ind in entity_indicators)
        # 如果查询看起来像 "XX怎么做" 且 XX 部分很短 → 可能是具体菜名查询
        if has_entity_indicator:
            # 去掉指示词和标点，如果剩余部分是 2-8 字的短语，可能是菜名
            remaining = query
            for ind in entity_indicators:
                remaining = remaining.replace(ind, " ")
            remaining = remaining.strip("？！。，,? \t")
            if 2 <= len(remaining) <= 10:
                return False  # 具体菜名查询，跳过 HyDE

        return True

    def _hyde_rewrite(self, query: str) -> Optional[str]:
        """
        用 LLM 生成假设答案（HyDE），用于增强检索语义。
        失败时返回 None，调用方回退到原始 query。
        """
        prompt = f"""你是一个中餐烹饪助手。用户提出了以下问题：

"{query}"

请以烹饪助手身份，直接给出一个简洁的推荐/回答（100-200字即可），
列出 3-5 道相关菜谱的中文名称，并简述每道菜的特点（口味、做法、适合场景）。
不要解释你在做什么，直接给答案。"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8192,
            )
            hyde_text = response.choices[0].message.content.strip()
            if hyde_text and len(hyde_text) > 10:
                return hyde_text
            return None
        except Exception as e:
            logger.warning(f"HyDE 改写失败，回退到原始 query: {e}")
            return None

    def _extract_metadata_filters(self, query: str) -> Optional[str]:
        """从 query 中提取结构化过滤条件，返回 Milvus filter 表达式"""
        conditions = []

        # 难度过滤
        for rule_key, rule in self._filter_rules.items():
            if rule_key.startswith("difficulty_"):
                for kw in rule["keywords"]:
                    if kw in query:
                        conditions.append(rule["expr"])
                        logger.info(f"元数据过滤 - 难度: {rule['expr']} (匹配关键词: {kw})")
                        break

        # 菜系/分类/饮食偏好过滤
        for group_key in ["cuisine", "category", "diet"]:
            rules = self._filter_rules.get(group_key, {})
            for label, keywords in rules.items():
                for kw in keywords:
                    if kw in query:
                        if group_key == "cuisine":
                            conditions.append(f'cuisine_type == "{label}"')
                        elif group_key == "category":
                            conditions.append(
                                f'(category == "{label}" or cuisine_type == "{label}")'
                            )
                        elif group_key == "diet":
                            if label == "减肥":
                                conditions.append(
                                    f'(category == "轻食" or category == "减肥餐" or difficulty <= 3)'
                                )
                            elif label == "高蛋白":
                                conditions.append(f'(category == "高蛋白" or cuisine_type == "{label}")')
                            elif label == "素食":
                                conditions.append(f'(category == "素食" or category == "素菜")')
                        logger.info(f"元数据过滤 - {group_key}: {label} (匹配关键词: {kw})")
                        break

        if conditions:
            expr = " and ".join(conditions)
            logger.info(f"Milvus 过滤表达式: {expr}")
            return expr
        return None

    def initialize(self, chunks: List[Document]):
        """初始化检索系统"""
        logger.info("初始化混合检索模块...")
        
        # 连接Neo4j
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri, 
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )
        
        # 初始化BM25检索器
        if chunks:
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            logger.info(f"BM25检索器初始化完成，文档数量: {len(chunks)}")
        
        # 初始化图索引
        self._build_graph_index()
        
    def _build_graph_index(self):
        """构建图索引"""
        if self.graph_indexed:
            return
            
        logger.info("开始构建图索引...")
        
        try:
            # 获取图数据
            recipes = self.data_module.recipes
            ingredients = self.data_module.ingredients
            cooking_steps = self.data_module.cooking_steps
            
            # 创建实体键值对
            self.graph_indexing.create_entity_key_values(recipes, ingredients, cooking_steps)
            
            # 创建关系键值对（这里需要从Neo4j获取关系数据）
            relationships = self._extract_relationships_from_graph()
            self.graph_indexing.create_relation_key_values(relationships)
            
            # 去重优化
            self.graph_indexing.deduplicate_entities_and_relations()
            
            self.graph_indexed = True
            stats = self.graph_indexing.get_statistics()
            logger.info(f"图索引构建完成: {stats}")
            
        except Exception as e:
            logger.error(f"构建图索引失败: {e}")
            
    def _extract_relationships_from_graph(self) -> List[Tuple[str, str, str]]:
        """从Neo4j图中提取关系"""
        relationships = []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (source)-[r]->(target)
                WHERE source.nodeId >= '200000000' OR target.nodeId >= '200000000'
                RETURN source.nodeId as source_id, type(r) as relation_type, target.nodeId as target_id
                LIMIT 1000
                """
                result = session.run(query)
                
                for record in result:
                    relationships.append((
                        record["source_id"],
                        record["relation_type"],
                        record["target_id"]
                    ))
                    
        except Exception as e:
            logger.error(f"提取图关系失败: {e}")
            
        return relationships
            
    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """
        提取查询关键词：实体级 + 主题级
        """
        prompt = f"""
        作为烹饪知识助手，请分析以下查询并提取关键词，分为两个层次：

        查询：{query}

        提取规则：
        1. 实体级关键词：具体的食材、菜品名称、工具、品牌等有形实体
           - 例如：鸡胸肉、西兰花、红烧肉、平底锅、老干妈
           - 对于抽象查询，推测相关的具体食材/菜品

        2. 主题级关键词：抽象概念、烹饪主题、饮食风格、营养特点等
           - 例如：减肥、低热量、川菜、素食、下饭菜、快手菜
           - 排除动作词：推荐、介绍、制作、怎么做等

        示例：
        查询："推荐几个减肥菜" 
        {{
            "entity_keywords": ["鸡胸肉", "西兰花", "水煮蛋", "胡萝卜", "黄瓜"],
            "topic_keywords": ["减肥", "低热量", "高蛋白", "低脂"]
        }}

        查询："川菜有什么特色"
        {{
            "entity_keywords": ["麻婆豆腐", "宫保鸡丁", "水煮鱼", "辣椒", "花椒"],
            "topic_keywords": ["川菜", "麻辣", "香辣", "下饭菜"]
        }}

        请严格按照JSON格式返回，不要包含多余的文字：
        {{
            "entity_keywords": ["实体1", "实体2", ...],
            "topic_keywords": ["主题1", "主题2", ...]
        }}
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=8192
            )

            raw_content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            logger.debug(f"关键词提取 - finish_reason={finish_reason!r}")
            result = json.loads(_extract_json(raw_content))
            entity_keywords = result.get("entity_keywords", [])
            topic_keywords = result.get("topic_keywords", [])
            
            logger.info(f"关键词提取完成 - 实体级: {entity_keywords}, 主题级: {topic_keywords}")
            return entity_keywords, topic_keywords
            
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            # 降级方案：简单的关键词分割
            keywords = query.split()
            return keywords[:3], keywords[3:6] if len(keywords) > 3 else keywords
    
    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """
        实体级检索：专注于具体实体和关系
        使用图索引的键值对结构进行检索
        """
        results = []
        
        # 1. 使用图索引进行实体检索
        for keyword in entity_keywords:
            # 检索匹配的实体
            entities = self.graph_indexing.get_entities_by_key(keyword)
            
            for entity in entities:
                # 获取邻居信息
                neighbors = self._get_node_neighbors(entity.metadata["node_id"], max_neighbors=2)
                
                # 构建增强内容
                enhanced_content = entity.value_content
                if neighbors:
                    enhanced_content += f"\n相关信息: {', '.join(neighbors)}"
                
                results.append(RetrievalResult(
                    content=enhanced_content,
                    node_id=entity.metadata["node_id"],
                    node_type=entity.entity_type,
                    relevance_score=0.9,  # 精确匹配得分较高
                    retrieval_level="entity",
                    metadata={
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "index_keys": entity.index_keys,
                        "matched_keyword": keyword
                    }
                ))
        
        # 2. 如果图索引结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_entity_level_search(entity_keywords, top_k - len(results))
            results.extend(neo4j_results)
            
        # 3. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"实体级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]
    
    def _neo4j_entity_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j补充检索"""
        results = []
        
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                CALL db.index.fulltext.queryNodes('recipe_fulltext_index', keyword + '*') 
                YIELD node, score
                WHERE node:Recipe
                RETURN 
                    node.nodeId as node_id,
                    node.name as name,
                    node.description as description,
                    labels(node) as labels,
                    score
                ORDER BY score DESC
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })
                
                for record in result:
                    content_parts = []
                    if record["name"]:
                        content_parts.append(f"菜品: {record['name']}")
                    if record["description"]:
                        content_parts.append(f"描述: {record['description']}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Recipe",
                        relevance_score=float(record["score"]) * 0.7,  # 补充检索得分较低
                        retrieval_level="entity",
                        metadata={
                            "name": record["name"],
                            "labels": record["labels"],
                            "source": "neo4j_fallback"
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Neo4j补充检索失败: {e}")
            
        return results
    
    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """
        主题级检索：专注于广泛主题和概念
        使用图索引的关系键值对结构进行主题检索
        """
        results = []
        
        # 1. 使用图索引进行关系/主题检索
        for keyword in topic_keywords:
            # 检索匹配的关系
            relations = self.graph_indexing.get_relations_by_key(keyword)
            
            for relation in relations:
                # 获取相关实体信息
                source_entity = self.graph_indexing.entity_kv_store.get(relation.source_entity)
                target_entity = self.graph_indexing.entity_kv_store.get(relation.target_entity)
                
                if source_entity and target_entity:
                    # 构建丰富的主题内容
                    content_parts = [
                        f"主题: {keyword}",
                        relation.value_content,
                        f"相关菜品: {source_entity.entity_name}",
                        f"相关信息: {target_entity.entity_name}"
                    ]
                    
                    # 添加源实体的详细信息
                    if source_entity.entity_type == "Recipe":
                        newline = '\n'
                        content_parts.append(f"菜品详情: {source_entity.value_content.split(newline)[0]}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=relation.source_entity,  # 以主要实体为ID
                        node_type=source_entity.entity_type,
                        relevance_score=0.95,  # 主题匹配得分
                        retrieval_level="topic",
                        metadata={
                            "relation_id": relation.relation_id,
                            "relation_type": relation.relation_type,
                            "source_name": source_entity.entity_name,
                            "target_name": target_entity.entity_name,
                            "matched_keyword": keyword,
                            "index_keys": relation.index_keys
                        }
                    ))
        
        # 2. 使用实体的分类信息进行主题检索
        for keyword in topic_keywords:
            entities = self.graph_indexing.get_entities_by_key(keyword)
            for entity in entities:
                if entity.entity_type == "Recipe":
                    # 构建分类主题内容
                    content_parts = [
                        f"主题分类: {keyword}",
                        entity.value_content
                    ]
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=entity.metadata["node_id"],
                        node_type=entity.entity_type,
                        relevance_score=0.85,  # 分类匹配得分
                        retrieval_level="topic",
                        metadata={
                            "entity_name": entity.entity_name,
                            "entity_type": entity.entity_type,
                            "matched_keyword": keyword,
                            "source": "category_match"
                        }
                    ))
        
        # 3. 如果结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_topic_level_search(topic_keywords, top_k - len(results))
            results.extend(neo4j_results)
            
        # 4. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"主题级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]
    
    def _neo4j_topic_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j主题级检索补充"""
        results = []
        
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (r:Recipe)
                WHERE r.category CONTAINS keyword 
                   OR r.cuisineType CONTAINS keyword
                   OR r.tags CONTAINS keyword
                WITH r, keyword
                OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
                WITH r, keyword, collect(i.name)[0..3] as ingredients
                RETURN 
                    r.nodeId as node_id,
                    r.name as name,
                    r.category as category,
                    r.cuisineType as cuisine_type,
                    r.difficulty as difficulty,
                    ingredients,
                    keyword as matched_keyword
                ORDER BY r.difficulty ASC, r.name
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })
                
                for record in result:
                    content_parts = []
                    content_parts.append(f"菜品: {record['name']}")
                    
                    if record["category"]:
                        content_parts.append(f"分类: {record['category']}")
                    if record["cuisine_type"]:
                        content_parts.append(f"菜系: {record['cuisine_type']}")
                    if record["difficulty"]:
                        content_parts.append(f"难度: {record['difficulty']}")
                    
                    if record["ingredients"]:
                        ingredients_str = ', '.join(record["ingredients"][:3])
                        content_parts.append(f"主要食材: {ingredients_str}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Recipe",
                        relevance_score=0.75,  # 补充检索得分
                        retrieval_level="topic",
                        metadata={
                            "name": record["name"],
                            "category": record["category"],
                            "cuisine_type": record["cuisine_type"],
                            "difficulty": record["difficulty"],
                            "matched_keyword": record["matched_keyword"],
                            "source": "neo4j_fallback"
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Neo4j主题级检索失败: {e}")
            
        return results
        
    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """
        双层检索：结合实体级和主题级检索
        """
        logger.info(f"开始双层检索: {query}")
        
        # 1. 提取关键词
        entity_keywords, topic_keywords = self.extract_query_keywords(query)
        
        # 2. 执行双层检索
        entity_results = self.entity_level_retrieval(entity_keywords, top_k)
        topic_results = self.topic_level_retrieval(topic_keywords, top_k)
        
        # 3. 结果合并和排序
        all_results = entity_results + topic_results
        
        # 4. 去重和重排序
        seen_nodes = set()
        unique_results = []
        
        for result in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)
        
        # 5. 转换为Document格式
        documents = []
        for result in unique_results[:top_k]:
            # 确保recipe_name字段正确设置
            recipe_name = result.metadata.get("name") or result.metadata.get("entity_name", "未知菜品")
            
            doc = Document(
                page_content=result.content,
                metadata={
                    "node_id": result.node_id,
                    "node_type": result.node_type,
                    "retrieval_level": result.retrieval_level,
                    "relevance_score": result.relevance_score,
                    "recipe_name": recipe_name,  # 确保有recipe_name字段
                    "search_type": "dual_level",  # 设置搜索类型
                    **result.metadata
                }
            )
            documents.append(doc)
            
        logger.info(f"双层检索完成，返回 {len(documents)} 个文档")
        return documents
    
    def vector_search_enhanced(self, query: str, top_k: int = 5, filter_expr: Optional[str] = None) -> List[Document]:
        """
        增强的向量检索：结合图信息，支持元数据过滤
        """
        try:
            # 使用Milvus进行向量检索
            vector_docs = self.milvus_module.similarity_search(
                query, k=top_k * 2, filter_expr=filter_expr
            )
            
            # 用图信息增强结果并转换为Document对象
            enhanced_docs = []
            for result in vector_docs:
                # 从Milvus结果创建Document对象
                content = result.get("text", "")
                metadata = result.get("metadata", {})
                node_id = metadata.get("node_id")
                
                if node_id:
                    # 从图中获取邻居信息
                    neighbors = self._get_node_neighbors(node_id)
                    if neighbors:
                        # 将邻居信息添加到内容中
                        neighbor_info = f"\n相关信息: {', '.join(neighbors[:3])}"
                        content += neighbor_info
                
                # 确保recipe_name字段正确设置
                recipe_name = metadata.get("recipe_name", "未知菜品")
                
                # 调试：打印向量得分
                vector_score = result.get("score", 0.0)
                logger.debug(f"向量检索得分: {recipe_name} = {vector_score}")
                
                # 创建Document对象
                doc = Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "recipe_name": recipe_name,  # 确保有recipe_name字段
                        "score": vector_score,
                        "search_type": "vector_enhanced"
                    }
                )
                enhanced_docs.append(doc)
                
            return enhanced_docs[:top_k]
            
        except Exception as e:
            logger.error(f"增强向量检索失败: {e}")
            return []
    
    def _get_node_neighbors(self, node_id: str, max_neighbors: int = 3) -> List[str]:
        """获取节点的邻居信息"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n {nodeId: $node_id})-[r]-(neighbor)
                RETURN neighbor.name as name
                LIMIT $limit
                """
                result = session.run(query, {"node_id": node_id, "limit": max_neighbors})
                return [record["name"] for record in result if record["name"]]
        except Exception as e:
            logger.error(f"获取邻居节点失败: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        混合检索：并行执行多种检索策略（双重检索 + 向量检索 + BM25）
        """
        import concurrent.futures

        logger.info(f"开始并行混合检索: {query}")

        # 提取元数据过滤条件
        meta_filter_expr = self._extract_metadata_filters(query)

        # HyDE 查询改写：模糊查询生成假设答案用于检索
        search_query = query
        if self._should_hyde(query):
            hyde_text = self._hyde_rewrite(query)
            if hyde_text:
                search_query = hyde_text
                logger.info(f"HyDE 改写: {query[:40]}... → {search_query[:80]}...")

        # 粗召回：top_k * 4 为后续 rerank 留足候选
        coarse_k = max(top_k * 4, 20)
        dual_docs = []
        vector_docs = []
        bm25_docs = []

        def dual_search():
            nonlocal dual_docs
            try:
                # dual_level 用原始 query（它自己做关键词提取）
                dual_docs = self.dual_level_retrieval(query, coarse_k)
                logger.info(f"双层检索完成: {len(dual_docs)} 个结果")
            except Exception as e:
                logger.error(f"双层检索失败: {e}")
                dual_docs = []

        def vector_search():
            nonlocal vector_docs
            try:
                # 向量检索用 HyDE 改写后的 query（语义更丰富）
                vector_docs = self.vector_search_enhanced(
                    search_query, coarse_k, filter_expr=meta_filter_expr
                )
                logger.info(f"向量检索完成: {len(vector_docs)} 个结果")
            except Exception as e:
                logger.error(f"向量检索失败: {e}")
                vector_docs = []

        def bm25_search():
            nonlocal bm25_docs
            try:
                if self.bm25_retriever:
                    # BM25 用 HyDE 改写后的 query（包含更多关键词）
                    raw_docs = self.bm25_retriever.invoke(search_query)
                    for doc in raw_docs:
                        doc.metadata["search_method"] = "bm25"
                        doc.metadata.setdefault("retrieval_level", "bm25")
                    bm25_docs = raw_docs[:coarse_k]
                    logger.info(f"BM25检索完成: {len(bm25_docs)} 个结果")
                else:
                    logger.warning("BM25检索器未初始化，跳过")
            except Exception as e:
                logger.error(f"BM25检索失败: {e}")
                bm25_docs = []

        # 使用线程池并行执行三种检索
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_dual = executor.submit(dual_search)
            future_vector = executor.submit(vector_search)
            future_bm25 = executor.submit(bm25_search)

            concurrent.futures.wait(
                [future_dual, future_vector, future_bm25], timeout=30
            )

        # Round-robin 轮询合并三种检索结果
        merged_docs = []
        seen_content_hashes = set()
        all_doc_lists = [dual_docs, vector_docs, bm25_docs]
        max_len = max(len(dl) for dl in all_doc_lists)
        origin_len = sum(len(dl) for dl in all_doc_lists)

        for i in range(max_len):
            for doc_list in all_doc_lists:
                if i < len(doc_list):
                    doc = doc_list[i]
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content_hashes:
                        seen_content_hashes.add(content_hash)
                        doc.metadata.setdefault("search_method", "unknown")
                        doc.metadata["round_robin_order"] = len(merged_docs)
                        merged_docs.append(doc)

        logger.info(
            f"Round-robin合并：从总共{origin_len}个结果合并为{len(merged_docs)}个文档"
        )

        # 父子块解析：将 child chunk 映射回完整菜谱父文档
        merged_docs = self._resolve_parents(merged_docs)

        # Reranker 重排序
        if len(merged_docs) > top_k:
            merged_docs = self._rerank(query, merged_docs, top_k)

        logger.info(f"混合检索完成，返回 {len(merged_docs)} 个文档")
        return merged_docs
        
    def _resolve_parents(self, documents: List[Document]) -> List[Document]:
        """
        父子块解析：将 child chunk 映射回完整菜谱父文档，按 parent_id 去重。

        - vector_enhanced/BM25 检索到的 child chunk（doc_type=meta/ingredients/step）
          通过 parent_id 还原为完整菜谱
        - dual_level 检索到的是图实体卡片（无 doc_type / 非 child），直接保留
        """
        resolved = []
        seen_parents = set()

        for doc in documents:
            parent_id = doc.metadata.get("parent_id")
            doc_type = doc.metadata.get("doc_type", "")

            # 判断是否为 child chunk：有 parent_id 且 doc_type 是已知的子类型
            is_child = parent_id and doc_type in ("meta", "ingredients", "step", "content", "chunk")

            if is_child:
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)

                parent_doc = self.data_module.get_parent_document(parent_id)
                if parent_doc:
                    # 复制父文档，合并子文档的检索元信息
                    merged_meta = dict(parent_doc.metadata)
                    merged_meta["search_method"] = doc.metadata.get("search_method", "child_resolved")
                    merged_meta["rerank_score"] = doc.metadata.get("rerank_score", 0)
                    merged_meta["resolved_from_child"] = doc_type
                    resolved.append(Document(
                        page_content=parent_doc.page_content,
                        metadata=merged_meta,
                    ))
                else:
                    # 父文档不可用，保留 child 本身
                    resolved.append(doc)
            else:
                # 非 child chunk（dual_level 实体卡片等），直接保留
                resolved.append(doc)

        logger.info(
            f"父子块解析: {len(documents)} → {len(resolved)} "
            f"(唯一父文档: {len(seen_parents)})"
        )
        return resolved

    def _get_reranker(self):
        """延迟加载 Cross-encoder reranker"""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            model_name = "BAAI/bge-reranker-v2-m3"
            logger.info(f"加载 Reranker 模型: {model_name}")
            self._reranker = CrossEncoder(model_name, device='cpu')
        return self._reranker

    def _rerank(
        self, query: str, documents: List[Document], top_k: int
    ) -> List[Document]:
        """用 Cross-encoder 对候选文档重排序"""
        if not documents or len(documents) <= top_k:
            return documents

        try:
            reranker = self._get_reranker()
            pairs = [(query, doc.page_content) for doc in documents]
            scores = reranker.predict(pairs, show_progress_bar=False)

            for doc, score in zip(documents, scores):
                doc.metadata["rerank_score"] = float(score)

            documents.sort(key=lambda d: d.metadata.get("rerank_score", 0), reverse=True)
            logger.info(
                f"Reranker 重排序完成: {len(documents)} → {top_k}, "
                f"top3 scores: {[f'{s:.3f}' for s in scores[:3]]}"
            )
            return documents[:top_k]
        except Exception as e:
            logger.error(f"Reranker 失败，回退到 Round-robin 顺序: {e}")
            return documents[:top_k]

    def close(self):
        """关闭资源连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭") 