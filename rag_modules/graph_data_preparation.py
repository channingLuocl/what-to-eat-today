"""
图数据库数据准备模块
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from neo4j import GraphDatabase
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """图节点数据结构"""
    node_id: str
    labels: List[str]
    name: str
    properties: Dict[str, Any]

@dataclass
class GraphRelation:
    """图关系数据结构"""
    start_node_id: str
    end_node_id: str
    relation_type: str
    properties: Dict[str, Any]

class GraphDataPreparationModule:
    """图数据库数据准备模块 - 从Neo4j读取数据并转换为文档"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        初始化图数据库连接
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_docs: Dict[str, Document] = {}  # node_id → 完整菜谱文档
        self.recipes: List[GraphNode] = []
        self.ingredients: List[GraphNode] = []
        self.cooking_steps: List[GraphNode] = []
        
        self._connect()
    
    def _connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                database=self.database
            )
            logger.info(f"已连接到Neo4j数据库: {self.uri}")
            
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()
                if test_result:
                    logger.info("Neo4j连接测试成功")
                    
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
    
    def load_graph_data(self) -> Dict[str, Any]:
        """
        从Neo4j加载图数据
        
        Returns:
            包含节点和关系的数据字典
        """
        logger.info("正在从Neo4j加载图数据...")
        
        with self.driver.session() as session:
            # 加载所有菜谱节点，从Category关系中读取分类信息
            recipes_query = """
            MATCH (r:Recipe)
            WHERE r.nodeId >= '200000000'
            OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:Category)
            WITH r, collect(c.name) as categories
            RETURN r.nodeId as nodeId, labels(r) as labels, r.name as name, 
                   properties(r) as originalProperties,
                   CASE WHEN size(categories) > 0 
                        THEN categories[0] 
                        ELSE COALESCE(r.category, '未知') END as mainCategory,
                   CASE WHEN size(categories) > 0 
                        THEN categories 
                        ELSE [COALESCE(r.category, '未知')] END as allCategories
            ORDER BY r.nodeId
            """
            
            result = session.run(recipes_query)
            self.recipes = []
            for record in result:
                # 合并原始属性和新的分类信息
                properties = dict(record["originalProperties"])
                properties["category"] = record["mainCategory"]
                properties["all_categories"] = record["allCategories"]
                
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=properties
                )
                self.recipes.append(node)
            
            logger.info(f"加载了 {len(self.recipes)} 个菜谱节点")
            
            # 加载所有食材节点
            ingredients_query = """
            MATCH (i:Ingredient)
            WHERE i.nodeId >= '200000000'
            RETURN i.nodeId as nodeId, labels(i) as labels, i.name as name,
                   properties(i) as properties
            ORDER BY i.nodeId
            """
            
            result = session.run(ingredients_query)
            self.ingredients = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.ingredients.append(node)
            
            logger.info(f"加载了 {len(self.ingredients)} 个食材节点")
            
            # 加载所有烹饪步骤节点
            steps_query = """
            MATCH (s:CookingStep)
            WHERE s.nodeId >= '200000000'
            RETURN s.nodeId as nodeId, labels(s) as labels, s.name as name,
                   properties(s) as properties
            ORDER BY s.nodeId
            """
            
            result = session.run(steps_query)
            self.cooking_steps = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.cooking_steps.append(node)
            
            logger.info(f"加载了 {len(self.cooking_steps)} 个烹饪步骤节点")
        
        return {
            'recipes': len(self.recipes),
            'ingredients': len(self.ingredients),
            'cooking_steps': len(self.cooking_steps)
        }
    
    def build_recipe_documents(self) -> List[Document]:
        """
        构建菜谱文档，集成相关的食材和步骤信息
        
        Returns:
            结构化的菜谱文档列表
        """
        logger.info("正在构建菜谱文档...")
        
        documents = []
        
        with self.driver.session() as session:
            for recipe in self.recipes:
                try:
                    recipe_id = recipe.node_id
                    recipe_name = recipe.name
                    
                    # 获取菜谱的相关食材
                    ingredients_query = """
                    MATCH (r:Recipe {nodeId: $recipe_id})-[req:REQUIRES]->(i:Ingredient)
                    RETURN i.name as name, i.category as category, 
                           req.amount as amount, req.unit as unit,
                           i.description as description
                    ORDER BY i.name
                    """
                    
                    ingredients_result = session.run(ingredients_query, {"recipe_id": recipe_id})
                    ingredients_info = []
                    for ing_record in ingredients_result:
                        amount = ing_record.get("amount", "")
                        unit = ing_record.get("unit", "")
                        ingredient_text = f"{ing_record['name']}"
                        if amount and unit:
                            ingredient_text += f"({amount}{unit})"
                        if ing_record.get("description"):
                            ingredient_text += f" - {ing_record['description']}"
                        ingredients_info.append(ingredient_text)
                    
                    # 获取菜谱的烹饪步骤
                    steps_query = """
                    MATCH (r:Recipe {nodeId: $recipe_id})-[c:CONTAINS_STEP]->(s:CookingStep)
                    RETURN s.name as name, s.description as description,
                           s.stepNumber as stepNumber, s.methods as methods,
                           s.tools as tools, s.timeEstimate as timeEstimate,
                           c.stepOrder as stepOrder
                    ORDER BY COALESCE(c.stepOrder, s.stepNumber, 999)
                    """
                    
                    steps_result = session.run(steps_query, {"recipe_id": recipe_id})
                    steps_info = []
                    for step_record in steps_result:
                        step_text = f"步骤: {step_record['name']}"
                        if step_record.get("description"):
                            step_text += f"\n描述: {step_record['description']}"
                        if step_record.get("methods"):
                            step_text += f"\n方法: {step_record['methods']}"
                        if step_record.get("tools"):
                            step_text += f"\n工具: {step_record['tools']}"
                        if step_record.get("timeEstimate"):
                            step_text += f"\n时间: {step_record['timeEstimate']}"
                        steps_info.append(step_text)
                    
                    # 构建完整的菜谱文档内容
                    content_parts = [f"# {recipe_name}"]
                    
                    # 添加菜谱基本信息
                    if recipe.properties.get("description"):
                        content_parts.append(f"\n## 菜品描述\n{recipe.properties['description']}")
                    
                    if recipe.properties.get("cuisineType"):
                        content_parts.append(f"\n菜系: {recipe.properties['cuisineType']}")
                    
                    if recipe.properties.get("difficulty"):
                        content_parts.append(f"难度: {recipe.properties['difficulty']}星")
                    
                    if recipe.properties.get("prepTime") or recipe.properties.get("cookTime"):
                        time_info = []
                        if recipe.properties.get("prepTime"):
                            time_info.append(f"准备时间: {recipe.properties['prepTime']}")
                        if recipe.properties.get("cookTime"):
                            time_info.append(f"烹饪时间: {recipe.properties['cookTime']}")
                        content_parts.append(f"\n时间信息: {', '.join(time_info)}")
                    
                    if recipe.properties.get("servings"):
                        content_parts.append(f"份量: {recipe.properties['servings']}")
                    
                    # 添加食材信息
                    if ingredients_info:
                        content_parts.append("\n## 所需食材")
                        for i, ingredient in enumerate(ingredients_info, 1):
                            content_parts.append(f"{i}. {ingredient}")
                    
                    # 添加步骤信息
                    if steps_info:
                        content_parts.append("\n## 制作步骤")
                        for i, step in enumerate(steps_info, 1):
                            content_parts.append(f"\n### 第{i}步\n{step}")
                    
                    # 添加标签信息
                    if recipe.properties.get("tags"):
                        content_parts.append(f"\n## 标签\n{recipe.properties['tags']}")
                    
                    # 组合成最终内容
                    full_content = "\n".join(content_parts)
                    
                    # 创建文档对象
                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "node_id": recipe_id,
                            "recipe_name": recipe_name,
                            "node_type": "Recipe",
                            "category": recipe.properties.get("category", "未知"),
                            "cuisine_type": recipe.properties.get("cuisineType", "未知"),
                            "difficulty": recipe.properties.get("difficulty", 0),
                            "prep_time": recipe.properties.get("prepTime", ""),
                            "cook_time": recipe.properties.get("cookTime", ""),
                            "servings": recipe.properties.get("servings", ""),
                            "ingredients_count": len(ingredients_info),
                            "steps_count": len(steps_info),
                            "doc_type": "recipe",
                            "content_length": len(full_content)
                        }
                    )
                    
                    documents.append(doc)
                    # 存储完整菜谱作为父文档（用于父子块索引检索时还原）
                    self.parent_docs[recipe_id] = doc
                    
                except Exception as e:
                    logger.warning(f"构建菜谱文档失败 {recipe_name} (ID: {recipe_id}): {e}")
                    continue
        
        self.documents = documents
        logger.info(f"成功构建 {len(documents)} 个菜谱文档")
        return documents
    
    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """
        按章节类型创建父子块索引（Small-to-Big Retrieval）。

        每个菜谱拆分为聚焦的 child chunk（meta / ingredients / step），
        检索时命中 child 后通过 parent_id 还原完整菜谱文档。
        参数 chunk_size / chunk_overlap 保留兼容旧接口，不再使用。
        """
        logger.info("正在进行父子块索引分块（按章节类型）...")

        if not self.documents:
            raise ValueError("请先构建文档")

        chunks = []
        chunk_id = 0

        for doc in self.documents:
            parent_id = doc.metadata["node_id"]
            recipe_name = doc.metadata["recipe_name"]
            content = doc.page_content

            # ---- 解析章节 ----
            sections = self._parse_recipe_sections(content, recipe_name)

            # ---- child 1: meta ----
            meta_text = (
                f"菜谱简介: {recipe_name}\n"
                f"菜系: {doc.metadata.get('cuisine_type', '未知')}\n"
                f"分类: {doc.metadata.get('category', '未知')}\n"
                f"难度: {doc.metadata.get('difficulty', 0)}/5\n"
                f"准备时间: {doc.metadata.get('prep_time', '未知')}\n"
                f"烹饪时间: {doc.metadata.get('cook_time', '未知')}\n"
                f"份量: {doc.metadata.get('servings', '未知')}"
            )
            if sections.get("description"):
                meta_text += f"\n描述: {sections['description'][:200]}"
            if sections.get("tags"):
                meta_text += f"\n标签: {sections['tags']}"

            chunks.append(self._make_child(
                doc, parent_id, chunk_id, 0, "meta", meta_text
            ))
            chunk_id += 1

            # ---- child 2: ingredients ----
            if sections.get("ingredients"):
                ing_text = f"食材清单 - {recipe_name}:\n{sections['ingredients']}"
                chunks.append(self._make_child(
                    doc, parent_id, chunk_id, 1, "ingredients", ing_text
                ))
                chunk_id += 1

            # ---- children: steps ----
            steps = sections.get("steps", [])
            for si, step_text in enumerate(steps):
                full_step = f"制作步骤 - {recipe_name}:\n{step_text}"
                chunks.append(self._make_child(
                    doc, parent_id, chunk_id, 2 + si, "step", full_step
                ))
                chunk_id += 1

            if not steps:
                chunks.append(self._make_child(
                    doc, parent_id, chunk_id, 2, "content",
                    content[:800]
                ))
                chunk_id += 1

        self.chunks = chunks
        logger.info(
            f"父子块索引完成: {len(self.documents)} 个菜谱 → {len(chunks)} 个 child chunk "
            f"(meta/ingredients/step)"
        )
        return chunks

    def _parse_recipe_sections(self, content: str, recipe_name: str) -> Dict[str, Any]:
        """从菜谱 Markdown 中解析各章节"""
        sections: Dict[str, Any] = {"steps": []}

        current_section = None
        current_content: List[str] = []

        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("## 菜品描述"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "description"
                current_content = []
            elif stripped.startswith("## 所需食材"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "ingredients"
                current_content = []
            elif stripped.startswith("## 制作步骤"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "steps"
                current_content = []
            elif stripped.startswith("## 标签"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "tags"
                current_content = []
            elif stripped.startswith("### 第") and current_section == "steps":
                if current_content:
                    sections["steps"].append("\n".join(current_content).strip())
                current_content = [stripped]
            elif stripped.startswith("# ") and not current_section:
                continue
            else:
                current_content.append(line)

        # 收尾
        if current_section == "steps" and current_content:
            sections["steps"].append("\n".join(current_content).strip())
        elif current_section and current_section != "steps":
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _make_child(self, parent_doc: Document, parent_id: str,
                    chunk_id: int, chunk_index: int, doc_type: str,
                    content: str) -> Document:
        """创建子块 Document，继承父文档的结构化元数据"""
        return Document(
            page_content=content,
            metadata={
                "node_id": parent_doc.metadata.get("node_id", parent_id),
                "recipe_name": parent_doc.metadata.get("recipe_name", ""),
                "node_type": parent_doc.metadata.get("node_type", "Recipe"),
                "category": parent_doc.metadata.get("category", "未知"),
                "cuisine_type": parent_doc.metadata.get("cuisine_type", "未知"),
                "difficulty": parent_doc.metadata.get("difficulty", 0),
                "chunk_id": f"{parent_id}_chunk_{chunk_id}",
                "parent_id": parent_id,
                "chunk_index": chunk_index,
                "doc_type": doc_type,
                "chunk_size": len(content),
            },
        )

    def get_parent_document(self, node_id: str) -> Optional[Document]:
        """根据 node_id 获取完整菜谱父文档"""
        return self.parent_docs.get(node_id)
    

    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_recipes': len(self.recipes),
            'total_ingredients': len(self.ingredients),
            'total_cooking_steps': len(self.cooking_steps),
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks)
        }
        
        if self.documents:
            # 分类统计
            categories = {}
            cuisines = {}
            difficulties = {}
            
            for doc in self.documents:
                category = doc.metadata.get('category', '未知')
                categories[category] = categories.get(category, 0) + 1
                
                cuisine = doc.metadata.get('cuisine_type', '未知')
                cuisines[cuisine] = cuisines.get(cuisine, 0) + 1
                
                difficulty = doc.metadata.get('difficulty', 0)
                difficulties[str(difficulty)] = difficulties.get(str(difficulty), 0) + 1
            
            stats.update({
                'categories': categories,
                'cuisines': cuisines,
                'difficulties': difficulties,
                'avg_content_length': sum(doc.metadata.get('content_length', 0) for doc in self.documents) / len(self.documents),
                'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
            })
        
        return stats
    

    
    def __del__(self):
        """析构函数，确保关闭连接"""
        self.close() 