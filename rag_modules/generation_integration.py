"""
生成集成模块
"""

import logging
import os
import time
from typing import List, Optional

from openai import OpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """生成集成模块 - 负责答案生成"""

    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.moonshot.cn/v1")

        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

        logger.info(f"生成模块初始化完成，模型: {model_name}, API地址: {self.base_url}")

    def _build_prompt(self, question: str, context: str, enrichment: Optional[str] = None) -> str:
        """构建统一的提示词"""
        enrichment_section = ""
        if enrichment:
            enrichment_section = f"""
## 知识图谱补充信息
{enrichment}
"""

        return f"""你是一位专业的烹饪助手。请严格基于以下检索到的信息回答用户问题。

## 检索到的相关信息
{context}
{enrichment_section}
## 用户问题
{question}

## 回答要求
1. **严格基于检索内容**：只能使用上面「检索到的相关信息」中明确提到的内容，不要编造或添加未提及的食材、调料、步骤或细节。
2. **信息不足时明确告知**：如果检索内容不足以回答用户的问题，直接说"抱歉，目前的知识库中暂时没有找到相关信息"，不要猜测或编造。
3. **引用来源**：在回答中标注信息来源，例如「根据《宫保鸡丁》菜谱...」或「菜谱中提到...」。
4. **结构清晰**：
   - 如果是询问具体菜谱做法，请按「所需食材→制作步骤→小贴士」组织
   - 如果是询问推荐/列表类问题，请用清晰的列表呈现，每条注明出处
   - 如果是一般性咨询，请提供综合性的结构化回答

## 回答：
"""

    def _build_context(self, documents: List[Document]) -> str:
        """从文档列表构建上下文字符串"""
        context_parts = []
        for doc in documents:
            content = doc.page_content.strip()
            if content:
                level = doc.metadata.get("retrieval_level", "")
                if level:
                    context_parts.append(f"[{level.upper()}] {content}")
                else:
                    context_parts.append(content)
        return "\n\n".join(context_parts)

    def generate_adaptive_answer(
        self,
        question: str,
        documents: List[Document],
        enrichment_context: Optional[str] = None,
    ) -> str:
        """智能统一答案生成，支持图增强上下文"""
        context = self._build_context(documents)
        prompt = self._build_prompt(question, context, enrichment_context)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def generate_adaptive_answer_stream(
        self,
        question: str,
        documents: List[Document],
        enrichment_context: Optional[str] = None,
        max_retries: int = 3,
    ):
        """流式答案生成（带重试机制），支持图增强上下文"""
        context = self._build_context(documents)
        prompt = self._build_prompt(question, context, enrichment_context)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=60,
                )

                if attempt == 0:
                    print("开始流式生成回答...\n")
                else:
                    print(f"第{attempt + 1}次尝试流式生成...\n")

                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content

                return

            except Exception as e:
                logger.warning(f"流式生成第{attempt + 1}次尝试失败: {e}")

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"⚠️ 连接中断，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("流式生成完全失败，尝试非流式后备方案")
                    print("⚠️ 流式生成失败，切换到标准模式...")
                    try:
                        fallback = self.generate_adaptive_answer(
                            question, documents, enrichment_context
                        )
                        yield fallback
                        return
                    except Exception as fallback_error:
                        logger.error(f"后备生成也失败: {fallback_error}")
                        yield f"抱歉，生成回答时出现网络错误，请稍后重试。错误信息：{str(e)}"
                        return
