# RAG 评估系统

针对本项目（菜谱 RAG）设计的端到端评估流水线，覆盖检索质量、生成质量、拒答行为三大维度。

## 一、评估指标

### 1. RAGAS 软指标（LLM-as-judge，基于 MiniMax）

| 指标 | 含义 |
|------|------|
| `context_recall` | 检索内容覆盖 reference_answer 的程度 |
| `faithfulness` | 回答是否完全基于检索内容（无幻觉） |
| `answer_relevancy` | 回答与问题的相关性 |

### 2. 检索硬指标（确定性，无需 LLM）

| 指标 | 含义 |
|------|------|
| `recall@K` | 标准文档被检索到的比例 |
| `precision@K` | 检索结果中相关文档的比例 |
| `f1@K` | recall 和 precision 的调和平均 |
| `mrr` | 第一个相关文档的排名倒数 |
| `hit@K` | 至少有一个相关文档（0/1） |
| `ndcg@K` | 考虑排序的归一化打分 |

### 3. 拒答行为评估

| 指标 | 含义 |
|------|------|
| `correct_rejection` | 应拒答 + 实际拒答 |
| `false_answer_hallucination` | 应拒答 + 实际给了答案（幻觉） |
| `false_rejection` | 应回答 + 实际拒答（过度保守） |

### 4. 统计可信度

所有指标都附带 95% Bootstrap 置信区间（n_iter=1000），用于判断版本间差异是否显著。

---

## 二、文件结构

```
evaluation/
├── llm_factory.py            # MiniMax LLM + BGE embedding 包装，供 RAGAS 用
├── retrieval_metrics.py      # 检索硬指标（Recall@K / MRR / NDCG / 拒答评估）
├── ragas_evaluator.py        # RAGAS 评估器封装，含 EvaluationResult 数据结构
├── bootstrap_ci.py           # Bootstrap 置信区间
├── metrics_collector.py      # 指标汇总、分组统计、报告生成、对比
├── build_dataset.py          # 自动扫描 data/dishes/ 生成测试样本
├── merge_dataset.py          # 合并 _auto + _manual 为最终 eval_dataset.json
├── run_evaluation.py         # 主入口
├── smoke_test.py             # 不依赖 RAG/RAGAS 的冒烟测试
├── _auto_dataset.json        # 自动生成的样本（120 条）
├── _manual_edge_cases.json   # 手工编写的 edge case（80 条）
├── eval_dataset.json         # 评测集（100 条，按 category × difficulty 分层抽样）
├── eval_dataset.200.bak.json # 合并后的完整评测集备份（200 条）
├── eval_dataset.old.json     # 旧版数据集备份（35 条）
└── README.md                 # 本文件
```

---

## 三、快速开始

### 1. 安装依赖

```bash
pip install 'ragas>=0.2,<0.3' \
            'langchain-openai' \
            'langchain-huggingface' \
            'sentence-transformers' \
            'pandas'
```

### 2. 配置环境变量（在项目根目录的 `.env`）

```dotenv
LLM_MODEL=MiniMax-M2.7
OPENAI_API_KEY=your_minimax_api_key
OPENAI_BASE_URL=https://api.minimaxi.com/v1
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

### 3. 跑冒烟测试（不连数据库、不调 API，验证代码本身）

```bash
python evaluation/smoke_test.py
```

### 4. 生成最终评测集

```bash
python evaluation/merge_dataset.py
# 会把 _auto + _manual 合并成 eval_dataset.json
```

### 5. 跑完整评估

```bash
# 完整跑（需要 Milvus + Neo4j 在线，会调用 MiniMax API）
python evaluation/run_evaluation.py

# 只跑检索指标，不调 LLM（快、免费）
python evaluation/run_evaluation.py --mode retrieval

# 用 Mock RAG 跑评估流水线（验证代码用，不连 RAG）
python evaluation/run_evaluation.py --mock-rag --no-ragas --limit 20

# 限制样本数 debug
python evaluation/run_evaluation.py --limit 10
```

输出会写到 `./evaluation_results/` 目录，包括：
- `eval_results_<时间>.json` — 完整结果（每条样本所有指标）
- `eval_per_query_<时间>.csv` — 每条 query 一行的 CSV，方便人工 review
- `eval_report_<时间>.md` — Markdown 报告，含分类统计、置信区间、Top worst cases

### 6. 对比两次评估结果

```bash
python evaluation/run_evaluation.py --compare \
    evaluation_results/eval_results_20260501_120000.json \
    evaluation_results/eval_results_20260505_120000.json
```

会输出：
- 总体指标变化 + Bootstrap CI 显著性判断
- case-level 翻转：哪些从 miss 变 hit、哪些从 hit 变 miss

---

## 四、数据集 Schema

```json
{
  "id": "A001",
  "query": "怎么做清蒸鲈鱼？",
  "category": "entity_query",
  "difficulty": "easy",
  "should_answer": true,
  "expected_doc_paths": [
    "data/dishes/aquatic/清蒸鲈鱼/清蒸鲈鱼.md"
  ],
  "reference_answer": "清蒸鲈鱼蒸10分钟，淋蒸鱼豉油和热油即可。",
  "tags": ["aquatic", "entity"]
}
```

字段说明：

- `id` — 唯一标识，A* 是自动生成的，M* 是手工的
- `category` — 类型，详见下表
- `difficulty` — easy / medium / hard
- `should_answer` — `true` 表示系统应该回答；`false` 表示应该拒答（用于评估拒答行为）
- `expected_doc_paths` — 系统应该检索到的文档路径列表（相对项目根目录）
- `reference_answer` — 标准答案（用于 RAGAS context_recall 和 faithfulness）

**category 分布（100 条，按 category × difficulty 分层抽样自 200 条全集）**：

| category | 数量 | 说明 |
|----------|------|------|
| entity_query | 34 | 按菜名问做法 |
| ingredient_query | 12 | 按食材问相关菜 |
| cooking_tip_query | 7 | 烹饪技巧 |
| robustness_case | 6 | 改写/错别字鲁棒性 |
| health_query | 5 | 健康/特殊人群饮食 |
| scenario_query | 5 | 场景（带饭/约会等） |
| cooking_method_query | 5 | 按烹饪方法问 |
| category_query | 5 | 按品类问 |
| rejection_case | 4 | 应拒答（越界） |
| multi_hop_query | 4 | 多跳/约束推理 |
| empty_kb_case | 4 | 库里没有 |
| seasonal_query | 3 | 时令 |
| diet_query | 3 | 饮食偏好（素食/低卡） |
| ambiguous_query | 3 | 歧义 query |

> 200 条全集见 `eval_dataset.200.bak.json`，需要时可替换 `eval_dataset.json` 跑全量评测。

---

## 五、典型用法场景

### 场景 1：开发期快速迭代

只跑硬指标，不调 LLM：
```bash
python evaluation/run_evaluation.py --mode retrieval --no-ragas
```

### 场景 2：发版前完整评估

```bash
python evaluation/run_evaluation.py --output ./evaluation_results/v1.2
```

### 场景 3：A/B 对比

```bash
# 第一版
python evaluation/run_evaluation.py --output ./evaluation_results/v1.2

# 调参后
python evaluation/run_evaluation.py --output ./evaluation_results/v1.3

# 对比
python evaluation/run_evaluation.py --compare \
    ./evaluation_results/v1.2/eval_results_*.json \
    ./evaluation_results/v1.3/eval_results_*.json
```

### 场景 4：定位问题

打开 `eval_per_query_*.csv` 按 `recall_at_k` 升序排序，看检索系统漏掉了哪些菜；按 `faithfulness` 升序看生成阶段哪些回答跟检索内容对不上。

---

## 六、注意事项

1. **MiniMax API key 不要写进代码或仓库**。`llm_factory.py` 已经从 `os.getenv("OPENAI_API_KEY")` 读，请配置 `.env`。
2. **第一次跑 BGE embedding 会下载模型**（约 100MB），建议提前下载或设置 HF 镜像。
3. **RAGAS context_recall 依赖 reference_answer 的质量**。如果某条样本的 reference 写得不好，那条的指标也会偏低。
4. **拒答样本不会跑 RAGAS**（没有 reference 可以对比），只算拒答行为指标。
5. **简化 fallback (`--no-ragas`) 的分数不可信**，仅用于代码冒烟测试，**不要用来做版本对比**。
