# RAG 评估报告

生成时间: 2026-05-11 10:13:16
总查询数: **100**  (应答 93 / 拒答样本 7)

## 总体指标 (95% Bootstrap CI)

| 指标 | 平均值 [CI 下界, CI 上界] | std | n |
|------|---------------------------|-----|---|
| Context Recall (RAGAS) | 0.3269 [0.2439, 0.4082] | 0.3904 | 93 |
| Faithfulness (RAGAS) | 0.7554 [0.6991, 0.8062] | 0.2571 | 93 |
| Answer Relevancy (RAGAS) | 0.5779 [0.4981, 0.6610] | 0.4197 | 93 |
| Recall@K (硬) | 0.6063 [0.5200, 0.6993] | 0.4288 | 93 |
| Precision@K (硬) | 0.2421 [0.2014, 0.2891] | 0.2251 | 93 |
| F1@K (硬) | 0.3065 [0.2636, 0.3518] | 0.2202 | 93 |
| MRR (硬) | 0.6885 [0.5950, 0.7742] | 0.4356 | 93 |
| Hit@K (硬) | 0.7527 [0.6667, 0.8387] | 0.4314 | 93 |
| NDCG@K (硬) | 0.6115 [0.5251, 0.6943] | 0.4200 | 93 |

## 拒答行为评估

- 正确拒答率 (correct_rejection / 应拒答样本数): **71.43%**
- 幻觉率 (false_answer / 应拒答样本数): **28.57%**
- 过度拒答率 (false_rejection / 应答样本数): **37.63%**

## 按 category 分组

| category | n | ctx_recall | faithfulness | answer_relevancy | recall@k | mrr | hit@k |
|----------|---|------------|--------------|-------------------|----------|-----|-------|
| entity_query | 38 | 0.609 | 0.857 | 0.685 | 0.868 | 0.868 | 0.868 |
| ingredient_query | 12 | 0.000 | 0.719 | 0.640 | 0.424 | 0.753 | 0.917 |
| cooking_tip_query | 7 | 0.286 | 0.736 | 0.397 | 0.571 | 0.643 | 0.714 |
| robustness_case | 6 | 0.444 | 0.871 | 0.406 | 0.833 | 0.833 | 0.833 |
| category_query | 5 | 0.100 | 0.728 | 0.744 | 0.164 | 0.600 | 0.600 |
| cooking_method_query | 5 | 0.133 | 0.720 | 0.334 | 0.463 | 0.800 | 0.800 |
| health_query | 5 | 0.067 | 0.509 | 0.168 | 0.700 | 0.333 | 0.800 |
| scenario_query | 5 | 0.000 | 0.693 | 0.874 | 0.133 | 0.300 | 0.400 |
| multi_hop_query | 4 | 0.271 | 0.547 | 0.601 | 0.500 | 0.583 | 0.750 |
| rejection_case | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| diet_query | 3 | 0.000 | 0.621 | 0.604 | 0.000 | 0.000 | 0.000 |
| seasonal_query | 3 | 0.000 | 0.456 | 0.000 | 0.000 | 0.000 | 0.000 |
| ambiguous_query | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## 按 difficulty 分组

| difficulty | n | ctx_recall | faithfulness | recall@k | mrr |
|------------|---|------------|--------------|----------|-----|
| easy | 32 | 0.377 | 0.875 | 0.661 | 0.768 |
| hard | 16 | 0.385 | 0.672 | 0.615 | 0.641 |
| medium | 52 | 0.285 | 0.712 | 0.575 | 0.658 |

## Top 10 最差检索 cases (按 hit@k=0 + 低 recall)


- **A059** (`category_query`) **Q**: 主食有什么推荐？
  - expected: `['data/dishes/staple/蛋炒饭.md', 'data/dishes/staple/扬州炒饭/扬州炒饭.md', 'data/dishes/staple/炸酱面.md']`
  - retrieved: `['西红柿鸡蛋挂面', '炒方便面', '老干妈拌面']`
  - recall@k=0.00, mrr=0.00

- **A061** (`category_query`) **Q**: 有哪些常用调味料的做法？
  - expected: `['data/dishes/condiment/油泼辣子/油泼辣子.md', 'data/dishes/condiment/糖醋汁.md', 'data/dishes/condiment/葱油.md']`
  - retrieved: `['醋', '酱油', '豆瓣酱']`
  - recall@k=0.00, mrr=0.00

- **A075** (`robustness_case`) **Q**: 番茄炒蛋的做法
  - expected: `['data/dishes/vegetable_dish/西红柿炒鸡蛋.md']`
  - retrieved: `['西红柿鸡蛋汤', '番茄牛肉蛋花汤', '鸡蛋羹']`
  - recall@k=0.00, mrr=0.00

- **A088** (`entity_query`) **Q**: 排骨苦瓜汤怎么做？
  - expected: `['data/dishes/soup/排骨苦瓜汤/排骨苦瓜汤.md']`
  - retrieved: `['陈皮排骨汤', '罗宋汤', '苦瓜']`
  - recall@k=0.00, mrr=0.00

- **M008** (`diet_query`) **Q**: 有没有适合素食者的菜谱？
  - expected: `['data/dishes/vegetable_dish/凉拌黄瓜.md', 'data/dishes/vegetable_dish/酸辣土豆丝.md', 'data/dishes/vegetable_dish/炒青菜.md']`
  - retrieved: `['素炒豆角', '西红柿炒鸡蛋', '带把肘子']`
  - recall@k=0.00, mrr=0.00

- **M009** (`diet_query`) **Q**: 高蛋白低脂的菜谱有哪些？
  - expected: `['data/dishes/aquatic/清蒸鲈鱼/清蒸鲈鱼.md', 'data/dishes/aquatic/白灼虾/白灼虾.md', 'data/dishes/meat_dish/可乐鸡翅.md']`
  - retrieved: `['豆腐', '炒青菜', '牛肉']`
  - recall@k=0.00, mrr=0.00

- **M016** (`cooking_tip_query`) **Q**: 做馒头需要多长时间？
  - expected: `['data/dishes/breakfast/蒸花卷.md']`
  - retrieved: `['炒馍', '基础牛奶面包', '中式馅饼']`
  - recall@k=0.00, mrr=0.00

- **M019** (`scenario_query`) **Q**: 上班族带饭，有什么菜适合？
  - expected: `['data/dishes/meat_dish/红烧肉/简易红烧肉.md', 'data/dishes/meat_dish/宫保鸡丁/宫保鸡丁.md']`
  - retrieved: `['蛋炒饭', '蛋包饭', '姜炒鸡']`
  - recall@k=0.00, mrr=0.00

- **M020** (`scenario_query`) **Q**: 想给女朋友做顿浪漫晚餐，推荐什么？
  - expected: `['data/dishes/meat_dish/牛排/牛排.md', 'data/dishes/dessert/提拉米苏/提拉米苏.md']`
  - retrieved: `['萝卜炖羊排', '西红柿鸡蛋挂面', '烤鱼']`
  - recall@k=0.00, mrr=0.00

- **M022** (`scenario_query`) **Q**: 新手第一次下厨做什么菜不容易翻车？
  - expected: `['data/dishes/vegetable_dish/西红柿炒鸡蛋.md', 'data/dishes/vegetable_dish/凉拌黄瓜.md', 'data/dishes/meat_dish/可乐鸡翅.md']`
  - retrieved: `['蒜蓉空心菜', '带把肘子', '鸡蛋羹']`
  - recall@k=0.00, mrr=0.00

## Top 5 最低 faithfulness (可能幻觉) cases


- **M047** **Q**: 蜗牛怎么烹饪？
  - response: 抱歉，目前的知识库中暂时没有找到关于蜗牛烹饪的相关信息。
  - faithfulness=0.000

- **M066** **Q**: 贫血吃什么补血？
  - response: 抱歉，目前的知识库中暂时没有找到相关信息。
  - faithfulness=0.000

- **M072** **Q**: 广式煲汤怎么做？
  - response: 抱歉，目前的知识库中暂时没有找到关于“广式煲汤”的相关信息。
  - faithfulness=0.000

- **M079** **Q**: 汉堡包的做法
  - response: 抱歉，生成回答时出现错误：Error code: 403 - {'error': {'message': '预扣费额度失败, 用户剩余额度: ＄0.004018, 需要预扣费额度: ＄0.004248 (request id: 202605...
  - faithfulness=0.000

- **A016** **Q**: 酸梅汤的做法
  - response: 抱歉，目前的知识库中暂时没有找到关于酸梅汤做法的相关信息。检索到的内容中仅提到了“酸梅汤”这个菜品名称及其分类为饮料，但并未包含具体的食材、调料或制作步骤。
  - faithfulness=0.250