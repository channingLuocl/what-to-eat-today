"""
评测数据集自动生成脚本

扫描 data/dishes/ 下所有菜谱文件，按规则生成测试样本：
- entity_query     按菜名直接问做法 (~30)
- ingredient_query 按主食材问相关菜 (~20)
- category_query   按品类问 (~12)
- cooking_method   按烹饪方法（清蒸/红烧/凉拌等）问 (~15)

输出 JSON 列表，可直接合并到 eval_dataset.json。
人工 edge case 部分由 build_edge_cases.py 提供。

用法:
    python evaluation/build_dataset.py --out evaluation/_auto_dataset.json
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

# 项目根目录
ROOT = Path(__file__).parent.parent
DISHES_DIR = ROOT / "data" / "dishes"

# 品类中文名映射
CATEGORY_CN = {
    "aquatic": "水产",
    "breakfast": "早餐",
    "condiment": "调味料",
    "dessert": "甜品",
    "drink": "饮品",
    "meat_dish": "荤菜",
    "semi-finished": "半成品",
    "soup": "汤羹",
    "staple": "主食",
    "vegetable_dish": "素菜",
}

# 烹饪方法关键词 -> 类型
COOKING_METHODS = {
    "清蒸": "蒸",
    "红烧": "红烧",
    "凉拌": "凉拌",
    "糖醋": "糖醋",
    "蒜蓉": "蒜蓉",
    "葱油": "葱油",
    "椒盐": "椒盐",
    "干煸": "干煸",
    "黄焖": "黄焖",
    "麻婆": "麻婆",
    "酱": "酱卤",
    "卤": "酱卤",
    "炸": "炸",
    "烤": "烤",
    "炒": "炒",
    "蒸": "蒸",
    "焖": "焖",
    "炖": "炖",
    "煎": "煎",
}

# 主食材关键词（用于 ingredient_query）—— 简单关键词匹配，不准但够用
INGREDIENTS = [
    "鸡蛋", "鸡肉", "鸡腿", "鸡翅",
    "猪肉", "五花肉", "排骨", "猪蹄",
    "牛肉", "羊肉", "鸭",
    "鱼", "虾", "蟹",
    "豆腐",
    "土豆", "茄子", "西红柿", "黄瓜",
    "面", "饭", "粥",
]


def get_dish_title(md_path: Path) -> str:
    """从文件名/路径推断菜名（去掉扩展和括号注释）"""
    name = md_path.stem
    # 去掉括号中的内容如 (半成品加工)
    name = re.sub(r"[（(].*?[)）]", "", name).strip()
    # 如果父目录名和文件名一样，说明文件名就是菜名
    return name


def get_relative_path(md_path: Path) -> str:
    """获取相对于项目根目录的路径，统一用 / 分隔符"""
    rel = md_path.relative_to(ROOT)
    return str(rel).replace("\\", "/")


def get_category(md_path: Path) -> str:
    """根据路径推断品类"""
    parts = md_path.relative_to(DISHES_DIR).parts
    return parts[0] if parts else ""


def detect_cooking_method(title: str) -> str:
    """检测菜名中包含的烹饪方法"""
    for kw, method in COOKING_METHODS.items():
        if kw in title:
            return method
    return ""


def detect_ingredient(title: str) -> str:
    """检测菜名中包含的主食材"""
    for ing in INGREDIENTS:
        if ing in title:
            return ing
    return ""


# ============= 生成器 =============


def gen_entity_queries(dishes: List[Dict]) -> List[Dict]:
    """按菜名问：'怎么做X？' 'X的做法是什么？'"""
    templates = [
        "怎么做{}？",
        "{}的做法是什么？",
        "{}怎么做才好吃？",
        "请告诉我{}的烹饪步骤",
        "{}需要哪些食材？",
    ]
    # 选有代表性的菜（每个品类挑几个）
    by_cat: Dict[str, List[Dict]] = {}
    for d in dishes:
        by_cat.setdefault(d["category"], []).append(d)

    items = []
    rng = random.Random(42)
    target_per_cat = {
        "aquatic": 4, "breakfast": 4, "condiment": 2, "dessert": 3,
        "drink": 3, "meat_dish": 6, "semi-finished": 1, "soup": 3,
        "staple": 4, "vegetable_dish": 4,
    }
    for cat, need in target_per_cat.items():
        candidates = by_cat.get(cat, [])
        if not candidates:
            continue
        picks = rng.sample(candidates, min(need, len(candidates)))
        for d in picks:
            tmpl = rng.choice(templates)
            items.append({
                "query": tmpl.format(d["title"]),
                "category": "entity_query",
                "difficulty": "easy",
                "expected_doc_paths": [d["path"]],
                "reference_answer": f"{d['title']}属于{CATEGORY_CN.get(cat, cat)}类菜品。详见菜谱步骤。",
                "should_answer": True,
                "tags": [cat, "entity"],
            })
    return items


def gen_ingredient_queries(dishes: List[Dict]) -> List[Dict]:
    """按食材问：'X可以做什么菜？'"""
    templates = [
        "{}可以做什么菜？",
        "有没有用{}做的菜谱？",
        "{}有哪些好吃的做法？",
        "推荐几个{}的菜",
    ]
    # 按食材聚合
    by_ing: Dict[str, List[Dict]] = {}
    for d in dishes:
        if d["ingredient"]:
            by_ing.setdefault(d["ingredient"], []).append(d)

    items = []
    rng = random.Random(43)
    # 每个食材至少 2 个菜才有意义
    eligible = [(ing, dl) for ing, dl in by_ing.items() if len(dl) >= 2]
    eligible.sort(key=lambda x: -len(x[1]))  # 多的优先

    for ing, dl in eligible[:20]:
        tmpl = rng.choice(templates)
        # 取最相关的 5 个菜作为 expected
        sample = rng.sample(dl, min(5, len(dl)))
        items.append({
            "query": tmpl.format(ing),
            "category": "ingredient_query",
            "difficulty": "medium",
            "expected_doc_paths": [d["path"] for d in sample],
            "reference_answer": f"用{ing}可以做：{','.join(d['title'] for d in sample[:3])}等。",
            "should_answer": True,
            "tags": ["ingredient", ing],
        })
    return items


def gen_category_queries(dishes: List[Dict]) -> List[Dict]:
    """按品类问：'有哪些早餐推荐？'"""
    cat_templates = {
        "aquatic": ["有哪些好吃的鱼虾菜？", "推荐几道海鲜菜", "海产品有什么做法？"],
        "breakfast": ["早餐吃什么好？", "有什么简单的早餐推荐？", "快手早餐有哪些？"],
        "dessert": ["有哪些好做的甜品？", "推荐几个甜点", "想吃甜品有什么推荐？"],
        "drink": ["有什么好喝的饮品？", "推荐几款饮品", "夏天喝什么好？"],
        "meat_dish": ["推荐几道下饭的肉菜", "有什么好吃的荤菜？", "肉菜有哪些经典做法？"],
        "soup": ["有哪些好喝的汤？", "推荐几款汤", "想喝汤有什么推荐？"],
        "staple": ["主食有什么推荐？", "有哪些好吃的面食？", "推荐几款米饭做法"],
        "vegetable_dish": ["素菜有哪些做法？", "推荐几道家常素菜", "蔬菜有什么好做法？"],
        "condiment": ["有哪些常用调味料的做法？", "推荐几款酱料"],
        "semi-finished": ["有哪些半成品的做法？"],
    }

    by_cat: Dict[str, List[Dict]] = {}
    for d in dishes:
        by_cat.setdefault(d["category"], []).append(d)

    items = []
    rng = random.Random(44)
    for cat, queries in cat_templates.items():
        candidates = by_cat.get(cat, [])
        if not candidates:
            continue
        # 每个品类生成 1 条 query
        q = rng.choice(queries)
        sample = rng.sample(candidates, min(8, len(candidates)))
        items.append({
            "query": q,
            "category": "category_query",
            "difficulty": "medium",
            "expected_doc_paths": [d["path"] for d in sample],
            "reference_answer": f"{CATEGORY_CN.get(cat, cat)}类菜包括: {','.join(d['title'] for d in sample[:5])}等。",
            "should_answer": True,
            "tags": ["category", cat],
        })
    return items


def gen_cooking_method_queries(dishes: List[Dict]) -> List[Dict]:
    """按烹饪方法问：'有哪些清蒸的菜？' """
    method_query = {
        "蒸": "有哪些清蒸的菜？",
        "红烧": "推荐几个红烧菜",
        "凉拌": "凉拌菜有哪些做法？",
        "糖醋": "糖醋菜有什么推荐？",
        "炸": "炸菜有哪些做法？",
        "烤": "烤的菜有哪些？",
        "炒": "炒菜有哪些经典做法？",
        "焖": "有什么焖菜推荐？",
        "炖": "推荐几道炖菜",
        "煎": "煎菜有哪些做法？",
    }
    by_method: Dict[str, List[Dict]] = {}
    for d in dishes:
        if d["method"]:
            by_method.setdefault(d["method"], []).append(d)

    items = []
    rng = random.Random(45)
    for method, q in method_query.items():
        candidates = by_method.get(method, [])
        if len(candidates) < 2:
            continue
        sample = rng.sample(candidates, min(5, len(candidates)))
        items.append({
            "query": q,
            "category": "cooking_method_query",
            "difficulty": "medium",
            "expected_doc_paths": [d["path"] for d in sample],
            "reference_answer": f"{method}类菜包括: {','.join(d['title'] for d in sample[:3])}等。",
            "should_answer": True,
            "tags": ["cooking_method", method],
        })
    return items


def gen_entity_paraphrase(dishes: List[Dict]) -> List[Dict]:
    """改写型 query：换说法但语义相同，测检索鲁棒性"""
    rng = random.Random(46)
    paraphrase_templates = [
        ("蒸鲈鱼怎么做", "清蒸鲈鱼"),
        ("怎么做凉拌黄瓜", "凉拌黄瓜"),
        ("番茄炒蛋的做法", "西红柿炒鸡蛋"),
        ("如何做糖醋排骨", "糖醋排骨"),
        ("家常红烧肉", "红烧肉"),
        ("酸辣土豆丝怎么炒", "酸辣土豆丝"),
        ("可乐鸡翅怎么做", "可乐鸡翅"),
        ("蛋炒饭做法", "蛋炒饭"),
    ]
    title_to_path = {d["title"]: d["path"] for d in dishes}

    items = []
    for query, dish_title in paraphrase_templates:
        # 找匹配的菜
        matched = [d for d in dishes if dish_title in d["title"] or d["title"] in dish_title]
        if not matched:
            continue
        items.append({
            "query": query,
            "category": "robustness_case",
            "difficulty": "medium",
            "expected_doc_paths": [d["path"] for d in matched[:2]],
            "reference_answer": f"{dish_title}的做法详见菜谱。",
            "should_answer": True,
            "tags": ["paraphrase", "robustness"],
        })
    return items


# ============= 主流程 =============


def scan_dishes() -> List[Dict]:
    """扫描所有 .md 菜谱文件，提取元信息"""
    dishes = []
    for md_path in DISHES_DIR.rglob("*.md"):
        title = get_dish_title(md_path)
        category = get_category(md_path)
        dishes.append({
            "title": title,
            "path": get_relative_path(md_path),
            "category": category,
            "method": detect_cooking_method(title),
            "ingredient": detect_ingredient(title),
        })
    return dishes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(ROOT / "evaluation" / "_auto_dataset.json"))
    args = parser.parse_args()

    print(f"扫描 {DISHES_DIR}...")
    dishes = scan_dishes()
    print(f"  找到 {len(dishes)} 个菜谱")

    # 按品类统计
    by_cat: Dict[str, int] = {}
    for d in dishes:
        by_cat[d["category"]] = by_cat.get(d["category"], 0) + 1
    print("  品类分布:")
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {n}")

    # 生成各类 query
    all_items = []
    all_items.extend(gen_entity_queries(dishes))
    all_items.extend(gen_ingredient_queries(dishes))
    all_items.extend(gen_category_queries(dishes))
    all_items.extend(gen_cooking_method_queries(dishes))
    all_items.extend(gen_entity_paraphrase(dishes))

    # 加 ID（自动样本用 A 开头）
    for i, item in enumerate(all_items):
        item["id"] = f"A{i+1:03d}"

    print(f"\n生成 {len(all_items)} 条自动样本:")
    cat_count: Dict[str, int] = {}
    for it in all_items:
        cat_count[it["category"]] = cat_count.get(it["category"], 0) + 1
    for c, n in sorted(cat_count.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")

    # 写文件（保证 id 在最前面，便于阅读）
    ordered = []
    for it in all_items:
        new = {"id": it["id"]}
        new.update({k: v for k, v in it.items() if k != "id"})
        ordered.append(new)

    Path(args.out).write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
