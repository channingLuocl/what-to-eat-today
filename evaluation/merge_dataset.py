"""
合并 _auto_dataset.json + _manual_edge_cases.json -> eval_dataset.json

用法:
    python evaluation/merge_dataset.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
EVAL_DIR = ROOT / "evaluation"

AUTO = EVAL_DIR / "_auto_dataset.json"
MANUAL = EVAL_DIR / "_manual_edge_cases.json"
OUT = EVAL_DIR / "eval_dataset.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    auto = load(AUTO) if AUTO.exists() else []
    manual = load(MANUAL) if MANUAL.exists() else []

    print(f"  auto:   {len(auto)} 条")
    print(f"  manual: {len(manual)} 条")

    combined = []
    seen_ids = set()

    for item in manual + auto:
        if item.get("id") in seen_ids:
            print(f"  ⚠️ 重复 id: {item['id']}, 跳过")
            continue
        seen_ids.add(item.get("id"))
        combined.append(item)

    # 类型分布
    by_cat = {}
    by_diff = {}
    answered = sum(1 for x in combined if x.get("should_answer", True))
    rejected = len(combined) - answered

    for it in combined:
        cat = it.get("category", "?")
        diff = it.get("difficulty", "?")
        by_cat[cat] = by_cat.get(cat, 0) + 1
        by_diff[diff] = by_diff.get(diff, 0) + 1

    print(f"\n总计 {len(combined)} 条 (answered={answered}, rejected={rejected})\n")
    print("category 分布:")
    for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {c:25s} {n}")
    print("\ndifficulty 分布:")
    for d, n in sorted(by_diff.items(), key=lambda x: -x[1]):
        print(f"  {d:10s} {n}")

    OUT.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
