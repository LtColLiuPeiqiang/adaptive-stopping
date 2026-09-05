#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构造 llm_vote.py 所需的 subset.jsonl (题目子集 + gold 标准答案)。

输入 (只需存在其一):
  data/gsm8k_test.jsonl   行 = {"question": ..., "answer": ...}        (download_gsm8k.py 产出)
  data/math_test.jsonl    行 = {"subject": ..., "problem": ..., "solution": ...}  (download_math.py 产出)

gold 提取规则 (与论文一致):
  GSM8K: answer 字段最后一个数字
  MATH : solution 字段中最后一个 \\boxed{...} 的内容; 无 boxed 时取最后一个数字

用法:
    python3 make_subset.py                        # 默认各取 80 题 (论文配置), 写 data/subset.jsonl
    python3 make_subset.py --limit 160 --seed 42  # 自定义
"""
import argparse
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def last_number(s):
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else None


def gold_from_answer(answer):
    v = last_number(answer)
    return v if v is not None else answer.strip()


def gold_from_solution(solution):
    boxes = re.findall(r"\\boxed\{(.*?)\}", solution, re.S)
    if boxes:
        v = last_number(boxes[-1])
        return v if v is not None else boxes[-1].strip()
    v = last_number(solution)
    return v if v is not None else solution.strip()


def load_lines(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k", default=os.path.join(DATA, "gsm8k_test.jsonl"))
    ap.add_argument("--math", default=os.path.join(DATA, "math_test.jsonl"))
    ap.add_argument("--limit", type=int, default=80, help="每个数据集抽取的题数 (0=全部)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(DATA, "subset.jsonl"))
    args = ap.parse_args()

    gsm8k = load_lines(args.gsm8k)
    math = load_lines(args.math)
    if gsm8k is None and math is None:
        print("!! 未找到任何输入数据文件。请先运行:")
        print("   HF_ENDPOINT=https://hf-mirror.com python3 download_gsm8k.py")
        print("   HF_ENDPOINT=https://hf-mirror.com python3 download_math.py")
        return 1

    rng = random.Random(args.seed)
    out = []
    if gsm8k is not None:
        rows = gsm8k if args.limit == 0 else rng.sample(gsm8k, min(args.limit, len(gsm8k)))
        for r in rows:
            out.append({"q": r["question"], "gold": gold_from_answer(r["answer"]), "src": "gsm8k"})
        print(f"gsm8k: {len(rows)} 题")
    if math is not None:
        rows = math if args.limit == 0 else rng.sample(math, min(args.limit, len(math)))
        for r in rows:
            out.append({"q": r["problem"], "gold": gold_from_solution(r["solution"]), "src": "math"})
        print(f"math : {len(rows)} 题")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"subset -> {args.out} ({len(out)} 题)")


if __name__ == "__main__":
    main()