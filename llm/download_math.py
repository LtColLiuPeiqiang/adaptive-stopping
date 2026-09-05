#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 MATH 数据集 (EleutherAI/hendrycks_math, hf-mirror), 存 JSONL, 随机抽 10 条。
用法: HF_ENDPOINT=https://hf-mirror.com python3 download_math.py [--num 10] [--seed 42]
"""
import argparse
import json
import os
import random

SUBJECTS = ["algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra",
            "precalculus"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.expanduser(
        "~/llm-consensus-pruning/data/math_test.jsonl"))
    ap.add_argument("--sample-out", default=os.path.expanduser(
        "~/llm-consensus-pruning/data/math_sample10.json"))
    args = ap.parse_args()

    from datasets import load_dataset

    all_rows = []
    for subj in SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split="test")
        print(f"[{subj}] test={len(ds)} rows", flush=True)
        for row in ds:
            rec = {"subject": subj}
            rec.update({k: row[k] for k in row.keys()})
            all_rows.append(rec)
    print(f"TOTAL test: {len(all_rows)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in all_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"saved -> {args.out}")

    random.seed(args.seed)
    sample = random.sample(all_rows, min(args.num, len(all_rows)))
    with open(args.sample_out, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"sample -> {args.sample_out}")

    print("\n===== 字段示例 (第1条) =====")
    print(json.dumps(all_rows[0], ensure_ascii=False, indent=2)[:800])


if __name__ == "__main__":
    main()
