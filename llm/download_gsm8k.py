#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 GSM8K 数据集 (openai/gsm8k, 免认证), 存 JSONL 供 make_subset.py 使用。

用法 (中国大陆网络建议走 hf-mirror):
    HF_ENDPOINT=https://hf-mirror.com python3 download_gsm8k.py

依赖: pip install datasets huggingface_hub  (或见 README 的 requirements)
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "gsm8k_test.jsonl"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    print(f"[gsm8k] {args.split}: {len(ds)} rows", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            rec = {"question": row["question"], "answer": row["answer"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()