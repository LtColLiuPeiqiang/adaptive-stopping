#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 上游多次采样投票实验
=========================
用 ollama 本地 qwen2.5 推理 GSM8K / MATH 数据集子集:
  1. 每道题预生成 E* 个 iid 解码样本 (temperature=0.7), 每个样本的解析答案视为一个"球"
  2. 自适应算法对样本流在线决策 (探索→快/朴素/拒绝), 统计实际消费样本数 n_used
  3. 固定 N 众数基线 (N=1,2,3,4,6,8) 复用同一预生成池做公平对照
  4. 判定: 预测=众数, 正确=与 gold 答案匹配; 另以 pool 众数(8采样)作为"模型自身众数"参考

统计等价性说明: 样本为 iid 解码, 停止规则为定义在已见前缀上的停时, 故离线重放与在线
实时停止在决策分布上等价; 报告中"有效计算量"按 n_used 统计 (即真实部署时可跳过的调用数)。
"""
import argparse, json, os, re, sys, time, random, math
import requests
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "synthetic"))
from alg import Params, top2_stats, alg_width, run_choose_w, mode_of

HERE = os.path.dirname(os.path.abspath(__file__))
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

PROMPT_G = ("Solve the following math word problem step by step. "
            "At the very end, on its own line, write exactly: The final answer is: <number>\n\n"
            "Question: {q}\n")
PROMPT_M = ("Solve the following math problem step by step. "
            "At the very end, on its own line, write exactly: The final answer is: <expr>\n"
            "where <expr> is a single simplified mathematical expression (a number, fraction, or expression).\n\n"
            "Problem: {q}\n")


def gen_response(question, src, temperature=0.7, max_tokens=1100, timeout=300, tries=3):
    prompt = PROMPT_G.format(q=question) if src == "gsm8k" else PROMPT_M.format(q=question)
    for a in range(tries):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": MODEL, "prompt": prompt, "stream": False,
                "options": {"temperature": temperature, "top_p": 0.9,
                            "num_predict": max_tokens}},
                timeout=timeout)
            r.raise_for_status()
            return r.json()["response"]
        except Exception as e:
            if a == tries - 1:
                raise
            time.sleep(3)


def parse_answer(text, src):
    # 1) 优先取 "final answer is:" 之后的整段
    seg = None
    m = re.search(r"final answer is\s*:\s*(.+)$", text, re.I | re.S)
    if m:
        seg = m.group(1).strip()
    if seg:
        b = re.search(r"\\boxed\{(.*?)\}", seg, re.S)
        cand = b.group(1) if b else seg.split("\n")[0].strip()
    else:
        b = re.findall(r"\\boxed\{(.*?)\}", text, re.S)
        cand = b[-1] if b else text.strip()
    cand = re.sub(r"[$,]", "", cand).strip()
    cand = cand.replace("\\[", "").replace("\\]", "").strip()
    if src == "gsm8k":
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", cand)
        return nums[-1] if nums else cand
    return cand


def norm_number(s):
    try:
        from fractions import Fraction
        s = s.strip()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
            return float(s)
        f = Fraction(s.split()[0])
        return float(f)
    except Exception:
        return None


def canon(a, src):
    na = norm_number(a)
    if na is not None:
        return ("num", round(na, 9))
    if src == "math":
        sa = re.sub(r"[(){}\s\\frac^$,]", "", a)
        return ("str", sa)
    return ("str", a.strip())


def answers_equal(a, b, src):
    return canon(a, src) == canon(b, src)


def mode_of_counts(counts):
    return max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# 自适应运行 (对预生成 pool 的流式重放)
# ---------------------------------------------------------------------------
def adaptive_on_pool(pool_keys, params, gold_key=None):
    counts = {}
    t = 0
    last, run = None, 0
    branch, w, tau, n_used = None, None, None, 0
    # ---- 探索阶段 ----
    while True:
        k = pool_keys[t]; t += 1
        counts[k] = counts.get(k, 0) + 1
        phat, dhat, sigma2, K = top2_stats(counts, t)
        if t >= params.min_t:
            wt = alg_width(params, sigma2, t)
            if dhat - wt >= params.Delta_high:
                tau, branch = t, "fast"
                break
        if t >= params.T_cap:
            tau, branch = t, "naive"
            break
    phat, dhat, sigma2, K = top2_stats(counts, tau)
    N_rem = params.E - tau
    if branch == "fast":
        # ---- 快分支: Choose_w + w连击停止 ----
        w, winfo = run_choose_w(phat, N_rem, params.p_star,
                                pessimistic=params.pessimistic, tau=tau,
                                delta_cs=params.delta)
        if w is not None:
            while t < params.E:
                k = pool_keys[t]; t += 1
                counts[k] = counts.get(k, 0) + 1
                if k == last:
                    run += 1
                else:
                    run, last = 1, k
                if run >= w:
                    break
            n_used = t
        else:
            branch = "fast_fallback_naive"
    if branch == "naive" or branch == "fast_fallback_naive":
        # ---- 朴素分支判据 / 拒绝 ----
        wtc = alg_width(params, sigma2, tau) if params.use_pessimistic_check else 0.0
        d_use = max(0.0, dhat - wtc)
        K_use = max(params.K_floor, K)
        crit = (K_use - 1) * math.exp(min(0.0, -N_rem * d_use * d_use / 2))
        if crit <= params.p_star:
            while t < params.E:
                k = pool_keys[t]; t += 1
                counts[k] = counts.get(k, 0) + 1
            n_used = t
        else:
            branch = "reject"
            n_used = tau
    if branch is None:
        branch = "naive"  # E==tau 理论情形
        n_used = t
    pred = mode_of_counts(counts) if counts else None
    return dict(branch=branch, w=w, tau=tau, n_used=n_used,
                pred=pred, correct=(pred == gold_key) if gold_key is not None else None)


def fixed_majority(pool_keys, N, gold_key=None):
    counts = {}
    for k in pool_keys[:N]:
        counts[k] = counts.get(k, 0) + 1
    pred = mode_of_counts(counts)
    return dict(pred=pred, correct=(pred == gold_key))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def load_items(subset_path):
    with open(subset_path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run(args):
    items = load_items(args.subset)
    if args.limit:
        rng = random.Random(777)
        items = rng.sample(items, min(args.limit, len(items)))
    params = Params(E=args.pool, p_star=args.p_star, Delta_high=args.delta_high,
                    delta=args.delta, T_cap=args.t_cap, min_t=args.min_t,
                    pessimistic=bool(args.pessimistic), explore_mode="early",
                    width_mode=args.width_mode, z=args.z)
    print(f"items={len(items)} params={params}")
    t0 = time.time()
    results = []
    err_calls = 0
    for idx, it in enumerate(items):
        q, src, gold = it["q"], it["src"], it["gold"]
        gold_key = canon(gold, src)
        pool = []
        for j in range(args.pool):
            txt = gen_response(q, src)
            pool.append(canon(parse_answer(txt, src), src))
        ad = adaptive_on_pool(pool, params, gold_key)
        fixed = {n: fixed_majority(pool, n, gold_key) for n in args.fixed_ns}
        rec = dict(q=q, src=src, gold=gold, branch=ad["branch"], w=ad["w"], tau=ad["tau"],
                   n_used=ad["n_used"], acc=ad["correct"],
                   fixed={str(n): f["correct"] for n, f in fixed.items()})
        results.append(rec)
        if (idx + 1) % 5 == 0 or idx == len(items) - 1:
            acc_a = np.mean([r["acc"] for r in results])
            print(f"[{time.time()-t0:.0f}s] {idx+1}/{len(items)} adapt_acc={acc_a:.3f} "
                  f"avg_n={np.mean([r['n_used'] for r in results]):.2f}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"params": dict(E=params.E, p_star=params.p_star, Delta_high=params.Delta_high,
                                  delta=params.delta, T_cap=params.T_cap, min_t=params.min_t,
                                  pessimistic=params.pessimistic, width_mode=params.width_mode,
                                  z=params.z, model=MODEL),
                   "results": results}, f, ensure_ascii=False, indent=1)
    print("saved", args.out)
    return results


def summarize(path):
    d = json.load(open(path, encoding="utf-8"))
    res = d["results"]
    n = len(res)
    out = {"n": n, "params": d["params"]}
    for src in ("gsm8k", "math"):
        rs = [r for r in res if r["src"] == src]
        if not rs:
            continue
        out[src] = {
            "acc_adaptive": round(float(np.mean([r["acc"] for r in rs])), 4),
            "avg_n_used": round(float(np.mean([r["n_used"] for r in rs])), 3),
            "med_n_used": int(np.median([r["n_used"] for r in rs])),
            "branch": {b: round(sum(1 for r in rs if r["branch"] == b) / len(rs), 3)
                       for b in ["fast", "naive", "reject", "fast_fallback_naive"]},
            "acc_fixed": {k: round(float(np.mean([r["fixed"][k] for r in rs])), 4)
                          for k in rs[0]["fixed"]},
        }
    out["all"] = {
        "acc_adaptive": round(float(np.mean([r["acc"] for r in res])), 4),
        "avg_n_used": round(float(np.mean([r["n_used"] for r in res])), 3),
        "pool_cost": d["params"]["E"],
        "saved_frac": round(1.0 - float(np.mean([r["n_used"] for r in res])) / d["params"]["E"], 3),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=os.path.join(HERE, "data", "subset.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "llm_results.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--p_star", type=float, default=0.02)
    ap.add_argument("--delta_high", type=float, default=0.3)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--t_cap", type=int, default=4)
    ap.add_argument("--min_t", type=int, default=2)
    ap.add_argument("--pessimistic", type=int, default=1)
    ap.add_argument("--width_mode", type=str, default="practical")
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--fixed_ns", default="1,2,3,4,6,8")
    ap.add_argument("--summarize", type=str, default="")
    args = ap.parse_args()
    args.fixed_ns = [int(x) for x in args.fixed_ns.split(",")]
    if args.summarize:
        summarize(args.summarize)
    else:
        run(args)