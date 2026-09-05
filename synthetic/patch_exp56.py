#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补跑 exp5/exp6 的乐观(推荐)配置, 合并进 results.json, 并重绘对应图。
用法: 在 python3 sim.py 主跑完成后执行 python3 patch_exp56.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim as S
from alg import Params

HERE = os.path.dirname(os.path.abspath(__file__))


def patch():
    rj = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))

    # ---- 重跑 exp5 (early vs full), 乐观配置 ----
    print("== patch exp5 (pessimistic=False) ==")
    out5 = {}
    for d in [0.05, 0.1, 0.2, 0.4, 0.8]:
        p = S.delta_family(d)
        row = {"bag": f"d={d}"}
        for mode in ("early", "full"):
            pr = Params(E=200, p_star=0.05, Delta_high=0.1, delta=0.05,
                        T_cap=150, min_t=8, pessimistic=False, explore_mode=mode,
                        width_mode="practical", z=2.0)
            s = S.summarize(S.simulate(p, pr, n_seeds=1200, seed0=6))
            row[mode] = dict(err_total=s["err_total"], err_ans=s["err_answered"],
                             avg_n=s["avg_n"], ans_rate=s["ans_rate"], branch=s["branch"])
        out5[row["bag"]] = row
        print(" ", row)
    rj["exp5"] = out5
    S.RES["exp5"] = out5
    names = list(out5.keys())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(len(names)), [out5[n]["early"]["err_total"] for n in names], "o-", label="early err_total")
    ax[0].plot(range(len(names)), [out5[n]["full"]["err_total"] for n in names], "s--", label="full err_total")
    ax[0].plot(range(len(names)), [out5[n]["early"]["err_ans"] for n in names], "o:", alpha=.8, label="early err_ans")
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=15, fontsize=8)
    ax[0].set_title("错误率: 立即切换 vs 跑满探索 (乐观)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(range(len(names)), [out5[n]["early"]["avg_n"] for n in names], "o-", label="early avg_n")
    ax[1].plot(range(len(names)), [out5[n]["full"]["avg_n"] for n in names], "s--", label="full avg_n")
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=15, fontsize=8)
    ax[1].set_title("期望采样: early vs full (乐观)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    S.savefig(fig, "exp5_explore_mode.png")

    # ---- 重跑 exp6 (K x E 规模), 乐观配置 ----
    print("== patch exp6 (pessimistic=False) ==")
    out6 = []
    for K in [3, 10, 30]:
        for E in [50, 200, 1000]:
            for d in [0.1, 0.4]:
                p = S.delta_family(d, K=K)
                pr = Params(E=E, p_star=0.05, Delta_high=0.1, delta=0.05,
                            T_cap=min(50, E // 2), min_t=max(3, min(8, E // 4)),
                            pessimistic=False, explore_mode="early",
                            width_mode="practical", z=2.0)
                s = S.summarize(S.simulate(p, pr, n_seeds=800, seed0=7))
                o = S.oracle_plan(p, pr)
                Nmin = max(1, int(np.ceil(2 * np.log((K - 1) / pr.p_star) / max(d * d, 1e-9))))
                row = dict(K=K, E=E, delta=d,
                           adapt=dict(err_total=s["err_total"], err_ans=s["err_answered"],
                                      avg_n=s["avg_n"], ans_rate=s["ans_rate"],
                                      branch=s["branch"]),
                           oracle=dict(branch=o["branch"], cost=o["cost"], p_err=o["p_err"]),
                           hoeffding_Nmin=Nmin,
                           ratio=round(s["avg_n"] / max(o["cost"], 1e-9), 3) if o["cost"] > 0.05 else None)
                out6.append(row)
                print(" ", row)
    rj["exp6"] = out6
    S.RES["exp6"] = out6

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(rj, f, ensure_ascii=False, indent=1, default=str)
    print("merged -> results.json")


if __name__ == "__main__":
    patch()