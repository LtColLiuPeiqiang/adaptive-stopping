#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成摸球实验: 已知分布下测试自适应多数投票算法, 调参, 与 Oracle 比较。"""
import argparse, json, os, math, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from alg import Params, run_algorithm, make_picker, oracle_plan, run_choose_w, top2_stats, cs_width

import matplotlib as mpl
mpl.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans", "sans-serif"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 110

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
RES = {}

def savefig(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  fig ->", path)
    return path

# ---------------------------------------------------------------------------
# 分布族
# ---------------------------------------------------------------------------
def dists_meta():
    D, K = {}, 10
    D["uniform"] = np.ones(K) / K
    rng = np.random.default_rng(2026)
    for a in (0.2, 1.0, 10.0):
        D[f"dirichlet_a={a}"] = rng.dirichlet([a] * K)
    for p1 in (0.3, 0.5, 0.8, 0.95):
        rest = np.ones(K - 1) / K
        rest = rest / rest.sum() * (1 - p1)
        D[f"top{p1}_rest_uniform"] = np.concatenate([[p1], rest])
    r = np.ones(K - 2) / K
    r = r / r.sum() * 0.33
    D["two_close_0.34_0.33"] = np.concatenate([[0.34, 0.33], r])
    return D

def delta_family(dlt, K=10, rest_mass=0.2):
    """top 两色间隔 dlt, 其余 K-2 色均匀瓜分 rest_mass。"""
    u = rest_mass / max(1, K - 2)
    mid = (1 - rest_mass) / 2
    p1, p2 = mid + dlt / 2, mid - dlt / 2
    p = np.concatenate([[p1, p2], [u] * (K - 2)])
    p = np.clip(p, 0, None)
    return p / p.sum()

def rfract(x, n=5):
    return round(float(x), n)

# ---------------------------------------------------------------------------
# 模拟
# ---------------------------------------------------------------------------
def simulate(p, params, n_seeds=2000, seed0=0):
    p = np.asarray(p, float); p = p / p.sum()
    true_mode = int(np.argmax(p))
    rows = []
    for s in range(n_seeds):
        rng = np.random.default_rng(seed0 * 10_000_003 + s * 7919 + 13)
        res = run_algorithm(make_picker(p, rng), params, true_mode=true_mode)
        rows.append({"branch": res.branch, "n": res.n_samples, "correct": res.correct,
                     "w": res.w, "tau": res.tau, "nominal": res.p_err_nominal})
    return rows

def summarize(rows):
    n = len(rows)
    br = {b: 0 for b in ("fast", "naive", "reject")}
    for x in rows: br[x["branch"]] += 1
    ans = [x for x in rows if x["branch"] != "reject"]
    fast = [x for x in rows if x["branch"] == "fast"]
    return dict(
        branch={k: rfract(v / n, 6) for k, v in br.items()},
        err_total=rfract(1.0 - np.mean([x["correct"] for x in rows]), 6),
        ans_rate=rfract(len(ans) / n, 6),
        err_answered=rfract(1.0 - np.mean([x["correct"] for x in ans]), 6) if ans else 1.0,
        avg_n=rfract(np.mean([x["n"] for x in rows]), 4),
        med_n=rfract(np.median([x["n"] for x in rows]), 2),
        avg_tau=rfract(np.mean([x["tau"] for x in rows]), 4),
        avg_w=rfract(np.mean([x["w"] for x in rows if x["w"]]), 4) if fast else None,
        avg_fast_draws=rfract(np.mean([x["n"] - x["tau"] for x in fast]), 4) if fast else None,
    )

def simulate_fixed_n(p, N, n_seeds=2000, seed0=0):
    """固定 N 采样的多数投票基线。"""
    p = np.asarray(p, float); p = p / p.sum()
    true_mode = int(np.argmax(p)); err = 0; done = []
    for s in range(n_seeds):
        rng = np.random.default_rng(seed0 * 10_000_003 + s * 7919 + 17)
        cnt = np.zeros(len(p))
        for _ in range(N):
            cnt[int(rng.choice(len(p), p=p))] += 1
        err += (cnt.argmax() != true_mode)
        done.append(int(cnt.argmax()))
    return dict(err=rfract(err / n_seeds, 6), cost=N)

# ---------------------------------------------------------------------------
# 实验1: Δ 扫描 (误差与开销 vs Δ), 含 oracle 与固定N基线和理论/经验期望对照
# ---------------------------------------------------------------------------
def exp1_delta_sweep(params, n_seeds, quick=False):
    print("== exp1: Delta sweep ==")
    dlt_list = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8] if not quick else [0.1, 0.4]
    out = {"deltas": dlt_list, "adaptive": [], "oracle": [], "fixed": [], "theory_vs_emp": []}
    for d in dlt_list:
        p = delta_family(d)
        rows = simulate(p, params, n_seeds=n_seeds, seed0=1)
        s = summarize(rows); s["delta"] = d
        out["adaptive"].append(s)
        o = oracle_plan(p, params); o["delta"] = d
        out["oracle"].append(o)
        fx = simulate_fixed_n(p, params.E, n_seeds=min(n_seeds, 1500), seed0=2); fx["delta"] = d
        out["fixed"].append(fx)
        # 理论 vs 经验 (仅 fast 行的快分支采样数)
        theo = np.mean([q.get("E_fast", np.nan) for q in [oracle_plan(p, params)]])
        out["theory_vs_emp"].append({"delta": d, "oracle_pred_fast_cost": theo})
        print(f"  Delta={d}: adapt err_tot={s['err_total']} err_ans={s['err_answered']} avg_n={s['avg_n']} "
              f"branch={s['branch']} | oracle={o['branch']} cost={o['cost']} p_err={o['p_err']} | fixed err={fx['err']}")
    # --- 图: 误差 vs Δ ---
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ds = out["deltas"]
    ax[0].axhline(params.p_star, ls="--", c="gray", lw=1, label=f"目标 $p^*={params.p_star}$")
    ax[0].plot(ds, [a["err_total"] for a in out["adaptive"]], "o-", c="tab:blue", label="自适应(含拒答计非答)")
    ax[0].plot(ds, [a["err_answered"] for a in out["adaptive"]], "s--", c="tab:blue", alpha=.6, label="自适应(仅已作答)")
    ax[0].plot(ds, [o["p_err"] for o in out["oracle"]], "d-", c="tab:green", label=f"Oracle 名义错误(DP精确)")
    ax[0].plot(ds, [f["err"] for f in out["fixed"]], "x-", c="tab:red", label=f"固定 {params.E} 次众数")
    ax[0].set_xscale("log"); ax[0].set_xlabel("真实 top-2 概率差 Δ"); ax[0].set_ylabel("错误率")
    ax[0].set_title("错误率 vs Δ (K=10)")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].axhline(params.E, ls=":", c="gray", lw=1)
    ax[1].plot(ds, [a["avg_n"] for a in out["adaptive"]], "o-", c="tab:blue", label="自适应期望采样")
    ax[1].plot(ds, [o["cost"] for o in out["oracle"]], "d-", c="tab:green", label="Oracle 成本")
    ax[1].plot(ds, [params.E] * len(ds), "x-", c="tab:red", label=f"固定 $E^*$={params.E}")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Δ"); ax[1].set_ylabel("期望采样次数")
    ax[1].set_title("期望采样 vs Δ"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    RES["exp1_fig"] = savefig(fig, "exp1_delta_sweep.png")
    return out

# ---------------------------------------------------------------------------
# 实验2: 分支构成 (借用 exp1 rows)
# ---------------------------------------------------------------------------
def exp2_branch(exp1_out, params, n_seeds):
    print("== exp2: branch composition ==")
    ds = exp1_out["deltas"]
    fast = [exp1_out["adaptive"][i]["branch"]["fast"] for i in range(len(ds))]
    naive = [exp1_out["adaptive"][i]["branch"]["naive"] for i in range(len(ds))]
    rej = [exp1_out["adaptive"][i]["branch"]["reject"] for i in range(len(ds))]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(ds))
    ax.bar(x, fast, label="快分支", color="tab:blue")
    ax.bar(x, naive, bottom=fast, label="朴素分支", color="tab:orange")
    ax.bar(x, rej, bottom=[f + n for f, n in zip(fast, naive)], label="拒绝(预算内不可行)", color="tab:red")
    ax.set_xticks(x); ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlabel("真实 Δ"); ax.set_ylabel("种子占比"); ax.set_title("自适应算法分支构成 vs Δ")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    RES["exp2_fig"] = savefig(fig, "exp2_branch.png")

# ---------------------------------------------------------------------------
# 实验3: 参数调优热图
# ---------------------------------------------------------------------------
def exp3_tuning(params, n_seeds, quick=False):
    print("== exp3: tuning ==")
    bags = {"easy(Δ=0.4)": delta_family(0.4), "hard(Δ=0.05)": delta_family(0.05)}
    # (a) Delta_high x z (practical 模式); 指标: 条件错误率 err_answered + 采样
    Dhs = [0.05, 0.1, 0.2, 0.4]; zs = [1.5, 2.0, 2.5, 3.0]
    for bname, p in bags.items():
        M_err = np.zeros((len(Dhs), len(zs))); M_n = np.zeros_like(M_err); M_ans = np.zeros_like(M_err)
        for i, dh in enumerate(Dhs):
            for j, zv in enumerate(zs):
                pr = Params(E=params.E, p_star=params.p_star, Delta_high=dh, delta=0.05,
                            T_cap=params.T_cap, min_t=params.min_t, pessimistic=False,
                            explore_mode="early", width_mode="practical", z=zv)
                s = summarize(simulate(p, pr, n_seeds=n_seeds, seed0=3))
                M_err[i, j] = s["err_answered"]; M_n[i, j] = s["avg_n"]; M_ans[i, j] = s["ans_rate"]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for k, (M, ttl, cmap) in enumerate([(M_err, "条件错误率(仅作答)", "Reds"),
                                            (M_n, "期望采样次数", "viridis"),
                                            (M_ans, "回答率", "Blues")]):
            im = ax[k].imshow(M, aspect="auto", cmap=cmap, vmin=0)
            ax[k].set_xticks(range(len(zs))); ax[k].set_xticklabels(zs)
            ax[k].set_yticks(range(len(Dhs))); ax[k].set_yticklabels(Dhs)
            ax[k].set_xlabel("z (实用宽度倍数)"); ax[k].set_ylabel("Δ_high"); ax[k].set_title(f"{bname}  {ttl}")
            for i in range(len(Dhs)):
                for j in range(len(zs)):
                    ax[k].text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax[k], fraction=.046)
        RES[f"exp3_heat_Dh_z_{bname}"] = savefig(fig, f"exp3_heatmap_Dh_z_{bname.split('(')[0]}.png")
    # (b) T_cap x min_t
    if not quick:
        p = bags["easy(Δ=0.4)"]
        Tc = [10, 25, 50, 100]; mt = [3, 8, 15]
        M_err = np.zeros((len(Tc), len(mt))); M_n = np.zeros_like(M_err)
        for i, tc in enumerate(Tc):
            for j, m in enumerate(mt):
                pr = Params(E=params.E, p_star=params.p_star, Delta_high=params.Delta_high,
                            delta=0.05, T_cap=tc, min_t=m, pessimistic=True, explore_mode="early",
                            width_mode="practical", z=params.z)
                s = summarize(simulate(p, pr, n_seeds=n_seeds, seed0=4))
                M_err[i, j] = s["err_total"]; M_n[i, j] = s["avg_n"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, (M, ttl, cmap) in enumerate([(M_err, "总错误率", "Reds"), (M_n, "期望采样次数", "viridis")]):
            im = ax[k].imshow(M, aspect="auto", cmap=cmap)
            ax[k].set_xticks(range(len(mt))); ax[k].set_xticklabels(mt)
            ax[k].set_yticks(range(len(Tc))); ax[k].set_yticklabels(Tc)
            ax[k].set_xlabel("min_t"); ax[k].set_ylabel("T_cap"); ax[k].set_title(f"(Δ=0.4) {ttl}")
            for i in range(len(Tc)):
                for j in range(len(mt)):
                    ax[k].text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax[k], fraction=.046)
        RES["exp3_fig_Tcap_mint"] = savefig(fig, "exp3_heatmap_Tcap_mint.png")

# ---------------------------------------------------------------------------
# 实验4: 乐观 vs 悲观 Choose_w
# ---------------------------------------------------------------------------
def exp4_opt_vs_pess(params, n_seeds, quick=False):
    print("== exp4: optimistic vs pessimistic Choose_w ==")
    bags = {}
    for d in ([0.02, 0.05, 0.1, 0.2, 0.4, 0.8] if not quick else [0.05, 0.4]):
        bags[f"d={d}"] = delta_family(d)
    for bname, p in bags.items():
        d = float(np.sort(p)[::-1][0] - np.sort(p)[::-1][1])
        row = {"bag": bname, "true_delta": rfract(d, 4)}
        for po in (True, False):
            pr = Params(E=params.E, p_star=params.p_star, Delta_high=0.1, delta=0.05,
                        T_cap=params.T_cap, min_t=params.min_t, pessimistic=po, explore_mode="early",
                        width_mode=params.width_mode, z=params.z)
            s = summarize(simulate(p, pr, n_seeds=n_seeds, seed0=5))
            row["pess" if po else "opt"] = dict(err_total=s["err_total"], err_ans=s["err_answered"],
                                                avg_n=s["avg_n"], ans_rate=s["ans_rate"], avg_w=s["avg_w"])
        print(" ", row)
        RES["exp4_" + bname] = row

# ---------------------------------------------------------------------------
# 实验5: early 触发 vs full 探索 (TODO: 立即切 vs 跑满再估)
# ---------------------------------------------------------------------------
def exp5_explore_mode(params, n_seeds, quick=False):
    print("== exp5: explore mode early vs full ==")
    bags = {}
    for d in ([0.05, 0.1, 0.2, 0.4, 0.8] if not quick else [0.2]):
        bags[f"d={d}"] = delta_family(d)
    out = {}
    for bname, p in bags.items():
        row = {"bag": bname}
        for mode in ("early", "full"):
            pr = Params(E=params.E, p_star=params.p_star, Delta_high=0.1, delta=0.05,
                        T_cap=params.T_cap, min_t=params.min_t, pessimistic=False, explore_mode=mode,
                        width_mode=params.width_mode, z=params.z)
            s = summarize(simulate(p, pr, n_seeds=n_seeds, seed0=6))
            row[mode] = dict(err_total=s["err_total"], avg_n=s["avg_n"], branch=s["branch"])
        out[bname] = row
        print(" ", row)
    RES["exp5"] = out
    # 图
    names = list(out.keys()); modes = ("early", "full")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(len(names)), [out[n]["early"]["err_total"] for n in names], "o-", label="early: 置信达标即切")
    ax[0].plot(range(len(names)), [out[n]["full"]["err_total"] for n in names], "s--", label="full: 跑满 T_cap 再定")
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=15, fontsize=8)
    ax[0].set_title("错误率: 立即切换 vs 跑满探索"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(range(len(names)), [out[n]["early"]["avg_n"] for n in names], "o-", label="early")
    ax[1].plot(range(len(names)), [out[n]["full"]["avg_n"] for n in names], "s--", label="full")
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=15, fontsize=8)
    ax[1].set_title("期望采样: immediate vs full"); ax[1].legend(); ax[1].grid(alpha=.3)
    RES["exp5_fig"] = savefig(fig, "exp5_explore_mode.png")

# ---------------------------------------------------------------------------
# 实验6: K 与 E 的规模效应
# ---------------------------------------------------------------------------
def exp6_scaling(n_seeds, quick=False):
    print("== exp6: scaling K x E ==")
    out = []
    for K in ([3, 10, 30] if not quick else [10]):
        for E in ([50, 200, 1000] if not quick else [200]):
            for d in ([0.1, 0.4] if not quick else [0.4]):
                p = delta_family(d, K=K)
                pr = Params(E=E, p_star=0.05, Delta_high=0.1, delta=0.05,
                            T_cap=min(50, E // 2), min_t=max(3, min(8, E // 4)),
                            pessimistic=False, explore_mode="early",
                            width_mode="practical", z=2.0)
                s = summarize(simulate(p, pr, n_seeds=min(n_seeds, 800), seed0=7))
                o = oracle_plan(p, pr)
                # 理论最少采样(朴素分支可行性下界): min N s.t. (K-1)e^{-N d^2/2} <= p*
                Nmin = max(1, math.ceil(2 * math.log((K - 1) / pr.p_star) / max(d * d, 1e-9)))
                row = dict(K=K, E=E, delta=d, adapt=dict(err_total=s["err_total"], avg_n=s["avg_n"]),
                           oracle=dict(branch=o["branch"], cost=o["cost"]),
                           hoeffding_Nmin=Nmin,
                           ratio=rfract(s["avg_n"] / max(o["cost"], 1e-9), 3))
                out.append(row); print(" ", row)
    RES["exp6"] = out

# ---------------------------------------------------------------------------
# 实验7: certified 模式的预算充分性 (诚实拒绝 vs 回答率)
# ---------------------------------------------------------------------------
def exp7_certified_adequacy(n_seeds, quick=False):
    print("== exp7: certified-mode budget adequacy ==")
    dlt_list = [0.05, 0.1, 0.2, 0.4, 0.8] if not quick else [0.2, 0.4]
    E, Tcap, min_t = 1000, 250, 40
    out = []
    for d in dlt_list:
        p = delta_family(d)
        pr = Params(E=E, p_star=0.05, Delta_high=0.1, delta=0.05, T_cap=Tcap, min_t=min_t,
                    pessimistic=False, explore_mode="early", width_mode="certified")
        s = summarize(simulate(p, pr, n_seeds=n_seeds, seed0=8))
        ps = np.sort(p)[::-1]
        sigma2 = ps[0] * (1 - ps[0]) + ps[1] * (1 - ps[1]) + 2 * ps[0] * ps[1]
        Tcert = None
        for t in range(1, 300001):
            if (d - 0.1) >= cs_width(sigma2, t, 0.05):
                Tcert = t
                break
        out.append(dict(delta=d,
                        adapt=dict(ans_rate=s["ans_rate"], err_ans=s["err_answered"],
                                   avg_n=s["avg_n"], branch=s["branch"]),
                        Tcert=Tcert))
        print(" ", out[-1])
    RES["exp7"] = out
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ds = [o["delta"] for o in out]
    ax[0].plot(ds, [o["adapt"]["ans_rate"] for o in out], "o-", label="回答率 ans_rate")
    ax[0].plot(ds, [o["adapt"]["err_ans"] for o in out], "s--", label="条件错误率 err_answered")
    ax[0].set_xscale("log"); ax[0].set_xlabel("真实 Δ")
    ax[0].set_title("certified 模式: E*=1000, T_cap=250, Δ_high=0.1")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].loglog(ds, [o["Tcert"] for o in out], "d-", label="理论认证时间 T_cert(Δ)")
    ax[1].set_xlabel("真实 Δ"); ax[1].set_ylabel("t")
    ax[1].set_title("认证 Δ≥Δ_high=0.1 所需样本数 (δ=0.05)")
    ax[1].legend(); ax[1].grid(alpha=.3, which="both")
    RES["exp7_fig"] = savefig(fig, "exp7_certified_adequacy.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    params = Params(E=200, p_star=0.05, Delta_high=0.1, delta=0.05, T_cap=150, min_t=8,
                    pessimistic=False, explore_mode="early", width_mode="practical", z=2.0)
    e1 = exp1_delta_sweep(params, args.seeds, quick=args.quick)
    exp2_branch(e1, params, args.seeds)
    exp3_tuning(params, min(args.seeds, 600), quick=args.quick)
    exp4_opt_vs_pess(params, min(args.seeds, 1200), quick=args.quick)
    exp5_explore_mode(params, min(args.seeds, 1200), quick=args.quick)
    exp6_scaling(min(args.seeds, 800), quick=args.quick)
    exp7_certified_adequacy(min(args.seeds, 1000), quick=args.quick)
    RES["params_main"] = dict(E=params.E, p_star=params.p_star, Delta_high=params.Delta_high,
                              delta=params.delta, T_cap=params.T_cap, min_t=params.min_t,
                              width_mode=params.width_mode, z=params.z)
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RES, f, ensure_ascii=False, indent=1, default=str)
    print(f"done in {time.time()-t0:.1f}s -> results.json")

if __name__ == "__main__":
    main()