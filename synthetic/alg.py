# -*- coding: utf-8 -*-
"""
自适应多数投票算法 (球袋抽象) 核心实现
=======================================
把 LLM 多次采样多数投票抽象为: 颜色种类未知、颜色分布未知的袋子中有放回摸球。
算法在线估计 top-2 概率差 Delta = p_(1)-p_(2), 视判定结果进入:
  - 快分支  (fast): 连续 w 个同色球即停, w 由 Choose_w (DP) 按错误率预算挑选
  - 朴素分支 (naive): 跑满剩余预算, 众数作答; 若可行性判据不满足则诚实拒绝
  - 拒绝    (reject): 输出 "预算内不可行"

实现细节:
  * cs_width: 时间一致(time-uniform)置信半宽 w_t = sqrt(2 sigma2 (2 ln t + ln(pi^2/(3 delta)))/t)
      P(∃t: |Δ̂_t - Δ| ≥ w_t) ≤ δ  (经验Bernstein型; 常数见报告理论部分的推导)
  * sigma2 := Var(p̂_(1) - p̂_(2)) 的plug-in估计 = p̂1(1-p̂1) + p̂2(1-p̂2) + 2 p̂1 p̂2
  * Choose_w: 马尔可夫链DP 状态 (c, r, n): 当前连击颜色c、长度r、已采样n;
    w 从 3 起遍历, 计算"首次完成w连击的是非最大概率颜色的概率" ≤ p* 则返回。
    支持 plug-in(乐观) 与 置信集内最不利重分布(悲观, 含未见颜色质量合并成一个对抗色)
  * 悲观重分布: q1=max(0,p̂1-w_τ), 其余观察色 q_c=max(0,p̂_c-w_τ),
    剩余质量合并为单一"填充色" (非最大色, 计入错误), 保守覆盖真实分布。
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Dict, Any, List

import numpy as np

PI2 = math.pi ** 2


# ----------------------------------------------------------------------------
# 统计量与置信宽度
# ----------------------------------------------------------------------------
def top2_stats(counts: Dict[int, int], t: int) -> Tuple[np.ndarray, float, float, int]:
    """返回: 降序频率数组 phat, Δ̂=p̂1-p̂2, σ̂², 已观察到颜色数 K̂"""
    vals = np.array(sorted(counts.values(), reverse=True), dtype=float)
    phat = vals / t
    p1 = phat[0] if len(phat) >= 1 else 0.0
    p2 = phat[1] if len(phat) >= 2 else 0.0
    sigma2 = p1 * (1 - p1) + p2 * (1 - p2) + 2 * p1 * p2
    dhat = p1 - p2
    return phat, dhat, sigma2, len(phat)


def cs_width(sigma2: float, t: int, delta: float, c: float = 1.0) -> float:
    """时间一致置信半宽:
       w_t = sqrt( 2 σ̂² (2 ln t + ln(π²/(3δ)) + 2 ln c) / t )
    常数 c≥1 为理论留量(默认 1)。σ̂² = plug-in Var(p̂_(1)-p̂_(2))。"""
    t = max(int(t), 1)
    if sigma2 <= 0:
        return 0.0
    inner = 2 * math.log(t) + math.log(PI2 / (3 * delta)) + 2 * math.log(c)
    return math.sqrt(2 * sigma2 * inner / t)


def practical_width(sigma2: float, t: int, z: float = 2.0) -> float:
    """实用模式宽度: z 倍经验标准差 z·sqrt(σ̂²/t)。无严格时间一致保证,
    供小预算启发式使用(其保证见报告实验部分的经验验证)。"""
    t = max(int(t), 1)
    if sigma2 <= 0:
        return 0.0
    return z * math.sqrt(sigma2 / t)


# ----------------------------------------------------------------------------
# 连击停止过程的精确DP
# ----------------------------------------------------------------------------
def run_dp(phat: np.ndarray, w: int, N: int):
    """对给定分布 phat (K 个颜色) 与窗口 w、剩余预算 N, 精确计算:
         stop_mass[t] : 恰在第 t 次(相对起点)完成首次 w 连击并停止的概率, t=1..N
         stop_err[t]  : 该次停止发生在非最大概率颜色上的概率
         no_stop      : 预算跑满仍未触发连击停止的概率
    状态: (c, r, n) 上一次颜色c, 当前连击长度 r∈[1,w-1], 采样次数 n。
    DP列 j=0..w-2 对应运行长度 r=j+1。"""
    K = len(phat)
    phat = np.asarray(phat, dtype=float)
    N = int(N)
    if K == 0:
        return np.zeros(N + 1), np.zeros(N + 1), 1.0
    if w < 1:
        return np.zeros(N + 1), np.zeros(N + 1), 1.0
    if w == 1:
        # 1 连击即停: 首个球就停; 错误 = 非最大概率颜色被首采
        stop_mass = np.zeros(N + 1); stop_err = np.zeros(N + 1)
        if N >= 1:
            stop_mass[1] = 1.0
            stop_err[1] = max(0.0, 1.0 - phat[0])
        return stop_mass, stop_err, 0.0
    M = np.zeros((K, w - 1))
    M[:, 0] = phat
    stop_mass = np.zeros(N + 1)
    stop_err = np.zeros(N + 1)
    col_last = w - 2
    no_stop = 0.0
    for n in range(1, N + 1):
        if n == N:
            no_stop = float(M.sum())
            break
        # 1) 连击长度 w-1 的颜色再摸到同色 -> 停止
        stop_mass[n + 1] = float(M[:, col_last] @ phat)
        if K >= 2:
            stop_err[n + 1] = float(M[1:, col_last] @ phat[1:])
        elif col_last >= 0 and K == 1:
            stop_err[n + 1] = 0.0
        # 2) 同色延续: 列 j → j+1 (即 run length r → r+1)
        cont = np.zeros_like(M)
        if w >= 3:
            cont[:, 1:] = M[:, :-1] * phat[:, None]
        # 3) 换色 -> 新连击(列0): 颜色 d 获得 Σ_{c≠d,r} M[c,r]·p_d
        run_total = float(M.sum())
        col_sums = M.sum(axis=1)
        new_col0 = phat * (run_total - col_sums)
        M = cont
        M[:, 0] += new_col0
    return stop_mass, stop_err, no_stop


def _pessimistic_reweight(phat: np.ndarray, tau: int, delta_cs: float) -> np.ndarray:
    """置信集内最不利重分布 (保守): 每个观察色下调 w_τ, 剩余质量合并为单一对抗填充色。
    覆盖: 真实分布 p 满足 |p_c - p̂_c| ≤ w_τ 且未见颜色总质量 ≤ K̂ w_τ。"""
    phat = np.asarray(phat, dtype=float)
    p1 = phat[0] if len(phat) >= 1 else 0.0
    p2 = phat[1] if len(phat) >= 2 else 0.0
    sigma2 = p1 * (1 - p1) + p2 * (1 - p2) + 2 * p1 * p2
    wt = cs_width(sigma2, tau, delta_cs)
    q = np.maximum(0.0, phat - wt)
    filler = 1.0 - q.sum()
    if filler > 1e-12:
        q = np.append(q, filler)
    return q


def _hoeffding_majority_err(Delta: float, N: int, K: int) -> float:
    """众数错误上界 (Hoeffding): P(modê ≠ mode) ≤ min(1, (K-1) e^{-N Δ²/2})"""
    if K <= 1:
        return 0.0
    return min(1.0, (K - 1) * math.exp(-N * Delta * Delta / 2))


def run_choose_w(phat: np.ndarray, N_rem: int, p_star: float,
                 pessimistic: bool = False, tau: int = 1, delta_cs: float = 0.05,
                 w_min: int = 3, w_max: Optional[int] = None):
    """Choose_w: 返回最小 w 使得连击停止错误率 ≤ p*。
    返回 (w, info) 或 (None, info) 表示快分支在预算内不可行。
    info: p_err_run, p_err_norun, P_stop, E_fast_draws(期望快分支采样数),
          w, ptilde"""
    phat = np.asarray(phat, dtype=float)
    if N_rem < w_min:
        return None, dict(reason="budget too small")
    if w_max is None:
        w_max = min(N_rem, 80)
    ptilde = _pessimistic_reweight(phat, max(tau, 1), delta_cs) if pessimistic else phat
    K_eff = len(ptilde)
    # 悲观 Δ 与 K 用于无停止事件的众数错误
    if len(ptilde) >= 2:
        d_eff = max(0.0, ptilde[0] - ptilde[1])
    else:
        d_eff = ptilde[0]
    for w in range(w_min, w_max + 1):
        stop_mass, stop_err, no_stop = run_dp(ptilde, w, N_rem)
        P_stop = float(stop_mass.sum())
        p_err_run = float(stop_err.sum())
        # 未触发连击 -> 跑满预算取众数, 用 Hoeffding 界
        p_err_norun = no_stop * _hoeffding_majority_err(d_eff, N_rem, K_eff)
        p_err = min(1.0, p_err_run + p_err_norun)
        if p_err <= p_star:
            n_stop = np.arange(N_rem + 1, dtype=float)
            E_fast = float(n_stop @ stop_mass) + N_rem * (1 - P_stop)
            info = dict(w=w, p_err=p_err, p_err_run=p_err_run, p_err_norun=p_err_norun,
                        P_stop=P_stop, E_fast=E_fast, ptilde=ptilde.copy())
            return w, info
    return None, dict(reason="no w within cap", ptilde=ptilde.copy())


# ----------------------------------------------------------------------------
# 主算法
# ----------------------------------------------------------------------------
@dataclass
class Params:
    E: int = 200                 # 最大采样次数预算 E*
    p_star: float = 0.05         # 允许最大错误率 p*
    Delta_high: float = 0.1      # 快分支判定阈值 Δ_high
    delta: float = 0.05          # 误触发/剪枝失败概率 δ
    T_cap: int = 50              # 探索上限 T_cap
    min_t: int = 5               # 允许触发快分支的最小探索次数(防退化plug-in)
    pessimistic: bool = True     # Choose_w 是否用最不利重分布(悲观)
    explore_mode: str = "early"  # 'early': 置信区间满足立即进快分支; 'full': 跑满T_cap再定
    K_floor: int = 2             # (K-1) 中 K 的下界, 防御 K̂=1
    use_pessimistic_check: bool = True  # 朴素分支可行性判据用 Δ̂-宽度 (诚实) 还是 Δ̂
    width_mode: str = "certified"  # 'certified' 时间一致CS; 'practical' z·sd 启发式
    z: float = 2.0               # practical 模式的宽度倍数


def alg_width(params: "Params", sigma2: float, t: int) -> float:
    """按模式取宽度。"""
    if params.width_mode == "practical":
        return practical_width(sigma2, t, params.z)
    return cs_width(sigma2, t, params.delta)


@dataclass
class Result:
    branch: str = ""             # fast / naive / reject
    n_samples: int = 0
    tau: int = 0
    correct: bool = False
    w: Optional[int] = None
    trigger_t: Optional[int] = None
    p_err_nominal: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)
    seq: List[int] = field(default_factory=list)


def mode_of(counts: Dict[int, int]) -> int:
    return max(counts, key=counts.get)


def run_algorithm(sample_fn: Callable[[int], int], params: Params,
                  true_mode: Optional[int] = None) -> Result:
    """sample_fn(t): 第 t 次摸球返回颜色(callable 或者支持 next 的生成器)。
    返回 Result。"""
    if not hasattr(sample_fn, "__next__"):
        class _G:
            def __init__(self, f, p): self.f = f; self.p = p; self.i = 0
            def __next__(self): v = self.f(self.i); self.i += 1; return v
        gen = _G(sample_fn, params)
    else:
        gen = sample_fn

    counts: Dict[int, int] = {}
    res = Result()
    t = 0
    tau = None

    # ---- 探索阶段 ----
    while True:
        c = next(gen)
        t += 1
        counts[c] = counts.get(c, 0) + 1
        res.seq.append(c)
        phat, dhat, sigma2, K = top2_stats(counts, t)
        if params.explore_mode == "early":
            if t >= params.min_t:
                wt = alg_width(params, sigma2, t)
                if dhat - wt >= params.Delta_high:
                    tau = t
                    break
        if t >= params.T_cap:
            tau = t
            break

    res.tau = tau if tau is not None else t
    res.trigger_t = tau

    # ---- 分支决策 ----
    phat, dhat, sigma2, K = top2_stats(counts, tau)
    N_rem = params.E - tau
    if N_rem < 1:
        # 预算已尽, 直接众数
        res.branch = "naive"
        res.n_samples = tau
        ans = mode_of(counts)
        res.correct = (ans == true_mode) if true_mode is not None else True
        res.p_err_nominal = 0.0
        return res

    if (params.explore_mode == "early" and tau < params.T_cap) or params.explore_mode == "full":
        # ---- 尝试快分支 (early: 触发即切; full: 探索满后仍考虑快分支) ----
        w, winfo = run_choose_w(
            phat, N_rem, params.p_star,
            pessimistic=params.pessimistic, tau=tau,
            delta_cs=params.delta, w_min=3, w_max=None)
        if w is not None:
            res.branch = "fast"
            res.w = w
            res.p_err_nominal = winfo["p_err"]
            res.info = winfo
            # 快分支运行: 连续 w 同色或预算耗尽
            run = 0
            last = None
            stopped = False
            while t < params.E:
                c = next(gen)
                t += 1
                res.seq.append(c)
                counts[c] = counts.get(c, 0) + 1
                if c == last:
                    run += 1
                else:
                    run = 1
                    last = c
                if run >= w:
                    stopped = True
                    break
            res.n_samples = t
            if true_mode is not None:
                res.correct = (mode_of(counts) == true_mode)
            return res
        # 快分支不可行 -> 落入朴素判据(剩余预算)
    # else: 探索跑满

    # ---- 朴素分支 / 拒绝 ----
    wt_c = alg_width(params, sigma2, tau) if params.use_pessimistic_check else 0.0
    d_use = max(0.0, dhat - wt_c)
    K_use = max(params.K_floor, K)
    crit = (K_use - 1) * math.exp(-N_rem * d_use * d_use / 2)
    if crit <= params.p_star:
        res.branch = "naive"
        res.p_err_nominal = min(1.0, crit)
        while t < params.E:
            c = next(gen)
            t += 1
            res.seq.append(c)
            counts[c] = counts.get(c, 0) + 1
        res.n_samples = t
        if true_mode is not None:
            res.correct = (mode_of(counts) == true_mode)
        return res
    # 拒绝
    res.branch = "reject"
    res.n_samples = tau
    res.p_err_nominal = 0.0
    if true_mode is not None:
        res.correct = False  # 未作答, 计为"非成功回答"(区别于错误答案)
    return res


# ----------------------------------------------------------------------------
# Oracle: 已知真实分布的预言机
# ----------------------------------------------------------------------------
def oracle_plan(p: np.ndarray, params: Params) -> Dict[str, Any]:
    """预言机: 已知真实分布 p。
    决策: 先试快分支(τ=0, 全预算 E), 不可行则试朴素(全预算E), 再拒绝。
    oracle fast 的 p_err 由 DP 对真实分布精确计算(plug-in==真实)。"""
    p = np.asarray(p, dtype=float)
    p = np.sort(p)[::-1]
    K = len(p)
    Delta = max(0.0, p[0] - p[1]) if K >= 2 else p[0]

    # 1) 快分支: 真实分布下找最小可行 w
    wo, oinfo = run_choose_w(p, params.E, params.p_star, pessimistic=False,
                             tau=1, delta_cs=params.delta, w_min=3, w_max=None)
    if wo is not None:
        return dict(branch="fast", w=wo, cost=round(float(oinfo["E_fast"]), 3),
                    p_err=oinfo["p_err"], E_fast=oinfo["E_fast"], p_err_run=oinfo["p_err_run"])
    # 2) 朴素: (K-1) e^{-E Δ²/2} ≤ p* ?
    crit = (K - 1) * math.exp(-params.E * Delta * Delta / 2)
    if crit <= params.p_star:
        return dict(branch="naive", w=None, cost=float(params.E), p_err=crit,
                    E_fast=float(params.E), p_err_run=0.0)
    # 3) 拒绝
    return dict(branch="reject", w=None, cost=0.0, p_err=0.0, E_fast=0.0, p_err_run=0.0)


# ----------------------------------------------------------------------------
# 工具: 按真实分布 p 的采样器
# ----------------------------------------------------------------------------
def make_picker(p: np.ndarray, rng: np.random.Generator):
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    def pick(i: int) -> int:
        return int(rng.choice(len(p), p=p))
    return pick