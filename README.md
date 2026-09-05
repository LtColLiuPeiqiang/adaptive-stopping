# Adaptive Majority Voting with Certified Rejection — Experiment Reproduction Package
# 预算感知自适应多数投票 + 认证式拒绝 —— 实验复现包

This package contains the **complete experiment code** (synthetic bag-drawing experiments
and LLM upstream experiments) for the paper:

> *Budget-Aware Adaptive Majority Voting with Certified Refusal* (ESWA submission)

It reproduces every table and figure number reported in the paper's experimental section
(§5 *Experiments and Verification*). Plotting and lecture-note scripts are intentionally
excluded; the code here is the minimal, dependency-light core that generates `results.json`
and the raw numbers cited in the paper.

本包收录论文(ESWA 投稿《预算感知自适应多数投票 + 认证式拒绝》)实验部分(§5)的全部代码:
合成摸球实验 + LLM 上游实验。已剔除绘图与讲义脚本,仅保留生成论文全部表格/图数值的
最小核心代码。

---

## 1. Directory layout / 目录结构

```
adaptive-stopping-experiments/
├── README.md                       # this file / 本文件
├── synthetic/                      # 合成摸球实验 (known-distribution bag drawing)
│   ├── alg.py                      #   核心算法: 探索→三分支、Choose_w DP、certified/practical 宽度、Oracle
│   ├── sim.py                      #   exp1–exp7 全实验 (Δ 扫描/分支构成/调参/乐观vs悲观/early vs full/规模/certified)
│   └── patch_exp56.py              #   历史补丁(可选): 早期版本合并乐观 exp5/exp6 结果, 当前主流程已内置, 一般无需运行
├── llm/                            # LLM 上游实验 (ollama + Qwen2.5-7B, GSM8K / MATH)
│   ├── llm_vote.py                 #   论文 LLM 实验: 池重放 + 自适应停止 vs 固定 N 众数基线
│   ├── make_subset.py              #   从原始数据构造 llm_vote 所需的 subset.jsonl (含 gold 提取)
│   ├── download_gsm8k.py           #   GSM8K test 下载 (hf-mirror 兼容)
│   └── download_math.py            #   MATH test 下载 (EleutherAI/hendrycks_math, 7 学科合并)
└── data/                           # 参考运行日志 (非论文数据, 仅演示真实输出)
    ├── sim_run_final.log           #   合成实验全量日志 (2000 seeds, 66 min)
    └── llm_run.log                 #   LLM 实验 160 题日志 (87 min)
```

---

## 2. Environment / 环境要求

- Python 3.10+ (`numpy`); 合成实验另需 `matplotlib` (仅用于保存附图, Agg 后端, 无需显示器);
  LLM 实验另需 `requests`。数据下载脚本需要 `datasets` (可选)。
- 中国大陆网络: 建议 pip 走清华 TUNA (`-i https://pypi.tuna.tsinghua.edu.cn/simple`),
  HuggingFace 走 `hf-mirror.com` (`export HF_ENDPOINT=https://hf-mirror.com`)。
- LLM 实验需要 [Ollama](https://ollama.com) + 模型 `qwen2.5:7b`
  (`ollama pull qwen2.5:7b`); 推理需要 GPU (论文环境: RTX 5070 12 GB)。

```bash
python3 -m venv venv && . venv/bin/activate
pip install numpy matplotlib requests          # 合成 + LLM 运行
pip install datasets huggingface_hub           # 仅数据下载需要
```

---

## 3. Part A — Synthetic bag-drawing experiments / 合成摸球实验

**Problem.** An unknown multinomial "bag" over colors with unknown K; drawing with
replacement; the algorithm must either certify the mode with error rate ≤ p* within budget,
or honestly refuse.

**Recommended configuration (paper's headline config):**
`E*=200, p*=0.05, Δ_high=0.1, δ=0.05, T_cap=150, min_t=8, practical width z=2,
optimistic (plug-in) Choose_w, early trigger`, distribution family `delta_family(Δ)`
(top-2 gap Δ, remaining K−2 colors split a 0.2 mass), K=10, 2000 seeds per cell.

**Run.**

```bash
cd synthetic

# smoke test (a few minutes)
python3 sim.py --quick

# full reproduction (all of exp1–exp7, 2000 seeds; ≈ 60–70 min on a desktop CPU)
python3 sim.py --seeds 2000
```

Outputs: `synthetic/results.json` (all numbers in the paper's synthetic tables) and
`figs/*.png` (regenerated figures).

**Expected headline numbers** (paper Fig. 4–7, Tables 3–4; exp1 sweep, optimistic):

| Δ (true gap) | fast | naive | reject | err_answered | avg_n | Oracle |
|---|---|---|---|---|---|---|
| 0.02 | 0.149 | 0 | 0.852 | 0.451 | 132.5 | reject |
| 0.05 | 0.159 | 0 | 0.841 | 0.315 | 131.6 | reject |
| 0.1 | 0.204 | 0 | 0.796 | 0.155 | 127.4 | reject |
| 0.2 | 0.45 | 0 | 0.55 | 0.026 | 106.1 | reject |
| 0.4 | 0.969 | 0 | 0.032 | 0.0005 | 40.6 | fast / 16.4 |
| 0.8 | 1.0 | 0 | 0 | 0.0 | 14.1 | fast / 4.8 |

Key takeaways: easy bags (Δ ≥ 0.4) are answered at ≈2.4× Oracle cost with near-zero error;
hard bags (Δ ≤ 0.1) are dominated by honest refusal — the fixed-N=200 majority baseline
errs 22–38% on them, which is exactly why refusal is the rational output.

---

## 4. Part B — LLM upstream experiments / LLM 上游实验

**Protocol (pool replay).** For each question, pre-generate E*=8 i.i.d. decoding samples
(temperature 0.7, top_p 0.9, num_predict 1100) via Ollama. Each parsed answer is a "ball".
The adaptive stopping rule replays the pool online (a stopping time on the observed prefix
is statistically equivalent to live online stopping); the fixed-N majority baselines
(N = 1,2,3,4,6,8) reuse the same pool for a fair comparison. *Effective compute* is
reported as `n_used` — the calls a real deployment can actually skip.

**Recommended configuration (paper Table 2):**
`E*=8, p*=0.02, Δ_high=0.3, δ=0.05, T_cap=4, min_t=2, practical width z=2,
optimistic plug-in` (i.e. `--pessimistic 0`).

**Step 1 — data.** Download GSM8K + MATH and build the question subset:

```bash
cd llm
HF_ENDPOINT=https://hf-mirror.com python3 download_gsm8k.py   # -> ../data/gsm8k_test.jsonl (1319 题)
HF_ENDPOINT=https://hf-mirror.com python3 download_math.py    # -> ../data/math_test.jsonl (5000 题)
python3 make_subset.py --limit 80                              # -> ../data/subset.jsonl (80+80 = 160 题)
```

**Step 2 — serve the model.**

```bash
ollama serve &        # 或 systemd 服务
ollama pull qwen2.5:7b
ollama ps             # 确认 100% GPU (Q4_K_M ≈ 4.7 GB)
```

**Step 3 — run the experiment.**

```bash
cd llm
# smoke test: 2 题
python3 llm_vote.py --subset ../data/subset.jsonl --out ../data/llm_results_smoke.json \
    --limit 2 --pessimistic 0

# paper config, 160 题 (≈ 1.5 h on a single RTX 5070; ≈ 20 s / question)
python3 llm_vote.py --subset ../data/subset.jsonl --out ../data/llm_results.json \
    --pessimistic 0

# print the summary table (accuracy / avg n_used / branch mix / fixed-N baselines)
python3 llm_vote.py --summarize ../data/llm_results.json
```

**Expected numbers** (paper Table 2; Qwen2.5-7B, temperature 0.7):

| Dataset | ans_rate | acc_answered (conditional) | avg n_used / 8 | fixed-N accuracy (1/2/3/4/6/8) |
|---|---|---|---|---|
| GSM8K (80) | 0.887 | 0.915 | 5.25 (省 34%) | 0.862 / 0.862 / 0.912 / 0.912 / 0.912 / 0.925 |
| MATH (80) | 0.50 | 0.85 | 4.89 | 0.537 / 0.537 / 0.612 / 0.60 / 0.675 / 0.6875 |

Adaptive ≈ fixed N = 3–4 accuracy at lower average cost; the rejected fraction on MATH is
concentrated on "torn" questions (partial-mode hit rate only ≈0.375), i.e. rejection
purifies the answer stream.

---

## 5. Parameters / 参数速查

| Symbol | Meaning / 含义 | Synthetic / 合成 | LLM |
|---|---|---|---|
| E* | sampling budget / 采样预算 | 200 | 8 |
| p* | max error rate / 允许最大错误率 | 0.05 | 0.02 |
| Δ_high | fast-branch trigger threshold | 0.1 | 0.3 |
| δ | mis-trigger probability cap | 0.05 | 0.05 |
| T_cap | exploration cap | 150 | 4 |
| min_t | min exploration before trigger | 8 | 2 |
| width_mode | certified (CS) vs practical (z·sd) | practical, z=2 | practical, z=2 |
| pessimistic | plug-in vs worst-case reweight | False (推荐) | False (推荐) |
| explore_mode | early trigger vs full exploration | early | early |

Certified mode (exp7) uses `E*=1000, T_cap=250, min_t=40`: certification time scales as
T_cert ∝ Δ⁻² (Δ=0.8→8 samples, Δ=0.2→3079, Δ≤0.1→never), which quantifies the
guarantee-vs-practicality trade-off.

---

## 6. Notes / 注意事项

1. **Metrics.** Always report `ans_rate` + `err_answered` (conditional error on answered)
   *and* `err_total` separately — the latter is dominated by refusals and hides
   mis-trigger risk. Do not tune on `err_total` alone.
2. **Choosing the width.** The certified time-uniform width
   `w_t = sqrt(2σ̂²(2ln t + ln(π²/3δ))/t)` is very conservative at practical budgets
   (≈ all bags refused). Use it only when a formal guarantee is required at large budgets;
   the paper's default is the practical width `z·sqrt(σ̂²/t)` with z=2 (≈
   exp3 tuning; z≥3 if you need stricter control on near-tie bags).
3. **DP correctness.** `run_dp` (run-window stopping) was cross-validated against 400k
   Monte-Carlo simulations; the fix that matters is shifting the column on same-color
   continuation (`cont[:, 1:] = M[:, :-1] * phat`). Do not "optimize" it back.
4. **Repro hygiene.** One script = one log file; check `ps aux | grep` for leftover
   processes before rerunning; remove stale `results.json` / `figs` before a full rerun.
5. **Answer parsing.** GSM8K gold = last number in `answer`; MATH gold = the last
   `\boxed{...}` in `solution` (see `make_subset.py`). The model often puts `\boxed{1}`
   on its own line after "The final answer is:" — `parse_answer` handles this.
6. **Ollama front/back-end version notes.** Any modern Ollama works; keep the service on
   `localhost:11434`. If the GPU driver is mismatched, Ollama's bundled CUDA still works —
   do not touch kernel drivers.
7. **China mainland mirrors.** pip: TUNA; HF: `HF_ENDPOINT=https://hf-mirror.com`
   (both datasets below are open, no token needed).

---

## 7. Provenance / 出处

Canonical sources:
- Core algorithm + synthetic experiments: `~/adaptive-voting/` (alg.py, sim.py)
- LLM experiments: `~/adaptive-voting/llm_vote.py`; data pipeline roots:
  `~/llm-consensus-pruning/` (inference_ollama.py, download_math.py)
- Paper (LaTeX, ESWA submission): `~/eswa-adaptive-stopping/`

These are static reproductions of the exact files that produced the paper's numbers
(2026-08 final round). Where a path had to change for this package layout, the change is
minimal and documented (llm_vote.py adds `../synthetic` to `sys.path`; data files are not
bundled — regenerate with the download/make_subset scripts).

License: MIT (reuse freely; cite the paper).

---

# 中文速览 / Quick start (中文)

```bash
# A. 合成实验
pip install numpy matplotlib
cd synthetic && python3 sim.py --quick        # 冒烟
cd synthetic && python3 sim.py --seeds 2000   # 全量 exp1-7, ~1 小时

# B. LLM 实验 (需 ollama + qwen2.5:7b, GPU)
pip install numpy requests
cd llm
HF_ENDPOINT=https://hf-mirror.com python3 download_gsm8k.py
HF_ENDPOINT=https://hf-mirror.com python3 download_math.py
python3 make_subset.py --limit 80
python3 llm_vote.py --subset ../data/subset.jsonl --out ../data/llm_results.json --pessimistic 0
python3 llm_vote.py --summarize ../data/llm_results.json
```