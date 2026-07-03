# skill-backtest-overfit

简体中文 | English

判断一个回测的 Sharpe 是不是多重检验/数据挖掘挑出来的噪声。计算 Deflated Sharpe Ratio、回测过拟合概率（PBO）、多重检验 haircut 与最小可信样本长度（MinTRL），输出 PASS/FAIL 结论。

> A statistical gate for backtests: given a strategy's returns and how many
> configurations were tried, it computes the Deflated Sharpe Ratio, the
> Probability of Backtest Overfitting (PBO), a multiple-testing Sharpe haircut,
> and the Minimum Track Record Length — then returns a PASS/FAIL verdict.

## 为什么需要它

整条因子流水线在批量造因子、批量调参。**只要试得够多，纯随机数里也能"挖"出一个 Sharpe 1.6 的策略。** `skill-backtest` 的 health-check 是启发式的，抓不到这种选择偏差。本 Skill 用统计方法把它揪出来。

下面是仓库自带 demo 的真实输出（`python examples/run_demo.py`，全合成、无需凭证）：

```
CASE A：从 200 个纯噪声策略里挑最好的（过拟合）
  观测 Sharpe(年化) = 1.65   →  DSR 0.63 < 0.95   →  FAIL
  PBO 0.34 · haircut -61%

CASE B：同一个配置，但注入了真实 edge
  观测 Sharpe(年化) = 2.58   →  DSR 0.98 ≥ 0.95   →  PASS
  PBO 0.06 · haircut -22%
```

同样"好看"的 Sharpe，一个是噪声、一个是真信号——本 Skill 区分得出来。

## 快速开始

```bash
pip install -r requirements.txt

# 离线对照 demo（推荐先跑这个）
python examples/run_demo.py

# 对你自己的回测出报告
python scripts/overfit_report.py \
  --returns selected.csv \
  --trials  trials.csv \
  --n-trials 200 \
  --out report.json
```

`selected.csv` 是被选中策略的逐期收益（单列）；`trials.csv` 是所有试过的配置组成的 `T×N` 收益矩阵（PBO 需要）。

## 方法与文献

| 统计量 | 回答的问题 | 文献 |
|--------|-----------|------|
| Deflated Sharpe Ratio | 扣掉"试了 N 次"后，真 SR 还 >0 吗 | Bailey & López de Prado (2014) |
| PBO (via CSCV) | 样本内最优在样本外有多大概率平庸 | Bailey, Borwein, López de Prado & Zhu (2017) |
| Purged & Embargoed CV | 防标签重叠导致的信息泄漏 | López de Prado, *AFML* (2018), ch.7 |
| Haircut Sharpe | 多重检验下 Sharpe 该打几折 | Harvey & Liu (2015) |

详见 [`references/methodology.md`](references/methodology.md) 与 [`references/anti-patterns.md`](references/anti-patterns.md)。

## 数据接入

逻辑不依赖外部数据即可运行（只需收益序列）。如需取基准/真实收益，`scripts/data_source.py` 封装 panda_data（`get_stock_daily` + `get_adj_factor`），**无凭证时自动回退到确定性合成数据**，方便离线验证。配置真实凭证：

```bash
export DEFAULT_USERNAME=...      # 或 ~/.pandadata/pandadata.env
export DEFAULT_PASSWORD=...
export JAVA_SERVICE_BASE_URL=...
```

## 许可证

GPL-3.0 · Copyright (C) 2026.
