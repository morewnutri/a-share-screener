# A股主板双模式日线筛选器

这是一个面向 **收盘后或次日开盘前** 的 A 股主板观察池项目。程序使用 `requests` 获取公开行情，不依赖 AkShare，不做盘中触发。代码可放在 GitHub，缓存、状态和每日结果可持久化到 Google Drive。

输出是研究候选，不是买卖指令。默认参数是可回测的起点，不是已证明有效的参数。

## 两类筛选模式

### 1. 强资金运作型（埋伏吸筹末期）

目标是发现仍处在平台或启动边缘、但量价行为已经出现资金主导痕迹的股票。模型按五组证据评分，不要求所有形态同时发生：

- 平台结构：中低位、20 日区间相对 60 日区间收敛、关键压力位反复测试、120 日涨幅不过热。
- 量价与成本代理：上涨量优于下跌量、OBV 改善、温和换手、低位换手峰、60 日成交量加权成本离散度较低、中小流通盘轻量加分。
- 行为与波动：ATR/布林带/成交量至少两项收缩、假跌破快速收回、下沿承接、上影试盘、压缩后定向扩张。
- 独立性：20/60 日相对沪深300更强，在全市场横截面中的相对强度较高。
- 风险：突破后 1 至 3 日快速跌回平台、爆量大振幅但价格停滞、近 5 日派发型 K 线会扣分或拦截。

默认硬门槛只保留历史长度、流动性、基础数据、不过度上涨、无明显快速失败和最低分数。其余条件通过证据组与总分表达，避免多个布尔条件连续相与造成过度漏股。

### 2. 强资金运作型（主升浪阶段）

目标是发现趋势已经形成、当日出现短线资金重新发力的股票：

- 核心趋势：收盘价在 MA60 上方、MA60 向上、收盘价在 MA20 上方且 MA20 高于 MA30。
- 趋势加分：MA5 > MA20 > MA30、多条均线斜率向上。
- 资金触发：阳线上穿 MA5；为避免漏掉连续主升，也接受收盘在 MA5 上方、阳线且 MACD 柱继续增强。
- 量价配合：上涨量优于下跌量、上涨日量能合理放大、近期回调量低于 20 日基准、OBV 改善。
- 环境与风险：沪深300没有系统性破位、不过度偏离 MA20、10 日涨幅不过热、无明显快速失败或派发风险。

行业没有走坏属于可选加分证据，不会因为缺少可靠行业历史数据而直接淘汰股票。

## 资金、筹码和基本面数据边界

仅靠日线 OHLCV 不能可靠还原真实筹码分布，也不能推导龙虎榜机构席位、融资余额、股东户数、减持公告或行业利润增速。程序不会伪造这些字段：

- `cost_concentration_60_pct` 是 60 日成交量加权成交成本离散度代理，不是真实筹码分布。
- `main_net_inflow_ratio_pct` 可从当日股票池快照获取；接口不可用时保持缺失。
- 龙虎榜、融资、股东、财务和减持信息通过可选 CSV 导入，只影响评分，不是硬条件。
- “套牢盘小于 30%”目前未实现，因为免费日线数据不足以严谨计算。

可将 [示例文件](examples/funding_signals.example.csv) 放到：

```text
DATA_DIR/external/funding_signals.csv
```

字段如下：

| 字段 | 含义 | 评分规则 |
| --- | --- | --- |
| `code`, `date` | 股票代码、证据有效日期 | 每只股票只取筛选日及之前的最新一行 |
| `main_net_inflow_ratio_pct` | 主力净流入/成交额 | 大于等于 20% 加分 |
| `main_net_inflow_positive_days_5` | 最近 5 日主力净流入为正的天数 | 等于 5 加分 |
| `institution_net_buy_amount` | 龙虎榜机构净买额 | 大于等于 5000 万加分 |
| `margin_balance_growth_3d_pct` | 融资余额 3 日增幅 | 大于等于 10% 加分 |
| `shareholder_count_change_pct` | 股东户数变化 | 小于等于 -5% 加分 |
| `sector_profit_growth_median_pct` | 板块利润增速中位数 | 大于等于 20% 加分 |
| `industry_risk_ok` | 行业是否未明显走坏，1/0 | 1 轻量加分 |
| `has_reduction_plan` | 是否存在减持计划，1/0 | 1 扣分 |

未来日期的数据会被忽略，缺失值保持中性。

## 为什么有时只有 0 至 2 只

不能只凭候选数量判断“股市不行”或“条件太严”。按以下顺序看：

1. 看终端和 `coverage_report.json`。覆盖率低于 90% 时先排查抓取，空结果不能解释为市场结论。
2. 看 `screening_funnel.csv`。若大量股票最后只卡在 `埋伏型评分` 或 `主升浪评分`，阈值可能偏严；若早期就卡在趋势共振、位置或风险，说明当日真正匹配该形态的股票较少。
3. 同一交易日分别运行默认和高召回配置。高召回结果明显增加，说明参数选择性较强；两个配置都很少，才更支持“当日匹配结构少”的判断。
4. 看 `near_miss_top100.csv` 的分项得分，不要为了增加数量把所有阈值一起放宽。
5. 最终用滚动样本外回测比较命中率、最大回撤、候选数量和不同市场阶段的稳定性。

## Colab 使用

1. 将整个仓库上传到 GitHub。
2. 打开 `notebooks/colab_daily_scan.ipynb`。
3. 修改 `REPO_URL`，挂载 Google Drive 后运行全部单元格。
4. 默认使用 `config/default.yaml`；需要对比召回率时改为 `config/high_recall.yaml`。

CLI 在扫描完成后会直接打印两类候选、筛选漏斗和近似入选股。Notebook 也会把两类完整结果的前 100 行显示出来。空 CSV 只有表头表示当日确实没有入选，不表示保存失败。

首次运行会下载全市场历史数据，后续优先复用缓存。建议代码保存在 GitHub，以下目录保存在 Drive：

```text
cache/
external/
state/
runs/
backtests/
```

## 本地运行

```bash
python -m pip install -e .
python -m ashare_scanner --config config/default.yaml expected-date
python -m ashare_scanner --config config/default.yaml --data-dir data run --print-top 30
```

高召回对照：

```bash
python -m ashare_scanner --config config/high_recall.yaml --data-dir data run --print-top 50
```

使用已建立的历史缓存回测：

```bash
python -m ashare_scanner --config config/default.yaml --data-dir data backtest \
  --start 2024-01-01 --end 2025-12-31
```

## 每次扫描结果

结果位于 `DATA_DIR/runs/YYYY-MM-DD/`：

- `accumulation_late_all.csv`：埋伏吸筹末期的全部候选。
- `main_wave_all.csv`：主升浪阶段的全部候选。
- 对应的 `_top100.csv`：便于查看的排名结果，不影响全量候选数。
- `indicators_scored.csv`：所有有效股票的指标、分项得分和两类总分。
- `screening_funnel.csv`：每个硬门槛后剩余数量。
- `near_miss_top100.csv`：最接近入选的股票及首个未通过步骤。
- `fetch_status.csv`：抓取、历史长度和最新日期状态。
- `coverage_report.json`：覆盖率、真实候选数、外部证据状态和自动诊断。
- `watchlist_active.csv`、`state_transitions.csv`：跨日观察状态。

## 配置原则

`config/default.yaml` 使用中等召回设置；`config/high_recall.yaml` 只用于对照和扩充观察池。优先一次只调整一个维度，并通过回测观察变化：

- `accumulation_score_min`、`main_wave_score_min`：两类最低分数。
- `accumulation_min_evidence_groups`：埋伏型至少需要多少组独立证据。
- `accumulation_max_position_250`：埋伏型允许的长期区间位置。
- `main_wave_max_extension_ma20_pct`：主升型允许偏离 MA20 的最大幅度。
- `min_amount_ma20`：20 日平均成交额门槛。
- `top_n`：只控制 Top 文件和展示，不截断 `_all.csv`。

## 数据与研究限制

公开 Web 接口可能限流或变更；项目会记录来源、覆盖率和失败原因，但不能保证永久可用。当前回测使用现有缓存股票池，存在幸存者偏差，也未模拟涨停排队深度、佣金、滑点、公告可用时间和完整行业历史。实盘使用前必须自行验证数据授权、复权一致性、交易限制和风险控制。