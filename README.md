# A股主板低位筹码峰日线筛选器

本项目用于每天收盘后或次日开盘前，在 Google Colab 扫描沪深 A 股主板。它只使用 `requests` 获取公开日线行情，不依赖 AkShare 或 Tushare Pro，不做盘中触发。代码适合保存在 GitHub，历史缓存和每日结果可持久化到 Google Drive。

输出是研究观察池，不是买卖指令。默认参数是中等召回的起点，需要用后续滚动回测和实际样本持续校准。

## 筛选目标

模型针对的是同一类结构的两个阶段，不再用“均线已经完全多头排列”作为前提：

### 1. 低位横盘 + 筹码峰（待启动）

- 此前有明显下跌或从 120 日高点显著回撤。
- 最近约 20 个交易日以横盘为主，平台宽度和平台涨跌幅受控。
- 模拟筹码主要集中在历史价格区间的中低位，70% 成本区不能过宽。
- 主筹码峰附近有足够质量，低位筹码占比较高，不能只是成交死寂形成的假集中。
- 当前价格仍在主筹码峰附近，尚未明显脱离平台。

### 2. 低位横盘 + 筹码峰（刚启动）

- 必须先满足上面的下跌、平台和低位筹码峰结构。
- 最近 5 日开始向上，价格站上 MA5、MA5 转向上，并接近或突破启动前平台上沿。
- 量能、MACD 改善用于加分，不要求 MA20、MA30、MA60 已经全部转为多头。
- 排除 10 日涨幅过大、快速假突破以及高换手大振幅但价格停滞的情况，避免追到加速末段。

这两个输出是同一生命周期的两个观察阶段，不是互相无关的两套策略。

## 筹码分布口径

免费公开日线无法取得账户级持仓成本，因此项目不会声称算出“精确筹码分布”。程序实现的是 `modeled_cyq`：

1. 用 2048 个固定对数价格格保存最近 250 个交易日的筹码质量，避免多年极端价格压低分辨率。
2. 每日旧筹码按当日换手率等比例衰减。
3. 当日新增筹码按最低价、最高价和 `(开+高+低+收)/4` 构造三角分布。
4. 按时间顺序逐日递推，只使用当日及以前数据。
5. 从最终分布提取主峰、次峰、峰值附近筹码占比、70%/90% 成本区、低位筹码占比、获利盘和上方套牢盘等特征。

这种算法与常见行情软件的日线筹码估算属于同一类模型，但不是交易所或券商账户的真实持仓。真实卖方成本不可观察，“换手后各成本区同比例退出”也是模型假设。

公开实现可参考 [AKShare 的 `stock_cyq_em` 源码](https://github.com/akfamily/akshare/blob/main/akshare/stock_feature/stock_cyq_em.py)：它同样使用 OHLC 均价、三角日内分布、换手衰减和近期日线自行计算，而不是下载账户级筹码明细。Tushare 也提供 [`cyq_chips`](https://tushare.pro/document/2?doc_id=294)，但需要相应积分且仍属于筹码分布估算；本项目不依赖它。

关键输出字段：

| 字段 | 含义 |
| --- | --- |
| `chip_peak_price` | 模拟主筹码峰价格 |
| `chip_peak_band_share_pct` | 主峰上下约 4% 价格带内的筹码占比 |
| `chip_70_width_pct` | 70% 模拟筹码成本区宽度，越小越集中 |
| `chip_low_zone_share_pct` | 历史价格区间下部 35% 内的筹码占比 |
| `chip_peak_position` | 主峰在历史最低到最高价格区间中的位置 |
| `chip_peak_distance_pct` | 当前收盘价相对主峰的距离 |
| `chip_overhead_ratio_pct` | 当前价上方的模拟筹码占比 |
| `chip_model_coverage_pct` | 历史换手累计后模型覆盖度 |

旧的 `cost_concentration_60_pct` 成交成本代理仍保留在全量指标文件中，但不再代替筹码峰参与硬筛选。

## 防止过拟合

默认策略把条件分为少量必要结构和分项评分：

- 必要结构：历史与流动性完整、下跌后处在低位、存在横盘平台、存在低位集中筹码峰、没有明显失败或派发风险。
- 评分：平台深度和紧凑度、筹码峰质量、低位筹码比例、启动量价和风险情况。
- 默认配置是中等召回；`config/high_recall.yaml` 只用于对照，不建议直接把所有阈值同时放宽。

用户提供的中天科技、远东股份、兴发集团等 16 只股票只放在 Colab 的样本审计列表中，不会被硬编码为候选，也不会改变评分。每次运行后 notebook 会显示这些股票当前是否命中，以及最先卡在哪个筛选步骤。样本形态会随时间变化，所以校准必须指定观察日期，不能把股票名称永久当成正样本标签。

## 为什么有时只有 0 至 2 只

不能只根据候选数量判断“股市不好”或“条件太严”：

1. 先看 `coverage_report.json`。覆盖率低于 90% 时应先排查抓取，空结果不能解释为市场结论。
2. 看 `screening_funnel.csv`。大量股票卡在最后的评分步骤，通常说明分数阈值偏严；大量股票卡在“下跌后低位”或“低位集中筹码峰”，说明当天真正符合目标结构的股票较少。
3. 看 `near_miss_top100.csv`。它给出最接近入选的股票、首个未通过步骤和所有筹码分项。
4. 同日再运行 `config/high_recall.yaml`。高召回候选显著增加，说明默认参数选择性较强；两份配置都很少，才更支持“市场中该形态较少”。
5. 不要为了固定候选数量每天改阈值。最终应比较不同市场阶段的样本外命中率和回撤。

CLI 和 notebook 都会直接打印筛选结果。空 CSV 只有表头表示当日确实没有入选，不代表保存失败。

## Colab 使用

1. 将整个项目上传到 GitHub。
2. 打开 `notebooks/colab_daily_scan.ipynb`。
3. 修改 `REPO_URL`，挂载 Google Drive，运行全部单元格。
4. 默认使用 `config/default.yaml`；需要判断是否过严时改为 `config/high_recall.yaml` 再跑一次。

首次运行会下载主板股票历史，后续优先复用 Drive 缓存。建议将以下数据目录保存在 Drive：

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

- `chip_base_ready_all.csv`：低位横盘且筹码峰形成、仍在峰值附近的候选。
- `chip_base_launch_all.csv`：同样结构中最近刚开始向上的候选。
- 对应的 `_top100.csv`：便于查看的排名文件；`top_n` 不会截断 `_all.csv`。
- `indicators_scored.csv`：所有有效股票的原始指标、筹码特征和分项得分。
- `screening_funnel.csv`：每个硬门槛后的剩余数量。
- `near_miss_top100.csv`：最接近入选的股票及首个未通过步骤。
- `fetch_status.csv`：每只股票的抓取、历史长度和最新日期状态。
- `coverage_report.json`：数据覆盖率、真实候选数、数据口径和自动诊断。
- `watchlist_active.csv`、`state_transitions.csv`：跨日观察状态。
- `reference_examples_audit.csv`：Colab 根据 16 只参考样本生成的诊断文件，不参与选股。

## 主要配置

| 参数 | 作用 |
| --- | --- |
| `chip_base_ready_score_min` | 待启动阶段最低评分 |
| `chip_base_launch_score_min` | 刚启动阶段最低评分 |
| `chip_max_base_width_pct` | 启动前 20 日平台最大宽度 |
| `chip_max_base_abs_return_pct` | 平台期允许的最大绝对涨跌幅 |
| `chip_max_70_width_pct` | 70% 筹码成本区最大宽度 |
| `chip_max_peak_position` | 主峰在长期价格区间中的最高位置 |
| `chip_min_peak_band_share_pct` | 主峰附近最低筹码占比 |
| `chip_min_low_zone_share_pct` | 低位区域最低筹码占比 |
| `chip_ready_max_peak_distance_pct` | 待启动股票允许偏离主峰的最大幅度 |
| `chip_launch_max_return_10d_pct` | 刚启动阶段的 10 日最大涨幅，防止追高 |
| `min_amount_ma20` | 20 日平均成交额门槛 |

## 外部资金数据

龙虎榜、融资余额、股东户数、减持和行业利润等数据不能从 OHLCV 推导。项目仍可读取 `DATA_DIR/external/funding_signals.csv` 并写入全量结果供人工核对，但这些不完整、低频的数据目前不参与筹码峰策略的硬筛选或评分。程序不会用缺失值伪造资金结论。

## 数据与研究限制

公开 Web 接口可能限流或变更；程序会记录来源、覆盖率和失败原因，但不能保证永久可用。模拟筹码会受换手率口径、复权方式和日内分布假设影响。当前回测使用现有缓存股票池，存在幸存者偏差，也未模拟涨停排队、佣金、滑点、公告可用时间和真实可成交性。实盘使用前需要自行验证数据授权、复权一致性、交易限制和风险控制。
