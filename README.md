# A股主板低位筹码峰日线筛选器

本项目用于每天收盘后或次日开盘前，在 Google Colab 扫描沪深 A 股主板。它使用 BaoStock 获取日线，使用 `requests` 访问网页备用源和候选资金流，不依赖 AkShare 或 Tushare Pro，不做盘中触发。代码适合保存在 GitHub，历史缓存和每日结果可持久化到 Google Drive。

输出是研究观察池，不是买卖指令。默认参数是中等召回的起点，需要用后续滚动回测和实际样本持续校准。

## 筛选目标

模型针对的是同一类结构的三个阶段，不再用“均线已经完全多头排列”作为前提：

### 1. 低位横盘 + 筹码峰（待启动）

- 此前下跌、从阶段高点回撤或长期处于价格区间低位，至少具备一种低位背景。
- 同时比较 20、30、40、60 个交易日的横盘平台，平台宽度和平台涨跌幅受控。
- 模拟筹码主要集中在历史价格区间的中低位，70% 成本区不能过宽。
- 主筹码峰附近有足够质量，低位筹码占比较高，不能只是成交死寂形成的假集中。
- 当前价格仍在主筹码峰附近，尚未明显脱离平台。

### 2. 低位横盘 + 筹码峰（刚启动）

- 必须先满足上面的下跌、平台和低位筹码峰结构。
- 最近 5 日开始向上，价格站上 MA5、MA5 转向上，并接近或突破启动前平台上沿。
- 量能、MACD 改善用于加分，不要求 MA20、MA30、MA60 已经全部转为多头。
- 排除 10 日涨幅过大、快速假突破以及高换手大振幅但价格停滞的情况，避免追到加速末段。

### 3. 横盘后反弹（启动确认）

- 在最近 3 至 30 日内结束的平台中，自适应选择 20、30、40、60 日最佳横盘，识别已经离开平台一段时间的股票。
- 允许平台和筹码位置比“待启动”阶段略高、略宽，但仍要求此前有回撤、筹码峰可见且 70% 成本区不过分分散。
- 当前价格需要保持在历史平台上沿附近或上方，20 日涨幅必须为正但不能已经严重过热。
- 该阶段用于覆盖新洁能、赛腾股份以及横盘后反弹的京泉华、翔鹭钨业一类结构，不会放宽前两个阶段的条件。

三个输出是同一生命周期的不同观察阶段，不是互相无关的策略。

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

- 必要结构：历史与流动性完整、存在自适应横盘平台、存在明显筹码峰、没有严重失败或派发风险。
- 可替代证据：前期下跌、阶段高点回撤、低区间位置和低位筹码占比参与评分，不再要求全部同时满足。
- 评分：平台深度和紧凑度、筹码峰质量、低位筹码比例、启动量价和风险情况。
- 默认配置是中等召回；`config/high_recall.yaml` 只用于对照，不建议直接把所有阈值同时放宽。

用户提供的股票是诊断用验收样本，不会被硬编码为候选，也不会绕过评分。CLI 会优先抓取这些股票，并固定打印、保存逐股抓取状态、当前是否命中以及最先卡在哪个筛选步骤。样本形态会随时间变化，所以校准必须指定观察日期，不能把股票名称永久当成正样本标签。

## 为什么有时只有 0 至 2 只

不能只根据候选数量判断“股市不好”或“条件太严”：

1. 先看 `coverage_report.json`。覆盖率低于 90% 时应先排查抓取，空结果不能解释为市场结论。
2. 看 `screening_funnel.csv`。大量股票卡在最后的评分步骤，通常说明分数阈值偏严；大量股票卡在“下跌后低位”或“低位集中筹码峰”，说明当天真正符合目标结构的股票较少。
3. 看 `near_miss_top100.csv`。它给出最接近入选的股票、首个未通过步骤和所有筹码分项。
4. 同日再运行 `config/high_recall.yaml`。高召回候选显著增加，说明默认参数选择性较强；两份配置都很少，才更支持“市场中该形态较少”。
5. 不要为了固定候选数量每天改阈值。最终应比较不同市场阶段的样本外命中率和回撤。

CLI 和 notebook 都会直接打印筛选结果。空 CSV 只有表头表示当日确实没有入选，不代表保存失败。

覆盖率低于配置的 `min_coverage_pct`（默认 90%）时，本次扫描会被标记为无效：正式候选 CSV 保持为空，不更新跨日观察池，也不打印临时候选。临时命中只写入 `provisional_*.csv` 供排查，避免把很小的数据子集误当成全市场筛选结果。

## 日线数据源与限流

- 个股日线优先使用 BaoStock 前复权日 K，直接取得历史成交额和逐日换手率；腾讯前复权 K 线和东方财富多节点作为备用。
- 股票池优先使用东方财富，失败后依次使用新浪和 BaoStock；候选资金流仍使用东方财富，资金流失败只影响排序，不删除形态候选。
- BaoStock 客户端使用单会话保护；网页备用源共享请求间隔，并在连续失败后短暂冷却再探测，不会永久停用到本次运行结束。
- 首次运行下载完整历史。后续使用 Drive 缓存时只刷新最近 30 天，并校验重叠的前复权价格；发现除权差异时自动重建该股票全量缓存。
- `0.3.0` 会自动判定旧版腾讯换手率缓存不兼容并重建日线；不会删除旧的运行结果、资金流或观察池文件。
- 任何来源都可能不可用。覆盖率低于 90% 时结果会明确判为无效，并打印失败样例和参考股票抓取状态。

## Colab 使用

1. 将整个项目上传到 GitHub。
2. 打开 `notebooks/colab_daily_scan.ipynb`。
3. 修改 `REPO_URL`，挂载 Google Drive，运行全部单元格。
4. 默认使用 `config/default.yaml`；需要判断是否过严时改为 `config/high_recall.yaml` 再跑一次。
5. 安装后先运行 `python -m ashare_scanner --version`，确认打印版本至少为 `0.3.1`，避免 GitHub `main` 仍是旧代码。

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
- `chip_base_rebound_all.csv`：平台已经结束 4 至 20 日、目前仍处于反弹确认阶段的候选。
- 对应的 `_top100.csv`：便于查看的排名文件；`top_n` 不会截断 `_all.csv`。
- `indicators_scored.csv`：所有有效股票的原始指标、筹码特征和分项得分。
- `screening_funnel.csv`：每个硬门槛后的剩余数量。
- `near_miss_top100.csv`：最接近入选的股票及首个未通过步骤。
- `fetch_status.csv`：每只股票的抓取、历史长度和最新日期状态。
- `fund_flow_status.csv`：最终候选的资金流抓取日期、来源和失败原因。
- `shape_only_*_all.csv`：形态评分完成后立即写出的检查点；资金流接口超时或失败也不会丢失候选。
- `coverage_report.json`：数据覆盖率、真实候选数、数据口径和自动诊断。
- `watchlist_active.csv`、`state_transitions.csv`：跨日观察状态。
- `reference_examples_audit.csv`：CLI 生成的完整参考样本诊断，抓取失败的样本也会保留，不参与选股。
- `provisional_*.csv`：仅在覆盖率不足时保存的临时命中，不是正式候选。

## 主要配置

| 参数 | 作用 |
| --- | --- |
| `chip_base_ready_score_min` | 待启动阶段最低评分 |
| `chip_base_launch_score_min` | 刚启动阶段最低评分 |
| `chip_base_rebound_score_min` | 横盘后反弹阶段最低评分 |
| `chip_max_base_width_pct` | 启动前 20 日平台最大宽度 |
| `chip_max_base_abs_return_pct` | 平台期允许的最大绝对涨跌幅 |
| `chip_max_70_width_pct` | 70% 筹码成本区最大宽度 |
| `chip_max_peak_position` | 主峰在长期价格区间中的最高位置 |
| `chip_min_peak_band_share_pct` | 主峰附近最低筹码占比 |
| `chip_min_low_zone_share_pct` | 低位区域最低筹码占比 |
| `chip_ready_max_peak_distance_pct` | 待启动股票允许偏离主峰的最大幅度 |
| `chip_launch_max_return_10d_pct` | 刚启动阶段的 10 日最大涨幅，防止追高 |
| `chip_rebound_max_base_width_pct` | 历史反弹平台允许的最大宽度 |
| `chip_rebound_max_return_20d_pct` | 反弹确认阶段允许的 20 日最大涨幅 |
| `min_amount_ma20` | 20 日平均成交额门槛 |
| `min_coverage_pct` | 发布正式候选和更新观察池所需的最低日线覆盖率 |
| `request_min_interval_seconds` | 所有日线请求共享的最小时间间隔 |

## 主力资金排序

程序使用东方财富个股资金流历史接口，只对形态排名靠前且不超过 `fund_flow_max_candidates`（默认 120）的候选请求近 100 个交易日数据。三个形态列表采用轮询方式公平选取，未请求资金流的股票仍保留并按形态得分排序。资金请求默认 2 线程、单请求 5 秒超时、最多尝试 2 个主机；连续失败 10 次或阶段运行超过 20 分钟会自动停止。每 10 只打印进度和 ETA，成功数据保存在 `cache/fund_flow/`；当日缓存直接复用，刷新失败时允许使用明确标记为滞后的旧缓存。公开字段和解析口径可参考 [AKShare 的个股资金流实现](https://github.com/akfamily/akshare/blob/main/akshare/stock/stock_fund_em.py)。

形态筛选结束后，程序会先写出 `shape_only_*_all.csv` 并在日志中打印各模式前 10 只预览，再进入资金流排序。因此外部资金接口不可用时，主要筛选结果仍然已经保存在 Drive。

需要立即得到纯形态结果时，可在运行命令末尾加 `--no-fund-flow`。这个开关只跳过可选的资金流排序，不会跳过横盘、筹码峰和启动/反弹模式筛选：

```bash
python -m ashare_scanner --config config/default.yaml --data-dir "$DATA_DIR" run --no-fund-flow
```

候选排序优先级固定为：

1. 资金数据更新到本次筛选交易日。
2. 3 日、5 日、10 日、20 日累计主力净流入全部大于 0。
3. 四个周期中为正的周期数量。
4. 依次比较 3 日、5 日、10 日、20 日净流入金额。
5. 资金条件相同时再比较形态评分。

资金流接口失败、数据滞后或历史不足 20 日时，股票不会被删除，只退回形态评分排序。输出会保留 `fund_flow_rank_reason`、各周期净流入金额以及 `fund_flow_status.csv`，避免把缺失数据误判为净流出。

## 其他外部数据

龙虎榜、融资余额、股东户数、减持和行业利润等数据不能从 OHLCV 推导。项目仍可读取 `DATA_DIR/external/funding_signals.csv` 并写入全量结果供人工核对，但这些不完整、低频的数据目前不参与筹码峰策略的硬筛选或评分。程序不会用缺失值伪造资金结论。

## 数据与研究限制

公开 Web 接口可能限流或变更；程序会记录来源、覆盖率和失败原因，但不能保证永久可用。模拟筹码会受换手率口径、复权方式、腾讯历史换手率推导和日内分布假设影响。当前回测使用现有缓存股票池，存在幸存者偏差，也未模拟涨停排队、佣金、滑点、公告可用时间和真实可成交性。实盘使用前需要自行验证数据授权、复权一致性、交易限制和风险控制。
