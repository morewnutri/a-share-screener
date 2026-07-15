# A股主板日线观察池

这是一个面向 **收盘后或次日开盘前** 的 A 股主板日线扫描项目。它使用 `requests` 抓取公开行情，不依赖 AkShare，不处理盘中触发，也不把未复权备用数据混进同一排行榜。

项目输出的是研究观察池，不是买卖指令。默认阈值是可回测的起点，不是已经证明有效的参数。

## 解决了什么

- 股票池覆盖 `600/601/603/605/000/001/002/003`，包含深市 `002`、`003`。
- 股票列表按代码字段稳定分页，并检查分页重复和覆盖率。
- 使用 `exchange_calendars` 的 `XSHG` 交易日历判断最新完整交易日。
- 交易日 15:10 前只使用上一交易日，15:10 后才接受当天日线。
- K 线最后日期落后的股票标记为 `stale_hist`，不会参与策略。
- 历史策略数据只接受东方财富前复权 `fqt=1`；多 host 只做同源容灾。
- 股票列表接口不可用且没有缓存时，可用新浪批量报价扫描代码和名称；该兜底不提供策略 K 线。
- 缓存元数据包含来源、复权方式、schema、起始日期和最后完整交易日。
- RSI 正确处理连续上涨为 100、横盘为 50。
- 压力位和成交量基准都排除当前 K 线。
- 趋势、位置、量价、收缩、相对强度、流动性分别计分并各自封顶。
- 全量候选和 Top N 分开保存，报告中的候选数不受 Top N 截断。
- 候选跨日保留 `SETUP/TRIGGER/RETEST/INVALID/EXPIRED` 状态。
- 回测与每日扫描复用同一指标和信号函数。

## 四类输出

| 文件前缀 | 含义 |
| --- | --- |
| `setup_contraction` | 突破前，接近前高，至少两项波动/布林带/成交量收缩 |
| `setup_accumulation` | 突破前，上涨量强于下跌量，OBV 改善，相对强度较高 |
| `breakout_today` | 当天刚越过排除当前 K 线的 20 日前高，且没有明显追高或一字板 |
| `retest_after_breakout` | 突破后 2 至 10 个交易日缩量回踩突破位并收强 |

每类同时生成 `_all.csv` 和 `_top100.csv`。观察池在 `watchlist_active.csv`，状态迁移在 `state_transitions.csv`。

## Colab 使用

推荐把 **代码放 GitHub**，把 `cache/state/runs/backtests` 放 Google Drive。否则 Colab 运行时重启后会丢失缓存和跨日观察状态。

1. 将整个仓库上传到 GitHub。
2. 在 Colab 打开 `notebooks/colab_daily_scan.ipynb`。
3. 修改第一段代码中的 `REPO_URL`。
4. 运行全部单元格并授权挂载 Google Drive。

首次运行需要抓取全市场历史数据，后续交易日会优先复用已到最新完整交易日的缓存。由于采用前复权，缓存 schema 或起始日期改变时会自动失效并重抓。

## 本地使用

```bash
python -m pip install -e .
python -m ashare_scanner --config config/default.yaml expected-date
python -m ashare_scanner --config config/default.yaml --data-dir data run
```

收盘后测试指定时间：

```bash
python -m ashare_scanner --config config/default.yaml --data-dir data run \
  --as-of "2026-07-15T15:20:00+08:00"
```

回测使用扫描器已经建立的历史缓存：

```bash
python -m ashare_scanner --config config/default.yaml --data-dir data backtest \
  --start 2024-01-01 --end 2025-12-31
```

## 每次扫描的结果

结果位于 `DATA_DIR/runs/YYYY-MM-DD/`：

- `universe.csv`：当日主板股票池。
- `fetch_status.csv`：每只股票的抓取状态、日期、来源和错误。
- `indicators_scored.csv`：完整指标和六类因子分数。
- `<signal>_all.csv`：符合该信号的全部股票。
- `<signal>_top100.csv`：便于人工查看的 Top 榜。
- `watchlist_active.csv`：跨日活跃观察池。
- `state_transitions.csv`：本次状态变化。
- `coverage_report.json`：覆盖率、失败明细和真实候选数量。

持久状态位于 `DATA_DIR/state/`，历史缓存位于 `DATA_DIR/cache/`。

## 回测口径

默认把信号日的下一个交易日开盘价作为入场价。如果随后 10 个交易日内最高价达到 `+12%`，且达到目标之前的最低价回撤不超过 `-6%`，则记为成功。

输出包括：

- 观察级 precision 和 recall；
- 每类信号 precision；
- 每日只看 Top 20/50/100 时的 recall；
- 每条信号的未来最高涨幅和目标前回撤。

当前免费数据回测仍有明确限制：使用当前缓存股票池，存在幸存者偏差；没有历史行业成分；没有模拟涨停排队深度、佣金和滑点。要用于资金决策，应接入带退市股票、历史成分和公司行动时点的商业数据，再做滚动样本外验证。

## 配置建议

主要参数在 `config/default.yaml`。建议先保持默认值运行至少数月并回测，不要根据少数案例直接调权重。

- `min_amount_ma20` 应按资金规模调整。
- `setup_min_rs_percentile` 越低，观察池召回率通常越高，误报也越多。
- `top_n` 只控制人工查看榜，不会截断全量候选。
- `force_refresh` 仅用于排查缓存或数据源问题，不建议日常开启。

## 数据源边界

行情接口属于未承诺稳定性的公开 Web 接口，可能限流或变更。程序会记录失败并拒绝陈旧数据，但不能保证数据源永久可用。行业强度默认没有加入评分，因为免费来源缺少可靠的历史行业归属；用当前行业标签回填历史会引入未来信息。

## 免责声明

本项目只用于数据工程和量化研究，不构成投资建议。任何实盘使用都需要自行验证数据授权、复权正确性、交易限制、手续费、滑点和风险控制。
