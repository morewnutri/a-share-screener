# A-Share Screener

面向 **A 股主板日线筛选** 的工程化项目，适用于：

- 每天 **开盘前** 在 Google Colab 运行，查看上一完整交易日的观察池与候选名单
- 每天 **收盘后** 在 Google Colab 运行，生成最新完整日线筛选结果

本项目根据你给出的 review 重构，重点修复了原始版本中的关键问题：

- 完整主板股票池：支持 `000/001/002/003/600/601/603/605`
- 使用 **最新完整交易日** 判断缓存是否可用，避免慢一个交易日
- 区分 **完整日线** 与盘中未完成K线，不缓存不完整日K线
- 正式策略只使用 **东方财富前复权日线**，避免与未复权数据混算
- 修复 RSI 连续上涨时变成 NaN 的问题
- 压力位、成交量基准统一使用 `shift(1)`，避免把“今天自己”算进参考基准
- 输出 **全部候选** 与 **Top 榜单**，不再只保留前100个
- 增加 **波动收缩、相对强度、流动性、陈旧数据过滤**
- 增加 **候选状态跟踪** 和 **基础回测框架**

> 注意：本项目是 **日线系统**，不包含盘中触发交易逻辑，也不尝试解决盘中第一时间打板/追板问题。

---

## Stack

- Python 3.10+
- requests
- pandas
- numpy
- openpyxl
- tqdm

---

## 项目结构

```text
.
├── config.py                # 全局配置
├── calendar_utils.py        # 最新完整交易日判断
├── cache_manager.py         # K线缓存与元数据管理
├── data_sources.py          # 东方财富/新浪/指数数据源
├── universe.py              # 股票池构建与交叉校验
├── indicators.py            # 技术指标
├── features.py              # 特征构建
├── filters.py               # 可交易性/数据质量过滤
├── strategies.py            # setup / breakout / retest 分类与评分
├── state_tracker.py         # 观察池状态跟踪
├── backtest.py              # 基础滚动回测框架
├── report.py                # CSV / JSON / Excel 报表输出
├── main.py                  # 主入口
├── requirements.txt         # 依赖
└── scripts/
    ├── colab_setup.py       # Colab 环境准备
    └── run_colab.sh         # Colab 一键运行
```

---

## 快速开始

### 本地运行

```bash
pip install -r requirements.txt
python main.py
```

### Google Colab 运行

方式 1：

```bash
!bash scripts/run_colab.sh
```

方式 2：

```python
!python scripts/colab_setup.py
!python main.py
```

---

## 输出目录

默认输出：

- `output/all_ashare_raw.csv`
- `output/stock_universe.csv`
- `output/fetch_status.csv`
- `output/indicator_table.csv`
- `output/feature_table.csv`
- `output/scored_table.csv`
- `output/setup_candidates_all.csv`
- `output/setup_candidates_top.csv`
- `output/breakout_candidates_all.csv`
- `output/breakout_candidates_top.csv`
- `output/retest_candidates_all.csv`
- `output/retest_candidates_top.csv`
- `output/state_snapshot.csv`
- `output/coverage_report.json`
- `output/result.xlsx`（可选）

---

## 使用原则

- **开盘前运行**：读取上一完整交易日数据，建立观察池
- **收盘后运行**：读取当天完整日线，更新候选与状态
- **不建议盘中运行正式筛选**：因为本项目明确以“完整日K线”为基准

---

## 当前限制

- 正式策略结果只使用东方财富前复权数据；新浪仅作为探测/列表兜底
- 节假日判断目前基于“周末 + 完整K线日期校验”，未接入官方交易所节假日日历接口
- 行业强度目前保留接口位置，后续可继续扩展
- 回测为基础框架，便于后续继续细化指标与标签定义
