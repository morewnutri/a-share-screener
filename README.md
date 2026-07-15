# A-Share Screener

面向 A 股主板股票的工程化筛选系统，重点解决以下问题：

- 主板股票池完整性校验（含 000/001/002/003, 600/601/603/605）
- 统一的前复权日线数据口径（正式策略仅使用可比数据）
- 缓存与元数据管理，避免过期/错误缓存污染结果
- setup / breakout / retest 多阶段候选分类
- 相对强度、波动收缩、流动性、陈旧数据过滤
- 候选状态跟踪与基础回测框架
- Google Colab 运行脚本

## 目录

- `config.py`：全局配置
- `calendar_utils.py`：交易日与完整 K 线判断
- `cache_manager.py`：缓存与元数据
- `data_sources.py`：东方财富 / 新浪 / 指数数据源适配
- `universe.py`：股票池构建与交叉校验
- `indicators.py`：技术指标
- `features.py`：特征构建
- `filters.py`：过滤逻辑
- `strategies.py`：候选分类与评分
- `state_tracker.py`：观察池状态管理
- `intraday_monitor.py`：盘中触发器骨架
- `backtest.py`：滚动回测框架
- `report.py`：输出报表
- `main.py`：主入口
- `scripts/run_colab.sh`：Colab 一键运行脚本
- `scripts/colab_setup.py`：Colab 环境准备脚本

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

## Google Colab

可在 Colab 中执行：

```bash
bash scripts/run_colab.sh
```

或在 notebook 中：

```python
!python scripts/colab_setup.py
!python main.py
```

## 说明

- 正式日线策略只使用 **东方财富前复权** 且口径可比较的数据。
- 新浪数据仅作为兜底探测或辅助，不进入正式评分结果。
- 当前盘中触发器和行业强度为工程骨架，可继续扩展更细粒度数据源。
