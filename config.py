# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Config:
    start_date: str = "20240101"
    end_date: str = datetime.now().strftime("%Y%m%d")
    market_close_time: str = "15:10"

    output_dir: str = "output"
    cache_dir: str = "cache"
    list_cache_dir: str = os.path.join("cache", "list")
    hist_cache_dir: str = os.path.join("cache", "hist")
    state_dir: str = os.path.join("cache", "state")

    fqt: int = 1
    min_history_bars: int = 120
    top_n: int = 100
    exclude_st: bool = True
    save_excel: bool = True

    request_timeout: int = 12
    max_retries: int = 2
    max_workers: int = 12
    sleep_seconds: float = 0.02

    cache_expire_hours: int = 8
    list_cache_expire_hours: int = 8
    force_refresh_list: bool = False
    force_refresh_hist: bool = False

    eastmoney_ut: str = "bd1d9ddb04089700cf9c27f6f7426281"
    mainboard_prefixes: tuple[str, ...] = (
        "600", "601", "603", "605",
        "000", "001", "002", "003",
    )

    min_amount_ma20: float = 2e8
    min_price: float = 3.0
    max_extension_ma20_pct: float = 12.0
    max_return_10d_pct_for_setup: float = 18.0
    min_rs_percentile: float = 0.70
    setup_expire_days: int = 7

    eastmoney_push2_hosts: list[str] = field(default_factory=lambda: [
        "https://push2.eastmoney.com",
        "https://7.push2.eastmoney.com",
        "https://19.push2.eastmoney.com",
        "https://80.push2.eastmoney.com",
    ])
    eastmoney_push2his_hosts: list[str] = field(default_factory=lambda: [
        "https://push2his.eastmoney.com",
        "https://7.push2his.eastmoney.com",
        "https://19.push2his.eastmoney.com",
    ])
    user_agents: list[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    ])

    eastmoney_fs: str = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
    benchmark_index_code: str = "000001"


CONFIG = Config()
