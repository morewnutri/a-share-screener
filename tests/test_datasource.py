import pandas as pd
import pytest
from types import SimpleNamespace

from ashare_scanner.datasource import BaoStockDataSource, EastmoneyDataSource


class FakeHttp:
    def __init__(self):
        self.params = None

    def get_json(self, url, params, referer):
        self.params = params
        items = [
            {"f12": "002001", "f14": "sample-a", "f13": 0, "f62": 1000, "f184": 22.5},
            {"f12": "003001", "f14": "sample-b", "f13": 0},
            {"f12": "688001", "f14": "star", "f13": 1},
            {"f12": "600001", "f14": "ST sample", "f13": 1},
            {"f12": "600002", "f14": "退市 sample", "f13": 1},
        ]
        return {"data": {"diff": items, "total": len(items)}}


def test_universe_snapshot_is_stable_and_includes_optional_flow_fields():
    http = FakeHttp()
    source = EastmoneyDataSource(http, fqt=1)
    frame, label = source.fetch_universe(min_size=2)
    assert frame["code"].tolist() == ["002001", "003001"]
    assert frame.loc[0, "main_net_inflow_amount"] == 1000
    assert frame.loc[0, "main_net_inflow_ratio_pct"] == 22.5
    assert "f62" in http.params["fields"]
    assert "f184" in http.params["fields"]
    assert http.params["fid"] == "f12"
    assert http.params["pz"] == 10_000
    assert label.endswith(":snapshot")


def test_sina_quote_parser_extracts_code_and_name():
    text = 'var hq_str_sz002001="sample,10,10,10.2,10.3,9.9,0,0,123,456";\n'
    rows = EastmoneyDataSource._parse_sina_quotes(text)
    assert rows[0]["code"] == "002001"
    assert rows[0]["name"] == "sample"
    assert rows[0]["latest"] == 10.2
    assert "main_net_inflow_ratio_pct" in rows[0]


def test_fund_flow_parser_uses_main_inflow_fields():
    line = "2026-08-14,1000,-10,-20,400,600,5,-1,-2,2,3,12.3,1.2,0,0"
    frame = EastmoneyDataSource._parse_fund_flow([line], "600001")
    assert frame.loc[0, "code"] == "600001"
    assert frame.loc[0, "main_net_inflow_amount"] == 1000
    assert frame.loc[0, "main_net_inflow_ratio_pct"] == 5
    assert frame.loc[0, "close"] == 12.3


def test_fund_flow_request_uses_history_endpoint_and_limit():
    class FakeFundHttp:
        def __init__(self):
            self.url = ""
            self.params = {}

        def get_json(self, url, params, referer):
            self.url = url
            self.params = params
            line = "2026-08-14,1000,-10,-20,400,600,5,-1,-2,2,3,12.3,1.2,0,0"
            return {"data": {"klines": [line]}}

    http = FakeFundHttp()
    frame, source = EastmoneyDataSource(http).fetch_stock_fund_flow("600001", limit=80)
    assert len(frame) == 1
    assert "/api/qt/stock/fflow/daykline/get" in http.url
    assert http.params["lmt"] == 80
    assert http.params["secid"] == "1.600001"
    assert source.endswith(":fund_flow")


def test_tencent_kline_parser_infers_historical_turnover_from_quote():
    quote = [""] * 39
    quote[36] = "1000"
    quote[38] = "2.0"
    klines = [
        ["2026-08-13", "9.8", "10.0", "10.2", "9.7", "500"],
        ["2026-08-14", "10.0", "10.2", "10.3", "9.9", "1000"],
    ]

    frame, turnover_source = EastmoneyDataSource._parse_tencent_klines(
        klines,
        "603893",
        quote=quote,
    )

    assert turnover_source == "turnover_from_quote"
    assert frame["turnover"].round(2).tolist() == [1.0, 2.0]
    assert frame.loc[1, "amount"] > 1_000_000
    assert frame.loc[1, "code"] == "603893"


def test_stock_history_prefers_tencent_qfq_endpoint():
    class FakeTencentHttp:
        def __init__(self):
            self.url = ""
            self.params = {}

        def get_json(self, url, params, referer):
            self.url = url
            self.params = params
            quote = [""] * 39
            quote[36] = "1000"
            quote[38] = "2"
            return {
                "data": {
                    "sh603893": {
                        "qfqday": [
                            ["2026-08-13", "9.8", "10", "10.2", "9.7", "500"],
                            ["2026-08-14", "10", "10.2", "10.3", "9.9", "1000"],
                        ],
                        "qt": {"sh603893": quote},
                    }
                }
            }

    http = FakeTencentHttp()
    frame, source = EastmoneyDataSource(http).fetch_stock_history(
        "603893",
        "2025-01-01",
        pd.Timestamp("2026-08-14").date(),
    )

    assert len(frame) == 2
    assert source.startswith("tencent:")
    assert "2026-08-14" in http.params["param"]


def test_history_providers_stop_retrying_after_consecutive_stock_failures():
    class AlwaysFailHttp:
        def __init__(self):
            self.calls = 0

        def get_json(self, url, params, referer):
            self.calls += 1
            raise RuntimeError("blocked")

    http = AlwaysFailHttp()
    source = EastmoneyDataSource(http)
    end = pd.Timestamp("2026-08-14").date()
    for index in range(13):
        with pytest.raises(RuntimeError):
            source.fetch_stock_history(f"600{index:03d}", "2025-01-01", end)

    # Twelve stocks try 2 Tencent and 5 Eastmoney hosts; the 13th fails fast.
    assert http.calls == 12 * 7


class FakeBaoResult:
    error_code = "0"
    error_msg = "success"

    def __init__(self, frame):
        self.frame = frame

    def get_data(self):
        return self.frame.copy()


class FakeBaoStock:
    def __init__(self):
        self.history_args = None
        self.history_kwargs = None

    def login(self):
        return SimpleNamespace(error_code="0", error_msg="success")

    def logout(self):
        return SimpleNamespace(error_code="0", error_msg="success")

    def query_history_k_data_plus(self, *args, **kwargs):
        self.history_args = args
        self.history_kwargs = kwargs
        return FakeBaoResult(
            pd.DataFrame(
                [
                    {
                        "date": "2026-08-13",
                        "code": "sh.600001",
                        "open": "9.8",
                        "high": "10.2",
                        "low": "9.7",
                        "close": "10.0",
                        "preclose": "9.8",
                        "volume": "1000000",
                        "amount": "10000000",
                        "turn": "2.5",
                        "tradestatus": "1",
                        "pctChg": "2.04",
                        "isST": "0",
                    }
                ]
            )
        )

    def query_all_stock(self, expected):
        return FakeBaoResult(
            pd.DataFrame(
                [
                    {"code": "sh.000300", "tradeStatus": "1", "code_name": "index"},
                    {"code": "sh.600001", "tradeStatus": "1", "code_name": "sample-a"},
                    {"code": "sz.002001", "tradeStatus": "1", "code_name": "sample-b"},
                    {"code": "sz.300001", "tradeStatus": "1", "code_name": "chinext"},
                ]
            )
        )


def test_baostock_history_uses_forward_adjustment_and_reported_turnover():
    module = FakeBaoStock()
    source = BaoStockDataSource(module=module)
    frame, label = source.fetch_stock_history(
        "600001",
        "2026-01-01",
        pd.Timestamp("2026-08-14").date(),
    )

    assert module.history_args[0] == "sh.600001"
    assert module.history_kwargs["adjustflag"] == "2"
    assert frame.loc[0, "turnover"] == 2.5
    assert frame.loc[0, "volume"] == 1_000_000
    assert label == "baostock:qfq"


def test_baostock_universe_excludes_indices_and_non_mainboard_stocks():
    source = BaoStockDataSource(module=FakeBaoStock())
    frame, label = source.fetch_universe(pd.Timestamp("2026-08-14").date(), min_size=2)
    assert frame["code"].tolist() == ["002001", "600001"]
    assert label == "baostock:query_all_stock"
