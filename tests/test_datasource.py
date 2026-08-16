from ashare_scanner.datasource import EastmoneyDataSource


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
