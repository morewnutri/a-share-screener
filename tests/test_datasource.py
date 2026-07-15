from ashare_scanner.datasource import EastmoneyDataSource


class FakeHttp:
    def __init__(self):
        self.params = None

    def get_json(self, url, params, referer):
        self.params = params
        items = [
            {"f12": "002001", "f14": "sample-a", "f13": 0},
            {"f12": "003001", "f14": "sample-b", "f13": 0},
            {"f12": "688001", "f14": "star", "f13": 1},
            {"f12": "600001", "f14": "ST sample", "f13": 1},
        ]
        return {"data": {"diff": items, "total": len(items)}}


def test_universe_snapshot_is_stable_and_includes_002_003():
    http = FakeHttp()
    source = EastmoneyDataSource(http, fqt=1)
    frame, label = source.fetch_universe(min_size=2)
    assert frame["code"].tolist() == ["002001", "003001"]
    assert http.params["fid"] == "f12"
    assert http.params["pz"] == 10_000
    assert label.endswith(":snapshot")


def test_sina_quote_parser_extracts_code_and_name():
    text = 'var hq_str_sz002001="sample,10,10,10.2,10.3,9.9,0,0,123,456";\n'
    rows = EastmoneyDataSource._parse_sina_quotes(text)
    assert rows[0]["code"] == "002001"
    assert rows[0]["name"] == "sample"
    assert rows[0]["latest"] == 10.2
