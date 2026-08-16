from datetime import date

from ashare_scanner.cache import HistoryCache


def test_history_cache_validates_adjustment_metadata(tmp_path, trending_history):
    expected = trending_history["date"].max().date()
    cache = HistoryCache(tmp_path, "2022-01-01", 1)
    cache.write("000001", trending_history, "eastmoney:test", "stock", expected)
    assert cache.read("000001", expected, "stock").fresh

    incompatible = HistoryCache(tmp_path, "2022-01-01", 2)
    read = incompatible.read("000001", expected, "stock")
    assert not read.fresh
    assert read.reason == "incompatible_metadata"


def test_history_cache_rejects_stale_last_date(tmp_path, trending_history):
    last = trending_history["date"].max().date()
    cache = HistoryCache(tmp_path, "2022-01-01", 1)
    cache.write("000001", trending_history, "eastmoney:test", "stock", last)
    read = cache.read("000001", date.fromordinal(last.toordinal() + 1), "stock")
    assert not read.fresh
    assert read.reason == "stale_date"


def test_history_cache_accepts_tencent_forward_adjusted_source(tmp_path, trending_history):
    expected = trending_history["date"].max().date()
    cache = HistoryCache(tmp_path, "2022-01-01", 1)
    cache.write(
        "000001",
        trending_history,
        "tencent:test:turnover_from_quote",
        "stock",
        expected,
    )
    assert cache.read("000001", expected, "stock").fresh


def test_history_cache_accepts_baostock_forward_adjusted_source(tmp_path, trending_history):
    expected = trending_history["date"].max().date()
    cache = HistoryCache(tmp_path, "2022-01-01", 1)
    cache.write("000001", trending_history, "baostock:qfq", "stock", expected)
    assert cache.read("000001", expected, "stock").fresh
