from __future__ import annotations


# Diagnostic acceptance examples supplied by the user. These codes are fetched
# first and audited on every run, but never bypass strategy conditions.
REFERENCE_STOCKS = (
    ("605111", "新洁能"),
    ("603283", "赛腾股份"),
    ("600487", "亨通光电"),
    ("002922", "伊戈尔"),
    ("002645", "华宏科技"),
    ("000823", "超声电子"),
    ("603618", "杭电股份"),
    ("603601", "再升科技"),
    ("002560", "通达股份"),
    ("600105", "永鼎股份"),
    ("600961", "株冶集团"),
    ("002885", "京泉华"),
    ("002842", "翔鹭钨业"),
    ("600522", "中天科技"),
    ("600869", "远东股份"),
    ("600141", "兴发集团"),
    ("002428", "云南锗业"),
    ("002491", "通鼎互联"),
    ("002222", "福晶科技"),
    ("600498", "烽火通信"),
    ("002579", "中京电子"),
    ("002975", "博杰股份"),
    ("603773", "沃格光电"),
    ("600367", "红星发展"),
)

REFERENCE_CODES = frozenset(code for code, _ in REFERENCE_STOCKS)

# The first group is the user's current positive-label benchmark. It is used
# only for recall reporting; strategy masks never import or inspect this set.
PRIMARY_ACCEPTANCE_CODES = frozenset(code for code, _ in REFERENCE_STOCKS[:13])
