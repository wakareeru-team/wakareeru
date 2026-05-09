OPERATORS = [
    ("JR東日本", "JR East",    "JR東日本の車両形式"),
    ("JR東海",   "JR Central", "JR東海の車両形式"),
    ("JR西日本", "JR West",    "JR西日本の車両形式"),
    ('JR北海道', "JR Hokkaido", "JR北海道の車両形式"),
    ("JR四国",   "JR Shikoku", "JR四国の車両形式"),
    ('JR九州',   "JR Kyushu",   "JR九州の車両形式"),
    ("JR貨物", "JR Freight", "JR貨物の車両形式"),
]

HEADERS = {
    "User-Agent": "JapaneseTrainDatasetBuilder/1.0 (research project; fengyukunfyk@gmail.com)"
}

WIKI_PAGE_SKIP_SECTIONS = {"概要", "2010年4月1日時点の在籍貨車", "2010年度以降に新製が発表された貨車", "2009年度までに消滅した貨車", # JR货物用的排除项
                    "脚注", "注釈", "出典", "関連項目", "外部リンク"}

STATUS_MAP = {
    "現在の所属車両":   "現役",
    "過去の所属車両":   "廃止",
    "在来線現有車両":   "現役",
    "在来線廃止車両":   "廃止",
    "導入予定車両":     "導入予定",
}