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


EXCLUDED_TYPES = ["貨車", "客車"]

EXCLUDED_SUBTYPES = ["旧形営業用", "旧形事業用"]


OPERATOR_PREFIX = {
    "JR東日本": "JR East",
    "JR東海":   "JR Central",
    "JR西日本": "JR West",
    "JR九州":   "JR Kyushu",
    "JR北海道": "JR Hokkaido",
    "JR四国":   "JR Shikoku",
    "JR貨物":   "JR Freight",
}

_DIGRAPHS = {
    'キャ':'kya','キュ':'kyu','キョ':'kyo',
    'シャ':'sha','シュ':'shu','ショ':'sho',
    'チャ':'cha','チュ':'chu','チョ':'cho',
    'ニャ':'nya','ニュ':'nyu','ニョ':'nyo',
    'ヒャ':'hya','ヒュ':'hyu','ヒョ':'hyo',
    'ミャ':'mya','ミュ':'myu','ミョ':'myo',
    'リャ':'rya','リュ':'ryu','リョ':'ryo',
    'ギャ':'gya','ギュ':'gyu','ギョ':'gyo',
    'ジャ':'ja', 'ジュ':'ju', 'ジョ':'jo',
    'ビャ':'bya','ビュ':'byu','ビョ':'byo',
    'ピャ':'pya','ピュ':'pyu','ピョ':'pyo',
}
_SINGLE = {
    'ア':'a', 'イ':'i', 'ウ':'u', 'エ':'e', 'オ':'o',
    'カ':'ka','キ':'ki','ク':'ku','ケ':'ke','コ':'ko',
    'サ':'sa','シ':'shi','ス':'su','セ':'se','ソ':'so',
    'タ':'ta','チ':'chi','ツ':'tsu','テ':'te','ト':'to',
    'ナ':'na','ニ':'ni','ヌ':'nu','ネ':'ne','ノ':'no',
    'ハ':'ha','ヒ':'hi','フ':'fu','ヘ':'he','ホ':'ho',
    'マ':'ma','ミ':'mi','ム':'mu','メ':'me','モ':'mo',
    'ヤ':'ya','ユ':'yu','ヨ':'yo',
    'ラ':'ra','リ':'ri','ル':'ru','レ':'re','ロ':'ro',
    'ワ':'wa','ヲ':'wo','ン':'n',
    'ガ':'ga','ギ':'gi','グ':'gu','ゲ':'ge','ゴ':'go',
    'ザ':'za','ジ':'ji','ズ':'zu','ゼ':'ze','ゾ':'zo',
    'ダ':'da','デ':'de','ド':'do',
    'バ':'ba','ビ':'bi','ブ':'bu','ベ':'be','ボ':'bo',
    'パ':'pa','ピ':'pi','プ':'pu','ペ':'pe','ポ':'po',
}

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
COMMONS_HEADERS = {"User-Agent": "TrainDatasetBuilder/1.0 (research; fengyukunfyk@gmail.com)"}


# ========= STAGE 3 manifest爬取 =========

FILE_EXCLUDE_PATTERNS = (
    "interior", "inside", "seat", "seats", "seating", "reclining", "free-space",
    "cab", "cockpit", "toilet", "wc", "route map", "counter", "merchandising counter",
    "display", "lcd", "vvvf", "logo", "air cleaner", "antenna", "pantograph",
    "camera", "accident", "syanai", "車内", "運転台", "運転室", "トイレ", "便所","カメラ", "事故", "車内",
    "trainchannel",
    "運転台", "運転室", "トイレ", "便所",
    "洗面所", "洗面台", "モニター", "カウンター", "停車駅案内", "案内表示器",
    "パンタグラフ", "エアクリーナー", "集電装置", "エアコン", "クーラー",
)

CATEGORY_EXCLUDE_PATTERNS = (
    "interior", "inside", "parts", "seats", "information display", "mockup","green car"
)

SERIES_CATEGORY_EXCLUDE_PATTERNS = {
    #"E231系": ("tokyu", "tōkyū", "toei", "shibuya hikarie"),
}

POWER_TYPE_MAP = {
    "電車": "EMU",
    "新幹線電車": "EMU",
    "気動車": "DMU",
    "電気機関車": "Electric Locomotive",
    "ディーゼル機関車": "Diesel Locomotive",
    "蒸気機関車": "Steam Locomotive",
    "電気・ディーゼル両用（EDC方式）車両": "Electro-diesel Multiple Unit",
}


