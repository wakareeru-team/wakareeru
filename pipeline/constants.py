# Pipeline控制
STAGE_COMPLETED = 0
STAGE_INTERRUPT = 1
STAGE_PASS = 2


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


# ========= Pipeline enums =========

MANUAL_ACTION_SET = "set"
MANUAL_ACTION_KEEP = "keep"
MANUAL_ACTION_EXCLUDE = "exclude"
MANUAL_ACTIONS = {MANUAL_ACTION_SET, MANUAL_ACTION_KEEP, MANUAL_ACTION_EXCLUDE}

CATEGORY_SOURCE_SCOPE_ROOT = "root"
CATEGORY_SOURCE_SCOPE_RECURSIVE = "recursive"
CATEGORY_SOURCE_SCOPES = {
    CATEGORY_SOURCE_SCOPE_ROOT,
    CATEGORY_SOURCE_SCOPE_RECURSIVE,
}

FETCH_STATUS_PENDING = "pending"
FETCH_STATUS_OK = "ok"
FETCH_STATUS_ERROR = "error"
FETCH_STATUSES = {
    FETCH_STATUS_PENDING,
    FETCH_STATUS_OK,
    FETCH_STATUS_ERROR,
}

DOWNLOAD_STATUS_NOT_STARTED = "not_started"
DOWNLOAD_STATUS_DOWNLOADED = "downloaded"
DOWNLOAD_STATUS_FAILED = "failed"
DOWNLOAD_STATUS_MISSING_URL = "missing_url"
DOWNLOAD_STATUSES = {
    DOWNLOAD_STATUS_NOT_STARTED,
    DOWNLOAD_STATUS_DOWNLOADED,
    DOWNLOAD_STATUS_FAILED,
    DOWNLOAD_STATUS_MISSING_URL,
}

CROP_STATUS_PENDING = "pending"
CROP_STATUS_OK = "ok"
CROP_STATUS_REJECTED = "rejected"
CROP_STATUSES = {
    CROP_STATUS_PENDING,
    CROP_STATUS_OK,
    CROP_STATUS_REJECTED,
}

NOISE_REVIEW_LABELS = [
    "ok",
    "wrong_label",
    "out_of_label_space",
    "bad_crop",
    "ambiguous",
]

NOISE_REVIEW_LABEL_OK = "ok"
NOISE_REVIEW_LABEL_WRONG_LABEL = "wrong_label"
NOISE_REVIEW_LABEL_OUT_OF_LABEL_SPACE = "out_of_label_space"
NOISE_REVIEW_LABEL_BAD_CROP = "bad_crop"
NOISE_REVIEW_LABEL_AMBIGUOUS = "ambiguous"


# ========= STAGE 3 manifest爬取 =========

FILE_INTERIOR_PATTERNS = (
    "interior", "inside", "seat", "seats", "seating", "reclining", "free-space",
    "cab", "cockpit", "toilet", "wc", "route map", "counter", "merchandising counter",
    "display", "lcd", "trainchannel", "syanai", "camera",
    "車内", "運転台", "運転室", "トイレ", "便所", "洗面所", "洗面台",
    "モニター", "カウンター", "停車駅案内", "案内表示器", "カメラ", 'bus', 'Bus', 'バス', 
    'breakfast', 'lunch', 'dinner', 'bento', 'food', 'meal', 'cafe', 'café', 'catering',
    '駅弁', '食事', '飲食', 'カフェ', 'ケータリング', '朝ご飯', '昼ご飯', '夜ご飯', '弁当', '昼めし'
)

FILE_DETAIL_PATTERNS = (
    "vvvf", "logo", "air cleaner", "antenna", "pantograph", "accident",
    "パンタグラフ", "エアクリーナー", "集電装置", "エアコン", "クーラー", "事故",
)

CATEGORY_EXCLUDE_PATTERNS = (
    "interior", "inside", "parts", "seats", "information display", "mockup","green car", 'bus', 'Bus', 'バス'
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



# 图片过滤相关的常量
SIGLIP_VIEW_CANDIDATES = [
    "an image of the interior of a train",
    "an image of whole exterior view of a train",
    "an image of a detailed close-up of the exterior part of a train",
    "an image of information monitor or map of a train"
]
SIGLIP_PROMPT_TO_LABEL = {
    "an image of the interior of a train": "interior",
    "interior": "interior",
    "an image of whole exterior view of a train": "exterior",
    "an image of information monitor or map of a train": "display",
    "exterior": "exterior",
    "display": "display",
    "an image of a detailed close-up of the exterior part of a train": "detailed",
    "detailed": "detailed",
    "uncertain":"uncertain"
}

KEEP_LABELS = {"exterior", "uncertain"}



# ================ LLM解析车型信息 ================

LLM_LABEL_DETAIL_PROMPT = """
你是一个日本铁路图片分类助手，通过Wikipedia Commons图片的分类路径来判断图片所属铁路车辆的各种信息，包括子车型，番台，特殊编成，特殊涂装，运行线路，运营公司等。
请在必要时通过web_search查找相关信息辅助判断，不要随意猜测。
你将会收到category_path（一个字符串列表，表示图片在Wikipedia Commons上的分类路径）。请根据这些信息判断以下的内容，若不满足条件或没有该项相关内容请直接留空。
不需要做任何补充说明，直接输出JSON，不输出任何解释与备注。
- 子车型：一些较大型号车型家族下的细分型号，注意不是番台。例如，kiha 40系列下有kiha 40, kiha 47, kiha 147等车型。这一项输出请不要带任何series等后缀，也不要加上运营商前缀例如JRF。直接输出子车型的名称，若是番台区分则带完整车型，例如kiha 147，E231-1000等。罗马音写的表记请保持只有首字母大写，例如文件中可能的KiHa也写成Kiha。如果没有明确的子车型信息请复制其原车型，不要随意猜测，也不要直接写机车型号。
- 番台：某类车型的细分型号，一般会标出番台，例如JR东日本的E231-1000系列中有E231-1000番台和E231-3000番台。但请不要输出机车的车号，例如C57 180号机或者 DD51 1043号。有些图片也会以车号给出，例如E231-517号车，此时则为500番台。但请注意番台并不一定是百位千位，若不确定请查询相关信息，例如该车型的番台列表，然后判断。这里**只写番台本身且写成TEXT格式**，例如'0'，'100'，'8000'，不写前缀也不带番台后缀。
- 运营公司：例如国鉄，JR东日本，JR西日本，JR东海，JR北海道，JR九州，JR四国等。可以从category路径的备注中和文件名看出，请输出其英文名及日文名，例如JR East/JR東日本, JR West/JR西日本, JR Central/JR東海, JR Hokkaido/JR北海道, JR Kyushu/JR九州, JR Shikoku/JR四国, JNR/国鉄等。
- 特殊编成：以特殊名称命名的一些编成，例如etSETOra，Resort Shirakami Aoike。由于文件是由英语写就，请直接输出罗马字或英文标记的名字。不要写入与车辆特殊编成无关的特急或优等列车爱称，例如'Odoriko'.
- 特殊涂装：一些车型拥有的特殊涂装，常见以"livery"在路径或文件名中出现，例如"Hokutosei livery"，也请输入日文原名，例如北斗星，不要带涂装等字样。

关于子车型和番台的判断和填入格式的例子：
- E231系，但没有给出任何番台信息，submodel: "E231", bandai: ""
- E231系0番台，submodel: "E231", bandai: "0"
- E231-2000番台，submodel: "E231-2000", bandai: "2000"
- Kiha 147, 因为他是kiha 40的改造型，所以submodel: "Kiha 147", bandai: ""
- C57 1号机,不是子车型也不是番台 submodel: "C57", bandai: ""
- 新干线E5系没有子车型, submodel: "E5", bandai: ""
- 415系，在wiki下被归为113系，但他应该为新的子车型，submodel: "415", bandai: ""

以上内容请输出为JSON数组，每个元素格式如下：
{"submodel": <车型>, "bandai": <番台>, "operator_en": <运营公司英文名>, "operator_jp": <运营公司日文名>, "special_formation": <特殊编成>, "special_livery": <特殊涂装>}
"""

# ================ Crop过滤 SigLIP相关常量 ================
# 图片过滤相关的常量
SIGLIP_CROP_FILTER_CANDIDATES = [
    "an image of a train",
    "a photo without a train",
    "a photo of station signs or route maps",
    "a photo of a statue",
    "a photo of a decoration board"
]
SIGLIP_CROP_PROMPT_TO_LABEL = {
    "an image of a train": "train",
    "train": "train",
    "a photo without a train": "no_train",
    "no_train": "no_train",
    "a photo of station signs or route maps": "no_train",
    "station_signs": "no_train",
    "a photo of a statue": "no_train",
    "statue": "no_train",
    "a photo of a decoration board": "no_train",
    "decoration_board": "no_train",
}


LABEL_ASCII_REPLACEMENTS = (
    ("非貫通型", " non kantsu "),
    ("非貫通", " non kantsu "),
    ("貫通型", " kantsu "),
    ("貫通", " kantsu "),
    ("キハ", " kiha "),
    ("クモハ", " kumoha "),
    ("クハ", " kuha "),
    ("モハ", " moha "),
    ("サハ", " saha "),
    ("デハ", " deha "),
    ("ロハ", " roha "),
    ("サロ", " saro "),
    ("モロ", " moro "),
    ("カニ", " kani "),
    ("番台", " "),
    ("号機", " "),
    ("系", " "),
    ("形", " "),
    ("型", " "),
)