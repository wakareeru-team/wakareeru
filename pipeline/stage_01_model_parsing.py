import httpx
import re
import json
import pandas as pd
import utils
import asyncio
import constants
logger = utils.get_logger("stage_01_model_parsing")


def fetch_wikitext(page_title: str) -> str:
    """获取页面的原始Wikitext"""
    url = "https://ja.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "revisions",
        "rvprop": "content",     # 返回原始wikitext
        "rvslots": "main",
        "format": "json",
    }
    resp = httpx.get(url, params=params)
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page["revisions"][0]["slots"]["main"]["*"]

async def _fetch_one(
    client: httpx.AsyncClient, operator_jp: str, operator_en: str, page_title: str
) -> tuple[str, str, str, str]:
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
    }
    resp = await client.get("https://ja.wikipedia.org/w/api.php", params=params)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    logger.info(f"页面：{page_title} 请求成功")
    return page_title, operator_jp, operator_en, page["revisions"][0]["slots"]["main"]["*"]

async def fetch_all(operators: list[tuple[str, str, str]]) -> dict[str, tuple[str, str, str]]:
    async with httpx.AsyncClient(headers=constants.HEADERS, timeout=30) as client:
        results = await asyncio.gather(*[_fetch_one(client, jp, en, page) for jp, en, page in operators])
    # {page_title: (operator_jp, operator_en, wikitext)}
    return {page: (jp, en, wt) for page, jp, en, wt in results}


# ================== 解析车种信息 ==================

def parse_vehicle_wikitext(lines: list[str]) -> list[dict]:
    link_re = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]')

    # 首字符允许数字，覆盖 113系/271系 等纯数字开头的 JR West/Central 型号
    series_re = re.compile(
        r'^[A-Za-z゠-ヿ一-鿿\d]'  # 首字符（含数字）
        r'[A-Za-z゠-ヿ一-鿿\-]*'   # 前缀（含连字符，如HB-E）
        r'\d+(?:系|形)?$'           # 数字结尾，可选系/形
    )

    results = []
    current_h2 = ""
    current_h3 = ""
    current_subtype = ""
    # 跳过 概要，脚注，相关项等非车种列表部分
    skip_sections = constants.WIKI_PAGE_SKIP_SECTIONS

    for line in lines:
        line = line.rstrip('\n')

        m = re.match(r'^== (.+?) ==$', line)
        if m:
            current_h2 = m.group(1)
            current_h3 = ""
            current_subtype = ""
            continue

        m = re.match(r'^=== (.+?) ===$', line)
        if m:
            current_h3 = m.group(1)
            current_subtype = ""
            continue

        if current_h2 in skip_sections:
            continue

        if not line.lstrip().startswith('*'):
            continue

        # 单星开头的粗体行（* '''xxx'''）才更新 subtype，** 及以上层级不更新
        subtype = re.match(r'^\*\s*\'\'\'(.+?)\'\'\'', line)
        if subtype:
            current_subtype = subtype.group(1).strip().strip('[]')

        for m in link_re.finditer(line):
            page  = m.group(1)
            label = m.group(2) or page
            label = label.split('・')[0].split('（')[0].strip()

            if series_re.match(label):
                results.append({
                    "series":     label,
                    "wiki_title": page,
                    "status":     constants.STATUS_MAP.get(current_h2, current_h2),
                    "type":       current_h3,
                    "subtype":    current_subtype,
                })

    return results

def canonical_vehicle_key(entry: dict) -> tuple[str, str]:
    wiki_base = entry["wiki_title"].split("#", 1)[0]
    return entry["series"], wiki_base


def score_entry(entry: dict) -> int:
    return sum(bool(entry.get(field)) for field in ["type", "subtype", "status", "wiki_title"])


def add_unique(items: list, item):
    if item not in items:
        items.append(item)
        

def _as_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return [value]


def _extend_unique(items: list, values) -> None:
    for value in _as_list(values):
        if value not in items:
            items.append(value)


def _row_quality(row: pd.Series) -> tuple:
    title = row.get("wiki_title") or ""
    full_name = row.get("full_name") or ""
    return (
        title != row.get("series"),
        len(title),
        len(full_name),
        pd.notna(row.get("subtype")),
    )




# ================== pipeline主函数 ==================

def main(config = None):
    
    # === 初始化 ===
    config = config or utils.load_pipeline_config()
    utils.init_db(config=config)
    logger = utils.get_logger("stage_01_model_parsing")
    active_operatos = config['crawler']['active_operators']
    
    
    # === 获取车型 ===
    operators = [op for op in constants.OPERATORS if op[0] in active_operatos]
    logger.info(f"正在处理的运营公司：{', '.join(op[0] for op in operators)}")
    wikitexts = asyncio.run(fetch_all(operators=operators))
    
    raw_series = []
    for page_title, (operator_jp, operator_en, wt) in wikitexts.items():
        entries = parse_vehicle_wikitext(wt.splitlines("\n"))
        for e in entries:
            e["operator_page_title"] = page_title
            e["operator_jp"] = operator_jp
            e["operator_en"] = operator_en
        raw_series.extend(entries)

    # 同一车型跨 JR 来源页合并；operator/page_title 保留为列表，不再因为重复而丢掉来源。
    merged: dict[tuple[str, str], dict] = {}
    for e in raw_series:
        key = canonical_vehicle_key(e)

        if key not in merged:
            merged[key] = {
                "series": e["series"],
                "wiki_title": e["wiki_title"],
                "status": e["status"],
                "type": e["type"],
                "subtype": e["subtype"],
                "operator_page_title": [e["operator_page_title"]],
                "operator_jp": [e["operator_jp"]],
                "operator_en": [e["operator_en"]],
                "full_name": e["wiki_title"],
            }
            continue

        current = merged[key]
        if score_entry(e) > score_entry(current):
            for field in ["wiki_title", "status", "type", "subtype"]:
                current[field] = e[field]
            current["full_name"] = e["wiki_title"]

        add_unique(current["operator_page_title"], e["operator_page_title"])
        add_unique(current["operator_jp"], e["operator_jp"])
        add_unique(current["operator_en"], e["operator_en"])

    all_series = pd.DataFrame(list(merged.values()))
    
    logger.info(f"共解析出 {len(all_series)} 个车型")
    logger.info('子车型:' + all_series['type'].value_counts().to_string())
    
    all_df = pd.DataFrame(all_series)
    #滤除对象：货车，因为多为货列连挂，难以找到单独的车辆照片，一阶段暂时跳过；客车，理由类似，一阶段保留
    excluding_types = constants.EXCLUDED_TYPES
    filtered_df = all_df[~all_df['type'].isin(excluding_types)]
    #现在滤除二级，对象为旧式营业车和事业用车
    exclduing_subtypes = constants.EXCLUDED_SUBTYPES
    filtered_df = filtered_df[~filtered_df['subtype'].isin(exclduing_subtypes)]
    excluding_statuses = constants.EXCLUDED_STATUSES
    final_df = filtered_df[~filtered_df['status'].isin(excluding_statuses)]
    final_df['type'].value_counts()
    final_df['subtype'].value_counts()


    # 按 series 去重：同一车型可能同时出现在多个 JR 来源页，保留一行并合并 operator 信息。

    duplicate_rows = final_df[final_df.duplicated(subset=["series"], keep=False)]

    merged_rows = []
    for _, group in final_df.groupby("series", sort=False):
        if len(group) == 1:
            merged_rows.append(group.iloc[0].copy())
            continue

        # 主行选信息量更高的标题；operator/page 信息从所有重复行合并。
        base = group.loc[max(group.index, key=lambda idx: _row_quality(group.loc[idx]))].copy()
        for col in ["operator_page_title", "operator_jp", "operator_en"]:
            merged = []
            for value in group[col]:
                _extend_unique(merged, value)
            base[col] = merged

        for col in ["status", "type", "subtype", "wiki_title", "full_name"]:
            if pd.isna(base[col]) or base[col] == "":
                first_valid = group[col].dropna()
                if not first_valid.empty:
                    base[col] = first_valid.iloc[0]

        merged_rows.append(base)

    final_df = pd.DataFrame(merged_rows).reset_index(drop=True)
    logger.info(f"去重前重复行 {len(duplicate_rows)} 条；去重后剩余重复 series {final_df.duplicated(subset=['series']).sum()} 条")
    
    export_df = final_df.copy()
    # 对 operator这列列表进行json序列化
    logger.info("正在将 operator 列表进行 JSON 序列化")
    for col in ["operator_page_title", "operator_jp", "operator_en"]:
        export_df[col] = export_df[col].apply(lambda v: json.dumps(v, ensure_ascii=False))
    series_list_path = utils.join_data_root(config["path"]["series_list_path"], config=config)
    series_list_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(series_list_path, index=False, encoding="utf-8")
    logger.info(f"车型列表已保存到 {series_list_path},共 {len(export_df)} 条记录")
    
    
    
    
if __name__ == "__main__":
    main()
