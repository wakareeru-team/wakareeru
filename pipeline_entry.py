# run_pipeline.py
import argparse
import sys
from pathlib import Path
from pipeline.constants import STAGE_COMPLETED, STAGE_INTERRUPT, STAGE_PASS
from pipeline.utils import get_logger, load_pipeline_config

PIPELINE_DIR = Path(__file__).resolve().parent / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import pipeline.stage_01_model_parsing as stage_01  # noqa: E402
import pipeline.stage_02_model_fixing as stage_02  # noqa: E402
import pipeline.stage_03_manifest_crawling as stage_03  # noqa: E402
import pipeline.stage_04_img_crawler as stage_04  # noqa: E402
import pipeline.stage_05_siglip_image_filtering as stage_05  # noqa: E402
import pipeline.stage_06_llm_metadata_labeling as stage_06  # noqa: E402
import pipeline.stage_07_gdino_bbox as stage_07  # noqa: E402
import pipeline.stage_08_fine_grain_series as stage_08  # noqa: E402
import pipeline.stage_09_DINOv3_feature_extraction as stage_09  # noqa: E402
import pipeline.stage_10_train_loss_tracking as stage_10  # noqa: E402
import pipeline.stage_11_loss_analysis as stage_11  # noqa: E402
import pipeline.stage_12_logistic_regression_filter as stage_12  # noqa: E402
logger = get_logger("run_pipeline")

STAGES: dict[str, tuple[str, callable]] = { #type: ignore
    "model_parsing":          ("车辆型号解析",         stage_01.main),
    "model_fixing":           ("车型人工修正与 Commons 映射", stage_02.main),
    "manifest_crawling":      ("Commons图片关键词过滤后manifest 爬取", stage_03.main),
    "img_crawling":           ("图片爬取",            stage_04.main),
    "siglip_filter":          ("SigLIP2 image filtering", stage_05.main),
    "llm_labeling":          ("LLM 车型信息解析",    stage_06.main),
    "gdino_bbox":     ("Grounding-DINO 主体裁切与后处理", stage_07.main),
    "fine_grain_series":     ("细粒度车型标签构造", stage_08.main), # 独立出来是因为它直接影响后续的label空间和模型训练，调整后后面的feature分类和模型训练都要重跑
    "feature_extraction":    ("DINOv3 特征提取",     stage_09.main),
    "loss_tracking":         ("训练与损失跟踪",     stage_10.main),
    "loss_analysis":         ("损失分析",          stage_11.main),
    "logistic_regression_filter": ("基于人工标记的logistic regression过滤", stage_12.main),
}
STAGE_KEYS = list(STAGES.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wakareeru Railway Dataset 数据管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {i+1}. {k:<20} {desc}"
            for i, (k, (desc, _)) in enumerate(STAGES.items())
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="pipeline.yaml 路径，默认使用 config/pipeline.yaml",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only", type=str, metavar="STAGE",
        choices=STAGE_KEYS,
        help="只运行指定阶段",
    )
    group.add_argument(
        "--from", type=str, metavar="STAGE", dest="from_stage",
        choices=STAGE_KEYS,
        help="从指定阶段开始运行到最后",
    )
    group.add_argument(
        "--stages",
        type=str,
        metavar="STAGES",
        help=(
            "按编号运行阶段，支持单个编号、多个编号或范围，"
            "例如：--stages '3'、--stages '1 2 3'、--stages '1-8 10'"
        ),
    )

    args = parser.parse_args()
    if args.stages:
        try:
            resolve_stage_number_spec(args.stages)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def parse_stage_number(token: str) -> int:
    try:
        number = int(token)
    except ValueError as exc:
        raise ValueError(f"阶段编号必须是整数：{token!r}") from exc
    if not 1 <= number <= len(STAGE_KEYS):
        raise ValueError(f"阶段编号超出范围：{number}，可用范围是 1-{len(STAGE_KEYS)}")
    return number


def resolve_stage_number_spec(spec: str) -> list[str]:
    """Resolve a stage number spec like ``1-3`` or ``1 2 3``."""
    stage_numbers: list[int] = []
    for token in spec.split():
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start = parse_stage_number(start_text)
            end = parse_stage_number(end_text)
            step = 1 if start <= end else -1
            stage_numbers.extend(range(start, end + step, step))
        else:
            stage_numbers.append(parse_stage_number(token))

    seen: set[int] = set()
    ordered_numbers = []
    for number in stage_numbers:
        if number in seen:
            continue
        seen.add(number)
        ordered_numbers.append(number)

    return [STAGE_KEYS[number - 1] for number in ordered_numbers]


def resolve_stages(args) -> list[str]:
    """根据参数决定要运行哪些 stage"""
    if args.stages:
        return resolve_stage_number_spec(args.stages)
    if args.only:
        return [args.only]
    if args.from_stage:
        start = STAGE_KEYS.index(args.from_stage)
        return STAGE_KEYS[start:]
    return STAGE_KEYS   # 默认全跑


def main():
    args = parse_args()
    cfg  = load_pipeline_config(args.config)

    stages_to_run = resolve_stages(args)

    logger.info("管线启动，将运行 %d 个阶段：%s", len(stages_to_run), stages_to_run)

    for key in stages_to_run:
        desc, fn = STAGES[key]
        logger.info("━━━ [%s] %s 开始 ━━━", key, desc)
        try:
            returned = fn(cfg)
            if returned is None:
                logger.info("━━━ [%s] 完成 ━━━", key)
            elif returned == STAGE_COMPLETED:
                logger.info("━━━ [%s] 完成 ━━━", key)
            elif returned == STAGE_INTERRUPT:
                logger.warning("━━━ [%s] 中断，跳过后续阶段 ━━━", key)
                break
            elif returned == STAGE_PASS:
                logger.warning("━━━ [%s] 跳过，后续阶段继续 ━━━", key)   
        except Exception as e:
            logger.exception("[%s] 失败，管线中止：%s", key, e)
            sys.exit(1)

    logger.info("所有阶段完成")


if __name__ == "__main__":
    main()
