# run_pipeline.py
import argparse
import sys
from pathlib import Path
from pipeline.utils import get_logger, load_pipeline_config

PIPELINE_DIR = Path(__file__).resolve().parent / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import pipeline.stage_01_model_parsing as stage_01
import pipeline.stage_02_model_fixing as stage_02
import pipeline.stage_03_manifest_crawling as stage_03
import pipeline.stage_04_img_crawler as stage_04
import pipeline.stage_05_siglip_image_filtering as stage_05
import pipeline.stage_06_llm_metadata_labeling as stage_06
import pipeline.stage_07_gdino_bbox as stage_07
logger = get_logger("run_pipeline")

STAGES: dict[str, tuple[str, callable]] = { #type: ignore
    "model_parsing":          ("车辆型号解析",         stage_01.main),
    "model_fixing":           ("车型人工修正与 Commons 映射", stage_02.main),
    "manifest_crawling":      ("Commons图片关键词过滤后manifest 爬取", stage_03.main),
    "img_crawling":           ("图片爬取",            stage_04.main),
    "siglip_filter":          ("SigLIP2 image filtering", stage_05.main),
    "llm_labeling":          ("LLM 车型信息解析",    stage_06.main),
    "gdino_bbox":     ("Grounding-DINO 主体裁切与后处理", stage_07.main),
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

    return parser.parse_args()


def resolve_stages(args) -> list[str]:
    """根据参数决定要运行哪些 stage"""
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
            fn(cfg)
            logger.info("━━━ [%s] 完成 ━━━", key)
        except Exception as e:
            logger.exception("[%s] 失败，管线中止：%s", key, e)
            sys.exit(1)

    logger.info("所有阶段完成")


if __name__ == "__main__":
    main()
