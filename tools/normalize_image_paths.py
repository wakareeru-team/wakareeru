from __future__ import annotations

"""Normalize downloaded image paths to Unicode NFC.

This maintenance tool fixes cross-platform filename normalization drift,
especially paths copied from macOS to Linux where visually identical names can
use different Unicode byte sequences. It normalizes both the files under
``path.raw_img_dir`` and ``images.downloaded_path`` in SQLite.
"""

import argparse
import sys
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Project root not found from {start}")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_examples(title: str, rows: list, limit: int) -> None:
    if not rows:
        return
    print(title)
    for row in rows[:limit]:
        print(f"  {row}")
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize downloaded image filenames and DB paths to Unicode NFC."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline_config.yaml. Defaults to config/pipeline_config.yaml.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply filesystem renames and DB updates. Without this flag, "
            "only reports planned changes."
        ),
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Maximum examples to print per section.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from pipeline import path_normalization, utils

    config = utils.load_pipeline_config(args.config)
    utils.init_db(config=config)
    data_root = utils.get_data_root(config)
    img_root = utils.join_data_root(config["path"]["raw_img_dir"], config=config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)

    rename_plan = path_normalization.collect_rename_plan(img_root)
    db_updates = path_normalization.collect_db_updates(db_path)
    missing_paths = path_normalization.collect_missing_downloaded_paths(
        db_path,
        data_root=data_root,
        img_root=img_root,
    )

    print(f"Data root: {data_root}")
    print(f"Image root: {img_root}")
    print(f"Database: {db_path}")
    print(f"Mode: {'apply' if args.apply else 'dry-run'}")
    print(f"Filesystem paths to normalize: {len(rename_plan)}")
    print(f"DB downloaded_path rows to normalize: {len(db_updates)}")
    print(f"Downloaded DB paths missing after NFC normalization: {len(missing_paths)}")

    print_examples(
        "Filesystem rename examples:",
        [f"{plan.source} -> {plan.target}" for plan in rename_plan],
        args.sample_limit,
    )
    print_examples(
        "DB update examples:",
        [f"id={update.image_id}: {update.old_path} -> {update.new_path}" for update in db_updates],
        args.sample_limit,
    )
    print_examples(
        "Missing downloaded image examples:",
        [f"id={image_id}: {path}" for image_id, path in missing_paths],
        args.sample_limit,
    )

    if not args.apply:
        print("Dry run only. Re-run with --apply to write changes.")
        return

    path_normalization.normalize_downloaded_image_paths(db_path=db_path, img_root=img_root)

    missing_after_apply = path_normalization.collect_missing_downloaded_paths(
        db_path,
        data_root=data_root,
        img_root=img_root,
    )
    print(f"Applied filesystem renames: {len(rename_plan)}")
    print(f"Applied DB updates: {len(db_updates)}")
    print(f"Downloaded DB paths still missing after apply: {len(missing_after_apply)}")
    print_examples(
        "Still missing downloaded image examples:",
        [f"id={image_id}: {path}" for image_id, path in missing_after_apply],
        args.sample_limit,
    )


if __name__ == "__main__":
    main()
