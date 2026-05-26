from __future__ import annotations

"""Normalize downloaded image paths to Unicode NFC.

This maintenance tool fixes cross-platform filename normalization drift,
especially paths copied from macOS to Linux where visually identical names can
use different Unicode byte sequences. It normalizes both the files under
``path.raw_img_dir`` and ``images.downloaded_path`` in SQLite.
"""

import argparse
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass
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


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def normalize_rel_path(value: str) -> str:
    return "/".join(normalize_text(part) for part in value.replace("\\", "/").split("/"))


@dataclass(frozen=True)
class RenamePlan:
    source: Path
    target: Path


@dataclass(frozen=True)
class DbUpdate:
    image_id: int
    old_path: str
    new_path: str


def collect_rename_plan(root: Path) -> list[RenamePlan]:
    if not root.exists():
        return []

    plans: list[RenamePlan] = []
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        normalized_name = normalize_text(path.name)
        if normalized_name == path.name:
            continue
        plans.append(RenamePlan(source=path, target=path.with_name(normalized_name)))
    return plans


def validate_rename_plan(plans: list[RenamePlan]) -> None:
    planned_sources = {plan.source for plan in plans}
    planned_targets: set[Path] = set()
    for plan in plans:
        if plan.target in planned_targets:
            raise FileExistsError(f"multiple paths would normalize to {plan.target}")
        planned_targets.add(plan.target)
        if plan.target.exists() and plan.target not in planned_sources:
            raise FileExistsError(f"normalization collision: {plan.source} -> {plan.target}")


def apply_rename_plan(plans: list[RenamePlan]) -> None:
    validate_rename_plan(plans)
    for plan in plans:
        if not plan.source.exists():
            raise FileNotFoundError(f"source path disappeared before rename: {plan.source}")
        plan.source.rename(plan.target)


def collect_db_updates(db_path: Path) -> list[DbUpdate]:
    updates: list[DbUpdate] = []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, downloaded_path
            FROM images
            WHERE downloaded_path IS NOT NULL
              AND downloaded_path != ''
            ORDER BY id
            """
        ).fetchall()

    for image_id, old_path in rows:
        new_path = normalize_rel_path(str(old_path))
        if new_path != old_path:
            updates.append(DbUpdate(image_id=int(image_id), old_path=str(old_path), new_path=new_path))
    return updates


def apply_db_updates(db_path: Path, updates: list[DbUpdate]) -> None:
    if not updates:
        return
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            UPDATE images
            SET downloaded_path = ?
            WHERE id = ?
            """,
            [(update.new_path, update.image_id) for update in updates],
        )
        conn.commit()


def normalized_existing_image_paths(data_root: Path, img_root: Path) -> set[str]:
    if not img_root.exists():
        return set()
    paths: set[str] = set()
    for path in img_root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(data_root).as_posix()
        paths.add(normalize_rel_path(rel_path))
    return paths


def collect_missing_downloaded_paths(db_path: Path, data_root: Path, img_root: Path) -> list[tuple[int, str]]:
    existing_paths = normalized_existing_image_paths(data_root=data_root, img_root=img_root)
    missing: list[tuple[int, str]] = []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, downloaded_path
            FROM images
            WHERE download_status = 'downloaded'
              AND downloaded_path IS NOT NULL
              AND downloaded_path != ''
            ORDER BY id
            """
        ).fetchall()

    for image_id, downloaded_path in rows:
        normalized_path = normalize_rel_path(str(downloaded_path))
        if normalized_path not in existing_paths:
            missing.append((int(image_id), normalized_path))
    return missing


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
        help="Apply filesystem renames and DB updates. Without this flag, only reports planned changes.",
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

    from pipeline import utils

    config = utils.load_pipeline_config(args.config)
    utils.init_db(config=config)
    data_root = utils.get_data_root(config)
    img_root = utils.join_data_root(config["path"]["raw_img_dir"], config=config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)

    rename_plan = collect_rename_plan(img_root)
    db_updates = collect_db_updates(db_path)
    missing_paths = collect_missing_downloaded_paths(db_path, data_root=data_root, img_root=img_root)

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

    apply_rename_plan(rename_plan)
    apply_db_updates(db_path, db_updates)

    missing_after_apply = collect_missing_downloaded_paths(db_path, data_root=data_root, img_root=img_root)
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
