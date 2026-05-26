from __future__ import annotations

"""Unicode normalization helpers for downloaded image paths."""

import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class NormalizationReport:
    filesystem_renames: int
    db_updates: int


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
            updates.append(
                DbUpdate(image_id=int(image_id), old_path=str(old_path), new_path=new_path)
            )
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


def normalize_downloaded_image_paths(db_path: Path, img_root: Path) -> NormalizationReport:
    rename_plan = collect_rename_plan(img_root)
    db_updates = collect_db_updates(db_path)

    apply_rename_plan(rename_plan)
    apply_db_updates(db_path, db_updates)

    return NormalizationReport(
        filesystem_renames=len(rename_plan),
        db_updates=len(db_updates),
    )


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


def collect_missing_downloaded_paths(
    db_path: Path,
    data_root: Path,
    img_root: Path,
) -> list[tuple[int, str]]:
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
