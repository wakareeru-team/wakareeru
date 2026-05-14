from pathlib import Path
import logging
import re
import sqlite3
import sys
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
# ================ Config Utils ================

def load_pipeline_config(config_path: str | Path | None = None) -> dict:
    import yaml

    config_path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "pipeline_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ================ Path Utils ================

def get_project_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Project root not found from {current}")

PROJECT_ROOT = get_project_root()
IMG_ROOT = os.path.join(PROJECT_ROOT, load_pipeline_config()['path']['raw_img_dir'])

def join_root_path(relative_path: str) -> str:
    return os.path.join(PROJECT_ROOT, relative_path)

# ================ Image Processing Utils ================
def load_img_with_orientation(path):
    

    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # 确保方向正确
    return img


def _source_image_path(row: pd.Series | dict, img_root: Path = IMG_ROOT) -> Path: # type: ignore
    path = Path(str(row["downloaded_path"]).replace("\\", "/"))
    return path if path.is_absolute() else img_root / path


def _expanded_box(row: pd.Series | dict, image_size: tuple[int, int], pad_frac: float = 0.04):
    width, height = image_size
    x1, y1, x2, y2 = (float(row[k]) for k in ["box_x1", "box_y1", "box_x2", "box_y2"])
    pad = max(x2 - x1, y2 - y1) * pad_frac
    left = max(0, int(np.floor(x1 - pad)))
    top = max(0, int(np.floor(y1 - pad)))
    right = min(width, int(np.ceil(x2 + pad)))
    bottom = min(height, int(np.ceil(y2 + pad)))
    if right <= left or bottom <= top:
        raise ValueError(f"bad crop box for crop_id={row.get('crop_id', row.get('id'))}: {(left, top, right, bottom)}")
    return left, top, right, bottom


def crop_from_image(
    img: Image.Image,
    row: pd.Series | dict,
    pad_frac: float = 0.04,
    resize: int | tuple[int, int] | None = 224,
) -> Image.Image:
    crop = img.crop(_expanded_box(row, img.size, pad_frac=pad_frac))
    if resize is not None:
        size = (resize, resize) if isinstance(resize, int) else resize
        crop = crop.resize(size, Image.Resampling.BICUBIC)
    return crop


def load_crop(
    row: pd.Series | dict,
    img_root: Path = IMG_ROOT, # type: ignore
    pad_frac: float = 0.04,
    resize: int | tuple[int, int] | None = 224,
) -> Image.Image:
    img = load_img_with_orientation(_source_image_path(row, img_root=img_root))
    return crop_from_image(img, row, pad_frac=pad_frac, resize=resize)




# ================ Logger ================


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if logger.handlers:          # 避免重复添加 handler
        return logger
    
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 终端 — 在 Windows cp1252 终端下强制 utf-8，避免日文字符报错
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    # 文件
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)   # 文件记录更详细
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

# ================ Database Utils ================

def init_db(
    db_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> None:
    """Initialize the SQLite database from config/schema.sql without dropping data."""
    resolved_db_path = resolve_db_path(db_path)
    schema_path = resolve_project_path(schema_path or PROJECT_ROOT / "config" / "schema.sql")
    migrations_dir = resolve_project_path(migrations_dir or PROJECT_ROOT / "config" / "migrations")

    schema_sql = schema_path.read_text(encoding="utf-8")

    conn = sqlite3.connect(resolved_db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(schema_sql)
        # 按顺序执行migrations操作
        apply_migrations(conn, migrations_dir)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def connect_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection for callers that need to own DB lifecycle."""
    conn = sqlite3.connect(resolve_db_path(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def resolve_db_path(db_path: str | Path | None = None) -> str | Path:
    if db_path is None:
        config = load_pipeline_config()
        db_path = config["path"]["db_path"]

    if db_path == ":memory:":
        return db_path

    resolved_db_path = resolve_project_path(db_path)
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_db_path


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def migration_version(path: Path) -> int:
    match = re.match(r"^(\d+)[_\-].*\.sql$", path.name)
    if not match:
        raise ValueError(
            f"Migration file must be named like '001_description.sql': {path.name}"
        )
    return int(match.group(1))


def list_migration_files(migrations_dir: Path) -> list[tuple[int, Path]]:
    if not migrations_dir.exists():
        return []

    migrations = [(migration_version(path), path) for path in migrations_dir.glob("*.sql")]
    versions = [version for version, _ in migrations]
    duplicate_versions = sorted({version for version in versions if versions.count(version) > 1})
    if duplicate_versions:
        raise ValueError(f"Duplicate migration versions: {duplicate_versions}")

    return sorted(migrations, key=lambda item: item[0])


def apply_migrations(conn: sqlite3.Connection, migrations_dir: Path) -> None:
    current_version = conn.execute("PRAGMA user_version").fetchone()[0]

    for version, path in list_migration_files(migrations_dir):
        if version <= current_version:
            continue

        script = path.read_text(encoding="utf-8")
        conn.executescript(script)
        conn.execute(f"PRAGMA user_version = {version}")
        current_version = version
