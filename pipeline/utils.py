from pathlib import Path
import logging
import pathlib
import re
import sqlite3
import sys
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import time

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

def join_root_path(relative_path: str | Path) -> str:
    """Backward-compatible alias for project-root paths."""
    return str(join_project_root(relative_path))


def join_project_root(path: str | Path) -> Path:
    """Join code/config paths to the repository root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def get_data_root(config: dict | None = None) -> Path:
    """Return the generated-data root from config.

    If path.in_project_root is true, data_root is resolved under PROJECT_ROOT.
    If false, data_root must be an absolute path for external volume mounts.
    """
    config = config or load_pipeline_config()
    path_config = config.get("path", {})
    data_root = Path(path_config.get("data_root", "data")).expanduser()
    in_project_root = path_config.get("in_project_root", True)

    if in_project_root:
        return data_root if data_root.is_absolute() else PROJECT_ROOT / data_root
    if not data_root.is_absolute():
        raise ValueError("path.data_root must be absolute when path.in_project_root is false")
    return data_root


def join_data_root(path: str | Path, config: dict | None = None) -> Path:
    """Join generated-data paths to data_root."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = get_data_root(config) / path
    return path


# 为分轮次loss分析保存数据的helper
def create_new_loss_round_dir(config: dict) -> Path:
    loss_analysis_root = join_data_root(config['path']['loss_analysis_data_dir'], config=config)
    if not loss_analysis_root.is_dir():
        loss_analysis_root.mkdir(parents=True, exist_ok=True)
    timestr = time.strftime('%Y%m%d_%H%M%S',time.localtime())
    analysis_dir = loss_analysis_root / timestr
    analysis_dir.mkdir(parents=True, exist_ok=False)
    return analysis_dir


def update_latest_loss_round_pointer(config: dict, loss_round_dir: Path) -> None:
    loss_analysis_root = join_data_root(config['path']['loss_analysis_data_dir'], config=config)
    loss_round_dir = Path(loss_round_dir)
    pointer = config['loss_noise_tracking']['latest_loss_analysis_round_pointer']
    with open(loss_analysis_root / pointer, 'w') as f:
        f.write(loss_round_dir.name)


def get_current_loss_round_dir(config: dict) -> Path:
    loss_analysis_root = join_data_root(config['path']['loss_analysis_data_dir'], config=config)
    loss_config = config['loss_noise_tracking']
    active_round = loss_config['active_loss_analysis_round']
    pointer = loss_config['latest_loss_analysis_round_pointer']
    return get_loss_round_dir(config=config, active_round=active_round, pointer=pointer)


def get_loss_round_dir(config: dict, active_round: str, pointer: str | None = None) -> Path:
    loss_analysis_root = join_data_root(config['path']['loss_analysis_data_dir'], config=config)
    if pointer is None:
        pointer = config['loss_noise_tracking']['latest_loss_analysis_round_pointer']
    pointer_path = loss_analysis_root / pointer
    if active_round == 'latest':
        if pointer_path.exists():
            with open(pointer_path, 'r') as f:
                timestr = f.read().strip()
            analysis_dir = loss_analysis_root / timestr
            if analysis_dir.exists():
                return analysis_dir
            else: # 设计上去让他报错
                raise FileNotFoundError(f"Expected loss analysis dir not found: {analysis_dir}")
        else:
            raise FileNotFoundError(f"Latest loss analysis round pointer not found: {pointer_path}")
            
    else:
        
        analysis_dir = loss_analysis_root / active_round
        if analysis_dir.exists():
            return analysis_dir
        else: # 设计上去让他报错
            raise FileNotFoundError(f"Expected loss analysis dir not found: {analysis_dir}")



# ================ Image Processing Utils ================
def load_img_with_orientation(path):
    

    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # 确保方向正确
    return img


def _source_image_path(
    row: pd.Series | dict,
    config: dict | None = None,
) -> Path:
    path = Path(str(row["downloaded_path"]).replace("\\", "/"))
    if path.is_absolute():
        return path
    return join_data_root(path, config=config)


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
) -> Image.Image:
    return img.crop(_expanded_box(row, img.size, pad_frac=pad_frac))


def load_crop(
    row: pd.Series | dict,
    config: dict | None = None,
    pad_frac: float = 0.04,
) -> Image.Image:
    img = load_img_with_orientation(
        _source_image_path(row, config=config)
    )
    return crop_from_image(img, row, pad_frac=pad_frac)


# ================ Torch Cache Utils ================

def save_pt_shard(
    shard_dir: str | Path,
    shard_index: int,
    payload: dict,
    prefix: str = "part",
) -> Path:
    """Save one ordered PyTorch shard under ``shard_dir``."""
    import torch

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{prefix}_{shard_index:06d}.pt"
    torch.save(payload, shard_path)
    return shard_path


def aggregate_pt_shards(
    shard_dir: str | Path,
    output_path: str | Path,
    tensor_keys: list[str],
    metadata: dict | None = None,
    prefix: str = "part",
    dim: int = 0,
) -> dict:
    """Concatenate tensor values from ordered ``.pt`` shards and save one cache file."""
    import torch

    shard_dir = Path(shard_dir)
    output_path = Path(output_path)
    shard_paths = sorted(shard_dir.glob(f"{prefix}_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {shard_dir} with prefix {prefix!r}")

    buckets = {key: [] for key in tensor_keys}
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        for key in tensor_keys:
            if key not in shard:
                raise KeyError(f"Shard {shard_path} missing tensor key {key!r}")
            buckets[key].append(shard[key])

    aggregated = {
        key: torch.cat(values, dim=dim)
        for key, values in buckets.items()
    }
    if metadata:
        aggregated.update(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(aggregated, output_path)
    return aggregated




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
    config: dict | None = None,
) -> None:
    """Initialize the SQLite database from config/schema.sql without dropping data."""
    resolved_db_path = db_path if db_path == ":memory:" else join_data_root(
        db_path or (config or load_pipeline_config())["path"]["db_path"],
        config=config,
    )
    if resolved_db_path != ":memory:":
        Path(resolved_db_path).parent.mkdir(parents=True, exist_ok=True)
    schema_path = join_project_root(schema_path or "config/schema.sql")
    migrations_dir = join_project_root(migrations_dir or "config/migrations")

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


def connect_db(db_path: str | Path | None = None, config: dict | None = None) -> sqlite3.Connection:
    """Open a SQLite connection for callers that need to own DB lifecycle."""
    if db_path is None:
        db_path = (config or load_pipeline_config())["path"]["db_path"]
    resolved_db_path = db_path if db_path == ":memory:" else join_data_root(db_path, config=config)
    conn = sqlite3.connect(resolved_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


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
