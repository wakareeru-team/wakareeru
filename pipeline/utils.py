from PIL import Image, ImageOps
from pathlib import Path

# ================ Path Utils ================

def get_project_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Project root not found from {current}")



# ================ Image Processing Utils ================
def load_img_with_orientation(path):
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # 确保方向正确
    return img