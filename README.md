# わかれーる Wakareeru

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/Conda-environment.yml-44A833?logo=anaconda&logoColor=white)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-required-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![SQLite](https://img.shields.io/badge/SQLite-manifest-003B57?logo=sqlite&logoColor=white)](data/commons_image_manifest.sqlite)

`wakareeru` 是一个面向日本铁路车辆的细粒度图像分类模型以及其数据集与预处理管线项目。此项目**目前为数据预处理阶段**，我们从日文 Wikipedia 和 Wikimedia Commons 收集车辆系列、分类路径与图片，经过规则、SigLIP2过滤、LLM 元数据抽取和 Grounding-DINO 目标检测+主体裁切，DINOv3+分类头的小模型Small Loss Trick噪声过滤，最终为日本铁道车辆识别模型准备更干净的训练样本。

当前重点覆盖 JR 东日本与 JR 货物车辆，但同样包含继承JNR的国铁车型的其他JR公司车型。目标是区分外观相近的系列与番台，例如 113 系 / 115 系、E231 系 / E233 系等。

## 项目目标

- 从 Wikipedia 车辆列表解析标准化车型标签。
- 自动匹配 Wikimedia Commons 车辆分类并生成图片 manifest。
- 下载 Commons 图片并过滤内饰、局部细节、显示屏等不适合训练的图片。
- 使用 LLM 从 Commons 分类路径中抽取番台、子型号、特殊编成、涂装和运营公司等结构化元数据。
- 使用 Grounding-DINO 检测车辆主体并生成裁切框。
- 为后续 DINOv2 / DINOv3 特征缓存、噪声样本检测和细粒度分类训练准备数据。

## 技术栈

- Python 3.11+，推荐 Conda 环境中的 Python 3.12。
- `httpx` / `beautifulsoup4` / `lxml`：Wikipedia 与 Commons 数据抓取。
- `pandas` / `numpy` / `scikit-learn`：数据处理与实验分析。
- `torch` / `torchvision` / `transformers` / `accelerate`：SigLIP2、DINO、Grounding-DINO 等模型推理与训练实验。
- `openai`：Commons 分类路径的结构化标签抽取。
- SQLite：图片、分类、裁切框和过滤状态的主数据表。

完整依赖见 [pyproject.toml](pyproject.toml) 与 [environment.yml](environment.yml)。

## 仓库结构

```text
pipeline_entry.py          # 管线入口，可运行全部阶段或指定阶段
pipeline/                  # 主数据管线，每个 stage 可独立运行
  stage_01_model_parsing.py
  stage_02_model_fixing.py
  stage_03_manifest_crawling.py
  stage_04_img_crawler.py
  stage_05_siglip_image_filtering.py
  stage_06_llm_metadata_labeling.py
  stage_07_gdino_bbox.py
  stage_08_fine_grain_series.py
  constants.py
  utils.py
config/
  pipeline_config.yaml     # 路径、模型名、阈值、抓取范围等运行配置
  manual_series_overrides.csv
  schema.sql
  migrations/
data/
  commons_image_manifest.sqlite
  jr_east_freight_series.csv
  jr_east_freight_series_wiki_commons.csv
  feature_cache/
  review/
src/crawler/               # 探索性 notebook 与实验流程
docs/                      # 项目过程记录
```

## 快速开始

创建环境：

```bash
conda env create -f environment.yml
conda activate wakareeru
pip install -e ".[dev]"
```

如果需要运行 LLM 元数据阶段，请配置 OpenAI API Key：

```bash
cp .env.example .env
```

然后在 `.env` 中填写对应环境变量。

运行完整管线：

```bash
python pipeline_entry.py
```

只运行某一阶段：

```bash
python pipeline_entry.py --only siglip_filter
```

从某一阶段继续运行到最后：

```bash
python pipeline_entry.py --from manifest_crawling
```

可用阶段包括：

| Stage | 说明 |
| --- | --- |
| `model_parsing` | 从 Wikipedia wikitext 解析车辆系列 |
| `model_fixing` | 应用人工修正并生成 Commons 分类映射 |
| `manifest_crawling` | 调用 Commons API 生成分类与图片 manifest |
| `img_crawling` | 异步下载图片并回写下载状态 |
| `siglip_filter` | 用 SigLIP2 zero-shot 过滤内饰与干扰图片 |
| `llm_labeling` | 用 OpenAI API 从分类路径抽取结构化元数据 |
| `fine_grain_series` | 根据人工规则构造 `fine_grained_series` 标签 |
| `gdino_bbox` | 用 Grounding-DINO 检测车辆主体并写入裁切框 |

## 数据流

1. `stage_01_model_parsing.py` 从日文 Wikipedia 车辆列表解析 `series`、`wiki_title`、`status`、`type`、`operator` 等字段。
2. `stage_02_model_fixing.py` 根据 `config/manual_series_overrides.csv` 修正常见 Commons 分类合并和命名差异。
3. `stage_03_manifest_crawling.py` 查询 Wikimedia Commons 分类树，写入 `categories` 与 `images` 表。
4. `stage_04_img_crawler.py` 下载图片，保存到 `data/img/`，并更新 SQLite 状态。
5. `stage_05_siglip_image_filtering.py` 使用 `google/siglip2-base-patch16-512` 判断图片是否适合保留。
6. `stage_06_llm_metadata_labeling.py` 使用 OpenAI 模型解析分类路径中的番台、运营公司、特殊涂装等信息。
7. `stage_08_fine_grain_series.py` 根据 LLM 元数据和人工规则构造 `fine_grained_series`。
8. `stage_07_gdino_bbox.py` 使用 `IDEA-Research/grounding-dino-base` 生成车辆主体 bbox 与裁切记录。

主数据库为 `data/commons_image_manifest.sqlite`，关键表包括：

- `categories`：每个系列解析到的 Commons 分类。
- `images`：每张 Commons 图片的元数据、下载状态、过滤结果和 LLM 标签。
- `crops`：Grounding-DINO 生成的主体框、置信度、裁切状态和噪声分数。

## 配置

主要配置位于 [config/pipeline_config.yaml](config/pipeline_config.yaml)：

- `crawler.active_operators` 控制当前纳入的数据范围。
- `crawler.manifest_max_depth` 与 `manifest_max_files_per_category` 控制 Commons 分类递归与单分类文件上限。
- `image_filtering.siglip_model_name` 控制图片过滤模型。
- `llm_labeling.openai_model_name` 控制 LLM 元数据抽取模型。
- `fine_grain_series.rules_path` 控制细粒度车型标签规则 CSV。
- `gdino.model_name`、`box_threshold`、`nms_iou_threshold` 控制主体检测。
- `noise_detection` 用于后续 DINO 特征缓存与 small-loss trick 噪声检测实验。

## 开发命令

```bash
ruff check .
pytest
```

## 当前状态

项目仍处于数据集构建与清洗阶段。当前主线管线已经覆盖标签解析、Commons manifest、图片下载、SigLIP2 图片过滤、LLM 元数据标注与 Grounding-DINO 主体裁切。`src/crawler/` 下的 notebook 主要用于实验和验证，不是稳定入口。

后续方向包括：

- 扩展覆盖范围到 JR 本州三社、全 JR 公司与私铁车辆。
- 基于 DINOv2 / DINOv3 特征缓存进行噪声样本检测。
- 构建可增量扩展的 prototype bank，用于新增系列的检索式分类。
- 整理 Hugging Face Dataset 格式并发布可复现实验配置。
