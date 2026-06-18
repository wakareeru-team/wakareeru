# わかれーる Wakareeru

[![model tag](https://img.shields.io/github/v/tag/SniperPigeon/wakareeru?filter=v*&label=model)](https://github.com/SniperPigeon/wakareeru/tags)
[![inference tag](https://img.shields.io/github/v/tag/SniperPigeon/wakareeru-inference?filter=inference-v*&label=inference)](https://github.com/SniperPigeon/wakareeru-inference/tags)
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
- Docker / RunPod：GPU 运行环境；`Dockerfile.basepod` 基于 RunPod PyTorch 镜像安装非 PyTorch 运行依赖。
- `rclone` / `rsync`：RunPod volume 与 Cloudflare R2 等远端对象存储之间的数据同步。

完整依赖见 [pyproject.toml](pyproject.toml) 与 [environment.yml](environment.yml)。

## 仓库结构

```text
Dockerfile.basepod        # RunPod 基础镜像构建文件
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
docker/
  entry.sh                 # 容器启动脚本，设置 HF cache 与 rclone R2 remote
data/
  commons_image_manifest.sqlite
  jr_east_freight_series.csv
  jr_east_freight_series_wiki_commons.csv
  feature_cache/
  review/
tools/                     # 人工 review 等交互式辅助工具
trainer/                   # crop 图像训练与推理 artifact 导出
model_core/                # 训练与推理共享的分类模型、loader 和 crop 分类逻辑
src/crawler/               # 探索性 notebook 与实验流程
docs/                      # 项目过程记录
```

平级仓库 `wakareeru-inference` 是 serverless 推理后端。它通过 Git dependency 复用本仓库的 `model_core`，并从本地 `models/` 读取本仓库导出的分类模型 artifact。

## 快速开始

创建环境：

```bash
conda env create -f environment.yml
conda activate wakareeru
pip install -e ".[dev]"
```

RunPod 等 GPU 镜像若已经内置 PyTorch/torchvision，可安装轻量运行依赖：

```bash
pip install -r requirements-runpod.txt
pip install -e .
```

也可以基于 [Dockerfile.basepod](Dockerfile.basepod) 构建 RunPod 镜像。该镜像假定基础镜像已经提供 CUDA 版 PyTorch/torchvision，只额外安装系统工具、`requirements-runpod.txt` 和启动脚本：

```bash
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.basepod \
  -t wakareeru-basepod:local \
  --load \
  .
```

容器启动时会先执行 [docker/entry.sh](docker/entry.sh)，设置 Hugging Face cache 到 `/workspace/.cache`，并从运行时环境变量配置名为 `r2` 的 rclone remote。RunPod secrets 可注入为：

```bash
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
R2_ACCESS_ID={{ RUNPOD_SECRET_r2_access_key_id }}
R2_ACCESS_KEY={{ RUNPOD_SECRET_r2_secret_access_key }}
R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
```

本地测试示例：

```bash
docker run --platform linux/amd64 --rm -it \
  -e HF_TOKEN="..." \
  -e R2_ACCESS_ID="..." \
  -e R2_ACCESS_KEY="..." \
  -e R2_ENDPOINT="https://<account-id>.r2.cloudflarestorage.com" \
  wakareeru-basepod:local
```

进入容器后可用 `rclone lsd r2:` 检查 remote。工作目录默认是 `/workspace`；RunPod volume 建议挂载到 `/workspace`，数据目录可在 `config/pipeline_config.yaml` 中用绝对路径如 `/workspace/data` 指定。

如果需要运行 LLM 元数据阶段，请配置 OpenAI API Key：

```bash
cp .env.example .env
```

然后在 `.env` 中填写对应环境变量。

运行完整管线：

```bash
python pipeline_entry.py
```

只运行某一阶段或多个阶段：

```bash
python pipeline_entry.py --stages "5"
python pipeline_entry.py --stages "5 10 11"
python pipeline_entry.py --stages "9-13"
```

从某一阶段继续运行到最后：

```bash
python pipeline_entry.py --from manifest_crawling
```

启动人工噪声复核 UI：

```bash
python tools/noise_review_gradio.py
```

启动指定 loss round 的只读抽查 UI（支持高可疑、高错误率、按 label 均衡抽样等）：

```bash
python tools/loss_round_spotcheck_gradio.py
```

训练 crop 图像线性头：

```bash
python -m trainer.train
```

导出供 `wakareeru-inference` 使用的自包含分类模型 artifact：

```bash
python -m trainer.export_inference_model
```

导出路径由 `config/pipeline_config.yaml` 的 `trainer.export` 控制。训练完成后会在 `trainer.output_dir` 下更新 `trainer.latest_run_pointer`；`trainer.export.checkpoint_path: "latest_best"` 会导出最新训练 run 最后一个 phase 的 best checkpoint，也可以填具体 checkpoint 路径。分类 artifact 包含 `backbone/`、`processor/`、`classifier.safetensors`、`model_config.json`、`labels.json` 和 `manifest.json`；`model_config.json` 的 `image_size` 来自 checkpoint 保存的训练配置，导出时会同步 processor 默认 `size` / `crop_size`，推理侧也以该 artifact 配置为准。

导出/导入人工复核 CSV（路径相对 `path.data_root` 解析，使用 stable key + bbox IoU 匹配）：

```bash
python tools/export_noise_review_csv.py --output-csv-path review/noise_review_labels.csv
python tools/import_noise_review_csv.py --review-csv-path review/noise_review_labels.csv
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
| `feature_extraction` | 提取 crop 图像 DINOv3 特征，只缓存 `features` 与 `crop_ids` |
| `loss_tracking` | 按当前数据库标签动态生成本轮 label id，训练线性头并记录 loss |
| `loss_analysis` | 聚合 loss 特征并生成噪声筛查分数 |
| `logistic_regression_filter` | 基于人工复核标签训练 LR 噪声筛选器 |
| `lr_prediction` | 生成 LR 噪声预测 CSV，并可选同步数据库 |
| `store_crops` | 保存最终 crop 图像并生成 `metadata.csv` / `labels.csv`，其中 `manual_reviewed` 标记人工复核为 `ok` 的样本，人工错标纠正会优先覆盖导出标签 |

## 数据流

1. `stage_01_model_parsing.py` 从日文 Wikipedia 车辆列表解析 `series`、`wiki_title`、`status`、`type`、`operator` 等字段。
2. `stage_02_model_fixing.py` 根据 `config/manual_series_overrides.csv` 修正常见 Commons 分类合并和命名差异。
3. `stage_03_manifest_crawling.py` 查询 Wikimedia Commons 分类树，写入 `categories` 与 `images` 表。
4. `stage_04_img_crawler.py` 下载图片，保存到 `data/img/`，并更新 SQLite 状态。
5. `stage_05_siglip_image_filtering.py` 使用 `google/siglip2-base-patch16-512` 判断图片是否适合保留。
6. `stage_06_llm_metadata_labeling.py` 使用 OpenAI 模型解析分类路径中的番台、运营公司、特殊涂装等信息。
7. `stage_08_fine_grain_series.py` 根据 LLM 元数据和人工规则构造 `fine_grained_series`。
8. `stage_07_gdino_bbox.py` 使用 `IDEA-Research/grounding-dino-base` 生成车辆主体 bbox 与裁切记录。
9. `stage_09_DINOv3_feature_extraction.py` 提取 crop 图像特征，缓存 `features` 与 `crop_ids`，不绑定细粒度标签。
10. `stage_10_train_loss_tracking.py` 按当前数据库标签动态生成本轮 label id，并过滤人工噪声与上一轮 LR 预测噪声后训练线性头。
11. `stage_11_loss_analysis.py` 读取本轮 `label_map.json` 和 loss history，生成噪声筛查特征。
12. `stage_14_store_crops.py` 保存最终 crop 图像，并在 `metadata.csv` 中写入 `manual_reviewed` 供评估集筛选；人工错标纠正会通过 crop 级 `manual_corrected_label` 覆盖训练和导出标签。

噪声复核、人工纠正标签、LR 预测过滤和多轮 loss tracking 的完整设计见 [docs/noise_review_loop.md](docs/noise_review_loop.md)。

主数据库为 `data/commons_image_manifest.sqlite`，关键表包括：

- `categories`：每个系列解析到的 Commons 分类。
- `images`：每张 Commons 图片的元数据、下载状态、过滤结果和 LLM 标签。
- `crops`：Grounding-DINO 生成的主体框、置信度、裁切状态、噪声分数、人工复核标签、LR 预测标签和 crop 级人工纠正标签。

## 配置

主要配置位于 [config/pipeline_config.yaml](config/pipeline_config.yaml)：

- `crawler.active_operators` 控制当前纳入的数据范围。
- `crawler.manifest_max_depth` 与 `manifest_max_files_per_category` 控制 Commons 分类递归与单分类文件上限。
- `image_filtering.siglip_model_name` 控制图片过滤模型。
- `llm_labeling.openai_model_name` 控制 LLM 元数据抽取模型。
- `fine_grain_series.rules_path` 控制细粒度车型标签规则 CSV，默认是 `config/manual_fine_grained_series.csv`。
- `gdino.model_name`、`box_threshold`、`nms_iou_threshold` 控制主体检测。
- `noise_detection.exclude_manual_noise` / `exclude_predicted_noise` 控制下一轮 loss tracking 是否排除人工噪声与上一轮 LR 预测噪声；`manual_corrected_label` 会覆盖原标签并保留为训练样本。
- `noise_detection` 用于后续 DINO 特征缓存与 small-loss trick 噪声检测实验；特征缓存只绑定 crop 图像和 `crop_id`，训练标签 id 在 loss tracking 轮次中动态生成。
- `crops_storage.metadata_columns` 控制最终 `metadata.csv` 输出列，默认包含 `manual_reviewed`。

## 开发命令

```bash
ruff check .
pytest
```

处理内存占用较小的 metadata、manifest、review overlay 等表格数据时，优先全量读取后用 pandas/DataFrame 表达筛选和派生逻辑；复杂 SQL 拼接只在数据量或数据库侧约束确实需要时使用。

## 当前状态

项目仍处于数据集构建与清洗阶段。当前主线管线已经覆盖标签解析、Commons manifest、图片下载、SigLIP2 图片过滤、LLM 元数据标注与 Grounding-DINO 主体裁切。`src/crawler/` 下的 notebook 主要用于实验和验证，不是稳定入口。

后续方向包括：

- 扩展覆盖范围到 JR 本州三社、全 JR 公司与私铁车辆。
- 基于 DINOv2 / DINOv3 特征缓存进行噪声样本检测。
- 构建可增量扩展的 prototype bank，用于新增系列的检索式分类。
- 整理 Hugging Face Dataset 格式并发布可复现实验配置。
