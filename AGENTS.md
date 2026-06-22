# わかれーる Wakareeru

`wakareeru` 是一个面向日本铁路车辆的细粒度图像分类数据集与预处理管线项目。它从日文 Wikipedia 与 Wikimedia Commons 收集车辆系列、分类路径和图片，并通过规则、SigLIP2、LLM 元数据抽取、Grounding-DINO 主体检测与后续噪声筛查，为车辆识别模型准备训练样本。

当前重点是 JR 东日本与 JR 货物车辆，但是可能包含JNR车型同时由其他JR公司继承的车型。；覆盖范围由 `config/pipeline_config.yaml` 中的 `crawler.active_operators` 控制。长期目标是逐步扩展到 JR 本州三社、全 JR 公司，以及私铁车辆。

## 给 Agent 的工作原则
- 在没有明确要求实现的情形下请不要直接触碰代码，在用户向你询问技术细节或请求纠错时，主动说明用到的library等技术细节。
- 只实现用户明确要求的任务。这个项目包含很多数据集构建细节，不要主动扩大范围、改标签体系或重构管线。
- 对于你的工具函数，注意保持职责单一原则，即不要隐含函数名不提到的细节，例如不要在load图片函数中自行transform图片到某些pretrain模型的要求。
- 对于可视化函数，注意保持参数简洁性和可扩展性，优先使用动态扫描列名的写法，而不是钉死字段名称。
- 对于内存占用较小的 metadata、manifest、review overlay 等表格处理，优先全量读取后用 pandas/DataFrame 表达筛选和派生逻辑；比起动态拼接复杂 SQL，可读性和可维护性更重要。
- 优先遵循现有脚本、配置和数据库 schema；不要为了“更完整”而新增抽象、自动修复逻辑或额外阶段。
- 如果阶段脚本、数据库字段、配置项或入口命令发生非显然变化，同步更新本文件的相关段落。
- 不要把一次性运行结果、当前样本数、临时实验结论、具体模型分数等易过期信息写进本文件；这类信息应放在 docs、notebook、review 文件或实验记录里。
- `data/` 下通常包含生成数据、缓存、图片和 SQLite 数据库。除非用户明确要求，不要清理、重建或覆盖这些文件。
- 构建pipeline时注意将易修改配置放入config，而把较为固定的常量放入constants.py。在笔记本实验阶段不需要这么做
- pipeline 正式脚本读取运行配置时不要在代码里用 `.get(..., 默认值)` 静默兜底；新增或依赖配置项应写入 `config/pipeline_config.yaml`，缺失时直接报错，避免隐藏硬编码默认值。

## 稳定目标与建模方向

- 标签来源：从日文 Wikipedia 车辆列表解析标准化车辆系列。
- 图片来源：匹配 Wikimedia Commons 分类树并生成图片 manifest。
- 清洗流程：关键词过滤、SigLIP2 zero-shot 图片过滤、LLM 分类路径元数据抽取、Grounding-DINO 主体 bbox。
- 训练方向：以 DINO 系列特征为主，结合线性头、监督对比学习或 prototype bank retrieval 做细粒度识别。
- 增量扩展方向：新增系列尽量通过特征缓存与 prototype bank 检索支持，避免每次都完整重训。

## 仓库结构

```text
pipeline_entry.py          # 主入口：运行全部阶段、指定阶段或从某阶段继续
pipeline/                  # 主数据管线；编号 stage 可独立运行
  stage_01_model_parsing.py
  stage_02_model_fixing.py
  stage_03_manifest_crawling.py
  stage_04_img_crawler.py
  stage_05_siglip_image_filtering.py
  stage_06_llm_metadata_labeling.py
  stage_07_gdino_bbox.py
  stage_08_fine_grain_series.py
  constants.py             # 共享枚举、正则、提示词等
  utils.py                 # 路径、DB、配置、日志等辅助函数
config/
  pipeline_config.yaml     # 路径、模型名、阈值、抓取范围等运行配置
  manual_series_overrides.csv
  schema.sql               # 新数据库的基线 schema
  migrations/              # 既有数据库的增量迁移，按数字顺序执行
docker/
  entry.sh                 # RunPod 容器启动脚本；设置 HF cache 与 rclone R2 remote
data/                      # 生成数据、SQLite、图片、缓存与 review 输出
tools/                     # 人工 review 等交互式辅助工具；不是自动 pipeline stage
trainer/                   # crop 图像训练模块；当前从冻结 backbone + 线性头开始
model_core/                # 训练与推理共享的分类模型、artifact loader 和 crop 分类逻辑
src/crawler/               # 探索性 notebook；不是稳定管线入口
docs/                      # 项目过程记录与实验说明
```

平级仓库 `wakareeru-inference` 是 serverless 推理后端，通常通过 Git dependency 复用本仓库的 `model_core`，并从本地 `models/` 读取由本仓库导出的分类模型 artifact。改动 `model_core`、导出 artifact 结构或推理输入输出契约时，应同步检查 `wakareeru-inference` 的 README / AGENTS 与服务配置。

## 运行入口

创建环境：

```bash
conda env create -f environment.yml
conda activate wakareeru
pip install -e ".[dev]"
cp .env.example .env
```

RunPod 等 GPU 镜像若已经内置 PyTorch/torchvision，可用 `requirements-runpod.txt` 安装运行依赖；该文件刻意不包含 `torch`/`torchvision`，避免覆盖镜像自带 CUDA 版本。

RunPod Docker 镜像入口：

```bash
docker buildx build --platform linux/amd64 -f Dockerfile.basepod -t wakareeru-basepod:local --load .
```

`Dockerfile.basepod` 基于 RunPod PyTorch 镜像安装 `rsync`、`rclone` 与 Python 运行依赖，并设置 Hugging Face 与 pip cache 到 `/workspace/.cache`。容器启动时 `docker/entry.sh` 从运行时环境变量配置 rclone remote `r2`，不把 secret 写入镜像层；需要的环境变量为 `HF_TOKEN`、`R2_ACCESS_ID`、`R2_ACCESS_KEY`、`R2_ENDPOINT`。RunPod volume 建议挂载到 `/workspace`，生成数据可通过 `path.in_project_root: false` 与绝对 `path.data_root` 指到 `/workspace/data`。

运行完整管线：

```bash
python pipeline_entry.py
```

只运行一个阶段或多个阶段：

```bash
python pipeline_entry.py --stages "5"
python pipeline_entry.py --stages "5 10 11"
python pipeline_entry.py --stages "9-13"
```

从某阶段继续运行：

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

启动 label 分布与抽样复核 UI（显示 label 规模差距，并支持跨 label 抽样与 crop 级修正标签）：

```bash
python tools/label_review_gradio.py
```

启动 crop 图像线性头训练：

```bash
python -m trainer.train
```

导出新 label 翻译队列，或在填写完成后将翻译表回写 `label_metadata`：

```bash
python pipeline_entry.py --only label_metadata_translation
```

翻译表位于 `label_metadata_translation.review_file_path`。本阶段只导出数据库尚不存在的新 label；翻译未完成时中断后续 pipeline，填写 `label_en`、`label_zh` 与三语 operator JSON 数组后重跑即可事务写入。已有 `label_metadata` 行不会被翻译表覆盖。

导出供推理仓库使用的分类模型 artifact：

```bash
python -m trainer.export_inference_model
```

导出配置位于 `trainer.export`。训练完成后会在 `trainer.output_dir` 下更新 `trainer.latest_run_pointer` 指针；`trainer.export.checkpoint_path: "latest_best"` 会导出最新训练 run 最后一个 phase 的 best checkpoint，也可以填具体 checkpoint 路径。导出的分类模型目录是自包含 artifact，应包含 `backbone/`、`processor/`、`classifier.safetensors`、`model_config.json`、`labels.json`、`l10n_metadata.json` 和 `manifest.json`；其中 `l10n_metadata.json` 从 `path.dataset_dir` 下复制，缺失时导出直接报错。`model_config.json` 中的 `image_size` 来自 checkpoint 保存的训练配置；导出时会同步 processor 默认 `size` / `crop_size`，推理侧也以 `model_config.json` 为准，避免训练与推理 resize/crop 尺寸错位。

从人工复核 CSV 导入 review overlay（路径相对 `path.data_root` 解析，用 stable key + bbox IoU 匹配，不依赖自增 id）：

```bash
python tools/export_noise_review_csv.py --output-csv-path review/noise_review_labels.csv
python tools/import_noise_review_csv.py --review-csv-path review/noise_review_labels.csv
```

跨平台迁移图片后，如遇到视觉相同但 Unicode 字节不同的文件名（例如 macOS 与 Linux 的 NFC/NFD 差异），先 dry-run 检查，再显式应用路径规范化；该工具会把 `path.raw_img_dir` 下文件名和 `images.downloaded_path` 统一为 NFC：

```bash
python tools/normalize_image_paths.py
python tools/normalize_image_paths.py --apply
```

开发检查：

```bash
ruff check .
pytest
```

Python 版本要求见 `pyproject.toml`；Conda 环境见 `environment.yml`。

## Pipeline Stages

所有阶段默认读取 `config/pipeline_config.yaml`。代码、配置和规则文件始终相对项目根目录解析；数据库、图片、缓存、review 输出等生成数据相对 `path.data_root` 解析。默认主要写入项目内 `data/commons_image_manifest.sqlite` 或 `data/` 下的派生文件。

| Key | Script | 作用 |
| --- | --- | --- |
| `model_parsing` | `stage_01_model_parsing.py` | 从 Wikipedia wikitext 解析车辆系列 CSV，并排除 `導入予定` 等不纳入数据集的状态 |
| `model_fixing` | `stage_02_model_fixing.py` | 应用人工修正，生成 Commons 根分类映射 |
| `manifest_crawling` | `stage_03_manifest_crawling.py` | 查询 Commons 分类树，写入 `categories` 与 `images` |
| `img_crawling` | `stage_04_img_crawler.py` | 下载图片，更新 `images.download_status`，并将图片文件名与 `images.downloaded_path` 规范化为 Unicode NFC |
| `siglip_filter` | `stage_05_siglip_image_filtering.py` | 用 SigLIP2 过滤内饰、局部细节等不适合训练的图片 |
| `llm_labeling` | `stage_06_llm_metadata_labeling.py` | 用 OpenAI API 从分类路径抽取番台、子型号、运营公司等元数据，并在回写前应用运营者名称人工规范化；只回写 `llm_metadata_processed = 0` 的新图片，同路径已有 checkpoint 时直接复用，避免覆盖既有 metadata |
| `fine_grain_series` | `stage_08_fine_grain_series.py` | 根据 LLM 元数据和人工规则构造 `fine_grained_series` |
| `gdino_bbox` | `stage_07_gdino_bbox.py` | 用 Grounding-DINO 检测车辆主体并写入 `crops` |
| `feature_extraction` | `stage_09_DINOv3_feature_extraction.py` | 提取 crop 图像 DINOv3 特征，只缓存 `features` 与 `crop_ids`，不绑定标签体系 |
| `loss_tracking` | `stage_10_train_loss_tracking.py` | 从当前数据库标签动态生成本轮 label id，训练线性头并记录 loss |
| `loss_analysis` | `stage_11_loss_analysis.py` | 读取本轮 `label_map.json` 和 loss history，聚合噪声筛查特征 |
| `logistic_regression_filter` | `stage_12_logistic_regression_filter.py` | 基于人工复核标签训练 Logistic Regression 噪声筛选器 |
| `lr_prediction` | `stage_13_lr_prediction.py` | 对未复核样本生成 LR 噪声预测 CSV，并可选同步到数据库 |
| `label_metadata_translation` | `stage_13b_label_metadata_translation.py` | 在 Stage 13 与 crop 存储之间增量导出新 label 翻译表；填写后校验并回写 `label_metadata`，不覆盖既有规范行 |
| `store_crops` | `stage_14_store_crops.py` | 将 crop 图像保存为最终数据集，并生成 `metadata.csv` / `labels.csv`，同时从数据库 `label_metadata` 规范表导出 `l10n_metadata.json`；`metadata.manual_reviewed` 表示人工复核为 `ok` 的高确信样本 |

`pipeline/deprecated_stage_08_siglip_crop_filtering.py` 是弃用阶段，不应作为默认流程的一部分。

## 标签与 Commons 映射

`stage_01` 从日文 Wikipedia 车辆列表解析 `series`、`wiki_title`、`full_name`、`status`、`type`、`subtype`、`operator_jp`、`operator_en` 等字段。

`stage_02` 使用 `config/manual_series_overrides.csv` 处理 Commons 命名差异、系列合并和人工修正。与 Commons 分类名相关的规则集中在 `pipeline/constants.py` 和 stage 脚本中；需要修改时先读现有逻辑，不要只凭文件名字符串硬编码。

`fine_grained_series` 用于更细粒度的训练标签；规则来源见 `config/manual_fine_grained_series.csv` 和 `fine_grain_series.rules_path` 配置。DINOv3 特征缓存只绑定 crop 图像和 `crop_id`，修改细粒度标签规则后通常只需重跑 `fine_grain_series`、`loss_tracking` 与后续噪声分析阶段，不需要重跑 `feature_extraction`。

## 数据库概要

新数据库基线 schema 位于 `config/schema.sql`。既有数据库通过 `config/migrations/` 按 `PRAGMA user_version` 增量升级。

关键表：

- `categories`：Commons 分类树节点、父分类、抓取状态与错误信息。
- `images`：每个 Commons 文件在某个系列/分类下的 manifest 记录，包含标签来源、分类路径、图片元数据、过滤状态、下载状态、LLM 元数据与 `fine_grained_series`。
- `image_categories`：文件与分类的多对多归属关系。
- `crops`：Grounding-DINO bbox、检测置信度、裁切状态、噪声分数、人工噪声复核字段和 crop 级人工纠正标签。
- `label_metadata`：label 的英中翻译、三语 operator 数组与日文 Wikipedia title，是 `l10n_metadata.json` 的唯一规范来源。`images.wiki_title`、`images.operator_en`、`images.operator_jp` 保留为旧的逐图来源字段，供既有 pipeline/review 使用，但不得用于本地化 metadata 导出。

常用状态字段：

- `images.excluded` / `exclude_reason`：关键词与 SigLIP2 等过滤结果。
- `images.siglip_processed`：SigLIP2 图片过滤是否已处理。
- `images.llm_metadata_processed`：Stage 06 是否已将分类路径 metadata 回写到该图片；迁移时既有图片标记为已处理，新抓取图片默认未处理。
- `images.download_status`：`not_started`、`downloaded`、`failed`、`missing_url`。
- `images.downloaded_path`：相对 `path.data_root` 的图片路径，通常形如 `img/<series>/<file>`。
- `crops.crop_status`：`pending`、`ok`、`rejected`。
- `crops.noise_score_v1` 与 `noise_review_*`：Small Loss Trick / 人工复核相关字段；`manual_corrected_label` 是人工确认错标后写入的 crop 级正确标签，`loss_tracking` 与 `store_crops` 会优先使用它；LR 噪声筛选训练仍按 `noise_review_label` 对 `wrong_label` 建正样本。跨机器迁移人工复核结果时使用 `tools/import_noise_review_csv.py` 显式导入，不作为默认 pipeline stage。详细闭环见 `docs/noise_review_loop.md`。

## 配置要点

- `path.in_project_root` 与 `path.data_root` 控制生成数据根目录；`in_project_root: true` 时 `data_root` 相对项目根目录解析，`false` 时 `data_root` 必须是绝对路径，适合 RunPod volume 挂载。
- `path.db_path`、`path.raw_img_dir`、`path.cache_dir`、`path.model_dir`、CSV、review 输出和 checkpoint 路径相对 `path.data_root` 解析；`manual_series_overrides_path`、`fine_grain_series.rules_path` 等代码/配置文件仍相对项目根目录解析。
- `crawler.active_operators` 控制当前纳入的数据范围。
- `crawler.manifest_max_depth`、`manifest_max_files_per_category` 控制 Commons 分类递归与每分类文件上限；`crawler.manifest_reprocess` 控制是否忽略 category checkpoint 并覆盖重爬 manifest。
- `image_filtering.*` 控制 SigLIP2 图片过滤。
- `llm_labeling.*` 控制 OpenAI 元数据抽取，包括是否为 Responses API 启用 `web_search` 工具。
- `fine_grain_series.*` 控制细粒度车型标签规则。
- `gdino.*` 控制 Grounding-DINO 检测阈值、NMS 与批大小。
- `noise_detection.*` 控制后续 DINO 特征缓存和 small-loss 噪声检测实验；`image_size` 控制特征提取阶段输入 processor 的正方形 resize/crop 分辨率，修改后需要重跑 `feature_extraction`；`feature_cache_shard_size` 控制特征提取阶段 `.pt` 分片保存后再聚合为单文件缓存。训练标签 id 在 `loss_tracking` 每轮根据当前数据库标签动态生成，并保存到该轮 loss analysis 目录的 `label_map.json`。`loss_tracking` 会按 `noise_detection.exclude_manual_noise` / `exclude_predicted_noise` 过滤人工噪声与上一轮 LR 预测噪声；`manual_corrected_label` 会覆盖原标签并保留为训练样本。详细设计见 `docs/noise_review_loop.md`。
- `logistic_regression_filter.*` 控制人工复核标签上的 Logistic Regression 噪声筛选实验。
- `label_metadata_translation.review_file_path` 控制新 label 翻译队列 CSV 路径。该阶段从 crop 的当前有效 label 扫描缺项，稳定的 `wiki_title` 仅用于辅助翻译，operator 只从已有规范 label 继承；不会读取 Stage 06 的 `operator_en` / `operator_jp`。完整行使用 `INSERT ... ON CONFLICT DO NOTHING` 回写，已有规范记录不可被自动覆盖。
- `crops_storage.metadata_columns` 控制最终 `metadata.csv` 输出列；默认包含 `manual_reviewed`，用于筛选人工复核为 `ok` 的评估样本。`l10n_metadata_file_name` 控制多语言 metadata JSON 文件名；内容只从数据库 `label_metadata` 表读取并按当前 `labels.csv` 的 id 导出，不再从既有 JSON 或 `images` 旧字段回填。当前 label 缺少规范记录、翻译为空、operator 三语数组不对齐，或检测到链接/双语言污染时直接报错。`manual_correction_invalidate_metadata_columns` 控制人工纠正标签后需要清空的原图分类路径派生 metadata；随后 `manual_correction_refill_operator_columns` 中的 operator 字段只有同 label 唯一非空值时补齐，`manual_correction_refill_submodel_bandai_columns` 作为一对只有唯一非空组合时才一起补齐。
- `trainer.*` 控制 crop 图像训练入口；`trainer.image_size` 控制输入 processor 的正方形 resize/crop 分辨率，并写入 checkpoint。训练结束后 `trainer.latest_run_pointer` 指向最新 run 目录，`run_summary.json` 记录每个 phase 的 best checkpoint；`trainer.export.checkpoint_path` 可用 `"latest_best"` 指向最新 run 的 best checkpoint。导出推理 artifact 时，`model_config.json` 与 `processor/preprocessor_config.json` 会使用 checkpoint 中保存的 `image_size`，而不是当前工作区后来改动的配置。修改 `trainer.image_size`、backbone、pooling 方式或特征维度后需要重建 linear head feature cache；加载已有 feature cache 时会用当前 metadata 的 label id 刷新 cache 内 labels，避免 metadata/labels 重新生成后沿用旧 label id。分类特征由 CLS 与排除 register tokens 后的 patch mean 拼接而成；当前默认冻结 `backbone_model_name` 并只训练线性分类头，报告和 checkpoint 写入 `trainer.output_dir`。

## 维护边界

本文件应保持为 agent 快速接手项目的稳定地图。适合写：

- 稳定入口、目录职责、阶段职责、关键表和维护原则。
- 与代码长期一致的命名约定和数据流。

不适合写：

- 某次运行得到的图片数、样本数、准确率、loss、人工 review 进度。
- 已经修掉的临时 bug 列表。
- 可能频繁切换的实验模型结论，除非已经成为项目约定。
