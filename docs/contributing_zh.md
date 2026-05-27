# 贡献规范

Wakareeru 是一个用于构建日本铁路车辆细粒度图像数据集的数据管线仓库。贡献应当尽量小、明确、可复现，并且谨慎对待已经生成的数据。

修改代码前，请先判断这次改动属于哪一类：稳定 pipeline、人工 review 工具、配置、数据库 schema、文档，还是实验代码。

## 目录边界

- `pipeline/` 是稳定自动管线,notebook验证通过的代码会迁移到这里。修改这里时应保持stage边界，运行参数从配置读取，避免大范围重构。
- `tools/` 是人工 review、导入导出和维护脚本。这类脚本可以更偏具体任务，但不应静默覆盖生成数据。
- `config/` 存放运行配置、人工规则、数据库基线 schema 和迁移脚本。
- `docs/` 存放过程记录、设计说明、实验记录，以及面向贡献者的说明。
- `src/crawler/` 是探索性 notebook 和原型代码空间，不应视为稳定 pipeline 入口。
- `data/` 存放生成数据集、缓存、下载图片、review 输出和 SQLite 数据库。除非任务明确要求，不要删除、重建或覆盖这些文件。

新的稳定 pipeline stage 应沿用 `pipeline/stage_XX_*.py` 命名方式。只有当它确实要成为受支持流程的一部分时，才接入 `pipeline_entry.py`。

## 代码风格

- 遵循现有 Python 风格；涉及代码改动时，提交 review 前运行 `ruff check .`。
- 函数保持单一职责。函数名不应隐藏无关副作用。
- 在可行的情况下，将 I/O、配置解析、模型推理、数据库读取和数据库写入分开。
- 优先使用小而清晰的 helper 函数，避免在长循环里混合多种规则。
- 新增路径、配置、日志或数据库工具前，优先复用 `pipeline/utils.py` 中已有 helper。
- 日志风格遵循项目现有 logger，不要大量使用临时 `print`。
- 提交到正式 pipeline 的代码不要硬编码本地绝对路径。直接从utils的两个路径helper拼接。
- 不要把探索性 notebook 逻辑直接搬进 `pipeline/`；进入稳定管线前，应先整理成明确函数和清晰入口。
- 涉及到模型需要预处理推荐使用各类库的pipeline功能，例如hf的`pipeline`和torch的`transform compose`。


避免产生危险的副作用，例如图片和模型相关工具应避免隐藏 transform。例如，加载图片的 helper 不应偷偷为某个 pretrained model 做 resize 或 normalize，除非这个行为已经从函数名和调用处清楚表达，避免二重transform。

可视化工具的参数应保持简洁且可扩展。优先动态扫描可用列名，避免写死某次临时实验里的字段。

## Pipeline 规则

- 每个 stage 应只负责一个阶段职责：标签解析、映射修正、manifest 抓取、图片下载、图片过滤、元数据标注、细粒度标签构造、bbox 检测或 crop 导出。
- stage 的输入输出应通过配置项、数据库字段和明确路径表达清楚。
- 不要在某个 stage 中顺手修复无关的上游或下游数据问题，除非这正是本次改动的目的。
- 长耗时任务、GPU 推理、网络请求、OpenAI API 调用和重处理行为应由配置控制。尽量使用细粒度的tqdm进度条进行管理
- 缺少必需配置时，pipeline 代码应明确报错。

稳定 pipeline 脚本不应使用 `.get(..., default)` 为运行配置提供静默默认值。新增或依赖配置项时，应更新 `config/pipeline_config.yaml`。

## 配置

- 易调整的运行参数放在 `config/pipeline_config.yaml`。
- 稳定常量、正则、共享 prompt 和枚举放在 `pipeline/constants.py`。
- 人工车型规则和细粒度标签规则默认继续使用现有 CSV 规则文件；除非本次改动明确要更换存储方式。
- 新增配置项时，如果含义不明显，应在附近配置结构或相关文档中说明用途。
- notebook 和一次性实验可以临时写本地值；但逻辑进入 `pipeline/` 前，相关值应配置化。

## 数据库与迁移

数据库基线 schema 位于 `config/schema.sql`。已有本地数据库通过 `config/migrations/` 下的编号文件增量升级。

新增或修改数据库字段时，应同步更新：

- `config/schema.sql`
- `config/migrations/` 下的新 migration,注意按照已有代码更新数据库的版本。
- 读取或写入该字段的 stage 或 tool
- 描述该字段或流程的相关文档

不要通过要求贡献者删除并重新生成 SQLite 数据库来解决 schema 变更。migration 应描述预期的增量变更。

## 共享数据库与云端数据

主 SQLite 数据库过大且变化频繁，不适合在不引入 Git LFS 的情况下纳入 Git；即使使用 Git LFS，也不推荐把工作数据库作为版本追踪对象。Git 应追踪 schema、migration、pipeline 代码、配置和小型规则文件。大型生成状态应放在 Git 之外管理。

以 Cloudflare R2 中的数据库作为共享基准快照。需要当前工作数据库的贡献者应从 R2 拉取；如果某次改动确实更新了共享数据库，应在验证后把更新后的数据库同步回 R2，并说明它由什么命令或流程产生。

推荐约定：

- `data/commons_image_manifest.sqlite` 不进入 Git。
- R2 中的数据库视为当前共享工作快照的基准。
- 覆盖 current 数据库前，优先保留带日期或 commit 信息的快照备份。
- schema 改动仍必须通过 `config/schema.sql` 和 `config/migrations/` 表达；上传新的 SQLite 文件不能替代 migration。
- 不要用本地临时实验数据库覆盖 R2 中的共享基准数据库。

RunPod 或其他云端 GPU 环境应优先使用 `Dockerfile.basepod` 构建的项目 Docker 镜像。容器会把 Hugging Face 和 pip cache 放到 `/workspace/.cache`，并通过 `docker/entry.sh` 使用 Cloudflare R2 的 S3-compatible API 配置名为 `r2` 的 rclone remote。

运行时需要的环境变量包括：

- `HF_TOKEN`
- `R2_ACCESS_ID`
- `R2_ACCESS_KEY`
- `R2_ENDPOINT`

进入容器后，可以用下面的命令检查 remote：

```bash
rclone lsd r2:
```

数据库拉取和上传使用 rclone。下面命令中的 bucket 和对象路径应替换为项目约定路径：

```bash
rclone copyto r2:<bucket>/database/current/commons_image_manifest.sqlite data/commons_image_manifest.sqlite
rclone copyto data/commons_image_manifest.sqlite r2:<bucket>/database/snapshots/commons_image_manifest_YYYY-MM-DD_<git-sha>.sqlite
rclone copyto data/commons_image_manifest.sqlite r2:<bucket>/database/current/commons_image_manifest.sqlite
```

只有在生成数据库的命令、相关 git commit 和验证状态都清楚时，才执行最后一条命令更新 R2 上的 current 数据库。

## 数据安全

- 将 `data/` 视为用户持有的生成状态。
- 除非明确要求，不要清理、重建或覆盖 `data/`。
- 不要随意运行会触发全量图片下载、全量模型重处理、大量 API 调用或数据库全表重写的命令。
- 如果工具提供 dry-run 模式，优先先 dry-run，尤其是路径规范化、review 导入导出和维护脚本。
- review CSV 导入导出应使用现有 stable key 加 bbox IoU 的流程，不应依赖自增 ID。
- 跨平台图片路径问题应使用 `tools/normalize_image_paths.py`，并在应用修改前检查 dry-run 输出。

不要把一次性运行数量、临时模型分数、review 进度或短期 bug 记录写入 README 或 `AGENTS.md`。这类易过期信息应放在 `docs/`、notebook 或实验记录中。

## 标签与规则

标签体系、Commons 分类映射和细粒度系列规则是数据集设计的核心。不要在无关 cleanup 中顺手修改它们。

如果某次贡献确实要改变标签行为，应说明：

- 影响哪些标签或映射
- 修改了哪个规则文件或 stage
- 现有数据库记录是否需要重新生成
- 如何检查了这次改动

## 验证

使用与改动风险相匹配的最小验证。

- 纯文档改动不需要运行 pipeline。
- Python 代码改动应运行 `ruff check .`。
- 共享工具或 stage 逻辑改动，在可行时应运行 `pytest`。
- schema 改动在可行时应检查新建 schema 和已有数据库迁移。
- 依赖 GPU、网络或 OpenAI 的改动，应说明运行过哪个命令；如果本地没有运行，也应说明原因。

请求 review 前，请总结本次改动是否触及 `data/`、配置、schema、长耗时 pipeline 行为或外部服务。
