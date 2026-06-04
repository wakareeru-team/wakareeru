# 噪声复核与迭代训练闭环

本文记录 `stage_10` 到 `stage_14` 之间的设计思想和数据流。它描述的是稳定流程，不记录某次运行的样本数、分数或临时结论。

## 设计目标

这个闭环的目标不是让模型自动修正所有标签，而是把人工复核与保守的自动噪声筛选结合起来：

- 人工复核负责确认高风险样本的真实问题类型。
- Logistic Regression 噪声分类器只学习 `wrong_label` 这类“当前训练标签与图像内容不一致”的噪声。
- 下一轮 `loss_tracking` 使用更干净的训练集合重新训练线性头，使后续 loss / error-rate 特征更可靠。
- `manual_corrected_label` 只作为 crop 级 overlay，不覆盖 `images` 原始标签，便于追溯。

## 关键字段

- `crops.noise_review_label`
  人工复核标签，取值包括 `ok`、`wrong_label`、`out_of_label_space`、`bad_crop`、`ambiguous`。

- `crops.manual_corrected_label`
  当 `noise_review_label = wrong_label` 且正确标签在当前 label space 内时，由 Gradio 的 `Correct label` 下拉框写入。

- `crops.noise_predicted_label` / `noise_predicted_prob`
  `stage_13` 对未人工复核样本的 LR 噪声预测结果。若 `lr_prediction.sync_to_db=false`，预测结果保存在当前 loss round 的 `lr_predictions.csv`。

- `data/loss_analysis/latest_loss_analysis_round.txt`
  指向最近一次完整完成 `stage_10` 训练产物的 loss round。`stage_10` 只在 loss history、epoch history 和 checkpoint 都保存后才更新该指针。

## 轮次数据流

一轮完整清洗通常是：

```text
stage_10 loss_tracking
  - 读取 DINO feature cache 中的 crop_id 和 feature
  - 从 DB 读取当前标签、人工复核和上一轮预测状态
  - 用 manual_corrected_label 覆盖训练标签
  - 排除人工噪声和上一轮预测噪声
  - 训练线性头，写入本轮 loss history、epoch history、label_map.json

stage_11 loss_analysis
  - 读取本轮 label_map.json 和 loss history
  - 聚合 mean loss、tail loss、error_rate、pred_label_rate 等特征
  - 写入本轮 demo_loss_feature.csv
  - 同步 noise_score_v1 到 DB，供 Gradio 抽样

Gradio 人工复核
  - ok：作为 clean 样本
  - wrong_label：作为错标噪声样本
  - wrong_label + Correct label：本轮 stage_12 仍是噪声正样本；下一轮 stage_10 和 stage_14 使用 Correct label
  - bad_crop / out_of_label_space：不参与 LR wrong_label 分类器训练
  - ambiguous：跳过，不参与 LR 训练

stage_12 logistic_regression_filter
  - 读取本轮 loss feature
  - 使用人工复核标签训练 wrong_label LR 分类器
  - wrong_label 即使有 manual_corrected_label，本轮仍作为噪声正样本
  - ok 作为 clean 负样本
  - 保存 LR 模型和 latest_lr_model 指针

stage_13 lr_prediction
  - 使用本轮 LR 模型对未人工复核样本预测 wrong_label 噪声概率
  - 可写入 DB，也可只写入本轮 lr_predictions.csv

下一轮 stage_10
  - 以上一轮 DB 预测字段或上一轮 lr_predictions.csv 为依据排除预测噪声
  - 使用 manual_corrected_label 作为正确训练标签
```

## 人工纠正标签的两种角色

`manual_corrected_label` 在不同阶段有不同含义：

- 对 `stage_12` 来说，它不把样本变成 clean。
  因为本轮 loss feature 是在纠正前的标签体系下产生的，`wrong_label + manual_corrected_label` 仍然是“当前标签下的错标噪声”，应作为 LR 的正样本。

- 对下一轮 `stage_10` 和最终 `stage_14` 来说，它是正确标签。
  这些阶段会优先使用 `manual_corrected_label`，没有时才回退到 `submodel`、`fine_grained_series` 或 `series`。

这样设计可以同时满足两个目的：本轮用它训练噪声分类器，下一轮把它作为干净监督样本。

## 训练过滤规则

`stage_10` 使用 `noise_detection.*` 控制训练集过滤：

- `exclude_manual_noise`
  是否排除人工确认的噪声标签。

- `manual_noise_labels`
  要排除的人工复核标签。通常包括 `wrong_label`、`out_of_label_space`、`bad_crop`。

- `exclude_predicted_noise`
  是否排除上一轮 LR 预测噪声。

- `predicted_noise_labels` / `predicted_noise_min_prob`
  控制哪些预测标签和最低概率会被排除。

保留规则：

- `ok` 保留。
- 未复核且未被预测为高置信噪声的样本保留。
- `wrong_label + manual_corrected_label` 保留，并用 `manual_corrected_label` 训练。

排除规则：

- `wrong_label` 且没有 `manual_corrected_label` 的样本排除。
- `out_of_label_space`、`bad_crop` 等配置在 `manual_noise_labels` 中的标签排除。
- 上一轮 LR 预测为噪声且概率超过阈值的未纠正样本排除。

## `sync_to_db` 对闭环的影响

`stage_10` 支持两种上一轮预测来源：

- `lr_prediction.sync_to_db=true`
  使用 DB 中的 `crops.noise_predicted_label` 和 `noise_predicted_prob`。

- `lr_prediction.sync_to_db=false`
  从上一轮 loss round 的 `lr_predictions.csv` 读取 prediction overlay。由于 `stage_10` 只在本轮成功完成后才更新 latest 指针，因此创建当前轮次目录后，`latest` 仍指向上一轮，可以安全读取上一轮预测文件。

如果预测 CSV 不存在，`stage_10` 会跳过预测噪声过滤；如果 CSV 缺必要列，则直接报错。

## 修改 Stage 8 Label Space 后的重跑

`stage_08_fine_grain_series.py` 会更新 DB 中的 `images.fine_grained_series`。如果只修改了细粒度标签规则或 `manual_fine_grained_series.csv`，DINO feature cache 通常仍然可复用，因为 `stage_09` 只缓存 crop 图像特征和 `crop_id`，不绑定标签体系。

已经生成的 loss round 不会因为 stage 8 规则变化而自动更新。该轮目录中的 `label_map.json`、loss history、loss feature 和 LR prediction 都仍然对应生成它们时的 label space。

推荐流程是先重跑新标签空间下的 loss round：

```bash
python pipeline_entry.py --stages "8 10 11"
```

人工复核当前新 round 的样本后，再继续：

```bash
python pipeline_entry.py --stages "12 13"
```

如果 `loss_analysis.request_manual_review=true`，管线会在 `stage_11` 后中断，等待人工复核；这时直接跑 `8-12` 不一定会进入 `stage_12`。

修改 label space 后，不建议继续用旧 label space 下训练出的 LR prediction 过滤新一轮样本。为了避免旧预测影响新 loss 特征，可在新一轮 `stage_10` 前临时关闭：

```yaml
noise_detection:
  exclude_predicted_noise: false
```

待新 label space 下完成 `stage_12` / `stage_13` 后，再恢复预测噪声过滤。另需注意，`manual_corrected_label` 是 crop 级 overlay，不会被 stage 8 自动改写；如果标签被改名、合并或拆分，应检查已有人工纠正标签是否仍在当前 label space 内。

## 为什么不默认沿用旧 LR 模型

每一轮 `stage_10` 都可能使用不同的训练集合重新训练线性头，因此 `stage_11` 生成的 loss feature 分布会变化。旧 LR 模型的阈值和概率校准可能不再适合新一轮 loss feature。

因此如果 `stage_12` 没有足够的正负样本重新拟合 LR，默认应中断并补充人工复核，而不是自动用旧 LR 模型继续预测。

## 最终导出

`stage_14_store_crops.py` 导出最终数据集时：

- 先按配置过滤人工噪声和预测噪声。
- 有 `manual_corrected_label` 的样本不会因为旧预测噪声而被过滤。
- 导出标签优先使用 `manual_corrected_label`。
- 人工纠正样本会按 `crops_storage.manual_correction_invalidate_metadata_columns` 清空原图分类路径派生的细节 metadata，避免旧标签语境下的番台、运营公司、特殊编成或特殊涂装继续污染导出。随后会从未纠正的同 label 导出候选中保守反查补齐：`manual_correction_refill_operator_columns` 中的 operator 字段只有唯一非空值时补齐；`manual_correction_refill_submodel_bandai_columns` 作为一对，只有唯一非空组合时才一起补齐。
- `metadata.manual_reviewed` 仍只表示人工复核为 `ok` 的高确信样本；人工纠正样本的正确标签通过 `label` 字段体现。
