# Contributing

Wakareeru is a data pipeline repository for building a fine-grained Japanese railway vehicle image dataset. Contributions should be small, explicit, reproducible, and careful around generated data.

Before changing code, first identify which part of the repository the change belongs to: stable pipeline, review tool, configuration, database schema, documentation, or experiment.

## Directory Boundaries

- `pipeline/` contains the stable automated pipeline. Code validated in notebooks may be migrated here after it is organized into stable functions and entry points. Changes here should preserve stage boundaries, read runtime settings from config, and avoid broad refactors.
- `tools/` contains manual review, import/export, and maintenance utilities. These scripts may be more task-specific, but they should not silently overwrite generated data.
- `config/` contains runtime configuration, manual rules, the baseline database schema, and migrations.
- `docs/` contains process notes, design notes, experiment records, and contributor-facing explanations.
- `src/crawler/` is exploratory notebook/prototype space. Do not treat it as a stable pipeline entry point.
- `data/` contains generated datasets, caches, downloaded images, review outputs, and SQLite databases. Do not delete, rebuild, or overwrite these files unless the task explicitly requires it.

New stable pipeline stages should follow the existing `pipeline/stage_XX_*.py` naming pattern and should be wired into `pipeline_entry.py` only when they are meant to become part of the supported workflow.

## Code Style

- Follow the existing Python style and run `ruff check .` before opening a pull request when code changes are involved.
- Keep functions single-purpose. A function name should not hide unrelated side effects.
- Keep I/O, config parsing, model inference, database reads, and database writes separated where practical.
- Prefer small readable helper functions over long loops that mix unrelated rules.
- Use existing helpers from `pipeline/utils.py` before adding new path, config, logging, or database utilities.
- Use the project logger style instead of large amounts of ad hoc `print`.
- Do not hard-code local absolute paths in committed pipeline code. Build paths through the existing path helpers in `pipeline/utils.py`.
- Do not move exploratory notebook logic directly into `pipeline/`; first turn it into explicit functions and a clear entry point.
- For model preprocessing, prefer library-provided pipelines where appropriate, such as Hugging Face `pipeline` APIs or torchvision transform composition.

Avoid dangerous hidden side effects. For image and model utilities, avoid hidden transforms. For example, an image loading helper should not silently resize or normalize for a specific pretrained model unless that behavior is clear from its name and call site, because that can lead to double transforms.

For visualization utilities, keep parameters simple and extensible. Prefer dynamically scanning available columns over hard-coding fields from one temporary experiment.

## Pipeline Rules

- Each stage should keep to one responsibility: parsing labels, fixing mappings, crawling manifests, downloading images, filtering images, labeling metadata, building fine-grained labels, detecting boxes, or exporting crops.
- A stage should make its inputs and outputs clear through config values, database fields, and explicit paths.
- Avoid fixing unrelated upstream or downstream data problems inside a stage unless that behavior is the purpose of the change.
- Long-running work, GPU inference, network requests, OpenAI API calls, and reprocessing behavior should be controlled by configuration. Prefer fine-grained `tqdm` progress bars for long-running stages.
- Pipeline code should fail clearly when required configuration is missing.

Stable pipeline scripts should not use `.get(..., default)` to silently provide runtime configuration defaults. Add or update the required key in `config/pipeline_config.yaml` instead.

## Configuration

- Put runtime-tunable values in `config/pipeline_config.yaml`.
- Put stable constants, regular expressions, shared prompts, and enums in `pipeline/constants.py`.
- Keep manual series and fine-grained label rules in the existing CSV rule files unless a different storage format is explicitly part of the change.
- When adding a config key, document its purpose through nearby config structure or related docs if the meaning is not obvious.
- In notebooks and one-off experiments, temporary values may be local. Before moving the logic into `pipeline/`, make the relevant values configurable.

## Database And Migrations

The baseline schema is `config/schema.sql`. Existing local databases are upgraded through numbered files in `config/migrations/`.

When adding or changing a database field, update all relevant pieces together:

- `config/schema.sql`
- a new migration in `config/migrations/`, with the database version updated according to the existing migration code
- the stage or tool that reads or writes the field
- any documentation that describes the field or workflow

Do not solve schema changes by asking contributors to delete and regenerate their SQLite database. Migrations should describe the intended incremental change.

## Shared Database And Cloud Data

The main SQLite database is too large and too mutable to be tracked in Git without Git LFS, and Git LFS is not recommended for the working database. Git should track the schema, migrations, pipeline code, config, and small rule files. Large generated state should live outside Git.

Use the database stored in Cloudflare R2 as the shared baseline snapshot. Contributors who need the current working database should pull it from R2. If a change intentionally updates the shared database, upload the updated database back to R2 after validating the change and documenting what produced it.

Recommended practice:

- Keep `data/commons_image_manifest.sqlite` out of Git.
- Treat R2 as the source of truth for the current shared database snapshot.
- Use dated or commit-associated backup snapshots before replacing the current shared database.
- Schema changes must still be represented by `config/schema.sql` and `config/migrations/`; uploading a new SQLite file is not a substitute for a migration.
- Do not overwrite the shared R2 database with a local experiment database.

For RunPod or other cloud GPU work, use the project Docker image built from `Dockerfile.basepod`. The container configures Hugging Face and pip caches under `/workspace/.cache` and uses `docker/entry.sh` to configure an rclone remote named `r2` through Cloudflare R2's S3-compatible API.

The required runtime environment variables are:

- `HF_TOKEN`
- `R2_ACCESS_ID`
- `R2_ACCESS_KEY`
- `R2_ENDPOINT`

Inside the container, verify the remote with:

```bash
rclone lsd r2:
```

Use rclone for database pull and push operations. Replace the bucket and object paths below with the agreed project paths:

```bash
rclone copyto r2:<bucket>/database/current/commons_image_manifest.sqlite data/commons_image_manifest.sqlite
rclone copyto data/commons_image_manifest.sqlite r2:<bucket>/database/snapshots/commons_image_manifest_YYYY-MM-DD_<git-sha>.sqlite
rclone copyto data/commons_image_manifest.sqlite r2:<bucket>/database/current/commons_image_manifest.sqlite
```

Only run the final command that updates the current R2 database after the producing command, relevant git commit, and validation status are clear.

## Data Safety

- Treat `data/` as user-owned generated state.
- Do not clean, rebuild, or overwrite `data/` unless explicitly requested.
- Do not casually run commands that trigger full image downloads, full model reprocessing, large API calls, or database-wide rewrites.
- Prefer dry-run modes when available, especially for path normalization, review import/export, and maintenance scripts.
- Review CSV import/export should use the existing stable key plus bbox IoU workflow rather than relying on autoincrement IDs.
- For cross-platform image path issues, use `tools/normalize_image_paths.py` and check the dry-run output before applying changes.

Do not put one-off run counts, temporary model scores, review progress, or transient bug notes in README or `AGENTS.md`. Use `docs/`, notebooks, or experiment records for time-sensitive notes.

## Labels And Rules

Label taxonomy, Commons category mapping, and fine-grained series rules are core dataset design choices. Do not change them as part of unrelated cleanup.

When a contribution intentionally changes labeling behavior, describe:

- which labels or mappings are affected
- which rule file or stage changed
- whether existing database rows need to be regenerated
- how the change was checked

## Validation

Use the smallest validation that matches the risk of the change.

- Documentation-only changes do not require pipeline runs.
- Python code changes should run `ruff check .`.
- Shared utility or stage logic changes should include `pytest` when practical.
- Schema changes should be checked against a fresh schema and an existing migrated database when practical.
- GPU, network, or OpenAI-dependent changes should say which command was run, or why it was not run locally.

Before asking for review, summarize whether the change touches `data/`, config, schema, long-running pipeline behavior, or external services.
