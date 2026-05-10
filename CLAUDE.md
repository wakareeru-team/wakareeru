# わかれーる — Japanese Train Image Classification Dataset

Fine-grained image classification dataset and model for Japanese rolling stock, targeting visually similar variants (e.g. 113系 vs 115系 commuter EMUs).

## Project Goal

Build a labeled image dataset from Wikimedia Commons, then train a fine-grained classifier using DINOv2-Large + SupCon Loss. New series should be addable via prototype-bank retrieval without retraining.

Coverage is phased:
1. **Phase 1** — JR東日本 + JR貨物 (in progress; active operators in `pipeline_config.yaml`)
2. **Phase 2** — JR本州三社 (JR East + JR Central + JR West)
3. **Phase 3** — All JR companies nationwide
4. **Phase 4** — Private railways across Japan

## Architecture Decisions

- **Backbone**: DINOv2-Large (preferred over CLIP — stronger linear separability, patch tokens preserve local detail, better for fine-grained)
- **Head**: Linear probe or prototype retrieval
- **Loss**: CrossEntropy + SupCon Loss
- **Dataset format**: HuggingFace Dataset (feature caching, stratified splits, push_to_hub)
- **Incremental extension**: prototype-bank retrieval so new series don't require retraining

## Repository Layout

```
pipeline/             # Numbered Python scripts; each stage is independently runnable
  stage_01_model_parsing.py       # Wikipedia wikitext → CSV label list
  stage_02_model_fixing.py        # Manual label corrections via CSV overrides
  stage_03_manifest_crawling.py   # Commons category lookup → SQLite manifest
  stage_04_img_crawler.py         # Async image download → SQLite
  stage_05_siglip_image_filtering.py  # SigLIP2 exterior/interior zero-shot filter
  stage_06_llm_metadata_labeling.py   # OpenAI → submodel/bandai/special_formation fields
  stage_07_gdino_bbox.py          # Grounding-DINO bbox detection → crops table
  constants.py                    # Shared enums, patterns, prompt strings
  utils.py                        # Path/DB/config helpers
config/
  pipeline_config.yaml            # Runtime config for all stages (models, thresholds, paths)
  manual_series_overrides.csv     # Hand-curated label corrections consumed by stage_02
  schema.sql                      # SQLite DB schema
  migrations/                     # Incremental DB schema migrations
src/crawler/          # Exploratory/prototype notebooks (not part of the main pipeline)
  model_parse.ipynb
  img_crawler.ipynb
  img_filter_v2.ipynb
  Small_Loss_Trick_Filter_Example.ipynb  # Noise label detection (post-pipeline)
data/
  jr_east_freight_series.csv      # Parsed label list; fields: series, wiki_title,
                                  #   full_name, status, type, subtype,
                                  #   operator_jp, operator_en
  jr_east_freight_series_wiki_commons.csv  # Commons root-category mapping per series
  commons_image_manifest.sqlite   # Main DB (categories / images / crops tables)
  img/<series>/                   # Downloaded originals; path stored in DB
  feature_cache/                  # DINOv2 feature tensors (.pt) for noise detection
```

## Label System

Labels parsed from Japanese Wikipedia vehicle list pages (H2/H3 headings + `[[link]]` extraction) by stage_01. Manual corrections applied by stage_02 via `config/manual_series_overrides.csv`.

CSV fields: `series`, `wiki_title`, `full_name`, `status`, `type`, `subtype`, `operator_jp`, `operator_en`

- `status` normalized via `STATUS_MAP` in `constants.py`: 現役 / 廃止 / 導入予定
- Duplicates within the same operator removed; entry with more complete type/subtype info retained
- 国鉄-inherited cars detected via `wiki_title.startswith("国鉄")` → `JNR` operator prefix

Commons category naming pattern (resolved in stage_03):
- 新幹線 → `Shinkansen`; 国鉄 (inherited by JR) → `JNR`
- JR東日本 → `JR East`, JR東海 → `JR Central`, JR西日本 → `JR West`, etc.
- Katakana prefix converted to romaji with space before number/letter code (キハ40 → `Kiha 40`, キヤE195 → `Kiya E195`)
- Special Commons merges handled via manual overrides in stage_02 (e.g. 481系 → `JNR 485`)
- Series with no matching Commons category are dropped from the dataset

## Pipeline Stages (implemented)

All stages read config from `config/pipeline_config.yaml` and write results to `data/commons_image_manifest.sqlite`.

| Stage | Script | Input → Output |
|-------|--------|----------------|
| 01 | `stage_01_model_parsing.py` | Wikipedia wikitext → `data/jr_east_freight_series.csv` |
| 02 | `stage_02_model_fixing.py` | CSV + manual overrides → `data/jr_east_freight_series_wiki_commons.csv` |
| 03 | `stage_03_manifest_crawling.py` | Commons API → `categories` + `images` tables in SQLite |
| 04 | `stage_04_img_crawler.py` | `images` table → downloaded originals in `data/img/`; `download_status` updated |
| 05 | `stage_05_siglip_image_filtering.py` | `google/siglip2-base-patch16-512` zero-shot → `images.siglip_view` / `images.excluded` |
| 06 | `stage_06_llm_metadata_labeling.py` | OpenAI API on `category_path_json` → `images.{submodel, bandai, operator_en, operator_jp, special_formation, special_livery}` |
| 07 | `stage_07_gdino_bbox.py` | `IDEA-Research/grounding-dino-base` + NMS → `crops` table |
| — | `Small_Loss_Trick_Filter_Example.ipynb` | frozen DINOv2 + cyclic-LR linear head → per-crop `noise_score`; human review → `crops.crop_status` |

Stage 05 excludes interior/cab/display images; keyword pre-filter runs in stage_03 (manifest crawl) before download.  
Stage 06 LLM fields are used downstream for subtype label refinement and review queue filtering.  
The Small Loss Trick notebook is run after stage_07, outputs write back to the `crops` table; can be re-run incrementally via the feature cache.

## DB Schema (key tables)

- **`images`**: one row per Commons file; key fields: `series`, `power_type`, `excluded`, `download_status`, `siglip_view`, `category_path_json`, `submodel`, `bandai`, `operator_en`, `operator_jp`, `special_formation`, `special_livery`
- **`crops`**: one row per detected bbox; key fields: `image_id`, `series`, `crop_status` (`pending`/`ok`/`rejected`), `detector_score`, `box_{x1,y1,x2,y2}`, `noise_score` (written by the notebook)
- **`categories`**: Commons categories discovered per series

## Known Issues & Decisions

| Issue | Resolution |
|-------|-----------|
| Digit-first series (113系 etc.) missing | Added `\d` to `series_re` first-char class |
| subtype leaking across H3 sections | Reset `current_subtype` on every H2/H3 change |
| status not normalized for JR Central | Replaced inline replace with `STATUS_MAP` dict |
| 国鉄-inherited cars using wrong operator prefix | Detect via `wiki_title.startswith("国鉄")` → `JNR` |
| Commons merges related series (481→485) | Manual override in stage_02; empty-result series dropped |
| Katakana+E-prefix ambiguity (キヤE195) | Generate two prefix candidates, try both, keep first hit |

## Dev Setup

```bash
conda activate wakareeru
pip install -e ".[dev]"
cp .env.example .env
```

Linting: `ruff check src/`  
Tests: `pytest`  
Python ≥ 3.11 required.

## Working Style Notes

- Only implement tasks explicitly requested — dataset specifics involve details that can't be fully specified in advance
- Do not add features, abstractions, or error handling beyond what the immediate task requires
- **Keep this document accurate**: when pipeline stages are added, renamed, or their behavior changes in a non-obvious way, update the relevant section here as part of the same change. The pipeline table and DB schema sections are the highest-signal parts — keep them in sync with the actual scripts and `constants.py`.
