# Data Pipeline

The **data pipeline** constructs training trajectories for Unify-Agent by converting a list of IP (Intellectual Property) entries into multi-turn agentic search trajectories paired with generated images. The pipeline consists of three sequential stages, orchestrated by a unified entry point.

## Overview

```
IP CSV / JSON  ─►  Stage 1  ─►  Stage 2  ─►  Stage 3  ─►  Training Data
                  (Prompts)   (Trajectories)  (Images)
```

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| **1 — Prompt Generation** | `stage1_generate_prompt.py` | IP CSV file | `ip_prompts.json` |
| **2 — Trajectory Generation** | `stage2_generate_trajectory.py` | `ip_prompts.json` | Trajectory JSONs + reference images |
| **3 — Image Generation** | `stage3_generate_image.py` | Trajectory JSONs | Generated PNG images |
| **Orchestrator** | `pipeline_unify_data.py` | All of the above | End-to-end run |

## Prerequisites

### Python Dependencies

All dependencies are listed in the top-level `requirements.txt`. Key packages:

- `openai` — LLM and image generation API client
- `requests` — HTTP calls to Serper / Jina APIs
- `pandas` — CSV reading (Stage 1)
- `Pillow` — Image processing and compression
- `tqdm` — Progress bars

### API Keys

The pipeline uses external APIs that require keys set as environment variables:

| Variable | Required | Used By | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | **Yes** | All stages | OpenAI-compatible API key for LLM and image generation |
| `SERPER_API_KEY` | **Yes** (Stage 2) | Stage 2 | [Serper](https://serper.dev) API key for web & image search |
| `JINA_API_KEY` | Optional | Stage 2 | [Jina AI](https://jina.ai) reader API key for full-page content extraction; falls back to search snippets if unset |

### Model Configuration

Each stage uses configurable model names. Override via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | *(OpenAI default)* | Custom base URL for OpenAI-compatible endpoints |
| `STAGE1_MODEL` | `gpt-4o` | Model used in Stage 1 for prompt generation |
| `AGENT_REASONING_MODEL` | `gpt-4o` | Model used in Stage 2 for reasoning, summarization, and image judging |
| `AGENT_MULTITURN_MODEL` | `gpt-4o` | Model used in Stage 2 for multi-turn agent dialogue |
| `IMAGE_GEN_MODEL` | `gpt-image-1` | Model used in Stage 3 for image generation |

## Quick Start

### Option A: Unified Entry Point (Recommended)

```bash
export OPENAI_API_KEY="sk-..."
export SERPER_API_KEY="..."

bash data_pipeline/run_pipeline.sh \
    --source_ip celebrity \
    --ip_data data/ip_prompts.json \
    --output_dir ./output/pipeline
```

### Option B: Orchestrator Script Directly

```bash
python data_pipeline/pipeline_unify_data.py \
    --source_ip celebrity \
    --ip_data data/ip_prompts.json \
    --output_dir ./output/pipeline
```

### Option C: Run Stages Individually

```bash
# Stage 1: Generate prompts from IP CSV
python data_pipeline/stage1_generate_prompt.py \
    --ip_csv data/ip_list.csv \
    --output_json data/ip_prompts.json

# Stage 2: Generate trajectories
python data_pipeline/stage2_generate_trajectory.py \
    --ip_data data/ip_prompts.json \
    --output_dir ./output/trajectories

# Stage 3: Generate images from trajectories
python data_pipeline/stage3_generate_image.py \
    --input_dir ./output/trajectories \
    --output_dir ./output/images
```

## Stage Details

### Stage 1: Prompt Generation

**Script:** `stage1_generate_prompt.py`

Reads an IP CSV file and generates structured image-generation prompts of the form *"Who is doing What in Which scene"* for each IP.

**Input CSV format:**

| Column | Description |
|--------|-------------|
| `index` | Unique identifier for the IP |
| `name_zh` | Chinese name |
| `name_en` | English name |
| `country` | Nationality (determines prompt language: `"中国"` → Chinese, else → English) |
| `category` | IP category (e.g., Celebrity, Animation) |
| `remark` | Optional description |

**Output JSON format:**

```json
{
  "ip_001": {
    "index": "ip_001",
    "country": "中国",
    "ip_name": "雷军",
    "ip_name_zh": "雷军",
    "ip_name_en": "Lei Jun",
    "ip_category": "Celebrity",
    "image_prompt": "雷军站在小米新品发布会的舞台中央...",
    "tag_name": "演讲",
    "language": "zh"
  }
}
```

**CLI flags:**

| Flag | Required | Description |
|------|----------|-------------|
| `--ip_csv` | Yes | Path to the input IP CSV file |
| `--output_json` | Yes | Path to write the output JSON file |
| `--existing_json` | No | Path to an existing JSON whose IPs should be skipped |

**Features:**
- Incremental processing: skips IPs that already exist in the output or existing JSON
- Checkpoint saving every 10 new prompts
- Automatic language detection (Chinese vs English) based on the IP's country

---

### Stage 2: Trajectory Generation

**Script:** `stage2_generate_trajectory.py`

For each IP, conducts a multi-turn agent dialogue that:

1. **Text Search** — Calls `text_search` (via Serper + Jina) to gather background knowledge about the IP
2. **Image Search** — Calls `search_image` (via Serper) to find visual references
3. **Quality Judging** — Scores each downloaded candidate image using an LLM judge, selecting the top-2
4. **Recaption** — Generates a detailed, evidence-grounded scene description referencing `image_1` and `image_2`

**Input:** IP prompts JSON (output of Stage 1)

**Output per IP:**
- `{ip_index}_trajectory.json` — Full multi-turn dialogue trajectory with tool calls and results
- `intermediate/{ip_index}/image_1.jpg` — Best reference image
- `intermediate/{ip_index}/image_2.jpg` — Second-best reference image
- `intermediate/{ip_index}/tmp/` — All candidate images before filtering

**Trajectory JSON structure:**

```json
{
  "ip_index": "ip_001",
  "ip_name": "Lei Jun",
  "image_prompt": "...",
  "language": "zh",
  "country": "中国",
  "turns": [
    {
      "turn": 0,
      "stage": 1,
      "input": "...",
      "response_text": "...",
      "tool_output": "..."
    }
  ],
  "recaption": "...",
  "full_response": [...]
}
```

**CLI flags:**

| Flag | Required | Description |
|------|----------|-------------|
| `--ip_data` | Yes | Path to the IP prompts JSON file |
| `--output_dir` | Yes | Output directory for trajectory files |
| `--ip_index` | No | Process only this specific IP index |

**Key parameters (hardcoded, adjustable in source):**
- `MAX_TURNS = 20` — Maximum dialogue turns
- `MAX_RETRIES = 3` — API retry count
- `MAX_IMAGE_SIZE_BYTES = 5MB` — Images above this threshold are compressed

---

### Stage 3: Image Generation

**Script:** `stage3_generate_image.py`

Reads trajectory JSON files, extracts the `<recaption>` content as the generation prompt, and calls the OpenAI Images API to produce output images.

**CLI flags:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input_dir` | Conditional | — | Directory containing `*_trajectory.json` files |
| `--trajectory_file` | No | — | Process a single trajectory file |
| `--intermediate_dir` | No | `<input_dir>/intermediate` | Stage 2 intermediate directory |
| `--output_dir` | Yes | — | Directory to write generated images |
| `--n_generation` | No | `1` | Number of images to generate per IP |
| `--model` | No | env `IMAGE_GEN_MODEL` | Image generation model name |
| `--size` | No | `1024x1024` | Generated image dimensions |

**Output:**
- `{ip_index}_0.png` — Generated image (numbered if `n_generation > 1`)
- `generation_summary.json` — Summary of all processing results

---

### Orchestrator: pipeline_unify_data.py

Runs Stage 1 → Stage 2 → Stage 3 for each IP in sequence, with incremental progress tracking and the ability to skip/resume stages.

**CLI flags:**

| Flag | Required | Description |
|------|----------|-------------|
| `--source_ip` | Yes | Source IP name (used as output subdirectory) |
| `--ip_data` | Yes | Path to IP data JSON file |
| `--output_dir` | Yes | Base output directory |
| `--ip_index` | No | Process only a specific IP index |
| `--skip_stage1` | No | Skip Stage 1 (prompt loading) |
| `--skip_stage2` | No | Skip Stage 2 (reuse existing trajectories) |
| `--skip_stage3` | No | Skip Stage 3 (image generation) |

**Output directory layout:**

```
{output_dir}/{source_ip}/
├── traj/
│   ├── ip_001_trajectory.json
│   └── ip_002_trajectory.json
├── intermediate/
│   ├── ip_001/
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   └── tmp/
│   └── ip_002/
│       └── ...
├── images/
│   ├── ip_001_0.png
│   └── ip_002_0.png
└── processing_summary.json
```

**Incremental processing:**
- Tracks results in `processing_summary.json`
- Automatically skips fully successful IPs on re-run
- If Stage 2 succeeded but Stage 3 failed, only reruns Stage 3

## Code Structure

```
data_pipeline/
├── __init__.py                     # Package marker
├── _logging.py                     # Shared logging configuration
├── run_pipeline.sh                 # Bash entry point with env checks
├── pipeline_unify_data.py          # Orchestrator (Stage 1 → 2 → 3)
├── stage1_generate_prompt.py       # Stage 1: IP CSV → prompts JSON
├── stage2_generate_trajectory.py   # Stage 2: Prompts → multi-turn trajectories
├── stage3_generate_image.py        # Stage 3: Trajectories → generated images
└── DATA.md                         # This documentation
```

## Architecture Notes

### API Abstraction

All stages use the OpenAI Python SDK (`openai.OpenAI`) as the single LLM/image-generation client. By setting `OPENAI_BASE_URL`, you can point the pipeline at any OpenAI-compatible endpoint (e.g., vLLM, Azure OpenAI, local LLM servers).

### Search Stack

Stage 2 uses a two-layer search stack:

1. **Serper API** (`google.serper.dev`) — Provides Google search results (both text and image)
2. **Jina Reader API** (`r.jina.ai`) — Extracts clean text content from web pages found by Serper

If `JINA_API_KEY` is not set, the pipeline falls back to using search result snippets instead of full-page content.

### Image Quality Judging

Downloaded reference images are scored by an LLM judge on a 0–10 scale across four dimensions:
- **IP Consistency** — Does the image depict the correct character/entity?
- **Layout & Composition** — Is the subject the main focus with visible features?
- **Visual Quality** — Is the image sharp and clear?
- **Watermarks & Obstructions** — Are there distracting overlays?

The top-2 scoring images (score ≥ 0) are selected as reference images for the recaption stage.

### Recaption Format

The final recaption is wrapped in XML tags and follows a strict format:

```xml
<think>
[Analysis of image_1 and image_2, integration strategy]
</think>
<recaption>
[Detailed scene description with preservation statements for facial features]
</recaption>
```

Language is determined by the IP's country: Chinese for `"中国"`, English otherwise.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENAI_API_KEY is not set` | Export the key: `export OPENAI_API_KEY="sk-..."` |
| Stage 2 returns no search results | Verify `SERPER_API_KEY` is valid; check network connectivity |
| Images fail to download | Check proxy settings (`http_proxy` / `https_proxy` if needed) |
| Stage 3 returns empty images | Verify the image generation model supports the requested size |
| `No recaption found` in Stage 3 | Stage 2 may have failed to generate a recaption; check the trajectory JSON |
| Pipeline hangs on a single IP | Stage 2 has a 1-hour timeout, Stage 3 has a 2-hour timeout per subprocess |
