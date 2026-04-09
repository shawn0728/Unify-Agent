<div align="center">



  <h1 style="margin: 0; font-size: 1.8em;">
    <img src="./images/unify_agent_logo.png" alt="Logo Icon" width="50" style="vertical-align: middle; margin-right: 10px;">
    Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis
  </h1>

  [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.29620)
  [![alphaXiv](https://img.shields.io/badge/discussion-A42C25?style=for-the-badge&logo=arxiv&logoColor=white&color=blue)](https://www.alphaxiv.org/abs/2603.29620)
  [![Github](https://img.shields.io/badge/Unify_Agent-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/shawn0728/Unify-Agent)
  [![Hugging Face Collection](https://img.shields.io/badge/Unify_Agent_Collection-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/csfufu/unify-agent)
  [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/HuggingPapers/status/2040288191534543001)

  [![Awesome](https://awesome.re/badge.svg)](https://github.com/shawn0728/Unify-Agent)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  ![](https://img.shields.io/github/last-commit/shawn0728/Unify-Agent?color=green) 

</div>

## 📖 Introduction


This paper presents **Unify-Agent**, an end-to-end **unified multimodal agent** for **world-grounded image synthesis**. Unlike conventional text-to-image models that rely only on fixed parametric memory, Unify-Agent can actively access external world knowledge at inference time, enabling more faithful generation of **real people, cultural symbols, rare IPs, historical scenes, scientific concepts**, and other long-tail entities.

The core challenge of factual image generation is not just producing visually plausible images, but correctly capturing the target’s **identity-defining visual attributes**. Existing agentic systems usually connect retrieval, reasoning, and generation through loosely coupled pipelines, making it difficult to effectively transform external evidence into accurate visual guidance.

To address this, Unify-Agent unifies four capabilities in a single model:

1. **THINK**: understand the prompt and identify missing knowledge.
2. **RESEARCH**: retrieve relevant textual and visual evidence.
3. **RECAPTION**: convert retrieved knowledge into structured generation guidance.
4. **GENERATE**: synthesize the final grounded image.

A key insight is that **unifying understanding and generation improves both**. By combining high-level semantic representations with low-level generative priors, Unify-Agent can better interpret retrieved references and produce images that are more faithful to real-world knowledge.

To evaluate this setting, the paper introduces **FactIP**, a benchmark covering 12 categories focused on rare identities and long-tail concepts. Experiments show that Unify-Agent significantly improves factual visual synthesis, outperforming its base model and strong open-source baselines across **FactIP, WiSE, KiTTEN, and T2I-FactualBench**.

This work highlights a new paradigm for text-to-image generation: moving from **closed-book generation** to **open-book, agentic generation**, where models actively reason over external knowledge before synthesis.



---

## 🧮 Showcase

![Showcase](./images/showcase.png)

High-quality samples from our **Unify-Agent**, highlighting its excellence in unified multi-image generation and agentic search enhanced world knowledge integration. It delivers strong cross-image consistency, broad stylistic versatility, and more faithful, knowledge-grounded visual generation across diverse concepts and scenarios—even for up-to-date real-world queries, such as generating images of the top three finishers (Kimi Antonelli, George Russell, Lewis Hamilton) of the 2026 Chinese Grand Prix in Shanghai.



![Comparison](./images/comparison.png)


Qualitative comparison of multi-image generation results on knowledge-intensive prompts involving historical figures, fictional characters, products, and stylized toys. Our method consistently produces images that better preserve subject identity, fine-grained attributes, and prompt-specific details, while achieving stronger real-world knowledge grounding than competing baselines, including Flux-1, Bagel-7b, Hunyuan, and Stable Diffusion.

---

## 🍭 Pipeline

**Overview of the agentic pipeline of our method**.
Given an input prompt, our framework first performs prompt understanding and cognitive gap detection to identify missing but visually critical attributes. It then acquires complementary multimodal evidence through textual evidence searching and visual evidence searching. Based on the collected evidence, the model grounds the generation process with two types of constraints: identity-preserving constraints that capture character-specific visual traits, and scene-compositional constraints that specify pose, environment, garment, and overall mood. These grounded constraints are then integrated into an evidence-grounded recaptioning module, which produces a detailed caption for the downstream image generator to synthesize the final image.

![method_overview](./images/method.png)



**Overview of our data pipeline**.
Starting from long-tail IP collection, we construct user instructions and Ground Truth images, build multimodal research trajectories with textual and visual evidence, and finally perform evidence-grounded recaption annotation to obtain high-quality training samples. The resulting dataset supports both SFT trajectory learning and the FactIP benchmark, which evaluates generation quality in terms of clarity, content, aesthetics, and relevance.

![dataset](./images/dataset.png)


---

## 🏝️ Reasoning Example

1. Image generated for the prompt: **"The copper is burning, highlighting the color".**

![reasoning_example1](./images/case_1.png)


2. Image generated for **Grigory Perelman** scribbling mathematical equations.

![reasoning_example2](./images/case_2.png)

---

## 🎯 More details about Benchmark

### Benchmark Construction

Hierarchical category distribution of **FactIP** Bench, consisting of three major groups (Character, Scene, and Object) and 12 fine-grained subcategories. Category-wise comparison of different methods on FactIP Bench, where the radar chart presents the overall scores across all subcategories.

![dataset](./images/construction.png)

The full benchmark contains three major categories and 12 fine-grained subcategories, totaling **2,462 prompts**. We will also release **500 prompts** from the full benchmark as a **test mini** subset, with the hierarchical category distribution (Character, Scene, Object and their 12 fine-grained subcategories) **strictly maintained in proportion** to the full FactIP Bench.


| Category | Subcategory | Description | Num |
|---|---|---|---:|
| **CHARACTER** | Animation | Animated characters, creatures, equipment, and iconic locations from anime and animated media. | 438 |
| **CHARACTER** | Comic | Characters and visual elements originating from comic books and manga series. | 363 |
| **CHARACTER** | Celebrity | Prominent figures across diverse domains, including scientists, political leaders, business executives, athletes, and entertainment personalities. | 300 |
| **CHARACTER** | Game | Video game characters, weapons, equipment, and other in-game visual elements. | 272 |
| **CHARACTER** | Mascot | Official mascots representing Olympic Games, regional events, and corporate brands. | 77 |
| **CHARACTER** | Mythology | Universally recognized mythological narratives and legendary figures, e.g., Kuafu Chasing the Sun. | 50 |
| **OBJECT** | Food | Cuisines, regional delicacies, desserts, and beverages with cultural significance. | 316 |
| **OBJECT** | Cultural Relic / Art | National treasures, classical calligraphy, paintings, sculptures, and fine art pieces. | 126 |
| **OBJECT** | Toy | Collectible figures, designer toys, and model kits with cultural relevance, e.g., Labubu. | 123 |
| **OBJECT** | Animal / Plant | Individually notable animals and plants with distinct public recognition, e.g., Giant Panda Qizai. | 50 |
| **SCENE** | Landmark | Renowned scenic spots, architectural landmarks, monuments, and heritage sites. | 297 |
| **SCENE** | Festival / Celebration | Visual elements and symbols associated with well-known festivals and cultural celebrations. | 50 |

### Evaluation Example

An example of MLLM evaluation for Popovich drawing a play.


![eval_example](./images/eval_1.png)

---



## 🛠️ Installation

```bash
git clone https://github.com/shawn0728/Unify-Agent.git
cd Unify-Agent
pip install -r requirements.txt
```

> **Note:** [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) is recommended but commented out in `requirements.txt`. Install it separately if your GPU supports it:
> ```bash
> pip install flash-attn --no-build-isolation
> ```


## 🏋️ SFT Training

We provide a ready-to-use script for supervised fine-tuning (SFT) of the Unify-Agent model. The training is built on top of [Bagel](https://github.com/ByteDance-Seed/Bagel) and uses FSDP for distributed training with W&B logging.

### Prerequisites

| Requirement | Details |
|---|---|
| **GPU** | 8 × A100/H100 per node (80 GB recommended) |
| **Python** | 3.10+ |
| **PyTorch** | 2.5.1+ with CUDA support |
| **Base Model** | [Bagel-7B-MoT](https://huggingface.co/ByteDance-Seed/Bagel-7B-MoT) checkpoint |
| **ViT** | [SigLIP-SO400M](https://huggingface.co/google/siglip-so400m-patch14-384) (NaViT variant) checkpoint |

### Quick Start

**1. Set required environment variables:**

```bash
export WANDB_API_KEY="your_wandb_api_key"
export RESUME_FROM="/path/to/Bagel-7B-MoT"
export VIT_PATH="/path/to/siglip-so400m-14-980-flash-attn2-navit"
```

**2. Single-node training (8 GPUs):**

```bash
bash SFT/scripts/train.sh 1 0 127.0.0.1 29500
```

**3. Multi-node training (e.g., 4 nodes):**

```bash
# On each node, replace <node_rank> with 0, 1, 2, 3
bash SFT/scripts/train.sh 4 <node_rank> <master_ip> 29500
```

### Script Usage

```
bash SFT/scripts/train.sh <nnodes> <node_rank> <master_addr> <master_port>
```

| Argument | Description |
|---|---|
| `nnodes` | Total number of nodes |
| `node_rank` | Rank of the current node (0-indexed) |
| `master_addr` | IP address of the rank-0 node |
| `master_port` | Port for distributed rendezvous |


### Key Environment Variables

All training hyperparameters can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `RESUME_FROM` | *(required)* | Path to the base model checkpoint |
| `VIT_PATH` | *(required)* | Path to the SigLIP ViT checkpoint |
| `WANDB_API_KEY` | *(required)* | Weights & Biases API key |
| `NPROC_PER_NODE` | `8` | Number of GPUs per node |
| `RESULTS_DIR` | `./outputs/sft` | Directory for logs and metrics |
| `CHECKPOINT_DIR` | `./outputs/sft` | Directory for saving checkpoints |
| `WANDB_PROJECT` | `unify-agent-sft` | W&B project name |
| `WANDB_NAME` | `sft_run` | W&B run name |
| `NUM_WORKERS` | `1` | DataLoader workers per rank |
| `KEEP_LAST_N` | `8` | Number of recent checkpoints to keep |
| `LAUNCH_MODE` | `static` | Launch mode: `static` or `elastic` |
| `COMM_PROFILE` | `socket_safe` | NCCL comm profile: `socket_safe` \| `ib_min` \| `ib_perf` |

### Training Hyperparameters

The default SFT recipe uses the following hyperparameters:

| Hyperparameter | Value |
|---|---|
| Learning rate | `5e-5` |
| Warmup steps | `500` |
| Max gradient norm | `5.0` |
| Expected tokens per batch | `40240` |
| Max tokens per batch | `41520` |
| Max tokens per sample | `40240` |
| Save every N steps | `500` |
| CE loss weight | `1.0` |
| MSE loss weight | `1.0` |
| Special token CE weight | `3.0` |

### Dataset Configuration

The SFT training uses a YAML config file to specify datasets. See [`SFT/data/configs/agent_data.yaml`](SFT/data/configs/agent_data.yaml) for the agentic SFT config and [`SFT/data/configs/example.yaml`](SFT/data/configs/example.yaml) for the general training config.

For detailed information on data preparation, dataset format, and model/training configuration tables, refer to [`SFT/train/TRAIN.md`](SFT/train/TRAIN.md).


## 📊 FactIP Benchmark Evaluation

We provide the [**FactIP**](https://huggingface.co/datasets/csfufu/FactIP) benchmark for evaluating knowledge-grounded image generation. The evaluation pipeline consists of three stages: **Generate** → **Score** → **Calculate**.

### Step 1: Download the FactIP Dataset

```bash
# Option A: HuggingFace CLI (recommended)
huggingface-cli download csfufu/FactIP --repo-type dataset --local-dir ./FactIP

# Option B: Git LFS
git clone https://huggingface.co/datasets/csfufu/FactIP
```

The dataset contains prompts (`test.json` / `test_mini.json`), ground-truth reference images, and category metadata.

### Step 2: Generate Images

Use `eval/bagel_infer_batch.py` to generate images from the FactIP prompts:

```bash
python eval/bagel_infer_batch.py \
    --model_path /path/to/your/model \
    --prompt_json ./FactIP/test.json \
    --output_dir ./results/my_model \
    --num_gpus 8 \
    --seed 42 \
    --think  # optional: enable chain-of-thought reasoning
```

### Step 3: Score with MLLM Judge

Score each generated image against the ground-truth references using an OpenAI-compatible multimodal API:

```bash
export OPENAI_API_KEY="your_api_key"

python eval/score_factip.py \
    --base-dir ./results/my_model \
    --workers 8
```

You can customize the judge model via environment variable or CLI flag:

```bash
# Use a custom model
export FACTIP_EVAL_MODEL="gpt-4o"

# Point to ground-truth images (if stored separately)
export FACTIP_GT_DIR="./FactIP/images"
```

### Step 4: Calculate Aggregate Scores

Aggregate per-category and overall scores (reported on a **0–100 scale**):

```bash
python eval/calculate.py --base_dir ./results/my_model
```

This produces an `overall_score.txt` report with per-subtask breakdowns:

```
Overall = 0.05 × clarity + 0.10 × content_quality + 0.10 × aesthetics + 0.75 × text_relevance_ip
```

### One-Click Pipeline

We also provide a single script that runs all three stages end-to-end:

```bash
bash eval/run_factip_eval.sh \
    --model_path /path/to/your/model \
    --prompt_json ./FactIP/test.json \
    --gt_dir ./FactIP/images \
    --output_dir ./results/my_model \
    --num_gpus 8 \
    --think
```

| Flag | Description |
|---|---|
| `--model_path` | Path to the model checkpoint |
| `--prompt_json` | Path to the FactIP prompt JSON |
| `--gt_dir` | Path to the ground-truth reference images |
| `--output_dir` | Directory for generated images and scores |
| `--num_gpus` | Number of GPUs for parallel generation (default: 8) |
| `--think` | Enable chain-of-thought reasoning |
| `--score_workers` | Concurrent API workers for scoring (default: 4) |
| `--eval_model` | Override the MLLM judge model name |
| `--skip_generate` | Skip generation, run scoring + calculation only |
| `--only_calculate` | Skip generation and scoring, run calculation only |


## 🚧 TODO

All the code, benchmark, and checkpoints have entered the final approval stage. Stay tuned — once the approval process is complete, we will release them **ASAP**.


## 🙌 Acknowledgements
We thank the open-source community for the wonderful works of [Bagel](https://github.com/ByteDance-Seed/Bagel) that inspired this project.



## 📮 Contact

For questions, feedback, or collaboration opportunities, feel free to reach out: csfufu0728@gmail.com

## 📄Citation

If you find our works useful for your research, please consider citing:
```
@article{chen2026unify,
  title={Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis},
  author={Chen, Shuang and Shou, Quanxin and Chen, Hangting and Zhou, Yucheng and Feng, Kaituo and Hu, Wenbo and Zhang, Yi-Fan and Lin, Yunlong and Huang, Wenxuan and Song, Mingyang and others},
  journal={arXiv preprint arXiv:2603.29620},
  year={2026}
}


@article{feng2026gen,
  title={Gen-Searcher: Reinforcing Agentic Search for Image Generation},
  author={Feng, Kaituo and Zhang, Manyuan and Chen, Shuang and Lin, Yunlong and Fan, Kaixuan and Jiang, Yilei and Li, Hongyu and Zheng, Dian and Wang, Chenyang and Yue, Xiangyu},
  journal={arXiv preprint arXiv:2603.28767},
  year={2026}
}
```

## ⭐️ Star HistoryMore actions

[![Star History Chart](https://api.star-history.com/svg?repos=shawn0728/Unify-Agent&type=Date)](https://star-history.com/#shawn0728/Unify-Agent&Date)
