<div align="center">



  <h1 style="margin: 0; font-size: 1.8em;">
    <img src="./images/unify_agent_logo.png" alt="Logo Icon" width="50" style="vertical-align: middle; margin-right: 10px;">
    Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis
  </h1>

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

To evaluate this setting, the paper introduces **FactIP**, a benchmark of **2,462 curated prompts** focused on rare identities and long-tail concepts. Experiments show that Unify-Agent significantly improves factual visual synthesis, outperforming its base model and strong open-source baselines across **FactIP, WiSE, KiTTEN, and T2I-FactualBench**.

This work highlights a new paradigm for text-to-image generation: moving from **closed-book generation** to **open-book, agentic generation**, where models actively reason over external knowledge before synthesis.

## 🧮 Showcase

![Showcase](./images/showcase.png)

High-quality samples from our **Unify-Agent**, highlighting its excellence in unified multi-image generation and agentic search enhanced world knowledge integration. It delivers strong cross-image consistency, broad stylistic versatility, and more faithful, knowledge-grounded visual generation across diverse concepts and scenarios—even for up-to-date real-world queries, such as generating images of the top three finishers (Kimi Antonelli, George Russell, Lewis Hamilton) of the 2026 Chinese Grand Prix in Shanghai.



![Comparison](./images/comparison.png)


Qualitative comparison of multi-image generation results on knowledge-intensive prompts involving historical figures, fictional characters, products, and stylized toys. Our method consistently produces images that better preserve subject identity, fine-grained attributes, and prompt-specific details, while achieving stronger real-world knowledge grounding than competing baselines, including Flux-1, Bagel-7b, Hunyuan, and Stable Diffusion.