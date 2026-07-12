+++
title = "Vision-Language-Action (VLA) Models: A Review of Recent Progress"
date = "2025-09-16"
updated = "2026-07-11"
description = "Recent VLAs are moving from discrete to continuous control and from single-system to dual-system designs."
template = "blog-page.html"

[taxonomies]
tags = ["Review", "VLA", "Embodied AI"]

[extra]
katex = true
+++

> I am new to this field — feel free to discuss and bring up any questions! \
> This post is adapted from my [slides](/files/blog/vla-models-review/vla-review-0911.pdf).

## Background and Concepts

<!-- ### The Evolution of Embodied AI -->

### The Concept of Vision-Language-Action (VLA) Models

In my understanding, *Vision-Language-Action (VLA) models*[^1] are multimodal foundation models for embodied AI. They take **vision** (e.g., observations in video streams) and **language** (e.g., user instructions) as inputs, and generate low-level robot **actions** (i.e., the *control policy*) as outputs.
A VLA uses a *vision-language model (VLM)* for *vision-and-language-conditioned action generation*.

{{ image(src="/img/blog/vla-models-review/timeline.png", dimmable=true) }}

### Add VLM-Based Task Planners for Long-Horizon Tasks

Early VLAs focus on low-level robot control, which alone is insufficient for complex, long-horizon tasks unless the entire task is trained end to end.
One approach is to add an LLM/VLM-based *task planner* that decomposes a long-horizon task into simpler subtasks for the VLA to complete in sequence.
Earlier work usually uses a separate model as the task planner, while recent work shares one VLM backbone between task planning and control (i.e., a *dual-system* design).

{{ image(src="/img/blog/vla-models-review/hierarchical-policy.png", dimmable=true) }}

## Recent VLA Progress

I summarize recent VLA progress along two axes: **from *system-1-only* (control) to *dual-system* (planning + control), and from *discrete* actions to *continuous* actions.** This gives four quadrants:

{{ image(src="/img/blog/vla-models-review/quadrants.png", invertible=true) }}

The following sections introduce these categories in turn.

### Discrete VLA

A *discrete VLA* generates *discrete action tokens*. It maps low-level robot actions to discrete tokens, then trains the VLM to generate them *autoregressively*, much like text tokens.
This gives action and language a common representation and extends next-token prediction to next-action prediction.
However, autoregressive generation can introduce high latency and low control frequency because each new action token requires another pass through the VLA.

Some representative methods:

- **RT-2**[^2] (ViT + PaLI-X/PaLM-E): pioneering work that introduced and popularized the term "VLA."
- **OpenVLA**[^3] (DINOv2 & SigLIP + Llama 2 7B): an influential open-source VLA model (3.8k stars on [GitHub](https://github.com/openvla/openvla)).
- **FAST**[^4]: an action tokenizer that compresses action sequences with DCT (Discrete Cosine Transform).

{{ image(src="/img/blog/vla-models-review/openvla.png", dimmable=true) }}

### Continuous VLA

A *continuous VLA* samples from a *continuous action space*. This allows smoother, higher-precision control, but is harder to train on top of existing language models.
Physical Intelligence addressed this by adding a *flow-matching action expert* to a pretrained VLM, and trained $\pi_0$[^5] on top of a pretrained PaliGemma 2B VLM.

The pretrained VLM provides **semantic understanding and generalization** from internet-scale data, while the flow-matching action expert learns **high-frequency (up to 50 Hz) control** from cross-embodiment data. The model can then be fine-tuned for difficult or unseen tasks.

{{ image(src="/img/blog/vla-models-review/pi0.png", dimmable=true) }}

Similarly, NVIDIA Isaac trained GR00T N1(.5)[^6], which combines a pretrained Eagle-2 VLM with a diffusion-based action head as a foundation model for generalist humanoid robots. In both $\pi_0$ and GR00T, the VLM backbone and action expert communicate through attention modules, conditioning generated actions on the VLM hidden states (i.e., KV). There are two technical differences:

- **Attention mechanism**: $\pi_0$ concatenates the VL and action KV and conducts masked self-attention (a [blog](https://huggingface.co/blog/pi0) illustrates this clearly); GR00T directly conducts cross-attention between the two parts.
- **Number of VLM layers involved**: $\pi_0$ aligns the number of layers in the action expert to the VLM backbone, and conducts self-attention in each layer (MoE-like); GR00T only keeps the hidden states of the last layer in the VLM[^7], and conducts cross-attention with it for each layer.

### Dual-System VLA

Rather than pairing a system-1 VLA with a separate LLM/VLM task planner, a *dual-system VLA* also uses its VLM backbone for planning. System 2 (high-level planning) and system 1 (low-level control) therefore **share one VLM**.
The VLA learns to **predict subtasks directly from user instructions**, improving open-world generalization while using fewer resources than a separate planner model.

> Question: does sharing a VLM improve performance by aligning systems 1 and 2? On the other hand, could their objectives interfere with each other?

{{ image(src="/img/blog/vla-models-review/pi05.png", dimmable=true) }}

$\pi_{0.5}$[^8], trained by Physical Intelligence, is the first model in this category. Compared with $\pi_0$, its training data also includes object detection, instructions, subtask commands, and discrete actions.
At inference time, its VLM first predicts a subtask from the high-level prompt (system 2), then the VLM and action expert execute that subtask (system 1).
Recent VLAs such as Galaxea's G0[^9] and X Square Robot's WALL-OSS[^10] follow this training recipe and inference scheme.
While most of these models are continuous VLAs, WALL-OSS also includes a discrete version with FAST tokenization ([WALL-OSS-FAST](https://huggingface.co/x-square-robot/wall-oss-fast)).

Their repositories and open-source status:

- $\pi_{0.5}$: Weights are available, and part of the code is released at [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi). The VLM subtask-prediction inference code is not available.
- G0: Weights and the open-world dataset are available, and part of the code is released at [OpenGalaxea/G0](https://github.com/OpenGalaxea/G0). It currently supports only real-robot inference.
- WALL-OSS: Weights and code are available at [X-Square-Robot/wall-x](https://github.com/X-Square-Robot/wall-x).

## Summary and Outlook

Since RT-2 appeared in 2023, VLAs have rapidly evolved from discrete to continuous control, and from single-system to dual-system designs.
I expect *native multitasking* to become another trend: embodied agents should handle fundamentally different tasks (e.g., chat, memory, and navigation) rather than being restricted to actions.
Recent models already share an internet-scale pretrained VLM backbone between task planning and control. Although both are still action-oriented, this provides a foundation for sharing one VLM across a broader set of tasks.

I am currently working on this *multi-expert foundation model* for native multitasking in embodied agents — feel free to contact me for discussion and collaboration!

[^1]: Yueen Ma et al., ["A Survey on Vision-Language-Action Models for Embodied AI"](https://arxiv.org/abs/2405.14093), arXiv, 2024.

[^2]: Brianna Zitkovich et al., ["RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"](https://arxiv.org/abs/2307.15818), CoRL, 2023.

[^3]: Moo Jin Kim et al., ["OpenVLA: An Open-Source Vision-Language-Action Model"](https://arxiv.org/abs/2406.09246), CoRL, 2024.

[^4]: Karl Pertsch et al., ["FAST: Efficient Action Tokenization for Vision-Language-Action Models"](https://arxiv.org/abs/2501.09747), RSS, 2025.

[^5]: Kevin Black et al., ["$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control"](https://arxiv.org/abs/2410.24164), RSS, 2025.

[^6]: Johan Bjorck et al., ["GR00T N1: An Open Foundation Model for Generalist Humanoid Robots"](https://arxiv.org/abs/2503.14734), arXiv, 2025.

[^7]: Specifically, the language backbone of the VLM in GR00T N1.5 is fine-tuned from the first 14 layers of the pre-trained [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (28 layers in total), according to my test of similarity between the weights.

[^8]: Physical Intelligence et al., ["$\pi_{0.5}$: A Vision-Language-Action Model with Open-World Generalization"](https://arxiv.org/abs/2504.16054), arXiv, 2025.

[^9]: Tao Jiang et al., ["Galaxea Open-World Dataset and G0 Dual-System VLA Model"](https://arxiv.org/abs/2509.00576), arXiv, 2025.

[^10]: X Square Robot, ["WALL-OSS: Igniting VLMs toward the Embodied Space"](https://x2robot.cn-wlcb.ufileos.com/wall_oss.pdf), white paper, 2025.
