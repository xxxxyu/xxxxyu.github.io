+++
title = "视觉—语言—动作（VLA）模型：近期进展综述"
date = "2025-09-16"
updated = "2026-07-12"
description = "近期 VLA 正从离散控制走向连续控制，也从单系统架构转向双系统架构。"
template = "blog-page.html"

[taxonomies]
tags = ["Review", "VLA", "Embodied AI"]

[extra]
katex = true
og_image = "/img/blog/vla-models-review/quadrants.png"
og_image_alt = "按动作表示和系统架构划分的近期 VLA 模型"
ai_translation_source = "en"
+++

> 我刚开始接触这个领域，欢迎讨论，也欢迎提出任何问题！ \
> 本文根据我的[幻灯片](/files/blog/vla-models-review/vla-review-0911.pdf)整理而成。

## 背景与概念

<!-- ### 具身智能的演进 -->

### 视觉—语言—动作（VLA）模型的概念

在我的理解中，*视觉—语言—动作（Vision-Language-Action，VLA）模型*[^1]是面向具身智能的多模态基础模型。它以**视觉**（如视频流中的观测）和**语言**（如用户指令）为输入，输出机器人的底层**动作**（即*控制策略*）。
VLA 使用*视觉—语言模型（VLM）*来*生成以视觉和语言为条件的动作*。

{{ image(src="/img/blog/vla-models-review/timeline.png", dimmable=true, caption="VLA 的概念，以及汇聚到现代 VLA 系统的几条研究脉络。") }}

### 为长程任务加入基于 VLM 的任务规划器

早期 VLA 主要关注机器人的底层控制。除非对整个任务进行端到端训练，否则仅靠底层控制不足以完成复杂的长程任务。
一种做法是加入基于 LLM/VLM 的*任务规划器*，把长程任务分解为较简单的子任务，再交给 VLA 依次完成。
较早的工作通常采用独立模型作为任务规划器；近期工作则让任务规划和控制共享同一个 VLM 主干，即采用*双系统*架构。

{{ image(src="/img/blog/vla-models-review/hierarchical-policy.png", dimmable=true, caption="高层规划器将长程指令分解为多个子任务，再交给底层 VLA 控制策略执行。") }}

## VLA 的近期进展

我从两个维度概括 VLA 的近期进展：**从仅有*系统 1*（控制）走向*双系统*（规划 + 控制），以及从*离散*动作走向*连续*动作。**由此可以得到四个象限：

{{ image(src="/img/blog/vla-models-review/quadrants.png", invertible=true, caption="按动作表示和系统架构划分的近期 VLA。") }}

下面依次介绍这几类方法。

### 离散 VLA

*离散 VLA* 生成*离散动作 token*。它先将机器人的底层动作映射为离散 token，再训练 VLM 像生成文本 token 一样，以*自回归*方式生成动作 token。
这样，动作和语言便有了统一的表示，也把下一个 token 预测扩展为下一个动作预测。
不过，自回归生成可能带来较高延迟和较低的控制频率，因为每生成一个新的动作 token，都需要再次运行 VLA。

代表性方法包括：

- **RT-2**[^2]（ViT + PaLI-X/PaLM-E）：首次提出并推广了“VLA”这一术语的开创性工作。
- **OpenVLA**[^3]（DINOv2 & SigLIP + Llama 2 7B）：颇具影响力的开源 VLA 模型（在 [GitHub](https://github.com/openvla/openvla) 上有 3.8k stars）。
- **FAST**[^4]：一种使用 DCT（离散余弦变换）压缩动作序列的动作 tokenizer。

{{ image(src="/img/blog/vla-models-review/openvla.png", dimmable=true, caption="OpenVLA 将机器人动作视为离散 token，由预训练 VLM 以自回归方式生成。") }}

### 连续 VLA

*连续 VLA* 从*连续动作空间*中采样。这样能实现更平滑、精度更高的控制，但也更难在现有语言模型之上训练。
Physical Intelligence 的做法是在预训练 VLM 上加入一个*流匹配动作专家*，并基于预训练的 PaliGemma 2B VLM 训练 $\pi_0$[^5]。

预训练 VLM 从互联网规模的数据中获得**语义理解和泛化能力**，流匹配动作专家则从跨本体数据中学习**高频（最高 50 Hz）控制**。之后，还可以针对困难任务或未见过的任务对模型进行微调。

{{ image(src="/img/blog/vla-models-review/pi0.png", dimmable=true, caption="π₀ 将预训练 VLM 与流匹配动作专家结合，实现连续控制。") }}

类似地，NVIDIA Isaac 训练了 GR00T N1(.5)[^6]。它将预训练 Eagle-2 VLM 与基于扩散模型的动作头相结合，作为通用人形机器人的基础模型。在 $\pi_0$ 和 GR00T 中，VLM 主干与动作专家通过注意力模块交互，使生成的动作以 VLM 的隐状态（即 KV）为条件。二者在技术上有两点区别：

- **注意力机制**：$\pi_0$ 将视觉—语言部分与动作部分的 KV 拼接起来，再进行掩码自注意力计算（这篇[博文](https://huggingface.co/blog/pi0)给出了很清楚的图示）；GR00T 则直接在两部分之间进行交叉注意力计算。
- **参与计算的 VLM 层数**：$\pi_0$ 让动作专家的层数与 VLM 主干对齐，并在每一层进行自注意力计算（类似 MoE）；GR00T 只保留 VLM 最后一层的隐状态[^7]，动作专家的每一层都与它进行交叉注意力计算。

### 双系统 VLA

与“系统 1 VLA + 独立 LLM/VLM 任务规划器”的组合不同，*双系统 VLA* 还会使用自身的 VLM 主干进行规划。因此，系统 2（高层规划）和系统 1（底层控制）**共享同一个 VLM**。
VLA 学习**直接根据用户指令预测子任务**，由此提升开放世界中的泛化能力，同时比采用独立规划器模型消耗更少的资源。

> 问题：共享 VLM 是否能通过对齐系统 1 和系统 2 来提升性能？反过来，两者的目标是否也可能互相干扰？

{{ image(src="/img/blog/vla-models-review/pi05.png", dimmable=true, caption="π₀.₅ 在 π₀ 的基础上加入高层子任务预测，实现双系统规划与控制。") }}

Physical Intelligence 训练的 $\pi_{0.5}$[^8] 是这一类别中的首个模型。与 $\pi_0$ 相比，它的训练数据还包括目标检测、指令、子任务命令和离散动作。
推理时，其 VLM 先根据高层提示预测一个子任务（系统 2），随后由 VLM 和动作专家执行该子任务（系统 1）。
近期的 VLA，如 Galaxea 的 G0[^9] 和 X Square Robot 的 WALL-OSS[^10]，都沿用了这一训练方法和推理流程。
这类模型大多采用连续 VLA，但 WALL-OSS 也提供了使用 FAST tokenization 的离散版本（[WALL-OSS-FAST](https://huggingface.co/x-square-robot/wall-oss-fast)）。

各模型的仓库和开源情况如下：

- $\pi_{0.5}$：已公开模型权重，部分代码发布于 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)，但 VLM 子任务预测的推理代码尚未公开。
- G0：已公开模型权重和开放世界数据集，部分代码发布于 [OpenGalaxea/G0](https://github.com/OpenGalaxea/G0)，目前仅支持真机推理。
- WALL-OSS：模型权重和代码均已发布于 [X-Square-Robot/wall-x](https://github.com/X-Square-Robot/wall-x)。

## 总结与展望

自 RT-2 于 2023 年问世以来，VLA 已迅速从离散控制发展到连续控制，也从单系统架构发展到双系统架构。
我认为，*原生多任务*会成为下一个趋势：具身智能体不应局限于执行动作，还应能处理聊天、记忆和导航等本质上不同的任务。
近期的模型已经让任务规划和控制共享基于互联网规模数据预训练的 VLM 主干。尽管这两者仍然以动作为中心，但它们为多个更广泛的任务共享同一个 VLM 奠定了基础。

我目前正在研究这种面向具身智能体原生多任务的*多专家基础模型*，欢迎联系我讨论与合作！

[^1]: Yueen Ma et al., ["A Survey on Vision-Language-Action Models for Embodied AI"](https://arxiv.org/abs/2405.14093), arXiv, 2024.

[^2]: Brianna Zitkovich et al., ["RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"](https://arxiv.org/abs/2307.15818), CoRL, 2023.

[^3]: Moo Jin Kim et al., ["OpenVLA: An Open-Source Vision-Language-Action Model"](https://arxiv.org/abs/2406.09246), CoRL, 2024.

[^4]: Karl Pertsch et al., ["FAST: Efficient Action Tokenization for Vision-Language-Action Models"](https://arxiv.org/abs/2501.09747), RSS, 2025.

[^5]: Kevin Black et al., ["$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control"](https://arxiv.org/abs/2410.24164), RSS, 2025.

[^6]: Johan Bjorck et al., ["GR00T N1: An Open Foundation Model for Generalist Humanoid Robots"](https://arxiv.org/abs/2503.14734), arXiv, 2025.

[^7]: 具体而言，根据我对权重相似度的测试，GR00T N1.5 的 VLM 语言主干由预训练 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)（共 28 层）的前 14 层微调而来。

[^8]: Physical Intelligence et al., ["$\pi_{0.5}$: A Vision-Language-Action Model with Open-World Generalization"](https://arxiv.org/abs/2504.16054), arXiv, 2025.

[^9]: Tao Jiang et al., ["Galaxea Open-World Dataset and G0 Dual-System VLA Model"](https://arxiv.org/abs/2509.00576), arXiv, 2025.

[^10]: X Square Robot, ["WALL-OSS: Igniting VLMs toward the Embodied Space"](https://x2robot.cn-wlcb.ufileos.com/wall_oss.pdf), white paper, 2025.
