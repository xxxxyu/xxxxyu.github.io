+++
title = "“口袋里的 GPT”，离我们还有多远？"
date = 2023-11-21
updated = 2023-11-21
description = "唠一唠端侧大模型部署那些事。"

[taxonomies]
tags = ["LLM Deployment", "Edge AI"]

[extra]
footnote_backlinks = true
quick_navigation_buttons = true
toc = true
+++

# 初识 GPT

2022年11月30日，OpenAI 正式发布 ChatGPT。

2022年12月5日，ChatGPT 发布仅5天便突破100万用户。

在国内，ChatGPT 首先在2022年12月在科技圈引起关注；而时间来到2023年2月，ChatGPT 更是在国内彻底爆火出圈，一时间风头无两。最近一段时间，Humane 和 GPTs 等又再度引起了大家的关注。在 2023 年，你从不关心 AI 的父母也要问问你，**“GPT”是什么？**

GPT 全称为 **Generative Pre-Trained Transformer**，即生成式预训练 Transformer 模型。具体来说，Transformer 是最早于2017年提出的一种基于 Attention 的神经网络架构[1]，而 GPT 则是由 OpenAI 基于此架构开发的一系列经过大规模文本的预训练、可以生成自然语言文本的模型。当下倍受关注的“大模型”主要就是指以 GPT 为代表的这类基于 Transformer 架构的大规模生成式模型。

最近一年来，除了 GPT 外，新的大模型如雨后春笋一般发布，例如 Meta 开源的 LLaMa 以及基于 LLaMa 的诸多模型。这些模型整体上仍沿用类似于 GPT 的架构，但在模型超参、训练数据、训练方法等方面各有不同。除了“大模型”这个称谓外，大语言模型（Large Language Model, LLM）和基础模型（Foundation Model, FM）等通常也是指它们，只是对于其应用各有侧重。为了简便，本文中统称为**大模型**或 **LLM**。

## 口袋里的 GPT

现阶段的大模型在对话、写作、编程等任务上已经表现出接近甚至超越人类的水平。这让很多人觉得我们距离通用人工智能（Artificial General Intelligence, AGI）又近了一步，大模型也的确展现出了改变现有生产方式的可能性。人们的想象力在此时无限延展，以往只出现在科幻作品中的场景似乎也近在咫尺，颇有“未来已来”之感。

**让每个人都拥有一个定制的大模型，并且成为随身携带的私人助理**，就是其中颇具吸引力的一个想法——它也许是你口袋中的一个智能终端、也许是眼镜等穿戴式设备、亦或是任何更酷的形式。这个 “**口袋里的 GPT**” 可以帮你处理文书，安排日程，甚至代替你与他人（或他们的智能助理）交互。

然而现实与理想之间却总是存在差距。除了当前模型、算法上的差距外，**隐私安全与网络延迟**也是极为重要的因素。一个高度定制化、深度参与用户日常生活的智能助理，必然需要处理大量的用户隐私数据，而使用部署在云端的大模型服务则给个人隐私安全带来了严重的隐患。同时，未来海量用户产生的大量推理请求也会对网络传输带来更大的压力，难以满足某些高实时性应用的要求。**在端侧直接部署大模型，一方面便于解决上述问题，另一方面也将面临独特的挑战。**

# 大模型端侧部署的挑战

## 资源受限

**在端侧部署大模型的首要挑战，就是设备上有限的硬件资源难以满足大模型的需求。** 大模型出色的能力是建立在其庞大的参数量的基础上的。例如，视觉领域经典的 CNN 模型 ResNet 参数量大致在 10M~60M 之间，反观 ChatGPT 使用的 GPT-3.5 模型的参数量最高可达 175B，比前者要大 3~4 个数量级。于是，仅仅加载模型权重所需的内存就从100 MB 上下飙升至几百 GB，更不要提计算中间数据的保存了。

即便是像 LLaMa-7B 这样更适合部署在端侧的“小模型”，也比以 MobileNet 系列为代表的轻量级 CNN 参数量大了 3 个数量级。如果以半精度浮点数估计，需要至少 14 GB 的内存才能够勉强放下全部模型参数，而这已经远超市面上中端设备的内存上限。除了前向推理之外，如果希望根据用户数据在本地对模型进行微调训练，则会对内存与算力提出更高要求。

## 计算低效

**端侧大模型的部署以推理为主，而大模型本身推理计算的低效，是其在端侧部署的另一个重要挑战。**即便我们设法压缩模型并提供匹配的硬件资源，解决了**可行性**的问题，仍然还需要解决大模型运行的**延时和功耗**带来的**实用性**问题——我们当然不希望我们的数字小秘书以秒为单位龟速输出，同时还热得烫手（> <）。而当前大模型推理低效的主要原因，在于其**计算-访存比低，无法充分利用计算资源**。

所谓的计算-访存比，也就是系统领域著名的 **Roofline Model[2]** 中的 Operational Intensity（计算强度）。它定义为每访问 1 byte 内存平均需要完成的计算操作次数。对于给定的硬件平台，目标任务的计算-访存比越低，计算单元每完成一次计算平均需要访问的内存也就越多。当硬件的内存带宽被占满时，即便再减少计算量或提供更多算力，也无法带来实际的性能提升。这时称系统是内存瓶颈的（memory-bound）；反之则是计算瓶颈（compute-bound）。**端侧大模型的推理就是典型的内存瓶颈的任务。**

**端侧大模型推理的计算-访存比低（内存瓶颈），是由其工作负载的特点决定的。** 由于硬件资源和应用场景的限制，端侧大模型通常不会处理高并发的请求，且大部分时候仅有一个请求，即 batch size = 1。同时，模型处理的文本长度也较为有限。 于是在生成阶段，模型中的计算主要是矩阵向量乘和“高瘦”的矩阵乘，其模型权重每次加载后参与计算的次数较少，于是权重加载（内存访问）就成为了系统的瓶颈。

# 现有方法

为了解决前文所述的大模型在端侧部署的**资源受限**和**计算低效**两大问题，已有的工作可以分为模型压缩与推理优化两类方法。在实际部署中，两类方法往往需要综合使用，以达到最优效果。

## 模型压缩

要想降低模型的内存与算力开销，最直接的方法就是降低模型的大小。大模型常用的压缩方法包括量化、剪枝、蒸馏和低秩分解等。本节简单介绍这几种常用方法，也欢迎感兴趣的同学自行深入了解。

**量化[4-7]（Quantization）** 即用更少的 bit 数来表示模型参数，从而有效降低模型大小。按照量化后权重是否需要微调，量化方法还可进一步分类为训练后量化（Post-Training Quantization, PTQ）和量化感知训练（Quantization-Aware Training, QAT）。低 bit 量化（如 4-bit，3-bit）是当前研究探索的一个重要方向。

**剪枝[8-10]（Pruning）** 即去除模型中一部分不重要的权重，从而降低模型的存储与计算开销。剪枝方法也可进一步分为结构化（structured）与非结构化（unstructured）两种，其中结构化剪枝对硬件计算更加友好。

**蒸馏（Distillation）** 即使用已有的效果较好的 Teacher 模型（参数量大、精度高），去指导训练一个轻量的 Student 模型（参数量小，精度低），使得小模型输出与大模型接近。蒸馏方法常与量化和剪枝方法配合使用。

**低秩分解（Low-Rank Factorization）** 即通过两个低秩矩阵的乘积来近似原权重矩阵，从而降低模型参数量和计算量。LoRA[11]（Low-Rank Adaptation）及其变体就是一种利用低秩分解实现高效的模型微调的方法。

## 推理优化

除了模型的压缩外，端侧 LLM 推理的计算过程也存在许多优化空间。为了解决延时、内存等方面的问题，已有的工作中提出了如下的方法。

**KV Cache** 是一种以内存换时间的优化方法，在目前的推理框架中被普遍使用。其核心思想是将 Attention 计算过程中每次迭代包含重复计算的张量（也即 K 和 V）保存下来，并且随着序列的生成进行增量更新，从而避免重复计算。当序列长度增加时，KV Cache 的大小也会显著增长，需要采用适当的策略进行内存管理[12,13]。

**投机采样[12,13]（Speculative Sampling）** 能够 small-batch 场景下提高吞吐量。所谓投机采样，即先通过一个轻量的小模型（draft model）生成（猜测）一组 token，再交由大模型（oracle model）进行评估检验。对 draft model 一次生成的多个 token 的验证没有前后依赖关系，因此可以并行执行，从而增大 batch size，提高吞吐。

**算子优化**目前主要集中于对 **Attention** 计算的优化。在大模型推理中，FFN 的计算相当简单，优化空间也较小；而 Attention 计算中包含多次矩阵乘和非线性操作，优化空间也相对更大。近期代表工作有 Flash-Decoding[17] 和 FlashDecoding++[18] 等。

**流式加载（Stream Loading）** 即每次仅加载一部分权重到内存（或显存）中进行推理，从而降低整体的内存（显存）占用。FlexGen[19] 通过对 GPU-CPU offloading 策略的设计实现了高吞吐的单卡 LLM 推理。但在要求时延而非吞吐的场景下，由于 LLM 加载权重的开销过大，这类方法并不适用。

# 总结与展望

本文主要介绍了目前端侧大模型部署中的挑战与已有技术，其中涵盖了模型压缩与系统优化中的多个方向。**笔者希望通过阅读本文，感兴趣的同学能够快速补充背景知识、了解最新进展。**

虽然文中已列出许多已有工作，但如何通过模型压缩与系统角度的协同优化、进一步降低大模型推理的硬件需求和功耗、提升大模型在资源受限的设备上的运行效率，仍是一个重要的待解决问题。**欢迎感兴趣的同学深入交流讨论（以及写得不好的地方多拍砖 0.0）。**

# 参考文献

[1] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J]. Advances in neural information processing systems, 2017, 30.

[2] Williams S, Waterman A, Patterson D. Roofline: an insightful visual performance model for multicore architectures[J]. Communications of the ACM, 2009, 52(4): 65-76.

[3] Haotian Tang, Shang Yang, Ji Lin, et al. TinyChat: Large Language Model on the Edge[EB/OL]. 2023-09-06. Available: https://hanlab.mit.edu/blog/tinychat.

[4] Frantar E, Ashkboos S, Hoefler T, et al. Gptq: Accurate post-training quantization for generative pre-trained transformers[J]. arXiv preprint arXiv:2210.17323, 2022.

[5] Xiao G, Lin J, Seznec M, et al. Smoothquant: Accurate and efficient post-training quantization for large language models[C]//International Conference on Machine Learning. PMLR, 2023: 38087-38099.

[6] Lin J, Tang J, Tang H, et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration[J]. arXiv preprint arXiv:2306.00978, 2023.

[7] Liu Z, Oguz B, Zhao C, et al. LLM-QAT: Data-Free Quantization Aware Training for Large Language Models[J]. arXiv preprint arXiv:2305.17888, 2023.

[8] Frantar E, Alistarh D. SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot[J]. 2023.

[9] Ma X, Fang G, Wang X. LLM-Pruner: On the Structural Pruning of Large Language Models[J]. arXiv preprint arXiv:2305.11627, 2023.

[10] Sun M, Liu Z, Bair A, et al. A Simple and Effective Pruning Approach for Large Language Models[J]. arXiv preprint arXiv:2306.11695, 2023.

[11] Hu E J, Shen Y, Wallis P, et al. Lora: Low-rank adaptation of large language models[J]. arXiv preprint arXiv:2106.09685, 2021.

[12] Kwon W, Li Z, Zhuang S, et al. Efficient memory management for large language model serving with pagedattention[C]//Proceedings of the 29th Symposium on Operating Systems Principles. 2023: 611-626.

[13] Pope R, Douglas S, Chowdhery A, et al. Efficiently scaling transformer inference[J]. Proceedings of Machine Learning and Systems, 2023, 5.

[14] Leviathan Y, Kalman M, Matias Y. Fast inference from transformers via speculative decoding[C]//International Conference on Machine Learning. PMLR, 2023: 19274-19286.

[15] Chen C, Borgeaud S, Irving G, et al. Accelerating large language model decoding with speculative sampling[J]. arXiv preprint arXiv:2302.01318, 2023.

[16] Spector B, Re C. Accelerating llm inference with staged speculative decoding[J]. arXiv preprint arXiv:2308.04623, 2023.

[17] Tri Dao, Daniel Haziza, Francisco Massa, and Grigory Sizov. Flash-decoding for long-context inference[EB/OL].

2023-10-12. Available:  https://crfm.stanford.edu/2023/10/12/flashdecoding.html.

[18] Hong K, Dai G, Xu J, et al. FlashDecoding++: Faster Large Language Model Inference on GPUs[J]. arXiv preprint arXiv:2311.01282, 2023.

[19] Sheng Y, Zheng L, Yuan B, et al. FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU[J]. 2023.

[20] Chris Love. AI for Everyone: ChatGPT Prompts That Anyone Can Use[EB/OL]. 2023-04-22. Available: https://www.c-sharpcorner.com/article/ai-for-everyone-chatgpt-prompts-that-anyone-can-use/

[21] Yang J, Jin H, Tang R, et al. Harnessing the power of llms in practice: A survey on chatgpt and beyond[J]. arXiv preprint arXiv:2304.13712, 2023.