+++
title = "用向量查表真正加速低比特 LLM 的并行推理"
date = "2026-07-11"
updated = "2026-07-11"
description = "Vec-LUT 将重复的标量查表转化为连续的向量读取，在 x86 和 ARM CPU 上将并行三值 LLM 推理加速至多 4.2 倍。"
template = "blog-page.html"

[taxonomies]
tags = ["LLM Inference", "Edge AI", "Quantization", "GeMM"]

[extra]
toc = true
og_image = "/img/blog/vec-lut-parallel-low-bit-llm/scalar-vs-vector-lut.png"
og_image_alt = "标量查表与向量查表在并行 LLM 推理中的对比示意图"
ai_translation_source = "en"
+++

> 🏆 ACM MobiSys 2026 <strong class="highlight">最佳论文奖亚军</strong> \
> 🔗 [论文](https://doi.org/10.1145/3745756.3809200) | [代码](https://github.com/OpenBitSys/vlut.cpp) | [演示文稿](https://github.com/OpenBitSys/vlut.cpp/blob/master/media/veclut_slides_mobisys_2026.pdf) | [中文介绍（清华 AIR 公众号）](https://mp.weixin.qq.com/s/_vSD8_r9GVjCppdVG2MqAg)

如今的超低比特 LLM（通常每个权重低于 4 比特）占用的内存很小。例如，微软 2B BitNet 模型的权重只占约 0.4 GB[^1]。照理说，它们应该能在笔记本电脑或智能手机上高速运行，对吗？

**并不必然如此。** 更低的位宽可以减少内存占用，但只有底层推理内核与数据表示和工作负载相匹配时，才能真正转化为速度优势。**查找表（lookup table, LUT）**内核在单 token 生成时表现很好，但并行处理大量 token 时，这种优势会大幅减弱。Vec-LUT 针对的正是这种并行访问模式。在五款 x86 和 ARM 设备上，与每权重位数（bits per weight, BPW）相近的基线相比，它将端到端预填充加速至多 **4.2 倍**；在 Snapdragon 8 Elite 上，其 CPU 实现甚至只用两个 CPU 核心，就超过了手机 NPU（llama.cpp 的 Hexagon 后端）。

{{ video(src="/videos/vec-lut-parallel-low-bit-llm/demo.mp4", autoplay=true, loop=true, max_width="80%", caption="Vec-LUT 与 llama.cpp 对比：在单个 CPU 核心上进行 32 路并行解码，端到端速度提升 3 倍。") }}

---

## 在 CPU 上进行端侧 LLM 推理

移动助手、本地智能体和具身智能系统如今都在使用端侧模型，而这些设备的内存依然有限。量化研究已经从 8 比特[^2]推进到 4 比特[^3]、2 比特[^4]，再到如今的 **1.58 比特三值**模型。这类模型的权重取值为 `{-1, 0, 1}`，微软的 [BitNet](https://github.com/microsoft/BitNet)[^5] 让这种格式得到了广泛关注。

较小的权重取值集合也让**基于查找表（LUT）的推理**成为可能：内核不必在运行时反量化低比特权重、再与激活值相乘，而是可以预先计算一张小表，用查表替代大部分算术运算。此前，[bitnet.cpp](https://github.com/microsoft/BitNet)[^6] 和 [T-MAC](https://github.com/microsoft/T-MAC)[^7] 等基于 LUT 的系统已经证明，CPU 在单 token 生成场景中也能具备竞争力。Vec-LUT 要回答的问题是：当工作负载变为并行处理时，同样的思路能否继续保持高效？

---

## 基于 LUT 的推理如何工作

量化 LLM 推理中的一项主要开销是**混合精度通用矩阵乘法（mixed-precision general matrix multiplication, mpGeMM）**，例如用 1.58 比特权重乘以 INT8 或 FP16 激活值。通用 CPU 和许多边缘加速器并不原生支持这类混合精度运算，因此推理框架通常需要反量化或定制内核[^8]。

{{ image(src="/img/blog/vec-lut-parallel-low-bit-llm/lut-based-inference.png", dimmable=true, caption="基于 LUT 的 mpGeMM 在量化 Transformer 层中以查表取代反量化和乘法。") }}

基于 LUT 的推理首先将权重矩阵拆成若干小组。低比特权重只有很少的可能模式：四个三值权重仅有 `3^4 = 81` 种组合。对于给定的激活向量，内核可以**预先计算**每一种“权重模式 × 激活值”的结果，并将其存入表中。运行时，打包后的权重组会成为这张表的**索引**，内核因而可以在内层循环中跳过反量化和乘法，转而累加查表所得的值。

{{ image(src="/img/blog/vec-lut-parallel-low-bit-llm/lut-example.png", max_width="60%", dimmable=true, caption="一个具体的 LUT 示例：将三值权重组作为查表索引，计算 o = w × v。") }}

---

## 并行时，位宽更低 ≠ 速度更快

问题出在并行性上。现有 LUT 内核遵循一种我们称为**标量 LUT** 的范式，即 `1→1` 查表。每个 token 都有一张*自己的*表，因此处理 N 个 token 意味着构建或访问 N 张表，并执行 N 条独立的查表流。

在单 token 生成时，标量 LUT 效果很好，因为此时只有一张表和一条查表流，现有内核可以有效利用可用的内存带宽。但真实工作负载中，**并行推理**非常常见：中长提示词的预填充[^9]<sup class="footnote-separator">、</sup>[^10]、同时服务多个请求[^11]<sup class="footnote-separator">、</sup>[^12]、测试时并行扩展[^13]<sup class="footnote-separator">、</sup>[^14]以及推测解码[^15]<sup class="footnote-separator">、</sup>[^16]，都会同时处理多个 token。在这些场景中，标量 LUT 会失去大部分优势：

- 内存带宽利用率降至 **40% 以下**。
- 有时甚至比完全不用 LUT *还慢*。

根源在于内存访问。查表是一种**随机且不连续**的操作。在多张按 token 划分的表上重复查找，尤其是这些表的总工作集超过缓存容量时，大量时间会耗在搬运分散的数据上，而不是有效的累加运算。在我们对一个典型标量 LUT 内核的性能分析中，查表（包括加载权重）占 mpGeMM 延迟的近**一半**。

下表展示了 T-MAC 标量 LUT mpGeMM 内核的耗时分布，数据来自 Orange Pi 5 Plus 上的单线程测试：

| GeMM 形状（`M × K`） | 预计算 |      查表 | 累加 | 缩放 |
| -------------------- | -----: | --------: | ---: | ---: |
| `320 × 3200`         |   0.8% | **47.6%** | 25.0% | 26.6% |
| `128 × 8640`         |   0.7% | **47.3%** | 25.5% | 26.4% |

对于并行推理，仅仅优化算术运算无法消除这个由查表主导的内存瓶颈。

---

## 核心思路：一次查出 N 个结果

关键观察是：**查表索引来自权重，而所有 token 共享同一组权重。** 同一个权重索引适用于并行组中的每个 token，因此没有必要按 token 分别查表。

Vec-LUT 将查表单位从标量变为向量。它不再为每个 token 建一张表，而是为所有并行 token 构建一张**统一的表**。表中的每个条目不再存储单个标量结果，而是存储一个结果*向量*，其中每项对应一个 token。这样，一个权重索引就能触发一次 `1→N` 查表，取回整个 token 组的结果。

{{ image(src="/img/blog/vec-lut-parallel-low-bit-llm/scalar-vs-vector-lut.png", dimmable=true, caption="Vec-LUT：将按 token 执行的 1→1 查表转化为一次跨 token 的 1→N 查表。") }}

由此产生的访问模式有三个优点：

- 每个权重索引原本需要 N 次独立查表，现在只需一次向量查表。
- 随机、分散的查表变为连续的向量读取，随后执行向量累加。
- 内核不再依赖 ARM `TBL` 或 x86 `PSHUF` 等硬件查表指令，减少了对特定指令集架构（ISA）的依赖。

在我们的延迟分解中，查表所占比例从 T-MAC 的 **47%** 降到了 **1% 以下**。

---

## 让向量 LUT 真正实用

向量 LUT 在实践中能否高效，取决于两个工程细节。

- **张量布局。** 只有周围张量采用兼容的布局，统一查找表才能发挥作用。布局不匹配会使内核速度降低至多 **12 倍**。我们的**以向量 LUT 为中心的张量布局（Vector LUT-Centric Tensor Layout）**让激活、查找表和输出按 token 连续存储，打包权重则按分块连续存储。权重重排在线下完成，运行时的激活与输出变换则融合进内核，以尽量减少额外开销。

- **缓存局部性。** 向量 LUT 会使表的大小增加 N 倍。序列长度为 512 时，Llama3 8B 的一次 GeMM 所需的 2 比特 INT16 向量表将超过 **280 MiB**，远超边缘 CPU 的缓存容量。我们的**缓存感知流式查表（Cache-Aware Streamed Lookup）**将预计算和查表流水线切分为与缓存容量相适应的小块，再依次流式处理，使大部分 LUT 工作留在片上缓存中，避免反复访问一张巨大的随机访问表。

摆脱硬件查表指令后，位宽和形状限制也随之放宽。由此，我们可以采用**灵活的低于 2 比特打包**：`I1` 打包达到**每权重 1.60 比特**，同时支持范围广泛的权重形状。拓扑预计算、INT16–INT32 分层累加等进一步优化，则减少了剩余开销。

---

## 实验结果：5 款设备 × 3 个模型

我们在五款 x86/ARM 设备（台式机、笔记本电脑、单板计算机、智能手机和 CPU 服务器）以及三个三值 LLM——Falcon3 1B、HF BitNet 3B 和 Llama3 8B——上评估了 Vec-LUT。

- **端到端预填充：** 与 BPW 相近的基线相比，`I1`（1.60 BPW）加速至多 **4.2 倍**，`I2`（2.00 BPW）加速至多 **2.6 倍**。
- **CPU 对比 NPU：** 在 Snapdragon 8 Elite 上，Vec-LUT `I2` 仅使用 **2 个 CPU 核心**，进行 Llama3 8B 预填充时的吞吐量即可达到 llama.cpp Q4_0 Hexagon NPU 后端的 **1.05–1.12 倍**。
- **连续批处理：** 在一台 8 核、每小时 **0.50 美元**的 AWS Graviton 3 服务器上，Vec-LUT `I2` 面对 32 个并行请求时，以 **273.5 tokens/s** 的速度服务 Falcon3 1B，比 llama.cpp `TQ2_0` 快 **1.4 倍**。
- **能效：** 相比 llama.cpp，每焦耳生成的 token 数至多提高 **2.1 倍**；相比 T-MAC，至多提高 **1.8 倍**。

{{ image(src="/img/blog/vec-lut-parallel-low-bit-llm/eval-prefill.png", max_width="80%", dimmable=true, caption="不同模型、设备和线程数下的端到端预填充性能对比。") }}

---

## 实现与适用范围

我们的实现 **vlut.cpp** 是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的轻量扩展。它面向主流 x86 和 ARM CPU，沿用了大家熟悉的 llama.cpp 式构建和使用流程，并作为开源 [OpenBitSys](https://github.com/OpenBitSys) 项目的一部分发布。

当三值模型检查点可用（例如通过量化感知训练获得）、工作负载包含大量并行推理（预填充、批处理、推测解码或并行测试时扩展），且 CPU 内存占用和吞吐量是主要限制时，Vec-LUT 最能发挥作用。

Vec-LUT 将查表开销降到 1% 以下后，**向量加法**（即累加查表结果）会成为主要开销。在实测的 HF BitNet 3B 工作负载中，它约占端到端预填充延迟的 65%，自然也就成为未来**以向量 LUT 为中心的加速器**值得重点优化的目标。

---

*我们的团队正在招聘博士后和实习生，研究方向包括物理 AI 基础模型、LLM 训练与推理系统等。欢迎联系[曹汀老师](https://tingcao952.github.io)。*

[^1]: Shuming Ma et al., ["BitNet b1.58 2B4T Technical Report"](https://arxiv.org/abs/2504.12285), arXiv, 2025.

[^2]: Tim Dettmers et al., ["LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"](https://arxiv.org/abs/2208.07339), NeurIPS, 2022.

[^3]: Elias Frantar et al., ["GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"](https://arxiv.org/abs/2210.17323), ICLR, 2023.

[^4]: Mengzhao Chen et al., ["EfficientQAT: Efficient Quantization-Aware Training for Large Language Models"](https://arxiv.org/abs/2407.11062), ACL, 2025.

[^5]: Shuming Ma et al., ["The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"](https://arxiv.org/abs/2402.17764), arXiv, 2024.

[^6]: Jinheng Wang et al., ["bitnet.cpp: Efficient Edge Inference for Ternary LLMs"](https://aclanthology.org/2025.acl-long.457/), ACL, 2025.

[^7]: Jianyu Wei et al., ["T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge"](https://doi.org/10.1145/3689031.3696099), EuroSys, 2025.

[^8]: Shijie Cao, Lingxiao Ma, and Ting Cao, ["Advances to low-bit quantization enable LLMs on edge devices"](https://www.microsoft.com/en-us/research/blog/advances-to-low-bit-quantization-enable-llms-on-edge-devices/), Microsoft Research Blog, 2025.

[^9]: Hao Wen et al., ["AutoDroid: LLM-powered Task Automation in Android"](https://dl.acm.org/doi/10.1145/3636534.3649379), MobiCom, 2024.

[^10]: Leming Shen et al., ["AutoIOT: LLM-Driven Automated Natural Language Programming for AIoT Applications"](https://dl.acm.org/doi/10.1145/3680207.3723486), MobiCom, 2025.

[^11]: Zheyu Shen et al., ["EdgeLoRA: An Efficient Multi-Tenant LLM Serving System on Edge Devices"](https://dl.acm.org/doi/10.1145/3711875.3729144), MobiSys, 2025.

[^12]: Borui Li et al., ["MobiLoRA: Accelerating LoRA-Based LLM Inference on Mobile Devices via Context-Aware KV Cache Optimization"](https://aclanthology.org/2025.acl-long.1140/), ACL, 2025.

[^13]: Mouxiang Chen et al., ["Parallel Scaling Law for Language Models"](https://arxiv.org/abs/2505.10475), NeurIPS, 2025.

[^14]: Ryan Ehrlich et al., ["CodeMonkeys: Scaling Test-Time Compute for Software Engineering"](https://arxiv.org/abs/2501.14723), arXiv, 2025.

[^15]: Yaniv Leviathan et al., ["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192), ICML, 2023.

[^16]: Charlie Chen et al., ["Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318), arXiv, 2023.
