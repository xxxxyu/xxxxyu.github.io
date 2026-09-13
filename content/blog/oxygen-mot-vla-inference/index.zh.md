+++
title = "用统一的 KV cache 支撑机器人的多任务推理"
date = "2026-09-01"
updated = "2026-09-01"
description = "OxyGen 将 KV cache 作为跨任务、跨控制周期共享的同一份资源，让 Mixture-of-Transformers VLA 在不损失动作频率的前提下并发生成语言，在 RTX 4090 与 Jetson AGX Thor 上端到端加速最高 3.7×。"
template = "blog-page.html"

[taxonomies]
tags = ["Embodied AI", "VLA", "LLM Inference"]

[extra]
toc = true
og_image = "/img/blog/oxygen-mot-vla-inference/teaser.png"
og_image_alt = "OxyGen 总览：isolated 与 unified KV cache 管理下的并发动作与语言生成对比"
ai_translation_source = "en"
ai_translation_harness = "OpenCode"
ai_translation_model = "GLM-5.3"
ai_translation_effort = ""
+++

> 🔗 [论文](https://arxiv.org/abs/2603.14371) | [代码](https://github.com/air-embodied-brain/OxyGen) | [项目主页](https://air-embodied-brain.github.io/OxyGen)

一台有用的机器人，要做的远不止执行动作：机械臂在动的同时，它还应当能向用户汇报进展、记录场景里的变化，甚至重新规划下一个子任务。近期的 Mixture-of-Transformers（MoT）视觉-语言-动作模型（VLA），比如 π<sub>0.5</sub>[^1]，正朝这个方向发展：一个共享的 backbone，动作和语言各有独立的 expert 权重；原则上，同一个模型可以同时生成动作与语言。

但落到机器人实际携带的单块 GPU 上，两路输出就只能相互等待。我们把这个问题的根源追到了一个不太起眼的地方：每个任务各管各的 KV cache。**OxyGen** 的解法，是把 KV cache 作为跨任务、跨控制周期共享的同一份资源。在 RTX 4090 上，它能同时实现 200+ tokens/s 的语言生成与 **70 Hz**[^2] 的动作频率（Jetson AGX Thor 上 27 Hz），相对现有系统的 isolated execution，端到端加速最高 **3.7×**。

<a id="demo-video"></a>

{{ video(src="/videos/oxygen-mot-vla-inference/demo.mp4", autoplay=true, loop=true, max_width="100%", caption="OxyGen 与 isolated 基线（openpi）在 LIBERO 上的对比：机器人在持续输出动作的同时流式生成文字记忆。时间线按「实测模型推理 + 1× 仿真时间」绘制。") }}

---

## 既能操作、也能对话的 VLA

从 π<sub>0</sub>[^3] 这类 continuous VLA 开始，主流做法是在 VLM backbone 上挂一个轻量的 flow-matching 头：backbone 每个控制周期将观测和指令编码一次，动作头再基于这份上下文，通过去噪生成一段动作。MoT VLA[^4] 将这一思路又推进一步：backbone 共享，但每种输出模态有自己的 expert 权重，于是一套 checkpoint 就能同时产出动作块和自由文本。

单看模型结构，推理的分解方式相当规整。每个控制周期，共享 backbone 将观测 prefill 成一份 KV cache，而这份 cache 与模态无关——它编码的是"机器人刚看到了什么、被要求做什么"，不绑定任何输出形式，任何 expert 都可以读取：

- **动作 expert** 在这份 cache 上执行 `S` 步 flow-matching 去噪，输出一段 `H` 个动作的 chunk（全文统一取 `S = 10, H = 10`，与 openpi 的 π<sub>0.5</sub>-LIBERO 配置一致）；
- **语言 expert** 从同一份 cache 出发自回归地解码至多 `N` 个 token：子任务、叙述、记忆。

这两个任务的时间约束差别很大。动作必须在本周期内产出：灵巧操作大致要求 50 Hz 量级，晚到一个周期，轻则动作抖动，重则机器人停滞。语言则是软约束：一句记忆晚到一秒，通常仍然有用。高效的系统设计必须利用这种不对称，而不是与它对抗。

---

## 为什么两个任务同时运行会这么慢

就架构而言，一切早已就绪：一次 prefill、两个 expert 读同一份 cache。欠缺的是 serving 层的配合。openpi[^5] 以及我们知道的所有 MoT serving 方案，采用的都是我们所说的 **isolated execution**[^6]：每个任务都对同一模型各自触发一次完整前向，语言路径会把同一份观测重新编码、从头构建自己的 KV cache，尽管模型本身完全支持共享。代价体现在两个方面。

**冗余计算。** 同一份观测被每个任务分别编码一次，产出逐字节相同的 KV cache 条目，而这笔重复并不便宜：在 RTX 4090 上，一次经过 VLM backbone 的 prefix 前向约需 45 ms，是整个去噪循环（约 20 ms）的 **2.2×**[^7]。在语言输出较短的场景，仅重复 prefill 一项就会造成 **1.4×** 的减速。

**资源争用。** 简单地共享 cache 并不能解决调度问题。两个任务在一个进程里顺序执行，语言解码就会阻塞控制回路：随着语言输出变长，基线的动作频率从 **49.9 Hz** 跌至 **19.1 Hz**，损失 **2.6×**[^8]；拆成两个进程靠 MPS 共享 GPU，峰值显存几乎翻倍，收益却微乎其微。

"这不就是 prefix caching 加 continuous batching 吗？"——这是很自然的疑问，LLM serving 系统[^9]<sup class="footnote-separator">, </sup>[^10] 这两件事都做得很好。但它们的世界是同构的：同一种 cache 读取方、同一种请求、没有截止时间。MoT VLA 恰好在两个维度上都是异构的：

- **cache 访问方式异构。** 动作 expert 在每一步去噪中都把 prefix 当作*只读*上下文；语言 expert 每解码一个 token 就*追加*一条 KV。prefix caching 假设单一的自回归读取路径；标准 serving 系统中，并没有能协调两种访问规则共用同一份 cache 的机制。
- **时间约束异构。** 语言解码必须遵守每帧的有界步数预算：推进若干步后暂停，下一帧继续，且无需重算，因为动作的截止时间每一帧都会重现。continuous batching 将每个请求不间断地执行到完成；标准 serving 系统并不理解"机器人控制周期"为何物。

---

## 统一的 KV cache：跨任务、跨周期共享

OxyGen 的答案是让 KV cache 成为一等公民：由同一个 cache manager 跨任务、跨时间统一管理，前述两个缺口，各由一个机制来填补。

{{ image(src="/img/blog/oxygen-mot-vla-inference/teaser.png", dimmable=true, caption="OxyGen 总览：从一个共享 prefix 同时进行动作与语言生成，用统一的 KV cache 管理取代各任务独立的 cache。") }}

**跨任务 KV 共享。** 每个周期，backbone 只对新观测做一次 prefill。manager 将得到的 cache 以只读视图的形式分发给动作 expert，同时用同一份 prefix 初始化一个全新的语言请求，让它从这份 prefix 出发继续自回归解码。重复的 prefill 就此消失，这也解释了为什么语言输出越短，这项优化的收益越显著。

**跨周期 continuous batching。** 每个语言请求都是一个可恢复的状态：自己的 KV cache、token 缓冲区、一个完成标志。每个周期，manager 收集所有未完成的请求，拼成一个解码 batch 做一次前向，将每个请求至多推进 `k` 个 token，完成即淘汰，未完成则保留至下一周期。稳态下，长度为 `N` 的请求每周期推进 `k` 个 token，系统中同时活跃的请求约有 `N/k` 个，一次前向即可服务全部请求，这正是 isolated execution 未能利用的硬件并行度。

预算 `k` 限制每个请求每周期最多推进多远；未完成的部分在下一周期继续，语言永远不会阻塞动作的按时产出，我们测试中最长的请求也能在一秒左右完成。两个机制本身并不新鲜，真正让它们咬合在一起的是中间那层 cache 语义：只读的去噪读取与只追加的自回归解码，在每几十毫秒重现一次的截止时间下协同工作。

---

## 结果：从 RTX 4090 到 Jetson Thor

我们在 RTX 4090 和 Jetson AGX Thor 上，按 LIBERO[^11]、DROID[^12]、ALOHA[^13] 三种配置评测 π<sub>0.5</sub>：每个周期新增一个观测和一个语言请求，扫描 `N` 与 `k`。

{{ image(src="/img/blog/oxygen-mot-vla-inference/tradeoff-pi05-libero.png", dimmable=true, caption="LIBERO 配置下的动作频率与语言吞吐。基线随 N 增长只能以一轴换另一轴；OxyGen 将整个边界向外推，最多 3.7×。") }}

这张 tradeoff 图是全部结果的核心：随着语言负载变大，每条 isolated execution 曲线都只能以动作频率换取语言吞吐（或反过来）；OxyGen 则将边界整体外推，两个维度同时改善 **1.2–3.7×**。在默认的"每周期一个请求"之外，我们还测试了均匀突发、Poisson、随机长度等请求到达模式，结论一致；作为对照，朴素的 MPS 并行收效甚微。

{{ image(src="/img/blog/oxygen-mot-vla-inference/ablation-pi05-libero.png", dimmable=true, caption="LIBERO 上的消融：KV sharing 抬升所有工作点；随着解码变长，跨周期 batching 将动作频率拉成一条平线。") }}

消融实验把两个机制的贡献分开。单独的 KV sharing（`Ours w/o Batching`）在 `N=5` 时将平均端到端延迟从 200.3 ms 降到 145.4 ms（**1.38×**），并在 `N=1–5` 区间削减 25.5–36.4% 的推理延迟。输出变长后，由跨周期 batching 接管：`k = 5` 且两个机制全部启用时，即便 `N=30`，RTX 4090 上的动作频率仍保持在 **60 Hz** 附近、Jetson 上 **27 Hz**，而基线劣化 2.6×。

除速度之外，我们也检查了其余各项指标，均无回退：使用官方 π<sub>0.5</sub>-LIBERO checkpoint，成功率与 openpi 报告值相差不超过 **±0.8%**（10 组随机种子重跑、20,000 次 rollout，总体 **96.76%** [96.55, 96.97]，openpi 报告 96.85%）；显存仅增加 15%，平均功耗最高降 **47%**，单请求能耗最高降 **78%**——冗余计算减少，权重读取也随之减少。MPS 并行则恰恰相反：峰值显存几乎翻倍，单请求能耗甚至*更高*。

我们最看重的一次部署，是在一台机载 Jetson AGX Thor 的人形机器人上。三路 224×224 相机，`N=30, k=5`，操作与语言生成交错进行；每帧有 333 ms 的动作执行窗口，其余由实测数据说明：

| 阶段 | 基线 (ms) | OxyGen (ms) |
| ---- | --------: | ----------: |
| Prefill 与去噪 | 207.5 | 198.0 |
| 语言生成 | 822.3 | 195.4 |
| 动作执行窗口 | 333.0 | 333.0 |

基线每帧 1,030 ms 的推理远超执行窗口，控制回路完全跟不上；OxyGen 将总推理压到 **393 ms**。更关键的是，决定动作能否按时产出的 prefill 与去噪阶段仅需 198 ms，**正好落在执行窗口之内**；语言生成（822.3 → 195.4 ms，缩减 4.2×）在动作输出之后运行，大部分被动作执行所掩盖。

---

## 超越单一 backbone

我们也把 OxyGen 移植到另外两个开源 MoT VLA[^14]<sup class="footnote-separator">, </sup>[^15] 上，在它们的原生实现上重跑了同样的测量：加速效果可以迁移，并且随语言积压量增长。同一张 RTX 4090，每周期一个新请求，`N=30`：

| 模型 | 基线延迟 | 加速比（`k=1`） | 加速比（`k=5`） |
| ---- | -------: | --------------: | --------------: |
| Xiaomi-Robotics-0 (4.7B) | 1,540.9 ms | 5.50× | 3.14× |
| StarVLA Qwen3VL-PI_v3 (5.07B) | 1,351.5 ms | 6.18× | 3.41× |

有一个边界情况值得指出：当 `N=k`，每个请求都在到达的那一帧内完成，平均 batch 大小为 1，结果大致持平（0.92–1.07×）。跨周期 batching 只在存在积压时才有收益；托住这些点的，正是 KV sharing。

共享机制本身能扩展到多远？我们用 StarVLA 发布的几个动作头（flow-matching 的 PI_v3 头、GR00T flow 头、OFT one-shot 头）做实验，它们全部复用同一个 Qwen3-VL-4B prefix。expert 越多，平摊同一次 prefill 的读者就越多：

| expert 数 `E` | 加速比（`N=5, k=5`） | 加速比（`N=30, k=5`） |
| ------------: | -------------------: | -------------------: |
| 2 | 1.07× | 3.41× |
| 3 | 1.21× | 3.32× |
| 4 | 1.36× | 3.48× |

backbone 规模方面，我们扫描了 StarVLA 的 Qwen3.5 多模态系列，外加一组 PCIe 双卡 tensor-parallel 实验：加速比几乎不变。

| Backbone | 基线延迟 | 加速比（`N=30, k=5`） |
| -------- | -------: | -------------------: |
| Qwen3.5-0.8B | 1,257.5 ms | 3.53× |
| Qwen3.5-2B | 1,267.5 ms | 3.44× |
| Qwen3.5-4B | 1,728.0 ms | 3.60× |
| Qwen3.5-9B | 1,725.9 ms | 3.50× |
| Qwen3.5-9B, TP=2 | 1,983.5 ms | 3.71× |

需要说明的是：这些额外的头是分别训练的，所以这组实验验证的是系统层面的扩展性与共享 prefix 的扇出能力，而不是联合训练的多 expert 策略的质量。

---

## Demo：一个记得自己刚做过什么的 VLA

提升语言吞吐的意义并不在解说本身，而在于记忆：长程任务会逐渐偏离，除非有什么东西将已完成的事记录下来，这正是 PI 的 MEM[^16] 的出发点。公开的 π<sub>0.5</sub> checkpoint 无法直接生成这类文本，于是论文投稿之后，我们额外训练了一个轻量模块：在冻结的模型上加一个 suffix-only 的 LoRA[^17]，让它以 `Memory: ` 开头，在共享 prefix 之后以私有延续的方式生成一句有观测依据的话（"The alphabet soup is in the basket."）。它不会触碰共享 cache 与动作路径（将其置零后，动作输出逐位不变），serving 时每条记忆的中位开销只有 **8.16 ms**，而 isolated execution 需要重算的那次 prefix 前向是 47.66 ms。

监督标注从 LIBERO 原始演示构造而来，[数据集](https://huggingface.co/datasets/xxxxyu/libero-textual-memory-annotations)与 [adapter](https://huggingface.co/xxxxyu/oxygen-pi05-textual-memory-lora) 已上传 Hugging Face；在留出帧上，模型逐字复现标注的精确匹配率为 **87.25%**。这正是<a href="#demo-video">开头 demo 视频</a>里的负载：每次动作重规划发起一条新的记忆请求，未完成的请求在下一帧的语言 batch 中继续，动作频率不受影响。

---

## 适用边界与下一步

OxyGen 并不试图解决所有问题。每帧预算 `k` 在部署时固定，运行时不重算；多机 serving 未测，tensor-parallel 的验证只覆盖单机双卡；整个系统是模型之上的一层调度，与动作侧的异步管线[^18]<sup class="footnote-separator">, </sup>[^19]以及压缩、剪枝均正交。下一步计划：更多 backbone、更丰富的联合训练 expert 组合，以及将投机解码接入可恢复的语言状态，它应当能与之干净地组合。

代码（JAX 与 PyTorch 双后端）、textual-memory adapter 和标注数据集均已开源，入口见文首链接。如果你正在实时约束下部署 MoT VLA，欢迎和我们交流你的负载情况：[提 issue](https://github.com/air-embodied-brain/OxyGen/issues) 或[发邮件](mailto:lixiangy22@mails.tsinghua.edu.cn)。

*我们的团队正在招聘博士后和实习生，研究方向包括物理 AI 基础模型、LLM 训练与推理系统等。欢迎联系[曹婷老师](https://tingcao952.github.io)。*

[^1]: Physical Intelligence, Kevin Black et al., ["π0.5: A Vision-Language-Action Model with Open-World Generalization"](https://arxiv.org/abs/2504.16054), arXiv, 2025.
[^2]: 动作频率按输出的动作数计（每个 chunk 含 `H = 10` 个动作，故 `f = H / T`，`T` 为每帧延迟）；70 Hz 测自 RTX 4090 上的 `N = 30, k = 1` 设置。`S = 10` 是 openpi 的去噪默认值（[pi0.py:222](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py#L222)）；`H = 10` 沿自 openpi 的 π<sub>0.5</sub>-LIBERO 配置（[config.py:744](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py#L744-L745)）。
[^3]: Kevin Black et al., ["π0: A Vision-Language-Action Flow Model for General Robot Control"](https://arxiv.org/abs/2410.24164), arXiv, 2024.
[^4]: Weizhuang Liang et al., ["Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models"](https://arxiv.org/abs/2411.04996), arXiv, 2024.
[^5]: Physical Intelligence, ["openpi"](https://github.com/Physical-Intelligence/openpi), GitHub.
[^6]: openpi 并未开源 π<sub>0.5</sub> 的语言生成路径；我们的基线遵循其[社区复现](https://github.com/BrunoFANG1/openpi_subtask_generation)。
[^7]: 在 π<sub>0.5</sub>-LIBERO（`S = 10`，RTX 4090）上实测：prefix 前向约 45.1 ms，完整的 10 步去噪循环约 20.4 ms。
[^8]: π<sub>0.5</sub>-LIBERO，RTX 4090，`k = 5`，单请求解码长度 `N` 从 5 增至 30。
[^9]: Woosuk Kwon et al., ["Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180), SOSP, 2023.
[^10]: Lianmin Zheng et al., ["SGLang: Efficient Execution of Structured Language Model Programs"](https://arxiv.org/abs/2312.07104), NeurIPS, 2024.
[^11]: Bo Liu et al., ["LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning"](https://arxiv.org/abs/2306.03310), NeurIPS, 2023.
[^12]: Alexander Khazatsky et al., ["DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset"](https://arxiv.org/abs/2403.12945), arXiv, 2024.
[^13]: Tony Z. Zhao et al., ["Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"](https://arxiv.org/abs/2304.13705), arXiv, 2023.
[^14]: Xiaomi Robotics, ["Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution"](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0), GitHub.
[^15]: starVLA Team, ["starVLA"](https://github.com/starVLA/starVLA), GitHub.
[^16]: Mateo Torne et al., ["MEM: Multi-scale Embodied Memory for Vision Language Action Models"](https://www.pi.website/download/Mem.pdf), Technical Report, Physical Intelligence, 2025.
[^17]: Edward J. Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685), ICLR, 2022.
[^18]: Kevin Black, Mark Y. Galliker, and Sergey Levine, ["Real-Time Execution of Action Chunking Flow Policies"](https://arxiv.org/abs/2506.07339), arXiv, 2025.
[^19]: Yiran Zhao et al., ["VLA-RAIL: A Real-Time Asynchronous Inference Linker for VLA Models and Robots"](https://arxiv.org/abs/2512.24673), arXiv, 2025.
