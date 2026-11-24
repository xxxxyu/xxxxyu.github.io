+++
title = "LLM Agent 量化感知蒸馏：闭环指导下的能力恢复"
date = "2026-11-23"
updated = "2026-11-23"
description = "量化模型在 Agent 任务中常在细节上出错。SteerDistill 让全精度教师在学生出错的步骤给出指导，再把修正蒸馏回量化模型。"
template = "blog-page.html"

[taxonomies]
tags = ["LLM", "Quantization", "Agents", "Knowledge Distillation"]

[extra]
toc = true
og_image = "/img/blog/steerdistill-agent-quantization/teaser.png"
og_image_alt = "SteerDistill 投稿版论文图 1：工作目录读取案例、W4A16 主结果均值与训练曲线"
+++

> 🔗 [论文](https://arxiv.org/abs/2611.13410) | [代码](https://github.com/air-embodied-brain/SteerDistill) | [项目主页](https://air-embodied-brain.github.io/SteerDistill/)
>
> 本文由 Claude Opus 5.5 根据[知乎原文](https://zhuanlan.zhihu.com/p/2107070075614262595)改写。

LLM 量化发展到今天，许多研究围绕旋转、变换、误差补偿和校准展开，在几篇经典方法的基础上持续细化。**在对话、问答和补全任务上，这些方法已经能做到接近无损，但到了 Agent 场景，效果明显打折**。它们确实降低了量化误差，这种改进却未必能体现在任务成功率这类端到端指标上。在我们的评测中，Qwen3-8B 直接用 GPTQ 量化到 W4A16 后，五个 Agent benchmark 的平均分[^1]从 BF16 的 53.54 降到 46.36，下降超过 7 分。对成功率要求高的应用来说，这样的损失很难接受。

工业界则更依赖数据规模：只要目标领域的数据足够多，量化方法本身的设计就显得次要。与其精心设计 quantizer，不如收集更多回流数据、多迭代几版训练。不过，造数据和洗数据成本高、工作繁琐。是否有更优雅的做法？

我们的尝试是，在面向 Agent 任务的量化感知蒸馏（quantization-aware distillation，QAD）中，扩展教师模型（一般就是量化前的 BF16 模型）的职责。**除了像已有方法那样提供 loss 监督，教师还在训练中为学生提供闭环指导：学生在哪一步出错，就在这一步修正，然后继续执行**。教师不替学生执行动作，只指出哪里做错了、应当如何做对。

{{ image(src="/img/blog/steerdistill-agent-quantization/teaser.png", alt="SteerDistill 投稿版论文图 1：量化学生在工作目录读取案例中出错并在教师指导下修正，以及各方法 W4A16 平均分和训练曲线对比", dimmable=true, caption="投稿版论文图 1：工作目录读取案例、W4A16 主结果均值与训练曲线。") }}

我们在三个模型、五个 Agent benchmark 上做了实验。**W4A16 下，三个模型的五项平均分都与各自的 BF16 相差不到 1 分**。以 Qwen3-8B 为例，最强的 baseline QeRL 仍比 BF16 低两分多，直接 GPTQ 则低 7 分以上。除端到端对比外，我们还通过消融和 case study 检验了各项设计选择，详见论文和项目主页。

下文主要讨论其中几项观察和分析。

## Agent 量化的难点

这个问题可以分几层来看。

**第一，Agent 任务的交互轮次更多**。设想一个需要 20 轮的任务，每一轮都必须正确：若每轮成功率固定为 99%，端到端成功率约为 82%；若每轮为 90%，则只剩约 12%。这个算例用于说明，量化带来的能力损失会在多轮交互中累积放大。

**第二，某些关键轮次会导致异常的状态转移，拉低后续轮次的成功率**。这是 Agent 任务区别于一般多轮对话的地方，例如误删文件、进错目录，或者执行了其他不可逆操作。量化模型本身能力有所下降，往往更难从未见过的异常状态中恢复。很多时候，一轮出错就足以导致整个任务失败。

**第三，量化模型在单步决策中出错的概率也更高**。这是我们重点关注的问题，并为此做了大量 case study。在我们观察到的许多失败中，**量化模型的大方向并没有偏离，但更容易忽略完成任务所需的细节，包括任务要求以及环境和工具调用的约束**。

这类错误大多属于“低级错误”。以图 1 的工作目录案例为例，这个环境中的 `cat` 只能读取当前目录下的文件，量化模型却没有先确认自己所在的目录，就直接读取文件[^2]。

## 闭环教师指导

已有的量化方法，或者说量化后的性能恢复方法，大致可以分为两类：

- **依赖离线数据做校准或训练**。这是最常见的一类，包括 GPTQ、AWQ 等训练后量化（post-training quantization，PTQ）方法[^3]，以及 EfficientQAT、BitDistiller 等 QAT/QAD 方法[^4]。数据由人工、GPT 等模型或教师模型离线生成，不随量化模型在训练中的更新而变化。
- **依赖在线生成的轨迹**。一种是 RL 类方法，例如 QeRL[^5]，由量化模型在线 rollout，并根据 reward 更新参数；另一种是 on-policy distillation（OPD）类方法，例如 NVIDIA NeMo 的 on-policy QAD[^6]，同样由量化模型在线 rollout，但由教师提供逐 token 的密集监督。

前者成本低、实现简单，但离线数据往往覆盖不到量化模型特有的出错情况。在图 1 的例子中，全精度教师没有犯同样的错误，它的成功轨迹也就很难覆盖学生实际出错时的上下文，这部分错误因而难以纠正。后者让量化模型在自己的失败上下文中学习，但能力相对较弱的量化模型，未必能自行找到正确做法。

我们在 BFCL V4 多轮任务上做了测试：在量化学生与全精度教师的分歧点，恢复出错前的历史和环境状态，让量化模型从同一状态重试四次。没有指导时，分歧点上的 Pass@4（四次中至少一次给出正确动作）只有 9.6%；加入全精度模型的指导后，提升到 82.3%。这说明单纯增加尝试次数作用有限，学生缺少的是能够改变这一步判断的信息。

由此，思路就比较直接了：**既然学生无法独立改正，就由教师有针对性地引导，但引导要在学生的能力范围之内**。直接给出答案和让学生自行试错，效果都不如有针对性的提示。训练流程如图 2 所示。

{{ image(src="/img/blog/steerdistill-agent-quantization/method.png", alt="SteerDistill 论文图 2：一个训练周期内，学生 rollout、在分歧点回溯并接受教师指导修正、再将修正蒸馏进学生的完整流程", dimmable=true, caption="论文图 2：SteerDistill 的训练周期，包括学生 rollout、分歧点修正与内化蒸馏。") }}

**第一阶段：学生 rollout，并在教师指导下修正失败轨迹**。

学生先独立 rollout。对于失败的轨迹，我们找一条教师在相同任务、相同初始环境下的成功轨迹作为参考，按环境状态的变化对齐两条轨迹，定位第一个分歧点。随后把历史和环境回溯到出错之前，由教师对照学生的错误回复和参考轨迹写一段 steering prompt，指出学生忽略了什么；学生在这段指导下重新生成这一步。

修正后的回复执行完毕后，做一次局部核验，只检查这一步执行后的环境状态和返回信息是否与参考一致。若通过，就保留修正、撤去指导，让学生继续执行；任务仍然失败的话，再寻找下一个分歧点。若未通过，就把执行反馈交给教师细化指导，学生从同一状态重试，直到用完重试预算。

**第二阶段：学生蒸馏并内化修正后的轨迹，无论最终是否成功**。

每个训练周期开始时，先冻结一份当前学生的快照，用于采集轨迹和生成修正。在修正过的轮次上，我们参考上下文蒸馏[^7]的做法：由带 steering prompt 的冻结快照通过 forward KL 监督学生，而学生自身的上下文中不含指导。这样，指导带来的预测变化就被写入学生的参数。

其余轮次沿用标准 OPD 的做法，由教师通过 reverse KL 监督。训练完成后，部署时只使用量化学生本身，不需要教师，也不注入任何指导。

## 实验与消融

主实验在 BFCL V4、τ²-Bench、WebShop、ALFWorld 和 Search QA 上评测；对每个模型，所有方法都从该模型同一个经过目标任务后训练的 BF16 checkpoint 出发。在 Qwen3-8B、W4A16 下，SteerDistill 在五个 benchmark 上都优于所有量化 baseline，平均分为 53.19，比最强的 QeRL 高 2.18，与 BF16 仅差 0.35。在 Qwen3.5-4B 和 Gemma 4 12B-it 上，SteerDistill 的平均分分别比 QeRL 高 1.71 和 1.67。下表汇总了三个模型在 W4A16 下的最新五项平均分，BF16 作为参考。

| 方法 | Qwen3-8B | Qwen3.5-4B | Gemma 4 12B-it |
| --- | ---: | ---: | ---: |
| BF16（参考） | 53.54 | 61.68 | 65.13 |
| GPTQ | 46.36 | 54.22 | 57.66 |
| EfficientQAT | 47.87 | 55.94 | 59.44 |
| BitDistiller | 48.49 | 56.68 | 60.07 |
| On-Policy QAD | 50.40 | 58.59 | 62.02 |
| QeRL | 51.01 | 59.24 | 62.64 |
| **SteerDistill** | **53.19** | **60.95** | **64.31** |

更低精度的实验在 Qwen3-8B 上进行，SteerDistill 的优势依然存在：W4A8 下比 QeRL 高 1.93；W2A16 下比最强的 On-Policy QAD 高 3.39。不过，W2A16 下 42.97 的平均分离 BF16 仍有较大差距，2-bit 的损失并没有解决。所有设置中，只有 W2A16 的 Search QA 一项，BitDistiller 略高于 SteerDistill。

SteerDistill 的训练收敛也更快。在 Qwen3-8B、W4A16 的 τ²-Bench 上，SteerDistill 训练 32 GPU 小时后成功率达到 31.68%，QeRL 训练 96 GPU 小时后为 29.84%[^8]。推理阶段，按各方法各自成功的任务统计，SteerDistill 每个成功任务平均输出 4.0K token，与 BF16 的 3.9K 接近。

消融实验均在 Qwen3-8B、W4A16、BFCL V4 Overall 上进行（完整方法为 46.87），主要结论如下：

- **定位分歧应依据环境状态，仅看动作是否匹配并不够**。不同的动作如果带来一致的环境状态转移，就可以视为等效。“环境状态”的具体定义需要依环境而定，可以理解为环境中任何会影响 Agent 决策的信息。
- **教师应给出引导，而不直接给出答案**。目的是在数据层面尽量避免分布偏移。从同一个恢复状态出发，由教师直接接管得到 45.06，由教师指导、学生自行修正得到 46.87。
- **由教师模型给出结构化指导，效果最好**。我们要求教师按固定结构撰写指导[^9]。结构化格式优于自由文本，全精度教师写的指导也优于学生自己或更强的外部模型所写的指导。一个可能的解释是，教师与学生的分布更接近；如果不限制格式，强模型给出的指导，弱模型未必能够领会。
- **保留局部成功**。即使整条轨迹最终失败，中间通过核验的修正片段仍然是有价值的监督信号，也提高了数据利用率。去掉局部核验后，分数大幅降到 42.18；额外要求任务最终成功，也没有带来收益。
- **修正轮次使用 FKL，其他轮次使用 RKL**。这一设置沿用已有做法，消融主要起验证作用：其他组合的效果都不如当前设置。

## 局限与展望

本文关注的是量化模型在 Agent 任务上的能力恢复。也就是说，我们假设教师模型已经在目标任务上充分训练，学生只需尽可能接近教师。实际应用中，这类场景相当常见。

我们还在补充实验中考察了初始 GRPO 的作用[^10]。主实验的起点是经过目标任务 GRPO 的 BF16 checkpoint；这里改从原始 Qwen3-8B checkpoint 出发，不做目标任务上的初始 GRPO，直接训练 W4A16 模型。原始 BF16 模型在 BFCL V4 和 τ²-Bench 上分别为 40.40 和 30.00。

| 设置 | 总训练预算（GPU 小时） | BFCL V4 Overall (%) | τ²-Bench Pass@1 (%) |
| --- | ---: | ---: | ---: |
| 无初始 GRPO，QeRL | 32 | 36.07 | 24.95 |
| 无初始 GRPO，SteerDistill | 32 | 39.67 | 27.67 |
| 无初始 GRPO，QeRL（收敛） | 132 | 42.74 | 28.53 |
| BF16 GRPO → SteerDistill QAD | 约 105 | 46.87 | 31.79 |
| BF16 GRPO → QeRL | 约 169 | 44.95 | 29.84 |

同样训练 32 GPU 小时，SteerDistill 在两项指标上都高于 QeRL。QeRL 继续训练到 132 GPU 小时后，超过了直接训练的 SteerDistill，其 BFCL 还高于原始 BF16 的 40.40，体现出 RL 能从任务奖励中继续学习；SteerDistill 的目标则是让学生接近给定的教师。

如果先在 BF16 上做 GRPO、再用 SteerDistill 做 QAD，总计约 105 GPU 小时就得到表中最好的结果，比直接用 QeRL 训练到收敛少约 20% 的总预算。在这组实验中，把预算先用于训练 BF16 教师、再蒸馏到量化模型，是更合算的分配方式。

另外，这一方法依赖可执行、可回滚、可验证的环境，还需要教师的成功轨迹作为参考。在我们使用的模拟环境中，这些条件可以满足；但在一些开放、复杂的环境中，状态难以恢复、结果难以核验，可能会出现新的问题，这或许也是该领域尚待解决的问题。

此外，我们的工作集中在量化场景，尚未在一般的蒸馏设置下（大小模型之间，乃至基座模型层面的蒸馏）验证这套方法。后续我们可能会补充相关实验。代码已经公开，也欢迎在自己的场景中尝试。

[^1]: W4A16 表示 4-bit 权重、16-bit 激活。五项指标分别为 BFCL Overall、τ²-Bench Pass@1、WebShop 与 ALFWorld 成功率、Search QA EM，取无权重平均。
[^2]: 教师与学生在这个案例中的完整执行过程见[项目主页案例](https://air-embodied-brain.github.io/SteerDistill/cases/?group=training&case=directory-navigation)。
[^3]: Elias Frantar et al., ["GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"](https://arxiv.org/abs/2210.17323), ICLR, 2023; Ji Lin et al., ["AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978), MLSys, 2024.
[^4]: Mengzhao Chen et al., ["EfficientQAT: Efficient Quantization-Aware Training for Large Language Models"](https://arxiv.org/abs/2407.11062), ACL, 2025; DaYou Du et al., ["BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation"](https://aclanthology.org/2024.acl-long.7/), ACL, 2024.
[^5]: Wei Huang et al., ["QeRL: Beyond Efficiency — Quantization-enhanced Reinforcement Learning for LLMs"](https://arxiv.org/abs/2510.11696), arXiv, 2025.
[^6]: NVIDIA NeMo RL, ["Quantization-Aware Distillation (On-Policy QAD)"](https://docs.nvidia.com/nemo/rl/latest/guides/quantization-aware-rl.html#quantization-aware-distillation-on-policy-qad), documentation.
[^7]: Tianzhu Ye et al., ["On-Policy Context Distillation for Language Models"](https://arxiv.org/abs/2602.12275), arXiv, 2026.
[^8]: 训练使用 8 张 A100 80GB，GPU 小时按墙钟时间乘以 GPU 数计算。
[^9]: 指导需写明学生忽略了什么、适用条件以及如何纠正；格式与示例见[项目主页](https://air-embodied-brain.github.io/SteerDistill/#guidance)。
[^10]: 总成本由 73 GPU 小时的 BF16 GRPO，加上训练曲线估计的低比特耗时（SteerDistill 32、QeRL 96 GPU 小时）得到；得分采用最终主结果。
