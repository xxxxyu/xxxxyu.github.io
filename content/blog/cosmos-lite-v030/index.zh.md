+++
title = "在单张 RTX 4090 上运行 Cosmos 3 机器人策略"
date = "2026-08-21"
updated = "2026-08-21"
description = "通过量化、采样器调优与 kernel 优化，把 NVIDIA 的 16B Cosmos 3 机器人策略装进一张 24 GB 显卡，并把这套 runtime 变成并行 rollout 引擎。"
template = "blog-page.html"

[taxonomies]
tags = ["Embodied AI", "Quantization", "Edge AI"]

[extra]
toc = true
ai_translation_source = "en"
ai_translation_harness = "OpenCode"
ai_translation_model = "GLM-5.2"
ai_translation_effort = "max"
og_image = "/img/blog/cosmos-lite-v030/cover.jpg"
og_image_alt = "Side-by-side closed-loop rollout comparison of Edge BF16 and Edge GenW8A8"
+++

> 代码：[cosmos-lite v0.3.0](https://github.com/xxxxyu/cosmos-lite) \
> 模型：[Nano GenW8A8](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-GenW8A8) · [Edge GenW8A8](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-GenW8A8)

NVIDIA 的 Cosmos 3[^1] 带来了在 DROID[^2] 上训练的机器人策略。不过想自己把它跑起来并不容易：16B 的 Nano 策略光是加载就要约 33 GB 显存，[官方工作流](https://github.com/NVIDIA/Cosmos-Framework)也默认你有 A100 或 H100 这样的数据中心 GPU。单张 RTX 4090 之前完全跑不动。

[Cosmos Lite](https://github.com/xxxxyu/cosmos-lite) 就是为此做的。它构建在 [Cosmos Framework](https://github.com/NVIDIA/Cosmos-Framework) 之上，是一个面向 Cosmos 3 策略的高效推理 runtime。同一套服务覆盖两类用法：一是**本地部署**，以 batch-one 方式提供策略推理，可直接对接真机；二是**并行 rollout**，由一个共享的推理服务同时驱动多个仿真环境，并把 rollout 轨迹导出，用于后续训练。

{{ video(src="/videos/cosmos-lite-v030/demo.mp4", max_width="100%", caption="Edge BF16（官方默认配置）对比 Edge GenW8A8（优化 runtime），在 RoboLab 中闭环 rollout。") }}

评测覆盖 RoboLab[^3] 全部 120 个任务，每任务 10 个 episode；所有 profile 用同一组采样参数（guidance 3、两步 UniPC 去噪、shift 5）。延迟统计的是单张 RTX 4090 上每个动作请求的耗时。

| 模型     | 产物        | 显存 (GB) | 请求 p50 (ms) | SR (%)    |
| ------- | ----------- | --------: | ------------: | ---------: |
| Edge 4B | BF16（NVIDIA 报告[^4]） | — | — | 22.90 |
| Edge 4B | BF16        |      9.20 |         582.0 | 20.92     |
| Edge 4B | **GenW8A8** |      8.79 |     **331.4** | 19.25     |
| Nano 16B | BF16（NVIDIA 报告[^4]） | — | — | 36.80 |
| Nano 16B | W8A16       |     21.42 |       2,403.0 | 31.50     |
| Nano 16B | **GenW8A8** | **15.51** |     **958.5** | **31.67** |

Nano 这两行能出现，靠的是量化：BF16 装不进 24 GB 显卡。GenW8A8 在保住任务成功率的前提下，把 Nano 相对 W8A16 提速 **2.5×**，把 Edge 相对 BF16 提速 **1.76×**。有一点要说清楚：表里的成功率全部低于 NVIDIA 报告值（Nano 31.7% 对 36.8%，Edge 19.3% 对 22.9%）。所谓"保住成功率"，指的是*统计上不可区分*，不是*零损失*。真要上自己的任务，请先验证成功率。

---

## 用量化装进显存

第一个问题是把 16B 模型装进 24 GB 显存，怎么做取决于模型结构。Cosmos 3 策略采用 Mixture-of-Transformers（MoT）结构：每个解码层都有两条分支，一条是处理语言、状态和视觉条件的*理解*分支，另一条是把 diffusion latents 解成动作的*生成*分支。两条分支的 token 数、矩阵形状、精度敏感度都不一样，量化时必须区别对待。量化对象只限 MoT decoder 中的 `Linear` 层，vision、embedding、action adapter 与 VAE 全部保持 BF16。

| 产物               | MoT linears                     | 校准                 | 峰值显存 (GB) |
| ------------------ | ------------------------------- | -------------------- | ------------- |
| Nano 16B BF16      | BF16（上游基线）                | —                    | ~33（放不下） |
| Nano 16B W8A16     | 全部 W8A16（Marlin[^5]）        | 无                   | 21.42         |
| Nano 16B GenW8A8   | 生成分支 FP8 W8A8，其余 W4A16   | 128 个 DROID episode | 15.51         |
| Edge 4B BF16       | BF16（上游基线）                | —                    | 9.20          |
| Edge 4B GenW8A8    | 生成分支 FP8 W8A8，其余 W4A16   | 128 个 DROID episode | 8.79          |

在测过的所有方案里，混合式的 GenW8A8 tradeoff 最好。全模型 W8A8 我同样构建并评测过：Nano 上它只比 GenW8A8 快约 10 ms，显存却多占 3.5 GB；Edge 上干脆更慢。道理不复杂：大 GEMM 集中在生成分支，那正是 FP8 tensor core 的用武之地；理解分支用 128 个训练 episode 做过 AWQ 式校准[^6]之后，W4A16 的精度就够用了。

这一阶段还有条经验值得单独记下：**开环动作误差预测不了闭环成功率**。按 replay 误差挑出的最优产物，在配对 rollout 里反而比一个更简单的方案低了 12 个百分点；校准能改善误差分布，但压不掉偶发的动作尖峰。因此发布里的每一个精度决策，最终标准都是 rollout 成功率。

---

## 用采样器调优少算一点

显存解决之后，下一个杠杆是每个请求的计算量。上游 RoboLab 服务器的默认采样配置是 guidance 3 加四步 UniPC[^7]，[v0.1.0](https://github.com/xxxxyu/cosmos-lite/releases/tag/v0.1.0) 作为第一个可部署版本沿用了它。把去噪步数从四改成二，改的只是一个 YAML 覆盖项，每个请求的去噪器前向就少了一半。

| Guidance / steps | 请求 p50 (ms) | BananaOnPlate SR |
| ---------------- | ------------: | ---------------: |
| g3 / s4          |         4,110 |           43/50  |
| **g3 / s2**      |     **2,403** |       **45/50**  |
| g1 / s4          |         2,431 |           32/50  |
| g1 / s2          |         1,565 |           40/50  |

两步把 Nano 的请求延迟降低 41%，而在上面的配对 rollout 里，两个模型的成功率持平甚至更好。这件事没那么反直觉：UniPC 是数值求解器，步数多意味着积分更细，不意味着策略更好。学出的向量场与推理 schedule 只要有一点错配，多算的求解步就可能把动作推离好的轨迹。Edge 表现一致，从四步到两步，成功率从 21/50 涨到 34/50。

表格同时回答了 guidance 为什么留在 3：降到 1 能省掉 CFG[^8] 的双份前向，延迟上看似白赚，闭环成功率却损失过大，过不了 gate。当然，这些都是单任务上的配对 gate；全量评测自带统计不确定性，换一个环境或机器人，最优设置未必相同。

---

## 用 kernel 与编译再省一点

最后一个杠杆，是让剩下的计算本身更便宜。项目初期的 profiling 推翻了两个普遍预期：attention 只占 GPU kernel 时间的约 4%；通常被认为能提速的 weight-only 量化，在 Nano 真实形状上基本持平，在 Edge 上甚至是负收益。以下是通过测试留下的改动，按落地顺序：

| 改动                                                              | 实测效果                                      |
| ----------------------------------------------------------------- | --------------------------------------------- |
| 生成分支上的 CUTLASS[^9] FP8 W8A8 GEMM                            | Nano −20%、Edge −12% request p50（vs. W8A16） |
| 整个 language block 级别的 `torch.compile`                        | Nano −14.5%、Edge −12.2% request p50          |
| Q/K/V 与 gate/up 共享 FP8 激活量化                                | −2%，动作逐位一致                            |
| 按形状选 attention：长序列生成 attention 用 SageAttention[^10]，其余用 FlashAttention2；条件 K/V cache | Nano 约 −12%；cache 仅 Nano 启用 |
| Edge SM89 形状专属 Triton FP8 GEMM                                | 仅 Edge：−8.4%；Nano 上被否（见下）           |
| 稀疏输入变换（只 resize 观测帧，其余零填充）                      | sample build 约 37 ms → 2 ms                  |

单独说说 Triton 那一行。CUTLASS 在 Edge 的主导 SM89 形状上 occupancy 不足，换成按形状调优的 Triton kernel 后，端到端快了约 8%。同样的 kernel 在 Nano 上也更快，但配对 rollout 从 49/50 掉到 47/50。两个 episode 的差距多半是噪声（还需进一步验证），只是对照本身已经是 49/50 的水平，没有质量余量去换约 9% 的延迟，Nano 最终留在了 CUTLASS。

---

## 哪些没成

这轮探索大部分由 coding agent 驱动，失败尝试来得便宜，死胡同攒了一长串，值得记下来。按失败所在的层面分组：

*倒在算子或端到端层面：*

- 全模型 `torch.compile`：自定义 GEMM 处图断裂，没有收益，首个请求要 20 秒。
- 静态形状编译与整请求 CUDA Graphs：新 prompt 长度触发重编译；capture 的开销换不来稳定收益。
- 全局统一 attention 后端：SageAttention 在短条件 attention 上反而更慢；FlashInfer 始终没跑赢 FlashAttention2。
- 拼接 projection 权重：个别算子更快，但 Nano 峰值显存飙到 19 GB，端到端表现不稳定；后来被共享激活量化取代。
- Edge 的条件 K/V cache：数值上正确，但收益无法复现；cache 因此只随 Nano 发布。
- torchao INT8 weight-only：在模型真实形状上比 BF16 慢 3–4 倍。
- CFG 两个分支合并 batch：attention 调用次数更少、单次更大，端到端反而更慢。
- 更短的动作 chunk（32→16/8）：名义计算量更小，实测延迟更差，控制语义也变了。

*kernel 赢了，rollout 输了：*

- Guidance 1：请求更快，闭环失败更多（Nano 上 40/50 对 45/50）。
- Nano 的 Triton FP8 GEMM：kernel 和 replay 都更快，rollout 点估计下滑（49→47/50）。

*暂缓：*

- TensorRT-LLM：方案本身可信，但要把 MoT、vision、VAE 整体导出，工程量是另一个量级。

完整的决策地图，以及每项否决背后的测量数据，都在仓库的[优化报告](https://github.com/xxxxyu/cosmos-lite/blob/main/docs/cosmos_lite_optimization_report.md)里。

---

## 接下来

v0.3.0 是第一个完成完整测量的版本，RTX 4090 是目前的主要测试目标。接下来的计划：

- 更多 NVIDIA GPU 架构与推理后端。
- 更多 world-action 模型，不限于现在的 Nano 和 Edge DROID 策略。
- 真机评测与优化。

如果你在做策略部署、仿真数据生产，或者 MoT 类模型的推理，欢迎[提 issue](https://github.com/xxxxyu/cosmos-lite/issues)或[发邮件](mailto:lixiangy22@mails.tsinghua.edu.cn)聊聊。

[^1]: NVIDIA, ["Cosmos 3 Technical Report"](https://research.nvidia.com/labs/cosmos-lab/cosmos3/), 2026.
[^2]: Alexander Khazatsky et al., ["DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset"](https://arxiv.org/abs/2403.12945), arXiv, 2024.
[^3]: NVIDIA, ["RoboLab"](https://github.com/NVlabs/RoboLab), GitHub, 2026.
[^4]: NVIDIA 报告的 RoboLab-120 default-instruction 成功率，来源为技术报告 Tab. 19 与模型卡。
[^5]: Elias Frantar et al., ["MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models"](https://arxiv.org/abs/2408.11743), arXiv, 2024.
[^6]: Ji Lin et al., ["AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978), MLSys, 2024.
[^7]: Wenliang Zhao et al., ["UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models"](https://arxiv.org/abs/2302.04867), arXiv, 2023.
[^8]: Jonathan Ho and Tim Salimans, ["Classifier-Free Diffusion Guidance"](https://arxiv.org/abs/2207.12598), arXiv, 2022.
[^9]: vLLM, ["FP8 Quantization"](https://docs.vllm.ai/en/latest/features/quantization/fp8/), documentation.
[^10]: Jinnian Zhang et al., ["SageAttention: Accurate 8-Bit Attention for Plug-and-Play Inference Acceleration"](https://arxiv.org/abs/2410.02367), arXiv, 2024.
