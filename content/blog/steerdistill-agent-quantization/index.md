+++
title = "Quantization-Aware Distillation for LLM Agents: Recovering Capabilities with Closed-Loop Guidance"
date = "2026-11-23"
updated = "2026-11-23"
description = "Quantized models often fail on details in agent tasks. SteerDistill has the full-precision teacher provide guidance at the step where the student goes wrong, then distills the correction back into the quantized model."
template = "blog-page.html"

[taxonomies]
tags = ["LLM", "Quantization", "Agents", "Knowledge Distillation"]

[extra]
toc = true
og_image = "/img/blog/steerdistill-agent-quantization/teaser.png"
og_image_alt = "SteerDistill submission-version Figure 1: a working-directory file-reading case, mean W4A16 main results, and training curves"
ai_translation_source = "zh"
ai_translation_harness = "Claude Code"
ai_translation_model = "Claude Opus 5.5"
ai_translation_effort = ""
+++

> 🔗 [Paper](https://arxiv.org/abs/2611.13410) | [Code](https://github.com/air-embodied-brain/SteerDistill) | [Project page](https://air-embodied-brain.github.io/SteerDistill/)
>
> Adapted by Claude Opus 5.5 from the [original Zhihu post](https://zhuanlan.zhihu.com/p/2107070075614262595) (in Chinese).

LLM quantization has come a long way, and many research efforts now focus on rotations, transforms, error compensation, and calibration, refining a few classic methods in ever finer detail. **On chat, QA, and completion tasks these methods are close to lossless, but in agent settings their effectiveness drops noticeably.** They do reduce quantization error, yet the improvement does not necessarily carry over to end-to-end metrics such as task success rate. In our evaluation, quantizing Qwen3-8B directly to W4A16 with GPTQ lowers the average across five agent benchmarks[^1] from 53.54 (BF16) to 46.36, a drop of more than 7 points. For applications that depend on high success rates, a loss of this size is hard to accept.

Industry relies more on data scale: with enough data in the target domain, the design of the quantization method itself becomes secondary. Rather than carefully designing a quantizer, one can collect more production data and run a few more training iterations. Generating and cleaning data, however, is costly and tedious. Is there a more elegant approach?

Our approach is to expand the role of the teacher (usually the BF16 model before quantization) in quantization-aware distillation (QAD) for agent tasks. **Beyond providing loss supervision as existing methods do, the teacher also gives the student closed-loop guidance during training: at whichever step the student goes wrong, the student corrects that step and continues from there.** The teacher does not act on the student's behalf; it only points out what went wrong and how to get it right.

{{ image(src="/img/blog/steerdistill-agent-quantization/teaser.png", alt="SteerDistill submission-version Figure 1: a quantized student fails a working-directory file-reading case and corrects it under teacher guidance, alongside W4A16 average scores and training curves for each method", dimmable=true, caption="Submission-version Figure 1: the working-directory reading case, mean W4A16 main results, and training curves.") }}

We ran experiments on three models and five agent benchmarks. **At W4A16, all three models come within 1 point of their own BF16 five-benchmark average.** For Qwen3-8B, the strongest baseline, QeRL, remains more than two points below BF16, and plain GPTQ is more than 7 points below. Beyond the end-to-end comparison, we used ablations and case studies to examine each design choice; details are in the paper and on the project page.

The rest of this post discusses several of these observations and analyses.

## Why agents are harder

The question can be examined at several levels.

**First, agent tasks involve many more turns of interaction.** Consider a task that takes 20 turns, every one of which must be correct. At a fixed 99% success rate per turn, end-to-end success is about 82%; at 90% per turn, it falls to about 12%. The example illustrates how capability lost to quantization accumulates and is amplified over many turns.

**Second, certain critical turns cause abnormal state transitions that lower the success rate of subsequent turns.** This distinguishes agent tasks from ordinary multi-turn conversation: deleting the wrong file, entering the wrong directory, or taking an irreversible action. With its capability already reduced, a quantized model often has more difficulty recovering from abnormal states it has not seen before. In many cases, a single wrong turn is enough to make the whole task fail.

**Third, quantized models are also more likely to make errors within a single decision.** This is the issue we focused on most, and we carried out many case studies on it. In many of the failures we observed, **the quantized model's overall direction remains correct, but it is more likely to overlook details required to complete the task, including task requirements and constraints imposed by the environment and tool calls.**

Most of these are basic mistakes. In the working-directory case in Figure 1, `cat` in this environment can only read files in the current directory, yet the quantized model reads the file directly without first checking which directory it is in[^2].

## Closed-loop guidance

Existing quantization methods, or more precisely methods for recovering performance after quantization, fall roughly into two groups:

- **Calibration or training on offline data.** This is the most common group, including post-training quantization (PTQ) methods such as GPTQ and AWQ[^3], and QAT/QAD methods such as EfficientQAT and BitDistiller[^4]. The data is generated offline by humans, by models such as GPT, or by the teacher, and does not change as the quantized model is updated during training.
- **Training on online-generated trajectories.** One kind is RL-based, such as QeRL[^5]: the quantized model rolls out online and updates its parameters from the reward. The other is on-policy distillation (OPD), such as NVIDIA NeMo's on-policy QAD[^6]: the quantized model also rolls out online, but the teacher provides dense, token-level supervision.

The first group is inexpensive and simple to implement, but offline data often fails to cover the errors specific to the quantized model. In the Figure 1 example, the full-precision teacher does not make the same mistake, so its successful trajectories rarely cover the context in which the student actually fails, and such errors remain difficult to correct. The second group lets the quantized model learn in its own failure contexts, but a relatively weak quantized model may not find the correct action on its own.

We tested this on BFCL V4 multi-turn tasks. At each point where the quantized student diverges from the full-precision teacher, we restored the history and environment state from before the error and let the quantized model retry four times from that same state. Without guidance, Pass@4 at divergence points (at least one of four attempts producing the correct action) was only 9.6%; with guidance from the full-precision model, it rose to 82.3%. Additional attempts alone help little; what the student lacks is information that changes its decision at that step.

The approach then follows directly: **since the student cannot correct the error on its own, the teacher provides targeted guidance, kept within the student's capability.** Both giving the answer outright and leaving the student to trial and error work less well than targeted hints. The training procedure is shown in Figure 2.

{{ image(src="/img/blog/steerdistill-agent-quantization/method.png", alt="SteerDistill paper Figure 2: within one training cycle, the student rolls out, backtracks at a divergence point and corrects it under teacher guidance, and the corrections are then distilled into the student", dimmable=true, caption="Paper Figure 2: the SteerDistill training cycle, covering student rollout, divergence-point correction, and internalization through distillation.") }}

**Stage 1: the student rolls out, and failed trajectories are corrected under the teacher's guidance.**

The student first rolls out independently. For each failed trajectory, we take a successful teacher trajectory on the same task and from the same initial environment as a reference, align the two by changes in environment state, and locate the first divergence point. We then restore the history and environment to just before the error. The teacher compares the student's incorrect response with the reference and writes a steering prompt identifying what the student overlooked, and the student regenerates that step with this guidance.

After the corrected response is executed, we perform a local verification that checks only whether the environment state and returned output after this step match the reference. If it passes, the correction is kept, the guidance is removed, and the student continues; if the task still fails, we look for the next divergence point. If it does not pass, the execution feedback is returned to the teacher to refine the guidance, and the student retries from the same state until the retry budget is exhausted.

**Stage 2: the student distills and internalizes the corrected trajectories, whether or not they ultimately succeed.**

At the start of each training cycle, we freeze a snapshot of the current student and use it to collect trajectories and generate corrections. On corrected turns, we follow the approach of context distillation[^7]: the frozen snapshot with the steering prompt supervises the student through forward KL, while the student's own context contains no guidance. In this way, the change in predictions induced by the guidance is written into the student's parameters.

All other turns follow standard OPD, with the teacher providing supervision through reverse KL. After training, deployment uses only the quantized student, with no teacher and no injected guidance.

## Results and ablations

The main experiments are evaluated on BFCL V4, τ²-Bench, WebShop, ALFWorld, and Search QA; for each model, all methods start from the same BF16 checkpoint of that model post-trained on the target tasks. For Qwen3-8B at W4A16, SteerDistill outperforms every quantized baseline on all five benchmarks, with an average of 53.19, which is 2.18 above the strongest baseline, QeRL, and only 0.35 below BF16. On Qwen3.5-4B and Gemma 4 12B-it, SteerDistill's average is 1.71 and 1.67 above QeRL, respectively. The table below summarizes the updated five-benchmark averages of the three models at W4A16, with BF16 as a reference.

| Method | Qwen3-8B | Qwen3.5-4B | Gemma 4 12B-it |
| --- | ---: | ---: | ---: |
| BF16 (reference) | 53.54 | 61.68 | 65.13 |
| GPTQ | 46.36 | 54.22 | 57.66 |
| EfficientQAT | 47.87 | 55.94 | 59.44 |
| BitDistiller | 48.49 | 56.68 | 60.07 |
| On-Policy QAD | 50.40 | 58.59 | 62.02 |
| QeRL | 51.01 | 59.24 | 62.64 |
| **SteerDistill** | **53.19** | **60.95** | **64.31** |

The lower-precision experiments are run on Qwen3-8B, and SteerDistill's advantage persists: at W4A8 it is 1.93 above QeRL, and at W2A16 it is 3.39 above the strongest baseline, On-Policy QAD. The W2A16 average of 42.97, however, remains well below BF16, so the 2-bit loss is not resolved. Across all settings, the only exception is W2A16 Search QA, where BitDistiller scores slightly higher than SteerDistill.

SteerDistill also converges faster in training. On τ²-Bench with Qwen3-8B at W4A16, SteerDistill reaches a 31.68% success rate after 32 GPU hours, while QeRL reaches 29.84% after 96 GPU hours[^8]. At inference, computed over each method's own successful tasks, SteerDistill produces 4.0K output tokens per successful task on average, close to BF16's 3.9K.

All ablations are run on Qwen3-8B, W4A16, BFCL V4 Overall (46.87 for the full method). The main findings are as follows:

- **Divergence should be located by environment state; matching actions alone is insufficient.** Different actions that produce the same environment state transition can be treated as equivalent. What counts as "environment state" depends on the environment; it can be understood as any information in the environment that affects the agent's decisions.
- **The teacher should guide rather than supply the answer directly.** The aim is to avoid distribution shift at the data level as far as possible. From the same restored state, having the teacher take over yields 45.06, whereas teacher guidance with the student correcting the step itself yields 46.87.
- **Structured guidance from the teacher model works best.** We require the teacher to write guidance in a fixed structure[^9]. The structured format outperforms free text, and guidance written by the full-precision teacher outperforms guidance written by the student itself or by a stronger external model. One possible explanation is that the teacher's distribution is closer to the student's; without format constraints, guidance from a strong model may not be understood by a weak one.
- **Partial successes should be kept.** Even when a trajectory ultimately fails, the corrected segments that pass verification remain valuable supervision signals and improve data efficiency. Removing local verification lowers the score sharply to 42.18, and additionally requiring final task success brings no gain.
- **Forward KL on corrected turns, reverse KL on other turns.** This setting follows existing practice, and the ablation mainly serves as a check: every other combination performs worse.

## Limits and outlook

This work concerns recovering a quantized model's capability on agent tasks. That is, we assume the teacher has already been fully trained on the target tasks and the student only needs to approach the teacher as closely as possible. Such scenarios are common in practice.

In a supplementary experiment, we also examined the role of the initial GRPO[^10]. The main experiments start from a BF16 checkpoint that has undergone GRPO on the target tasks; here we instead start from the original Qwen3-8B checkpoint, skip the initial GRPO on the target tasks, and train the W4A16 model directly. The original BF16 model scores 40.40 on BFCL V4 and 30.00 on τ²-Bench.

| Setting | Total training budget (GPU hours) | BFCL V4 Overall (%) | τ²-Bench Pass@1 (%) |
| --- | ---: | ---: | ---: |
| No initial GRPO, QeRL | 32 | 36.07 | 24.95 |
| No initial GRPO, SteerDistill | 32 | 39.67 | 27.67 |
| No initial GRPO, QeRL (converged) | 132 | 42.74 | 28.53 |
| BF16 GRPO → SteerDistill QAD | ~105 | 46.87 | 31.79 |
| BF16 GRPO → QeRL | ~169 | 44.95 | 29.84 |

With the same 32 GPU hours of training, SteerDistill scores higher than QeRL on both metrics. Trained further to 132 GPU hours, QeRL surpasses directly trained SteerDistill, and its BFCL score also exceeds the original BF16's 40.40, showing that RL keeps learning from the task reward; SteerDistill, by contrast, aims to bring the student close to a given teacher.

Running GRPO on BF16 first and then SteerDistill QAD reaches the best results in the table with about 105 GPU hours in total, roughly 20% less total budget than training QeRL directly to convergence. In this set of experiments, spending the budget first on training the BF16 teacher and then distilling into the quantized model is the more cost-effective allocation.

The method also depends on environments that are executable, resettable, and verifiable, and it requires successful teacher trajectories as references. These conditions can be met in the simulated environments we used, but in more open and complex environments, where states are hard to restore and outcomes hard to verify, new problems may arise; this may also be an open problem for the field.

Finally, this work focuses on quantization, and we have not yet validated the approach in general distillation settings (between large and small models, or even distillation at the foundation-model level). We may add such experiments later. The code is publicly available, and we welcome attempts to apply it in other settings.

[^1]: W4A16 means 4-bit weights and 16-bit activations. The five metrics are BFCL Overall, τ²-Bench Pass@1, WebShop and ALFWorld success rates, and Search QA EM, combined as an unweighted mean.
[^2]: The full executions of the teacher and student in this case are shown in the [project page case](https://air-embodied-brain.github.io/SteerDistill/cases/?group=training&case=directory-navigation).
[^3]: Elias Frantar et al., ["GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"](https://arxiv.org/abs/2210.17323), ICLR, 2023; Ji Lin et al., ["AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978), MLSys, 2024.
[^4]: Mengzhao Chen et al., ["EfficientQAT: Efficient Quantization-Aware Training for Large Language Models"](https://arxiv.org/abs/2407.11062), ACL, 2025; DaYou Du et al., ["BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation"](https://aclanthology.org/2024.acl-long.7/), ACL, 2024.
[^5]: Wei Huang et al., ["QeRL: Beyond Efficiency — Quantization-enhanced Reinforcement Learning for LLMs"](https://arxiv.org/abs/2510.11696), arXiv, 2025.
[^6]: NVIDIA NeMo RL, ["Quantization-Aware Distillation (On-Policy QAD)"](https://docs.nvidia.com/nemo/rl/latest/guides/quantization-aware-rl.html#quantization-aware-distillation-on-policy-qad), documentation.
[^7]: Tianzhu Ye et al., ["On-Policy Context Distillation for Language Models"](https://arxiv.org/abs/2602.12275), arXiv, 2026.
[^8]: Training uses 8 A100 80GB GPUs, and GPU hours are computed as wall-clock time multiplied by the number of GPUs.
[^9]: The guidance states what the student overlooked, the conditions under which it applies, and how to correct it; the format and examples are shown on the [project page](https://air-embodied-brain.github.io/SteerDistill/#guidance).
[^10]: Total cost adds 73 GPU hours of BF16 GRPO to the low-bit training time estimated from the training curves (32 GPU hours for SteerDistill and 96 for QeRL); scores come from the final main results.
