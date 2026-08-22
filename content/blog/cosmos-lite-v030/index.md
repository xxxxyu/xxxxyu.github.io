+++
title = "Running Cosmos 3 Robot Policies on a Single RTX 4090"
date = "2026-08-21"
updated = "2026-08-21"
description = "How quantization, sampler tuning, and kernel work put NVIDIA's 16B Cosmos 3 robot policy on one 24 GB GPU — and turned the runtime into a parallel rollout engine."
template = "blog-page.html"

[taxonomies]
tags = ["Embodied AI", "Quantization", "Edge AI"]

[extra]
toc = true
og_image = "/img/blog/cosmos-lite-v030/cover.jpg"
og_image_alt = "Side-by-side closed-loop rollout comparison of Edge BF16 and Edge GenW8A8"
+++

> Code: [cosmos-lite v0.3.0](https://github.com/xxxxyu/cosmos-lite) \
> Models: [Nano GenW8A8](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-GenW8A8) · [Edge GenW8A8](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-GenW8A8)

NVIDIA's Cosmos 3[^1] shipped with robot policies trained on DROID[^2]. Running them yourself is harder than it sounds: the 16B Nano policy needs about 33 GB just to load, and the [reference workflow](https://github.com/NVIDIA/Cosmos-Framework) assumes a datacenter GPU like an A100 or H100. A single RTX 4090 could not run it at all.

[Cosmos Lite](https://github.com/xxxxyu/cosmos-lite) fills that gap. It is an efficient inference runtime for Cosmos 3 policies, built on top of [Cosmos Framework](https://github.com/NVIDIA/Cosmos-Framework). It covers two use cases: **local serving** as a batch-one policy server for real-robot deployment, and **parallel rollout** as a shared server feeding multiple simulator lanes, exporting rollout trajectories for further training.

{{ video(src="/videos/cosmos-lite-v030/demo.mp4", max_width="100%", caption="Edge BF16 with official defaults vs. Edge GenW8A8 with the optimized runtime, running closed-loop in RoboLab.") }}

The benchmark covers all 120 RoboLab[^3] tasks, 10 episodes each, all with the same sampler settings (guidance 3, two UniPC denoise steps, shift 5), and latency is measured per action request on a single RTX 4090.

| Model    | Artifact    | VRAM (GB) | Request p50 (ms) | SR (%)    |
| -------- | ----------- | --------: | ---------------: | --------: |
| Edge 4B  | BF16 (NVIDIA-reported[^4]) | — |      — | 22.90     |
| Edge 4B  | BF16        |      9.20 |            582.0 | 20.92     |
| Edge 4B  | **GenW8A8** |      8.79 |        **331.4** | 19.25     |
| Nano 16B | BF16 (NVIDIA-reported[^4]) | — |      — | 36.80     |
| Nano 16B | W8A16       |     21.42 |          2,403.0 | 31.50     |
| Nano 16B | **GenW8A8** | **15.51** |        **958.5** | **31.67** |

Both Nano rows require quantization because BF16 does not fit a 24 GB GPU. GenW8A8 holds task success while running Nano **2.5×** faster than W8A16 and Edge **1.76×** faster than BF16. One caveat, stated plainly: every rate here trails the NVIDIA-reported rows (31.7% vs. 36.8% for Nano, 19.3% vs. 22.9% for Edge). "Holds task success" means *statistically indistinguishable*, not *free*. Verify success on your own tasks before deploying.

---

## Make it fit via quantization

The first problem is fitting a 16B policy into 24 GB, and the model's structure determines how. A Cosmos 3 policy is a Mixture-of-Transformers (MoT): every decoder layer has an *understanding* branch for language, state, and visual conditioning, and a *generation* branch that turns diffusion latents into actions. The two branches differ in token counts, matrix shapes, and precision sensitivity, so they are quantized separately. We quantize the `Linear` modules inside the MoT decoder, while vision, embeddings, action adapters, and the VAE stay BF16.

| Artifact           | MoT linears                                  | Calibration          | Peak VRAM (GB)   |
| ------------------ | ------------------------------------------- | -------------------- | ---------------- |
| Nano 16B BF16      | BF16 (upstream baseline)                    | —                    | ~33 (does not fit) |
| Nano 16B W8A16     | all W8A16 (Marlin[^5])                 | none                 |            21.42 |
| Nano 16B GenW8A8   | generation FP8 W8A8, rest W4A16             | 128 DROID episodes   |            15.51 |
| Edge 4B BF16       | BF16 (upstream baseline)                    | —                    |             9.20 |
| Edge 4B GenW8A8    | generation FP8 W8A8, rest W4A16             | 128 DROID episodes   |             8.79 |

Mixed GenW8A8 shows the best tradeoff among evaluated configurations, better than full-model W8A8, which I also built and benchmarked. On Nano, full W8A8 saved only ~10 ms over GenW8A8 while using 3.5 GB more memory; on Edge it was outright slower than the mixed scheme. The generation branch holds the large GEMMs, which is where FP8 tensor cores pay off, while the understanding branch stays accurate at W4A16 with AWQ-style calibration[^6] over 128 training episodes.

One caveat from this stage: **open-loop action error does not predict closed-loop success.** The best artifact by replay error lost 12 points of success rate against a simpler one in matched rollouts, and calibration, while it improves the error distribution, does not remove rare action spikes. Every precision decision in the release is therefore gated on rollout success rate.

---

## Do less work via sampler tuning

After quantization, the next lever is how much computation each request performs. The upstream RoboLab server defaults to guidance 3 with four UniPC[^7] denoise steps, and [v0.1.0](https://github.com/xxxxyu/cosmos-lite/releases/tag/v0.1.0), the first deployable release, inherited that setting. Halving the step count to two, a YAML override and nothing more, halves the denoiser forwards per request.

| Guidance / steps | Request p50 (ms) | BananaOnPlate SR       |
| ---------------- | ---------------: | ---------------------: |
| g3 / s4          |            4,110 |                 43/50  |
| **g3 / s2**      |        **2,403** |             **45/50**  |
| g1 / s4          |            2,431 |                 32/50  |
| g1 / s2          |            1,565 |                 40/50  |

Two steps cut Nano request latency by 41%, and in the paired rollout gates above the success rate held or improved on both models. This is less surprising than it sounds: UniPC is a numerical solver, so more steps mean finer integration, not a better policy. When the learned vector field and the inference schedule are imperfectly matched, extra solver points can move actions away from good trajectories. Edge showed the same pattern, improving from 21/50 to 34/50 as steps went from four to two.

The table also explains why guidance stays at 3: dropping it to 1 removes the CFG[^8] double-forward and looks like free latency, but it loses enough closed-loop success to fail the gate. These are paired gates on one task; the full-suite results carry the usual uncertainty, and other environments or robots may prefer different settings.

---

## Do it cheaper via kernels and compilation

The last lever is the cost of the computation that remains. Early profiling invalidated two common assumptions here: attention took only ~4% of GPU kernel time, and weight-only quantization measured roughly neutral on Nano's real shapes and slightly negative on Edge. The changes that survived testing, in the order they landed:

| Change                                                            | Measured effect                                    |
| ----------------------------------------------------------------- | -------------------------------------------------- |
| CUTLASS[^9] FP8 W8A8 GEMMs on the generation branch               | Nano −20%, Edge −12% request p50 (vs. W8A16)       |
| `torch.compile` over whole language blocks                        | Nano −14.5%, Edge −12.2% request p50               |
| Shared FP8 activation quantization for Q/K/V and gate/up          | −2%, actions bit-identical                         |
| Shape-aware attention: SageAttention[^10] for long generation attention, FlashAttention2 elsewhere; condition K/V cache | ≈ −12% on Nano; the cache is Nano-only |
| Triton FP8 GEMM for Edge's SM89 shapes                            | Edge only: −8.4%; rejected on Nano (see below)     |
| Sparse input transform (resize the observed frame, zero-fill the rest) | sample build ~37 ms → ~2 ms                    |

About the Triton row: CUTLASS under-occupies Edge's dominant SM89 shapes, and the shape-tuned Triton kernel is ~8% faster end-to-end there. The same kernels were also faster on Nano, but Nano's paired rollout moved from 49/50 to 47/50. Two episodes is likely noise (needs further verification), yet with a 49/50 control there was no quality margin to trade for ~9% latency, so Nano kept CUTLASS.

---

## What didn't work

Most of the exploration was coding-agent-driven, so failed attempts were cheap to generate, and the dead-end list is long enough to be worth recording. Grouped by where they failed:

*Failed at the operator or end-to-end level:*

- Whole-model `torch.compile`: graph breaks around the custom GEMMs, no gain, 20 s first requests.
- Static-shape compile and whole-request CUDA Graphs: recompiles on new prompt lengths; capture overhead with no steady gain.
- One global attention backend: SageAttention loses on the short condition attention; FlashInfer never beat FlashAttention2.
- Concatenated projection weights: some isolated operator wins, 19 GB peak memory on Nano, inconsistent end-to-end; replaced by shared activation quantization.
- Edge condition K/V cache: numerically correct, no repeatable gain; the cache ships for Nano only.
- torchao INT8 weight-only: 3–4× slower than BF16 on the model's real shapes.
- Batching the CFG branches: fewer but larger attention calls, net slower end to end.
- Shorter action chunks (32→16/8): lower nominal compute, worse measured latency, different control semantics.

*Passed the kernels, failed the rollout:*

- Guidance 1: faster requests, more closed-loop failures (40/50 vs. 45/50 on Nano).
- Nano Triton FP8 GEMM: faster kernels and replay, rollout point estimate dipped (49→47/50).

*Deferred:*

- TensorRT-LLM: credible, but exporting the full MoT-plus-vision-plus-VAE stack is a different-scale project.

The full decision map, with the measurements behind each drop, is in the repo's [optimization report](https://github.com/xxxxyu/cosmos-lite/blob/main/docs/cosmos_lite_optimization_report.md).

---

## What's next

v0.3.0 is the first fully measured release, with the RTX 4090 as its primary tested target. Planned next:

- More NVIDIA GPU architectures and inference backends.
- More world-action models beyond the current Nano and Edge DROID policies.
- Real-robot evaluation and optimization.

If you are working on policy deployment, simulation data generation, or inference for MoT-style models, I'd love to hear from you: [open an issue](https://github.com/xxxxyu/cosmos-lite/issues) or [email me](mailto:lixiangy22@mails.tsinghua.edu.cn).

[^1]: NVIDIA, ["Cosmos 3 Technical Report"](https://research.nvidia.com/labs/cosmos-lab/cosmos3/), 2026.
[^2]: Alexander Khazatsky et al., ["DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset"](https://arxiv.org/abs/2403.12945), arXiv, 2024.
[^3]: NVIDIA, ["RoboLab"](https://github.com/NVlabs/RoboLab), GitHub, 2026.
[^4]: RoboLab-120 default-instruction success rates as reported by NVIDIA (technical report Tab. 19 and model cards).
[^5]: Elias Frantar et al., ["MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models"](https://arxiv.org/abs/2408.11743), arXiv, 2024.
[^6]: Ji Lin et al., ["AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978), MLSys, 2024.
[^7]: Wenliang Zhao et al., ["UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models"](https://arxiv.org/abs/2302.04867), arXiv, 2023.
[^8]: Jonathan Ho and Tim Salimans, ["Classifier-Free Diffusion Guidance"](https://arxiv.org/abs/2207.12598), arXiv, 2022.
[^9]: vLLM, ["FP8 Quantization"](https://docs.vllm.ai/en/latest/features/quantization/fp8/), documentation.
[^10]: Jinnian Zhang et al., ["SageAttention: Accurate 8-Bit Attention for Plug-and-Play Inference Acceleration"](https://arxiv.org/abs/2410.02367), arXiv, 2024.
