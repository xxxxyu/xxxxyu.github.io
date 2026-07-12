+++
title = "Making Low-Bit LLMs Actually Fast in Parallel via Vector Table Lookup"
date = "2026-07-11"
updated = "2026-07-11"
description = "Vec-LUT turns repetitive scalar table lookups into contiguous vector reads, accelerating parallel ternary LLM inference on x86 and ARM CPUs by up to 4.2×."
template = "blog-page.html"

[taxonomies]
tags = ["LLM Inference", "Edge AI", "Quantization", "GeMM"]

[extra]
toc = true
og_image = "/img/blog/vec-lut-parallel-low-bit-llm/scalar-vs-vector-lut.png"
og_image_alt = "Diagram comparing scalar and vector table lookup for parallel LLM inference"
+++

> 🏆 ACM MobiSys 2026 <strong class="highlight">Best Paper Award Runner-Up</strong> \
> 🔗 [Paper](https://doi.org/10.1145/3745756.3809200) | [Code](https://github.com/OpenBitSys/vlut.cpp) | [Slides](https://github.com/OpenBitSys/vlut.cpp/blob/master/media/veclut_slides_mobisys_2026.pdf) | [Chinese post (清华AIR公众号)](https://mp.weixin.qq.com/s/_vSD8_r9GVjCppdVG2MqAg)

Today's ultra-low-bit LLMs (typically below 4 bits per weight) have small memory footprints. For example, the weights of Microsoft's 2B BitNet model occupy about 0.4 GB[^1]. So they should run fast on a laptop or smartphone, right?

**Not automatically.** Fewer bits reduce memory footprint, but they only translate into speed when the underlying inference kernel matches the data representation and the workload. **Lookup-table (LUT)** kernels work well for single-token generation, yet lose much of that advantage when many tokens are processed in parallel. Vec-LUT addresses this parallel access pattern. Across five x86 and ARM devices, it delivers up to **4.2×** end-to-end prefilling speedup over baselines at similar bits per weight (BPW); on Snapdragon 8 Elite, its CPU implementation even outperforms the phone's NPU (llama.cpp's Hexagon backend) using only two CPU cores.

{{ video(src="/videos/vec-lut-parallel-low-bit-llm/demo.mp4", autoplay=true, loop=true, max_width="80%", caption="Vec-LUT vs. llama.cpp: 3× end-to-end speedup for 32-way parallel decoding on one CPU core.") }}

---

## On-device LLM inference on CPUs

On-device models now power mobile assistants, local agents, and embodied AI systems, where memory remains limited. Quantization research has moved from 8-bit[^2] to 4-bit[^3] to 2-bit[^4], and now to **1.58-bit ternary** models, whose weights take values in `{-1, 0, 1}`, a format popularized by Microsoft's [BitNet](https://github.com/microsoft/BitNet)[^5].

Their small weight alphabet also makes **lookup-table (LUT)-based inference** practical: instead of dequantizing low-bit weights and multiplying them with activations at runtime, a kernel can precompute a small table and replace much of the arithmetic with table lookup. Prior LUT-based systems such as [bitnet.cpp](https://github.com/microsoft/BitNet)[^6] and [T-MAC](https://github.com/microsoft/T-MAC)[^7] showed that CPUs can be competitive for single-token generation. Vec-LUT asks whether the same idea can remain fast when the workload becomes parallel.

---

## How LUT-based inference works

A major cost in quantized LLM inference is **mixed-precision general matrix multiplication (mpGeMM)**, such as multiplying 1.58-bit weights with INT8 or FP16 activations. Commodity CPUs and many edge accelerators do not natively support these mixed-precision operations, so inference frameworks usually need either dequantization or custom kernels[^8].

{{ image(src="/img/blog/vec-lut-parallel-low-bit-llm/lut-based-inference.png", dimmable=true, caption="LUT-based mpGeMM replaces dequantization and multiplication with table lookup in quantized Transformer layers.") }}

LUT-based inference starts by splitting the weight matrix into small groups. Low-bit weights have very few possible patterns: four ternary weights have only `3^4 = 81` combinations. For a given activation vector, the kernel can **precompute** every "weight pattern × activation" result and store the results in a table. At runtime, the packed weight group becomes an **index** into that table, so the kernel can skip dequantization and multiplication for the inner loop and accumulate the looked-up values instead.

{{ image(src="/img/blog/vec-lut-parallel-low-bit-llm/lut-example.png", max_width="60%", dimmable=true, caption="A concrete LUT example: compute o = w × v by using the ternary weight group as a lookup index.") }}

---

## Lower bits ≠ faster, in parallel

The catch is parallelism. Existing LUT kernels follow what we call a **scalar LUT** paradigm: a `1→1` lookup. Each token has its *own* table, so processing N tokens means building or accessing N tables and doing N separate lookup streams.

For single-token generation, scalar LUT works well because only one table and one lookup stream are involved, allowing existing kernels to use the available memory bandwidth effectively. But **parallel inference** is common in real workloads: prefilling medium/long prompts[^9]<sup class="footnote-separator">, </sup>[^10], serving multiple requests[^11]<sup class="footnote-separator">, </sup>[^12], parallel test-time scaling[^13]<sup class="footnote-separator">, </sup>[^14], and speculative decoding[^15]<sup class="footnote-separator">, </sup>[^16] all process many tokens at once. In these cases, scalar LUT loses much of its advantage:

- Memory bandwidth utilization drops **below 40%**.
- Sometimes it's *slower* than not using a LUT at all.

The root cause is memory access. Table lookup is a **random, non-contiguous** operation. Repeating it over multiple per-token tables, especially when their aggregate working set exceeds cache capacity, spends much of the time moving scattered data rather than doing useful accumulation. In our profiling of a representative scalar-LUT kernel, lookup (including weight loading) accounts for nearly **half** of the mpGeMM latency.

The breakdown below shows where the time goes in T-MAC's scalar-LUT mpGeMM kernel, measured on an Orange Pi 5 Plus with a single thread:

| GeMM shape (`M × K`) | Precompute |    Lookup | Accumulate | Scale |
| -------------------- | ---------: | --------: | ---------: | ----: |
| `320 × 3200`         |       0.8% | **47.6%** |      25.0% | 26.6% |
| `128 × 8640`         |       0.7% | **47.3%** |      25.5% | 26.4% |

For parallel inference, optimizing arithmetic alone cannot remove this lookup-dominated memory bottleneck.

---

## The core idea: look up N at once

The key observation is that **lookup indices come from the weights, and the weights are shared across all tokens.** The same weight index applies to every token in the parallel group, so independent per-token lookups are unnecessary.

Vec-LUT changes the lookup unit from a scalar to a vector. Instead of one table per token, it builds a single **unified table** across all parallel tokens. Each table entry stores not one scalar result, but a *vector* of results, one per token. A weight index now triggers one `1→N` lookup that fetches the results for the whole token group.

{{ image(src="/img/blog/vec-lut-parallel-low-bit-llm/scalar-vs-vector-lut.png", dimmable=true, caption="Vec-LUT: turning per-token 1→1 lookups into a single multi-token 1→N lookup") }}

The resulting access pattern has three useful properties:

- N independent lookup passes become one vector lookup pass per weight index.
- Random scattered lookups become contiguous vector reads, followed by vector accumulation.
- The kernel no longer depends on hardware lookup instructions such as ARM `TBL` or x86 `PSHUF`, making it less tied to a specific ISA.

In our latency breakdown, the lookup share drops from T-MAC's **47%** to **under 1%**.

---

## Making vector LUT practical

Two engineering details determine whether vector LUT is fast in practice.

- **Tensor layout.** A unified table only helps if the surrounding tensors follow compatible layouts. A mismatched layout can make the kernel up to **12×** slower. Our **Vector LUT-Centric Tensor Layout** stores activations, tables, and outputs in token-contiguous form, and packed weights in tile-contiguous form. It performs weight reordering offline and fuses the runtime activation/output transformations into the kernel to minimize overhead.

- **Cache locality.** Vector LUT increases the table size by N×. At a sequence length of 512, one 2-bit, INT16 vector table for a Llama3 8B GeMM would exceed **280 MiB**, far beyond the cache capacity of edge CPUs. Our **Cache-Aware Streamed Lookup** slices the precompute-and-lookup pipeline into cache-sized tiles and streams through them, so most LUT work stays in on-chip cache instead of repeatedly touching a large random-access table.

Dropping hardware lookup instructions also relaxes bit-width and shape constraints. That lets us use **flexible sub-2-bit packing**: our `I1` packing reaches **1.60 bits per weight**, while supporting a broad range of weight shapes. Additional optimizations, including topological precomputation and INT16–INT32 hierarchical accumulation, reduce the remaining overhead.

---

## Results: 5 devices × 3 models

We evaluated Vec-LUT across five x86/ARM devices (PC, laptop, SBC, smartphone, and CPU server) and three ternary LLMs: Falcon3 1B, HF BitNet 3B, and Llama3 8B.

- **End-to-end prefilling:** up to **4.2×** speedup with `I1` (1.60 BPW) and **2.6×** with `I2` (2.00 BPW) over baselines with similar BPWs.
- **CPU vs. NPU:** on Snapdragon 8 Elite, Vec-LUT `I2` reaches **1.05–1.12×** the throughput of llama.cpp's Q4_0 Hexagon NPU backend for Llama3 8B prefilling, using only **2 CPU cores**.
- **Continuous batching:** on an 8-core, **$0.50/h** AWS Graviton 3 server, Vec-LUT `I2` serves Falcon3 1B at **273.5 tokens/s** across 32 parallel requests, **1.4×** faster than llama.cpp `TQ2_0`.
- **Energy efficiency:** up to **2.1×** better tokens/Joule than llama.cpp and **1.8×** better than T-MAC.

{{ image(src="/img/blog/vec-lut-parallel-low-bit-llm/eval-prefill.png", max_width="80%", dimmable=true, caption="End-to-end prefilling comparison across models, devices and threads.") }}

---

## Implementation and scope

Our implementation, **vlut.cpp**, is a lightweight extension of [llama.cpp](https://github.com/ggml-org/llama.cpp). It targets mainstream x86 and ARM CPUs, follows a familiar llama.cpp-style build and usage flow, and is released as part of the open-source [OpenBitSys](https://github.com/OpenBitSys) project.

Vec-LUT is most useful when a ternary checkpoint is available (e.g., obtained via quantization-aware training), the workload includes substantial parallel inference (prefilling, batching, speculative decoding, or parallel test-time scaling), and CPU memory use and throughput are the main constraints.

Once Vec-LUT reduces lookup cost to below 1%, **vector addition** (accumulating the looked-up results) becomes the dominant cost. It takes about 65% of end-to-end prefilling latency in the measured HF BitNet 3B workload, making it a natural target for future **vector-LUT-centric accelerators**.

---

*Our team is hiring postdocs and interns (physical-AI foundation models, LLM training/inference systems, and more). Feel free to contact [Prof. Ting Cao](https://tingcao952.github.io).*

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
