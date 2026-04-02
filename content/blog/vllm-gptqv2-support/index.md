+++
title = "Enhancing GPTQv2 Format Support in vLLM: Analysis and Implementation"
date = "2025-10-12"
updated = "2025-10-28"
description = "Deep technical analysis of GPTQv2 format limitations in vLLM, and implementation of CUDA kernel adaptations to enable efficient low-bit/asymmetric quantization inference."
template = "blog-page.html"

[taxonomies]
tags = ["vLLM", "Development", "Quantization", "GPTQ", "LLM"]

[extra]
toc = true
katex = true
mermaid = true
+++

> Issue (closed): [#26343](https://github.com/vllm-project/vllm/issues/26343) \
> Pull request (merged): [#26092](https://github.com/vllm-project/vllm/pull/26092) \
> Commit: [`5cc6bdd`](https://github.com/vllm-project/vllm/commit/5cc6bddb6ef5e8e5c10de8122a43fd6e8c1e3b4b)

## Introduction

vLLM, one of the leading LLM inference frameworks, currently lacks robust support for **GPTQv2 format** (an upgraded version of GPTQv1 format) models, particularly those using **low-bit (2/3-bit) or asymmetric quantization**. While vLLM doesn't raise explicit errors when loading such models, it incorrectly treats them as GPTQv1 format, resulting in degraded inference quality and characteristic gibberish outputs (consisting of repeated `!!!`, details in [this issue](https://github.com/vllm-project/vllm/issues/26343)).

This limitation stems from differences in **zero point handling** between GPTQv1 and GPTQv2 checkpoint formats, which vLLM's existing GPTQ GeMM kernels don't account for. This post presents a comprehensive analysis of this limitation, and documents the implementation of kernel adaptations (i.e., in [this PR](https://github.com/vllm-project/vllm/pull/26092)), that enable proper GPTQv2 support while maintaining backward compatibility.

Through careful investigation of vLLM's quantization support and targeted CUDA kernel modifications, I enable robust inference for GPTQv2 format models, especially low-bit or asymmetric ones, with vLLM — contributing a step forward towards efficient LLM deployment.

## Background and Preliminaries

In my case, I use vLLM to serve some low-bit (e.g., 2-bit), asymmetrically quantized models, stored in GPTQv2 format, and encountered [this issue](https://github.com/vllm-project/vllm/issues/26343).

Before diving into technical details, I'll briefly introduce the background to do so (e.g., why GPTQv2), and some preliminaries (e.g., the checkpoint format) to help follow the technical parts. **Key takeaways:**

- **Asymmetric quantization benefits low-bit quantization**, by adjusting the zero point for each group of weights.
- **GPTQv2 format outperforms GPTQv1 in asymmetric quantization**, by better preserving zero point information.

### Weight Quantization of LLMs

Weight quantization that quantizes high-precision model weights (e.g., 16/32-bit) into fewer bits (e.g., 2/3/4/8-bit) has been a common practice in LLM deployment, especially in resource-constrained scenarios.

Technically, weight quantization maps the large range of high-precision weights (e.g., within $[-65504, 65504]$ for FP16) into a limited range of quantized weights (e.g., $[0, 2^b - 1]$ for $b$-bit unsigned integer).
This mapping typically involves a scaling factor ($scale$) that compresses the range, and a bias ($zero$) that shifts the zero point. We denote $w_o$ as the original weight within the range $[w_{min}, w_{max}]$ (usually for a group of weights), and $w_q$ as the quantized weight. Then, a simple $b$-bit integer quantization is formulated as:

\\[
    scale = \frac{w_{max} - w_{min}}{2^b - 1}
\\]

\\[
    zero = - \mathrm{round}(\frac{w_{min}}{scale})
\\]

\\[
    w_q = \mathrm{clamp}(\mathrm{round}(\frac{w_o}{scale}) + zero)
\\]

To recover the original weight $\hat{w}_o$ during dequantization:

\\[
    \hat{w}_o = (w_q - zero) \cdot scale
\\]

Based on this formulation, quantization methods are categorized by whether the zero point is required:

- *Symmetric quantization* assumes $w_{max} = - w_{min}$, so $zero = - \mathrm{round}(\frac{2^b - 1}{2})$ will not change. In this case, $zero$ doesn't provide additional information given the quantization bits.

- *Asymmetric quantization* doesn't have such assumption. So $zero$ varies across groups of weights, and is necessary for accurately recovering the original weights.

Note that most GPTQ implementations are 4-bit symmetric quantization. However, to reduce the quantization error in lower bits, asymmetric quantization is necessary.

### GPTQ: Quantization and Checkpoint Format

GPTQ[^1] is one of the most popular post-training **quantization methods** for generative transformers (mainly LLMs and VLMs).
It utilizes approximate second-order information (inverse layer Hessian) to reduce quantization errors.
Besides, GPTQ could also refer to the specific **checkpoint format** adopted by GPTQ-quantized models (e.g., by [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)), with details explained in [this blog](https://danieldk.eu/GPTQ-Checkpoint-Format).

GPTQ is widely supported by the community, including 1) quantization libraries that implement the GPTQ quantization method or support exporting to GPTQ format (although not implementing the quantization method), and 2) kernel libraries and inference frameworks that support inference with models of the GPTQ checkpoint format, as listed below:

Quantization libraries:

- [GPTQModel](https://github.com/ModelCloud/GPTQModel) is now the 1st choice for GPTQ quantization in replacement of AutoGPTQ, with richer model and backend support.
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) was the most popular library for GPTQ quantization, and is unmaintained now.
- [gptq](https://github.com/IST-DASLab/gptq) is the official code implementation of the GPTQ paper.
- Many other libraries implement GPTQ quantization, or support to save quantized models in GPTQ format.

Computing (CUDA) kernels:

- [Marlin](https://github.com/IST-DASLab/marlin) is the SOTA W4A16 mpGeMM kernel supporting 4-bit GPTQ models.
- [ExLlamaV2](https://github.com/turboderp-org/exllamav2) is an inference library with high-performance kernels for GPTQ models.
- [BitBLAS](https://github.com/microsoft/BitBLAS) is a high-performance low-bit mpGeMM kernel library supporting GPTQ models.
- Similarly, many other kernel libraries support GPTQ format models.

SOTA LLM inference frameworks, including [vLLM](https://github.com/vllm-project/vllm), are integrated with the above quantization and kernel libraries, to support efficient LLM deployment with GPTQ quantization.

### From GPTQv1 to GPTQv2

GPTQv2 is an upgraded version of GPTQ (by a different team though), in both the quantization method and checkpoint format. Specifically:

- Quantization method: GPTQv2 (also called GPTAQ) introduces asymmetric calibration, which effectively reduces the quantization error accumulated in previous layers[^2]. It's first implemented in [GPTAQ](https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ), and then integrated to [GPTQModel](https://github.com/ModelCloud/GPTQModel) (enabled by setting `v2=True`).
- Checkpoint format: GPTQv2 format stores the zero points differently with GPTQv1 (i.e., GPTQ) format — GPTQv1 stores $\mathrm{clamp}(zero - 1)$ in the checkpoint, and requires adding $1$ back before dequantization at runtime[^3]. GPTQv2 stores the exact $zero$ value, and doesn't require extra runtime adjustment. It is also supported by [GPTQModel](https://github.com/ModelCloud/GPTQModel) (enabled by setting `format="gptq_v2"`).

Just like GPTQv1, the quantization method and checkpoint format are not coupled. So you can:

- Store a non-GPTQ-quantized model in GPTQv2 format, like [BitDistiller/Qwen-8B-w2g64-gptq](https://huggingface.co/BitDistiller/Qwen-8B-w2g64-gptq).
- Store a GPTQv1-quantized model in GPTQv2 format, by setting `format="gptq_v2"` only, in GPTQModel.
- Store a GPTQv2-quantized model in GPTQv1 format, by setting `v2=True` only, in GPTQModel.

Note that the conversion between GPTQv2 and GPTQv1 format is **irreversible** — you can convert GPTQv1 to GPTQv2 losslessly, but not from GPTQv2 to GPTQv1. This is due to the "-1" issue of GPTQv1 as mentioned above. In this way, the actual zero point range in suppressed by clamping $0 - 1$ to $0$ in GPTQv1. For example, in INT2 quantization, the effective range shrinks from $[0,3]$ to $[1,3]$.
Therefore, **GPTQv2 format is a preferable choice in asymmetric quantization, especially for low-bit quantization.**

## Cause Analysis

After all the preparation, we can finally dive into the technical details about why and how vLLM fails for GPTQv2 in my case — even though it has some sort of support actually, which I found after careful investigation. **Key takeaways:**

- **vLLM has several GPTQ-compatible GeMM kernels with pre-defined priorities** — Marlin, BitBLAS, and fallbacks.
- **Performant kernels like Marlin support GPTQv2 format of limited bits and symmetry** — only 4/8-bit symmetric quantization.
- **Fallback kernels support more bits and symmetry but lacks GPTQv2 format support** — the reason why I encountered the issue.

<!-- markdownlint-disable MD046 -->
{% mermaid(invertible=false, full_width=false) %}
graph TD
    A[VllmConfig] --> B[ModelConfig._verify_quantization]

    B --> |"Priority: gptq_marlin > gptq_bitblas > gptq"| E[QuantizationConfig.get_quant_method]
    
    E -->|4/8-bit + sym| J[GPTQMarlinLinearMethod]
    E -->|4/8-bit + sym| K[GPTQBitBLASLinearMethod]
    E -->|2/3/4/8-bit + sym/asym| L[GPTQLinearMethod]
    
    J --> JJ[MarlinLinearKernel]
    K --> KK[BitBLASLinearKernel]
    L --> LL[Direct Kernel Call]
    
    JJ --> M[gptq_marlin_gemm<br/>CUDA kernel]
    KK --> N[bitblas.Matmul<br/>External library]
    LL --> O[gptq_gemm<br/>CUDA kernel]
    
    M --> Q["✅ Marlin: gptq/gptq_v2"]
    N --> R["✅ BitBLAS: gptq/gptq_v2"] 
    O --> S["❌ GPTQ: gptq only"]
    
    style J fill:#90EE90
    style K fill:#90EE90
    style L fill:#FFB6C1
    style Q fill:#90EE90
    style R fill:#90EE90
    style S fill:#FFB6C1

{% end %}
<!-- markdownlint-enable MD046 -->

### vLLM's Kernel Routing Hierarchy

The first step is to understand how vLLM routes computing kernels for different quantizations, like GPTQ. This is implemented in the model execution part of the [LLMEngine](https://docs.vllm.ai/en/stable/design/arch_overview.html#llm-engine). For simplicity, we only consider dense models (no MoE). It includes the following calling hierarchy:

**1. Model-level quantization configuration:**

- In model implementations (`vllm/model_executor/models`), `vllm_config: VllmConfig` is passed to the model at initialization, which contains `quant_config: QuantizationConfig`.
  - Each quantization extends `QuantizationConfig` with quantization-specific overrides (e.g., `GPTQConfig`). See [all quantizations](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/index.html).
  - If no quantization is specified in `quant_config`, `ModelConfig._verify_quantization` will select one from a priority list (see [below](#vllm-gptq-support-notes)).
- Each linear module of this model is initialized with `quant_config`.

**2. Layer-level quantization configuration (linear methods):**

- At initialization, each quantized linear module determines `quant_method = quant_config.get_quant_method`.
  - `get_quant_method` returns a specific linear method class (inherited from `LinearMethodBase`, e.g., `GPTQLinearMethod`) depending on the quantization configuration.
- Each linear module calls the computing kernel by overriding `LinearMethodBase.apply` (e.g., `GPTQLinearMethod.apply` calls `gptq_gemm`).

**3. (Optional) Kernel selection (linear kernels):**

vLLM supports several ways of routing to low-level CUDA kernels from a linear method class:

- Direct calling. For example, `GPTQLinearMethod` directly routes to `gptq_gemm`, a registered custom operand of vLLM (implemented in `csrc/quantization/gptq`).
- Indirect calling. When multiple kernels are available for a linear method, or a kernel is available for multiple linear methods, vLLM supports extending the `MPLinearKernel` class as an interface for routing to this kernel (in `vllm/model_executor/layers/quantization/kernels/mixed_precision`). For example, `GPTQBitBLASLinearMethod` routes to `BitBLASLinearKernel`, and `GPTQMarlinLinearMethod` calls `choose_mp_linear_kernel` for flexible routing.
- External calling (orthogonal to the above). vLLM supports calling kernels from external libraries. For example, `BitBLASLinearKernel` calls `Matmul` from the `bitblas` Python library.

### vLLM's Support for GPTQ(v2) Format

vLLM integrates several optimized kernels for GPTQ format models, as listed in [GPTQ: Quantization and Checkpoint Format](#gptq-quantization-and-checkpoint-format), including Marlin, ExLlamaV2, BitBLAS, etc. vLLM also has fallback kernels for unsupported quantization configurations of these kernels. Following the analysis in [vLLM's Kernel Routing Hierarchy](#vllm-s-kernel-routing-hierarchy), I summarize this support matrix (by linear methods):

| Method      | Bits    | Sym  | GPTQ Format     |
| ----------- | ------- | ---- | --------------- |
| GPTQMarlin  | 4,8     | True | `gptq, gptq_v2` |
| GPTQBitBLAS | 4,8     | True | `gptq, gptq_v2` |
| GPTQ        | 2,3,4,8 | Any  | `gptq`          |

<a id="vllm-gptq-support-notes"></a>

Notes:

- All methods support 4/8-bit symmetric quantization. vLLM selects the most performant method supported by the current configuration, according to predefined priorities of quantization overrides (in `ModelConfig._verify_quantization`):

```py
overrides = [
                ...
                # gptq_marlin_24 requires special format,
                # so we don't consider here.
                "gptq_marlin_24",
                # Priority: gptq_marlin > gptq_bitblas > gptq.
                "gptq_marlin",
                "gptq_bitblas",
                ...
            ]
```

- Only `GPTQLinearMethod` supports 2/3-bit quantization and asymmetric quantization. However, it lacks GPTQv2 support (both the other two supports).

As a result, neither 2/3-bit nor asymmetric quantization in GPTQv2 format are unsupported by vLLM, which motivates [this PR](https://github.com/vllm-project/vllm/pull/26092).

## Solution

Based on the above analysis, vLLM's GPTQ linear methods lack support for 2/3-bit quantization and asymmetric quantization in GPTQv2 format, and require adaption to robustly support GPTQv2 format models of various configurations.

### Approach: Adapt GPTQ Linear Method & Kernel

To add such support, at least one linear method should be added/adapted. To adapt existing linear methods:

- `GPTQMarlinLinearMethod` (lacking 2/3-bit and asymmetric support): It requires also modifying vLLM's Marlin CUDA kernel, which is dedicated for 4/8-bit symmetric quantization — not a good choice.
- `GPTQBitBLASLinearMethod` (lacking 2/3-bit and asymmetric support): It requires modifying only the linear method/kernel (Python code), since the `bitblas` library itself supports the `bits` and `sym` we want — a reasonable choice, but requires the optional `bitblas` package to be installed.
- `GPTQLinearMethod` (lacking GPTQv2 format support): It requires also modifying vLLM's `gptq_gemm` CUDA kernel, by only adapting the zero point handling logic — preferred.

So, the plan is to adapt `GPTQLinearMethod` (with `GPTQConfig`) and `gptq_gemm` to add proper GPTQv2 format support.

### Details of Adaption: Conditioned on Format

During this linear method & kernel adaption, there are 3 points to keep in mind:

1. Maintain compatibility for GPTQv1 format.
2. Keep the code and binary size impact down.
3. Make sure other kernels (e.g., Marlin) are not running with GPTQv2 incorrectly.

In response to Pt. 1 and 2:

- Add a `use_v2_format: bool` attribute to `GPTQLinearMethod` that indicates whether `checkpoint_format == "gptq_v2"`.
- Add a `bool use_v2_format` argument to `gptq_gemm`, which accepts `GPTQLinearMethod.use_v2_format` as input.
- In `gptq_gemm`, update the zero point handling logic to be conditioned on `use_v2_format`. For example:

```c
// In `reconstruct_exllama_2bit_kernel`:

// Previous: zeros[i] + 1 (hardcoded for GPTQv1)
dequant_2bit_16(load_int4.x, dq[0], size_n, zeros[0] + 1);
dequant_2bit_16(load_int4.y, dq[1], size_n, zeros[1] + 1);
dequant_2bit_16(load_int4.z, dq[2], size_n, zeros[2] + 1);
dequant_2bit_16(load_int4.w, dq[3], size_n, zeros[3] + 1);

// Now: zeros[i] + offset (conditioned on `use_v2_format`)
int zero_offset = use_v2_format ? 0 : 1;
...
dequant_2bit_16(load_int4.x, dq[0], size_n, zeros[0] + zero_offset);
dequant_2bit_16(load_int4.y, dq[1], size_n, zeros[1] + zero_offset);
dequant_2bit_16(load_int4.z, dq[2], size_n, zeros[2] + zero_offset);
dequant_2bit_16(load_int4.w, dq[3], size_n, zeros[3] + zero_offset);

```

> When testing, I found that the original `gptq_gemm` is buggy at 4-bit even with symmetrically quantized model of GPTQv1 format — out of scope of this PR.

To ensure Pt. 3, review [vLLM's Support for GPTQ(v2) Format](#vllm-s-support-for-gptq-v2-format) — both `GPTQMarlinLinearMethod` and `GPTQBitBLASLinearMethod` are not affected, as they already support GPTQv2 format, though limited to 4/8-bit symmetric quantization.

> TODO: Add some performance benchmarks.
> Currently I've found that 2-bit gptq_gemm is slower during decoding (GeMV) than prefilling (GeMM).

## Conclusion

This post details the development of GPTQv2 format support in vLLM, which addresses a significant gap in low-bit asymmetric quantization inference with SOTA LLM inference frameworks.

Questions and discussions are welcomed.

**Possible future works:**

- Extend optimized kernels (Marlin, BitBLAS) to support 2/3-bit or asymmetric quantization.
- Fix the 4-bit bug in `gptq_gemm`.
- Improve the decoding speed with `gptq_gemm`.

[^1]: Frantar, Elias, et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers." arXiv preprint arXiv:2210.17323 (2022).

[^2]: Li, Yuhang, et al. "GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration." arXiv preprint arXiv:2504.02692 (2025).

[^3]: de Kok, Daniël. "GPTQ Checkpoint Format." Daniël's Website, 7 Aug. 2024, danieldk.eu/GPTQ-Checkpoint-Format.
