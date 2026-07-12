+++
title = "Enhancing GPTQv2 Format Support in vLLM: Analysis and Implementation"
date = "2025-10-12"
updated = "2026-07-11"
description = "An analysis of GPTQv2 format limitations in vLLM and the CUDA kernel changes needed for low-bit asymmetric quantization inference."
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

Before this change, vLLM lacked robust support for models in **GPTQv2 format**, particularly those using **low-bit (2/3-bit) or asymmetric quantization**. vLLM loaded these models without an explicit error but treated them as GPTQv1, degrading inference quality and often producing repeated `!!!` tokens (see [issue #26343](https://github.com/vllm-project/vllm/issues/26343)).

The problem came from differences in **zero-point handling** between GPTQv1 and GPTQv2 checkpoints that vLLM's fallback GPTQ GeMM kernel did not account for. This post analyzes the failure and documents the kernel changes in [PR #26092](https://github.com/vllm-project/vllm/pull/26092), which added GPTQv2 support while preserving backward compatibility.

The resulting implementation supports GPTQv2 models across low-bit and asymmetric configurations in vLLM's fallback GPTQ path.

## Background and Preliminaries

I encountered the issue while using vLLM to serve low-bit (e.g., 2-bit), asymmetrically quantized models stored in GPTQv2 format.

Two points of background are useful before the implementation details:

- **Asymmetric quantization benefits low-bit quantization** by adjusting the zero point for each group of weights.
- **GPTQv2 better preserves zero-point information** in asymmetrically quantized checkpoints.

### Weight Quantization of LLMs

Weight quantization maps high-precision model weights (e.g., 16/32-bit) to fewer bits (e.g., 2/3/4/8-bit). It is common in LLM deployment, especially under tight resource constraints.

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

- *Symmetric quantization* assumes $w_{max} = -w_{min}$, so $zero = \mathrm{round}(\frac{2^b - 1}{2})$ is fixed. In this case, $zero$ provides no additional information once the bit width is known.

- *Asymmetric quantization* does not make this assumption. Its $zero$ varies across weight groups and must be stored to recover the quantized values accurately.

Most GPTQ implementations use 4-bit symmetric quantization. At lower bit widths, asymmetric quantization can reduce quantization error by adapting the zero point to each group.

### GPTQ: Quantization and Checkpoint Format

GPTQ[^1] is one of the most popular post-training **quantization methods** for generative transformers (mainly LLMs and VLMs).
It utilizes approximate second-order information (inverse layer Hessian) to reduce quantization errors.
GPTQ can also refer to the **checkpoint format** used by GPTQ-quantized models, such as those produced by [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ). Daniël de Kok's [GPTQ Checkpoint Format](https://danieldk.eu/GPTQ-Checkpoint-Format) explains this representation in detail.

The GPTQ ecosystem includes quantization libraries that implement the method or export its checkpoint format, as well as kernels and inference frameworks that execute those checkpoints.

Quantization libraries:

- [GPTQModel](https://github.com/ModelCloud/GPTQModel) is the actively maintained successor to AutoGPTQ, with broader model and backend support.
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) was a popular GPTQ quantization library but is no longer maintained.
- [gptq](https://github.com/IST-DASLab/gptq) is the official implementation of the GPTQ paper.
- Other libraries either implement GPTQ quantization or export quantized models in GPTQ format.

Computing (CUDA) kernels:

- [Marlin](https://github.com/IST-DASLab/marlin) is a high-performance W4A16 mpGeMM kernel for 4-bit GPTQ models.
- [ExLlamaV2](https://github.com/turboderp-org/exllamav2) is an inference library with high-performance kernels for GPTQ models.
- [BitBLAS](https://github.com/microsoft/BitBLAS) is a high-performance low-bit mpGeMM kernel library supporting GPTQ models.
- Many other kernel libraries also support GPTQ-format models.

LLM inference frameworks such as [vLLM](https://github.com/vllm-project/vllm) integrate these quantization and kernel libraries to serve GPTQ models efficiently.

### From GPTQv1 to GPTQv2

The name GPTQv2 is used for changes to both the quantization method and the checkpoint format:

- Quantization method: GPTQv2 (also called GPTAQ) introduces asymmetric calibration to reduce quantization error propagated from earlier layers[^2]. It was first implemented in [GPTAQ](https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ), then integrated into [GPTQModel](https://github.com/ModelCloud/GPTQModel) behind `v2=True`.
- Checkpoint format: GPTQv2 stores zero points differently from GPTQv1. GPTQv1 stores $\mathrm{clamp}(zero - 1)$ and adds $1$ before runtime dequantization[^3]. GPTQv2 stores the exact $zero$ value and needs no adjustment. GPTQModel enables this format with `format="gptq_v2"`.

Just like GPTQv1, the quantization method and checkpoint format are not coupled. So you can:

- Store a model quantized by another method in GPTQv2 format, as in [BitDistiller/Qwen-8B-w2g64-gptq](https://huggingface.co/BitDistiller/Qwen-8B-w2g64-gptq).
- Store a GPTQv1-quantized model in GPTQv2 format by setting only `format="gptq_v2"` in GPTQModel.
- Store a GPTQv2-quantized model in GPTQv1 format by setting only `v2=True` in GPTQModel.

The conversion is lossless from GPTQv1 to GPTQv2, but not in the reverse direction. GPTQv1's subtract-one representation clamps $0 - 1$ to $0$, suppressing part of the zero-point range. For INT2 quantization, for example, the effective range shrinks from $[0,3]$ to $[1,3]$.
Therefore, **GPTQv2 is preferable for asymmetric checkpoints, especially at low bit widths.**

## Cause Analysis

The failure comes from how vLLM routes different GPTQ configurations to its available kernels:

- **vLLM has several GPTQ-compatible GeMM kernels with predefined priorities** — Marlin, BitBLAS, and fallbacks.
- **Optimized kernels such as Marlin support GPTQv2 only for limited configurations** — 4/8-bit symmetric quantization.
- **The fallback kernel supports more bit widths and asymmetric quantization, but lacked GPTQv2 support** — the source of this issue.

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

The first step is to understand how vLLM routes compute kernels for quantization methods such as GPTQ. This logic lives in the model-execution portion of the [LLMEngine](https://docs.vllm.ai/en/stable/design/arch_overview.html#llm-engine). For simplicity, this discussion covers only dense models (no MoE). The call hierarchy is:

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

vLLM integrates several optimized kernels for GPTQ-format models, including Marlin, ExLlamaV2, and BitBLAS, plus fallback kernels for unsupported configurations. The resulting support matrix by linear method is:

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

- Only `GPTQLinearMethod` supports 2/3-bit and asymmetric quantization. However, it lacked GPTQv2 support; the other two methods already supported the format.

As a result, vLLM supported neither 2/3-bit nor asymmetric quantization in GPTQv2 format through this fallback path, which motivated [PR #26092](https://github.com/vllm-project/vllm/pull/26092).

## Solution

Based on this analysis, one GPTQ linear method and its kernel needed adaptation to support GPTQv2 models across these configurations.

### Approach: Adapt the GPTQ Linear Method and Kernel

Adding this support requires adapting at least one existing linear method:

- `GPTQMarlinLinearMethod` lacks 2/3-bit and asymmetric support. Extending it would also require changes to vLLM's Marlin CUDA kernel, which targets 4/8-bit symmetric quantization.
- `GPTQBitBLASLinearMethod` also lacks 2/3-bit and asymmetric support at the linear-method layer. Although the `bitblas` library supports the required `bits` and `sym` values, this route depends on the optional `bitblas` package.
- `GPTQLinearMethod` already supports the required bit widths and symmetry modes. It only needs format-aware zero-point handling in the `gptq_gemm` CUDA kernel, making it the preferred route.

The implementation therefore adapts `GPTQLinearMethod` (with `GPTQConfig`) and `gptq_gemm` for GPTQv2.

### Adaptation Details: Conditioned on Format

The linear-method and kernel adaptation has three requirements:

1. Maintain compatibility for GPTQv1 format.
2. Minimize the impact on code and binary size.
3. Preserve the existing routing of GPTQv2 models to other kernels such as Marlin.

For the first two requirements:

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

> During testing, I found that the original `gptq_gemm` is buggy at 4-bit even with a symmetrically quantized GPTQv1 model. This is outside the scope of the PR.

For the third requirement, both `GPTQMarlinLinearMethod` and `GPTQBitBLASLinearMethod` remain unchanged: they already support GPTQv2, although only for 4/8-bit symmetric quantization.

> TODO: Add some performance benchmarks.
> I found that 2-bit `gptq_gemm` is slower during decoding (GeMV) than during prefilling (GeMM).

## Conclusion

This change adds GPTQv2 support to vLLM's fallback GPTQ path, covering low-bit and asymmetric configurations while preserving GPTQv1 compatibility.

Questions and discussion are welcome.

**Possible future work:**

- Extend optimized kernels (Marlin, BitBLAS) to support 2/3-bit or asymmetric quantization.
- Fix the 4-bit bug in `gptq_gemm`.
- Improve the decoding speed with `gptq_gemm`.

[^1]: Elias Frantar et al., ["GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers"](https://arxiv.org/abs/2210.17323), ICLR, 2023.

[^2]: Yuhang Li et al., ["GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration"](https://arxiv.org/abs/2504.02692), arXiv, 2025.

[^3]: Daniël de Kok, ["GPTQ Checkpoint Format"](https://danieldk.eu/GPTQ-Checkpoint-Format), Daniël's Website, 2024.
