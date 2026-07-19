+++
title = "增强 vLLM 对 GPTQv2 格式的支持：分析与实现"
date = "2025-10-12"
updated = "2026-07-11"
description = "分析 vLLM 对 GPTQv2 格式支持的局限，以及低比特非对称量化推理所需的 CUDA 内核改动。"
template = "blog-page.html"

[taxonomies]
tags = ["vLLM", "Development", "Quantization", "GPTQ", "LLM"]

[extra]
toc = true
katex = true
mermaid = true
ai_translation_source = "en"
+++

> Issue（已关闭）：[#26343](https://github.com/vllm-project/vllm/issues/26343) \
> Pull request（已合并）：[#26092](https://github.com/vllm-project/vllm/pull/26092) \
> Commit：[`5cc6bdd`](https://github.com/vllm-project/vllm/commit/5cc6bddb6ef5e8e5c10de8122a43fd6e8c1e3b4b)

## 引言

在这次改动之前，vLLM 对 **GPTQv2 格式**模型的支持并不完善，采用**低比特（2/3-bit）或非对称量化**的模型尤其如此。vLLM 加载这些模型时不会明确报错，却会将其按 GPTQv1 处理，导致推理质量下降，并经常重复生成 `!!!` token（参见 [issue #26343](https://github.com/vllm-project/vllm/issues/26343)）。

问题来自 GPTQv1 与 GPTQv2 checkpoint 在**零点处理**方式上的差异，而 vLLM 的 fallback GPTQ GeMM 内核没有考虑这一点。本文分析了故障原因，并记录 [PR #26092](https://github.com/vllm-project/vllm/pull/26092) 中的内核改动：在保持向后兼容的同时加入 GPTQv2 支持。

最终实现让 vLLM 的 fallback GPTQ 路径能够支持采用低比特和非对称配置的 GPTQv2 模型。

## 背景与预备知识

我是在使用 vLLM 部署以 GPTQv2 格式存储的低比特（如 2-bit）非对称量化模型时遇到这个问题的。

介绍实现细节前，先说明两点背景：

- **非对称量化会为每组权重调整零点，因此对低比特量化很有帮助。**
- **GPTQv2 能在非对称量化 checkpoint 中更完整地保留零点信息。**

### LLM 的权重量化

权重量化将高精度模型权重（如 16/32-bit）映射为较少的比特（如 2/3/4/8-bit）。这种方法常用于 LLM 部署，尤其适合资源紧张的场景。

从技术上说，权重量化会把范围较大的高精度权重（例如 FP16 的 $[-65504, 65504]$）映射到有限的量化权重范围（例如 $b$-bit 无符号整数的 $[0, 2^b - 1]$）。
这一映射通常需要一个用于压缩取值范围的缩放因子（$scale$），以及一个用于平移零点的偏置（$zero$）。记 $w_o$ 为 $[w_{min}, w_{max}]$ 范围内的原始权重（通常对应一组权重），$w_q$ 为量化后的权重，则一种简单的 $b$-bit 整数量化可写为：

\\[
    scale = \frac{w_{max} - w_{min}}{2^b - 1}
\\]

\\[
    zero = - \mathrm{round}(\frac{w_{min}}{scale})
\\]

\\[
    w_q = \mathrm{clamp}(\mathrm{round}(\frac{w_o}{scale}) + zero)
\\]

反量化时，按下式恢复原始权重 $\hat{w}_o$：

\\[
    \hat{w}_o = (w_q - zero) \cdot scale
\\]

根据是否需要零点，可以将量化方法分为两类：

- *对称量化*假设 $w_{max} = -w_{min}$，因此 $zero = \mathrm{round}(\frac{2^b - 1}{2})$ 是固定值。此时只要知道比特宽度，$zero$ 就不再提供额外信息。

- *非对称量化*不作这一假设。它的 $zero$ 会随权重组而变化，必须将其存储下来，才能准确恢复量化值。

大多数 GPTQ 实现采用 4-bit 对称量化。在更低的比特宽度下，非对称量化可以让零点适应每组权重，从而减小量化误差。

### GPTQ：量化方法与 Checkpoint 格式

GPTQ[^1] 是生成式 Transformer（主要是 LLM 和 VLM）最常用的训练后**量化方法**之一。
它利用近似二阶信息（层 Hessian 的逆）来减小量化误差。
GPTQ 也可以指 GPTQ 量化模型所使用的 **checkpoint 格式**，例如 [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) 生成的格式。Daniël de Kok 的 [GPTQ Checkpoint Format](https://danieldk.eu/GPTQ-Checkpoint-Format) 对这种表示方式有详细说明。

GPTQ 生态既包括实现该量化方法或导出相应 checkpoint 格式的量化库，也包括执行这些 checkpoint 的内核与推理框架。

量化库：

- [GPTQModel](https://github.com/ModelCloud/GPTQModel) 是 AutoGPTQ 仍在积极维护的后继项目，支持更多模型和后端。
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) 曾是常用的 GPTQ 量化库，但现已停止维护。
- [gptq](https://github.com/IST-DASLab/gptq) 是 GPTQ 论文的官方实现。
- 其他量化库也会实现 GPTQ 量化，或以 GPTQ 格式导出量化模型。

计算（CUDA）内核：

- [Marlin](https://github.com/IST-DASLab/marlin) 是面向 4-bit GPTQ 模型的高性能 W4A16 mpGeMM 内核。
- [ExLlamaV2](https://github.com/turboderp-org/exllamav2) 是一个推理库，为 GPTQ 模型提供高性能内核。
- [BitBLAS](https://github.com/microsoft/BitBLAS) 是支持 GPTQ 模型的高性能低比特 mpGeMM 内核库。
- 还有许多其他内核库也支持 GPTQ 格式模型。

[vLLM](https://github.com/vllm-project/vllm) 等 LLM 推理框架会集成这些量化库和内核库，从而高效部署 GPTQ 模型。

### 从 GPTQv1 到 GPTQv2

GPTQv2 这一名称同时用于指代量化方法和 checkpoint 格式的改动：

- 量化方法：GPTQv2（也称 GPTAQ）引入非对称校准，以减小从前面各层传播而来的量化误差[^2]。它最早在 [GPTAQ](https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ) 中实现，之后集成进 [GPTQModel](https://github.com/ModelCloud/GPTQModel)，可通过 `v2=True` 启用。
- Checkpoint 格式：GPTQv2 与 GPTQv1 存储零点的方式不同。GPTQv1 存储 $\mathrm{clamp}(zero - 1)$，并在运行时反量化前加 $1$[^3]；GPTQv2 则直接存储准确的 $zero$ 值，无需调整。在 GPTQModel 中可通过 `format="gptq_v2"` 启用这一格式。

与 GPTQv1 一样，量化方法和 checkpoint 格式之间并不绑定。因此，可以：

- 将其他方法量化的模型存为 GPTQv2 格式，例如 [BitDistiller/Qwen-8B-w2g64-gptq](https://huggingface.co/BitDistiller/Qwen-8B-w2g64-gptq)。
- 只在 GPTQModel 中设置 `format="gptq_v2"`，把采用 GPTQv1 方法量化的模型存为 GPTQv2 格式。
- 只在 GPTQModel 中设置 `v2=True`，把采用 GPTQv2 方法量化的模型存为 GPTQv1 格式。

从 GPTQv1 转换到 GPTQv2 是无损的，反向转换则不是。GPTQv1 的减一表示会把 $0 - 1$ clamp 到 $0$，因此损失一部分零点范围。以 INT2 量化为例，有效范围会从 $[0,3]$ 缩小到 $[1,3]$。
所以，**GPTQv2 更适合非对称 checkpoint，尤其是在低比特宽度下。**

## 原因分析

故障源于 vLLM 如何将不同的 GPTQ 配置路由到已有内核：

- **vLLM 有多个兼容 GPTQ 的 GeMM 内核，并为它们预先规定了优先级**——包括 Marlin、BitBLAS 和 fallback 内核。
- **Marlin 等优化内核仅在有限配置下支持 GPTQv2**——即 4/8-bit 对称量化。
- **fallback 内核支持更多比特宽度和非对称量化，却不支持 GPTQv2**——这正是本次问题的根源。

<!-- markdownlint-disable MD046 -->
{% mermaid(invertible=false, full_width=false) %}
graph TD
    A[VllmConfig] --> B[ModelConfig._verify_quantization]

    B --> |"优先级: gptq_marlin > gptq_bitblas > gptq"| E[QuantizationConfig.get_quant_method]

    E -->|4/8-bit + sym| J[GPTQMarlinLinearMethod]
    E -->|4/8-bit + sym| K[GPTQBitBLASLinearMethod]
    E -->|2/3/4/8-bit + sym/asym| L[GPTQLinearMethod]

    J --> JJ[MarlinLinearKernel]
    K --> KK[BitBLASLinearKernel]
    L --> LL[直接调用内核]

    JJ --> M[gptq_marlin_gemm<br/>CUDA 内核]
    KK --> N[bitblas.Matmul<br/>外部库]
    LL --> O[gptq_gemm<br/>CUDA 内核]

    M --> Q["✓ Marlin: gptq/gptq_v2"]
    N --> R["✓ BitBLAS: gptq/gptq_v2"]
    O --> S["✗ GPTQ: 仅 gptq"]

    style J fill:#90EE90
    style K fill:#90EE90
    style L fill:#FFB6C1
    style Q fill:#90EE90
    style R fill:#90EE90
    style S fill:#FFB6C1

{% end %}
<!-- markdownlint-enable MD046 -->

### vLLM 的内核路由层级

首先需要了解 vLLM 如何为 GPTQ 等量化方法选择计算内核。相关逻辑位于 [LLMEngine](https://docs.vllm.ai/en/stable/design/arch_overview.html#llm-engine) 的模型执行部分。为便于说明，本文只讨论稠密模型，不涉及 MoE。调用层级如下：

**1. 模型级量化配置：**

- 在模型实现（`vllm/model_executor/models`）中，初始化模型时会传入 `vllm_config: VllmConfig`，其中包含 `quant_config: QuantizationConfig`。
  - 每种量化方式都会扩展 `QuantizationConfig`，提供与该量化方式相关的 override（例如 `GPTQConfig`）。参见[全部量化方式](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/index.html)。
  - 如果 `quant_config` 中没有指定量化方式，`ModelConfig._verify_quantization` 会按优先级列表选择一种（见[下文](#vllm-gptq-support-notes)）。
- 该模型的每个线性模块都以 `quant_config` 初始化。

**2. 层级量化配置（线性方法）：**

- 初始化时，每个量化线性模块都会确定 `quant_method = quant_config.get_quant_method`。
  - `get_quant_method` 会根据量化配置返回具体的线性方法类（继承自 `LinearMethodBase`，例如 `GPTQLinearMethod`）。
- 每个线性模块通过 override `LinearMethodBase.apply` 来调用计算内核（例如 `GPTQLinearMethod.apply` 调用 `gptq_gemm`）。

**3.（可选）内核选择（线性内核）：**

vLLM 支持通过多种方式从线性方法类路由到底层 CUDA 内核：

- 直接调用。例如，`GPTQLinearMethod` 直接路由到 `gptq_gemm`，这是 vLLM 注册的自定义算子（实现在 `csrc/quantization/gptq`）。
- 间接调用。如果一个线性方法有多个可用内核，或一个内核可供多个线性方法使用，vLLM 可以扩展 `MPLinearKernel` 类，并将其作为路由到该内核的接口（位于 `vllm/model_executor/layers/quantization/kernels/mixed_precision`）。例如，`GPTQBitBLASLinearMethod` 路由到 `BitBLASLinearKernel`，`GPTQMarlinLinearMethod` 则调用 `choose_mp_linear_kernel` 进行灵活路由。
- 外部调用（与上述方式正交）。vLLM 支持调用外部库中的内核。例如，`BitBLASLinearKernel` 会调用 `bitblas` Python 库中的 `Matmul`。

### vLLM 对 GPTQ(v2) 格式的支持

vLLM 集成了多个适用于 GPTQ 格式模型的优化内核，包括 Marlin、ExLlamaV2 和 BitBLAS，同时也为不受支持的配置提供 fallback 内核。按线性方法划分，支持矩阵如下：

| 方法        | Bits    | Sym  | GPTQ 格式       |
| ----------- | ------- | ---- | --------------- |
| GPTQMarlin  | 4,8     | True | `gptq, gptq_v2` |
| GPTQBitBLAS | 4,8     | True | `gptq, gptq_v2` |
| GPTQ        | 2,3,4,8 | Any  | `gptq`          |

<a id="vllm-gptq-support-notes"></a>

说明：

- 所有方法都支持 4/8-bit 对称量化。vLLM 会根据预定义的量化 override 优先级（位于 `ModelConfig._verify_quantization`），选择当前配置下支持且性能最好的方法：

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

- 只有 `GPTQLinearMethod` 支持 2/3-bit 和非对称量化，但它此前不支持 GPTQv2；另外两种方法已经支持该格式。

因此，vLLM 的这条 fallback 路径既无法支持 GPTQv2 格式的 2/3-bit 量化，也无法支持其非对称量化，这正是 [PR #26092](https://github.com/vllm-project/vllm/pull/26092) 的动机。

## 解决方案

根据以上分析，需要调整一个 GPTQ 线性方法及其内核，才能让 GPTQv2 模型覆盖这些配置。

### 方案：调整 GPTQ 线性方法与内核

要加入这项支持，至少需要调整一种现有的线性方法：

- `GPTQMarlinLinearMethod` 不支持 2/3-bit 和非对称量化。扩展它还需要修改 vLLM 的 Marlin CUDA 内核，而该内核面向 4/8-bit 对称量化。
- `GPTQBitBLASLinearMethod` 在线性方法层面同样不支持 2/3-bit 和非对称量化。虽然 `bitblas` 库支持所需的 `bits` 与 `sym` 取值，但这条路径依赖可选的 `bitblas` 包。
- `GPTQLinearMethod` 已经支持所需的比特宽度和对称/非对称模式，只需在 `gptq_gemm` CUDA 内核中根据格式处理零点，因此是首选方案。

因此，实现中选择调整 `GPTQLinearMethod`（及 `GPTQConfig`）和 `gptq_gemm`，为它们加入 GPTQv2 支持。

### 调整细节：根据格式选择处理方式

线性方法与内核的调整需要满足三项要求：

1. 保持与 GPTQv1 格式的兼容性。
2. 尽量减小对代码和二进制大小的影响。
3. 保持现有路由，让 GPTQv2 模型仍可选择 Marlin 等其他内核。

为满足前两项要求：

- 为 `GPTQLinearMethod` 添加 `use_v2_format: bool` 属性，用于表示 `checkpoint_format == "gptq_v2"`。
- 为 `gptq_gemm` 添加 `bool use_v2_format` 参数，以 `GPTQLinearMethod.use_v2_format` 作为输入。
- 在 `gptq_gemm` 中，根据 `use_v2_format` 调整零点处理逻辑。例如：

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

> 测试期间，我发现原有的 `gptq_gemm` 即使处理对称量化的 GPTQv1 模型，在 4-bit 下也存在 bug。这超出了该 PR 的范围。

对于第三项要求，`GPTQMarlinLinearMethod` 和 `GPTQBitBLASLinearMethod` 都不作改动：它们已经支持 GPTQv2，只是仅限于 4/8-bit 对称量化。

> TODO：补充一些性能基准测试。
> 我发现 2-bit `gptq_gemm` 在解码（GeMV）时比预填充（GeMM）时更慢。

## 总结

这项改动为 vLLM 的 fallback GPTQ 路径加入了 GPTQv2 支持，在保持 GPTQv1 兼容性的同时，覆盖了低比特和非对称配置。

欢迎提问与讨论。

**后续可做的工作：**

- 扩展优化内核（Marlin、BitBLAS），使其支持 2/3-bit 或非对称量化。
- 修复 `gptq_gemm` 的 4-bit bug。
- 提升 `gptq_gemm` 的解码速度。

[^1]: Elias Frantar 等，[“GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers”](https://arxiv.org/abs/2210.17323)，ICLR，2023。

[^2]: Yuhang Li 等，[“GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration”](https://arxiv.org/abs/2504.02692)，arXiv，2025。

[^3]: Daniël de Kok，[“GPTQ Checkpoint Format”](https://danieldk.eu/GPTQ-Checkpoint-Format)，Daniël's Website，2024。
