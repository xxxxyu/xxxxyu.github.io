+++
title = "A Unified KV Cache for Multi-Task Inference on Robots"
date = "2026-09-01"
updated = "2026-09-01"
description = "OxyGen treats the KV cache as one shared resource across tasks and control frames, letting Mixture-of-Transformers VLAs keep full action frequency while generating language concurrently, up to 3.7× faster on an RTX 4090 and Jetson AGX Thor."
template = "blog-page.html"

[taxonomies]
tags = ["Embodied AI", "VLA", "LLM Inference"]

[extra]
toc = true
og_image = "/img/blog/oxygen-mot-vla-inference/teaser.png"
og_image_alt = "OxyGen overview: concurrent action and language generation with isolated versus unified KV cache management"
+++

> 🔗 [Paper](https://arxiv.org/abs/2603.14371) | [Code](https://github.com/air-embodied-brain/OxyGen) | [Project Page](https://air-embodied-brain.github.io/OxyGen)

A useful robot has to do more than act. While its arms are moving, it should be narrating progress to the user, writing down what changed in the scene, maybe replanning the next subtask. Recent Mixture-of-Transformers (MoT) Vision-Language-Action Models (VLAs) such as π<sub>0.5</sub>[^1] are a step in this direction: one shared backbone, separate experts for actions and language, so the model can in principle act and talk at the same time.

In practice, on the single GPU a robot actually carries, the two outputs wait in line for each other. We traced the problem to something unglamorous: every task manages its own KV cache. **OxyGen** is our fix: treat the KV cache as one shared resource across tasks and control frames. On an RTX 4090 it sustains **over 200 tokens/s** of language generation while holding a **70 Hz** action frequency[^2] (27 Hz on a Jetson AGX Thor), with **up to 3.7×** end-to-end speedup over the isolated execution that existing systems use.

<a id="demo-video"></a>

{{ video(src="/videos/oxygen-mot-vla-inference/demo.mp4", autoplay=true, loop=true, max_width="100%", caption="OxyGen vs. the isolated openpi baseline on LIBERO: the robot keeps acting while textual memory streams in. The timeline uses measured model inference plus 1× simulator time.") }}

---

## VLAs that act and talk

Continuous VLAs such as π<sub>0</sub>[^3] attach a lightweight flow-matching head to a VLM backbone: the backbone consumes the observation and instruction once per control frame, and the head denoises an action chunk from that context. MoT VLAs[^4] take the idea one step further. The backbone is shared, but each output modality gets its own expert weights, so a single checkpoint can emit both action chunks and free-form language.

On paper, inference factorizes neatly. At every control frame, the shared backbone prefills the observation into a single KV cache, and that cache is modality-agnostic: it encodes what the robot just saw and was told, without committing to any particular output. Any expert can read from it:

- the **action expert** runs `S` flow-matching denoise steps over that cache and emits a chunk of `H` actions (`S = 10, H = 10` throughout, as in openpi's π<sub>0.5</sub>-LIBERO configuration);
- the **language expert** decodes up to `N` tokens autoregressively from the same cache: subtasks, narration, memory.

The two tasks live under very different deadlines. Actions are due within the frame: dexterous manipulation wants on the order of 50 Hz, and a late chunk means jerky motion or a stalled robot. Language is soft: a memory sentence arriving a second late is still useful. Any efficient design has to exploit that asymmetry rather than fight it.

---

## Why running both tasks crawls

The architecture is ready for all of this: one prefill, two experts reading one cache. The serving stack is not. The standard approach in openpi[^5] and every other MoT serving setup we know is what we call **isolated execution**[^6]: each task triggers its own forward pass of the same model, so the language path re-encodes the same observation and builds its own KV cache from scratch, even though the model would happily share one. This costs you in two ways.

**Redundant computation.** The shared observation gets encoded once per task, producing byte-identical KV cache entries. That duplicate is not cheap: on an RTX 4090, one prefix forward through the VLM backbone takes ~45 ms, about **2.2×** the ~20 ms the entire denoise loop costs[^7]. For short language outputs, the duplicate prefill alone accounts for a **1.4×** slowdown.

**Resource contention.** Sharing the cache naively doesn't fix the schedule. Run the two tasks sequentially in one process and language decoding blocks the control loop: as language outputs grow long, baseline action frequency falls from **49.9 Hz to 19.1 Hz**, a **2.6×** loss[^8]. Run them as separate processes sharing the GPU via MPS and you nearly double peak memory while saving very little.

"Isn't this just prefix caching and continuous batching?" is the natural question, and LLM servers[^9]<sup class="footnote-separator">, </sup>[^10] do both well. But their world is homogeneous: one kind of cache reader, one kind of request, no deadlines. A MoT VLA is heterogeneous on both axes:

- **Heterogeneous cache access.** The action expert consumes the prefix as *read-only* context on every denoise step; the language expert *appends* a new KV entry per decoded token. Prefix caching assumes a single autoregressive reader path, and a standard server has no semantics for coordinating two experts with different access rules on the same cache.
- **Heterogeneous timing.** Language decoding must advance by a bounded budget per control frame, pause, and resume next frame without recomputation, because the action deadline comes back every frame. Continuous batching runs each request to completion; nothing in a standard server knows what a robot control frame is.

---

## One cache, shared across tasks and frames

OxyGen's answer is to make the KV cache a first-class shared resource, managed by one cache manager across tasks and time, with one mechanism for each of the gaps above.

{{ image(src="/img/blog/oxygen-mot-vla-inference/teaser.png", dimmable=true, caption="OxyGen overview: concurrent action and language generation from one shared prefix, with unified KV cache management replacing isolated per-task caches.") }}

**Cross-task KV sharing.** Each frame, the backbone prefills the new observation once. The manager fans the resulting cache out to the action expert as a read-only view, and to a freshly initialized language request that starts its autoregressive suffix from the same prefix. The duplicate prefill simply disappears, which is why the gain is largest exactly when language outputs are short.

**Cross-frame continuous batching.** Every language request is a resumable state: its own KV cache, its token buffer, a done flag. At each frame, the manager collects all unfinished requests, batches them into a single decoding forward, advances each by up to `k` tokens, evicts the finished ones, and persists the rest for the next frame. In steady state, with `N`-token requests advancing `k` tokens per frame, about `N/k` requests are in flight and one forward pass serves them all, exactly the hardware parallelism that isolated execution leaves on the table.

The budget `k` caps how far each request advances per frame; anything unfinished resumes in the next, so language never holds the action deadline hostage, and even the longest requests we test finish within about a second. The two mechanisms are standard pieces. What makes them work together is the cache semantics in between: read-only denoising reads alongside append-only decoding, coordinated under a deadline that recurs every few dozen milliseconds.

---

## Results: from an RTX 4090 to a Jetson Thor

We evaluated π<sub>0.5</sub> across LIBERO[^11], DROID[^12], and ALOHA[^13] configurations on an RTX 4090 and a Jetson AGX Thor, one new observation and one new language request per frame, sweeping `N` and `k`.

{{ image(src="/img/blog/oxygen-mot-vla-inference/tradeoff-pi05-libero.png", dimmable=true, caption="Action frequency vs. language throughput on the LIBERO configuration. Baselines trade one axis for the other as N grows; OxyGen pushes the frontier out by up to 3.7×.") }}

The tradeoff plot is the headline: every isolated-execution curve has to give up action frequency to buy language throughput (or vice versa) as the language workload grows. OxyGen expands the frontier instead, improving **both axes by 1.2–3.7×**. Beyond the one-request-per-frame default, we swept arrival patterns (uniform bursts, Poisson arrivals, random request lengths), and the picture is the same; naive MPS parallelization, by comparison, buys very little.

{{ image(src="/img/blog/oxygen-mot-vla-inference/ablation-pi05-libero.png", dimmable=true, caption="Ablation on LIBERO: KV sharing lifts every operating point; cross-frame batching holds action frequency flat as decoding grows long.") }}

The ablation separates the two mechanisms. KV sharing alone (`Ours w/o Batching`) cuts mean end-to-end latency from 200.3 to 145.4 ms at `N=5` (a **1.38×** speedup) and shaves 25.5–36.4% off profiled inference latency across `N=1–5`. Cross-frame batching then takes over as outputs lengthen: with both enabled at `k = 5`, action frequency stays near **60 Hz** on the RTX 4090 and **27 Hz** on the Jetson even at `N=30`, while the baseline degrades 2.6×.

We also checked everything besides speed, and nothing regressed: with the official π<sub>0.5</sub>-LIBERO checkpoint, success rates stay within **±0.8%** of openpi's reported numbers (a 10-seed rerun, 20,000 rollouts, lands at 96.76% overall [96.55, 96.97] against openpi's 96.85%); memory grows by only 15%, while average power drops by up to **47%** and energy per request by up to **78%**. Less redundant work means fewer weight reads. MPS parallelization moves the wrong way on every count: nearly double the peak memory, slightly *more* energy per request.

The deployment we care most about is a real humanoid robot with the Jetson AGX Thor on board, 3-way 224×224 cameras, `N=30, k=5`, running manipulation interleaved with language generation. Each frame has a 333 ms action-execution window, and the measured breakdown fills in the rest:

| Stage | Baseline (ms) | OxyGen (ms) |
| ----- | ------------: | ----------: |
| Prefill & denoise | 207.5 | 198.0 |
| Language generation | 822.3 | 195.4 |
| Action execution window | 333.0 | 333.0 |

The baseline's 1,030 ms of per-frame inference far exceeds the window, so the control loop cannot keep up at all. OxyGen brings total inference to **393 ms**; more importantly, the action-critical prefill-plus-denoise stage takes 198 ms and **fits inside the execution window**, while language generation (822.3 → 195.4 ms, a 4.2× cut) runs after the action is emitted and hides behind execution.

---

## Beyond a single backbone

We also ported OxyGen to two more open MoT VLAs[^14]<sup class="footnote-separator">, </sup>[^15] and repeated the measurements on their native implementations: the speedups transfer, and they grow with the language backlog. Same RTX 4090, one new request per frame, `N=30`:

| Model | Baseline latency | Speedup (`k=1`) | Speedup (`k=5`) |
| ----- | ---------------: | --------------: | --------------: |
| Xiaomi-Robotics-0 (4.7B) | 1,540.9 ms | 5.50× | 3.14× |
| StarVLA Qwen3VL-PI_v3 (5.07B) | 1,351.5 ms | 6.18× | 3.41× |

One edge case is worth flagging: when `N=k`, each request finishes in its arrival frame, the average batch is one, and results sit near break-even (0.92–1.07×). Cross-frame batching pays only when there is a backlog to batch; KV sharing alone is what keeps those points above water.

To see how far the sharing itself scales, we used StarVLA's released action heads (a flow-matching PI_v3 head, a GR00T flow head, an OFT one-shot head), all reusing one shared Qwen3-VL-4B prefix. More experts means more readers amortizing the same prefill:

| Experts `E` | Speedup (`N=5, k=5`) | Speedup (`N=30, k=5`) |
| ----------: | -------------------: | -------------------: |
| 2 | 1.07× | 3.41× |
| 3 | 1.21× | 3.32× |
| 4 | 1.36× | 3.48× |

For backbone scale, we swept StarVLA's Qwen3.5 multimodal backbones plus a tensor-parallel run over PCIe: the speedup barely moves.

| Backbone | Baseline latency | Speedup (`N=30, k=5`) |
| -------- | ---------------: | -------------------: |
| Qwen3.5-0.8B | 1,257.5 ms | 3.53× |
| Qwen3.5-2B | 1,267.5 ms | 3.44× |
| Qwen3.5-4B | 1,728.0 ms | 3.60× |
| Qwen3.5-9B | 1,725.9 ms | 3.50× |
| Qwen3.5-9B, TP=2 | 1,983.5 ms | 3.71× |

One caveat: the extra heads were trained separately, so these experiments validate system-level scaling and shared-prefix fan-out, not the quality of a jointly trained multi-expert policy.

---

## Demo: a VLA that remembers what it just did

The point of all that language throughput is not narration for its own sake. It is memory: long-horizon tasks drift unless something records what has been done, which is the motivation behind PI's MEM[^16]. The public π<sub>0.5</sub> checkpoint cannot produce such text out of the box, so after submitting the paper we trained a small addition that can: a suffix-only LoRA[^17] on the frozen model that appends one observation-grounded line ("The alphabet soup is in the basket.") after the shared prefix as a private `Memory: ` continuation. It never touches the shared cache or the action path (zeroing it leaves actions bit-identical), and it adds a median **8.16 ms** per memory at serving time, against the 47.66 ms prefix forward that isolated execution would recompute.

We supervised it with annotations derived from the raw LIBERO demonstrations; the [dataset](https://huggingface.co/datasets/xxxxyu/libero-textual-memory-annotations) and [adapter](https://huggingface.co/xxxxyu/oxygen-pi05-textual-memory-lora) are on Hugging Face, and on held-out frames the model reproduces the annotation exactly **87.25%** of the time. That is the workload in <a href="#demo-video">the demo video</a> at the top of this post: one new memory request per action replan, unfinished ones resuming in the next frame's language batch, action frequency never noticing.

---

## Scope and what's next

OxyGen does not try to be everything. The per-frame budget `k` is fixed per deployment rather than recomputed at runtime; multi-node serving is untested, with the tensor-parallel check covering two GPUs within one node; and the system is a scheduling layer over the model, orthogonal to action-side async pipelines[^18]<sup class="footnote-separator">, </sup>[^19] and to compression or pruning. What we would like next: more backbones, richer jointly-trained expert sets, and speculative decoding hooked onto the resumable language state, where it should compose cleanly.

Code (JAX and PyTorch backends), the textual-memory adapter, and the annotation dataset are all public; pointers are in the links above. If you're deploying MoT VLAs under real-time constraints, we'd love to hear what your workload looks like: [open an issue](https://github.com/air-embodied-brain/OxyGen/issues) or [email me](mailto:lixiangy22@mails.tsinghua.edu.cn).

*Our team is hiring postdocs and interns (physical-AI foundation models, LLM training/inference systems, and more). Feel free to contact [Prof. Ting Cao](https://tingcao952.github.io).*

[^1]: Physical Intelligence, Kevin Black et al., ["π0.5: A Vision-Language-Action Model with Open-World Generalization"](https://arxiv.org/abs/2504.16054), arXiv, 2025.
[^2]: Action frequency counts emitted actions (`H = 10` per chunk, so `f = H / T` for per-frame latency `T`); the 70 Hz figure is measured at `N = 30, k = 1` on the RTX 4090. `S = 10` is openpi's denoising default ([pi0.py:222](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py#L222)); `H = 10` follows openpi's π<sub>0.5</sub>-LIBERO configuration ([config.py:744](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py#L744-L745)).
[^3]: Kevin Black et al., ["π0: A Vision-Language-Action Flow Model for General Robot Control"](https://arxiv.org/abs/2410.24164), arXiv, 2024.
[^4]: Weizhuang Liang et al., ["Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models"](https://arxiv.org/abs/2411.04996), arXiv, 2024.
[^5]: Physical Intelligence, ["openpi"](https://github.com/Physical-Intelligence/openpi), GitHub.
[^6]: openpi does not release the π<sub>0.5</sub> language-generation path; our baseline follows the [community reproduction](https://github.com/BrunoFANG1/openpi_subtask_generation).
[^7]: Profiled on π<sub>0.5</sub>-LIBERO (`S = 10`, RTX 4090): prefix forward ~45.1 ms, full 10-step denoise loop ~20.4 ms.
[^8]: π<sub>0.5</sub>-LIBERO on the RTX 4090, `k = 5`, as the per-request decoding length `N` grows from 5 to 30.
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
