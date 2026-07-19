+++
title = "Dario Amodei 博客阅读笔记"
date = "2025-08-02"
updated = "2026-07-11"
description = "阅读 Dario Amodei 关于强大 AI、可解释性与 AI 政策文章的笔记。"
template = "blog-page.html"

[taxonomies]
tags = ["Essay", "Reading"]

[extra]
ai_translation_source = "en"
+++

> 本文仍在更新，但现有笔记已经可以阅读。

最近，我读了 [Anthropic](https://www.anthropic.com/) CEO [Dario Amodei](https://www.darioamodei.com/) 的几篇文章[^1]。作为一名 AI 博士生，也是 Anthropic Claude 模型的常用用户，我从中获得了不少启发。

这些文章拓宽了我对学术界和产业界的认识，内容也常常横跨多个学科。
这篇笔记记录了我阅读时想到的一些问题。

按我的阅读顺序，涉及的文章包括：

- [*On DeepSeek and Export Controls*](https://www.darioamodei.com/post/on-deepseek-and-export-controls)（2025 年 1 月）
- [*The Urgency of Interpretability*](https://www.darioamodei.com/post/the-urgency-of-interpretability)（2025 年 4 月）
- [*Machines of Loving Grace*](https://www.darioamodei.com/essay/machines-of-loving-grace)（2024 年 10 月）

## 畅想强大 AI

在 [*Machines of Loving Grace*](https://www.darioamodei.com/essay/machines-of-loving-grace) 中，Amodei 描绘了“强大 AI”（他更偏爱这一说法，而非 *AGI*）可能具有的形态，以及它问世后 5–10 年间可能给社会带来的变化。他预测，“它最早可能在 2026 年出现”。在我看来，他设想的 AI 在形态上与今天的大语言模型相似，但实现方式未必相同，主要具有以下特点：

- **超级智能。** 单论智力，它“比诺贝尔奖得主更聪明”。例如，它能够证明尚未解决的数学定理，也能写出极为出色的小说。
- **能够使用各种界面。** 除了聊天，它还可以使用人类完成虚拟工作所需的一切界面。
- **自主工作。** 它能够长时间执行分配给它的任务，只在确有必要时向人类询问。
- **通过虚拟方式与现实交互。** 它没有实体，但可以借助计算机控制乃至创造现实中的工具。
- **规模庞大、速度极快。** 原本用于训练的资源可以转而运行数百万个实例；这些实例学习和行动的速度都远超人类（例如快 10–100 倍）。
- **既独立又协作。** 数百万个实例既可以各自处理互不相关的任务，也可以协同解决同一个问题；其中一些实例还可以通过微调，尤其擅长特定任务（我会称之为“专家”）。

他将其概括为“数据中心里的天才之国”：它能迅速解决棘手问题，却依然受到外部世界的制约。这些制约包括物理过程的速度、对数据的需求（例如粒子物理研究）、问题本身的复杂性（例如[三体问题](https://en.wikipedia.org/wiki/Three-body_problem)）、法律与人类意愿，以及晶体管密度和擦除单位信息所需能量的物理极限。

因此，他借用经济学中的“智力边际回报”来思考一个拥有极其强大 AI 的世界。
他的预测建立在这样一个问题之上：更高的智能能在哪些领域发挥作用、作用有多大，又需要多长时间？
我很喜欢这种分析方式。如果没有一种合理的方法衡量智能的影响，我们很容易走向过度乐观或过度悲观。

上面的基本假设和分析框架[^2]并不是 *Machines of Loving Grace* 的主体，却是最打动我的部分。
毕竟，最难的并不是在一个扎实的框架下推导出合理答案，而是提出这个框架本身。

文章余下的篇幅分别从五个领域讨论强大 AI 问世后 5–10 年间可能发生的变化：生物学与健康、神经科学与心智、经济发展与贫困、和平与治理，以及工作与意义。
他的核心预测是：**强大 AI 将把人类 50–100 年的进步压缩到 5–10 年内完成**，他称之为“压缩的 21 世纪”。悬而未决的问题是，这样的 AI 究竟何时才会出现。

与[“基本假设和分析框架”](https://www.darioamodei.com/essay/machines-of-loving-grace#basic-assumptions-and-framework)一节相比，我觉得有些预测更带有个人色彩，比如关于自由民主与威权主义的讨论[^3]。
即使采用同一套分析框架，人们也会因为背景和经历不同，对某些问题产生不同的理解与想象。
这正是我喜欢阅读不同背景作者作品的重要原因之一，也是我写下这些笔记的原因。

## 用机制可解释性保障 AI 安全

随着现代 AI 逐渐接近上文所说的“强大 AI”，治理与“安全”问题会愈发重要（而如何定义安全，本身就是一个重大问题）。
强大的 AI 系统需要技术层面的安全机制，可解释性是其中的重要一环。
在 [*The Urgency of Interpretability*](https://www.darioamodei.com/post/the-urgency-of-interpretability) 中，Amodei 介绍了机制可解释性的重要性与发展历程，以及 Anthropic 最近在这一方向上的进展。

可解释性之所以重要，是因为现代深度学习系统不同于由明确规则编写的传统系统，在很大程度上仍是“黑箱”。就连开发者也无法完全解释，为什么某个输入会产生某个输出。
Amodei 引用了联合创始人 [Chris Olah](https://colah.github.io/about.html) 的说法：生成式 AI 系统与其说是被“建造”出来的，不如说是被“培育”出来的（我很喜欢这个词）——它们的内部机制从训练中涌现，并非由人直接设计。
这类系统可能对人们产生越来越大的影响，因此，要让它们得到更广泛的应用，透明度与可解释性不可或缺。

*机制可解释性*试图通过逆向工程，理解神经网络在内部学到了哪些算法和计算机制。
它不同于关注输入输出关系或高层行为的方法，而是试图从底层解释各层和神经元中究竟发生了什么。
这一领域早期主要研究 CNN，如今则越来越关注大语言模型。
Anthropic 联合创始人 Chris Olah 从在 Google 和 OpenAI 工作时起，就一直在研究这一领域。

对大语言模型而言，一个早期步骤是找出可解释的神经元[^4]。研究人员发现，有些神经元的含义一目了然，但大多数神经元表达的却是杂乱混合的词语与概念。这种“叠加”让模型能表达比神经元数量更多的概念。为了将它们分离，研究人员使用*稀疏自编码器*[^5]寻找对应于更纯粹、可由人理解的概念的神经元组合，并将这些概念称为“特征”。有些特征非常微妙，例如“不论字面还是比喻意义上的闪烁其词或犹豫不决”。借助这种方法，Anthropic 在 Claude 3 Sonnet 中识别出了超过 3000 万个特征。

有了这些特征，研究人员就可以进一步探查模型的工作方式。人为放大某项特征可以显著改变模型行为（[“Golden Gate Claude”](https://www.anthropic.com/news/golden-gate-claude)）；追踪并操纵成组的特征（即“回路”[^6]），则有助于解释推理过程的某些部分。
Anthropic 的研究文章[^7]更详细地介绍了这项工作。

机制可解释性试图逆向分析神经网络，甚至是拥有数十亿参数的大语言模型，并为模型内部的工作方式提供具体证据；我很欣赏这一点。
这个方向看起来很有前景，但我不认为在庞大的神经网络里寻找“漏洞”，就是通往强大且安全的 AI 的完整路径——那就像在一头蓝鲸体内清除癌细胞一样。
至少目前，我不觉得我们的模型已经强大到足以构成“危险”，不过机制可解释性也可以帮助提升模型能力。

## AI 与政治

这一节尚待完成（这个话题也实在太大）——等有时间时我会补上。

[^1]: 我最初读到的是一篇介绍 [*On DeepSeek and Export Controls*](https://www.darioamodei.com/post/on-deepseek-and-export-controls) 的中文文章，之后又阅读了英文原文，以及 [Amodei 个人网站](https://www.darioamodei.com/)上的其他文章。

[^2]: 这些内容整理自原文的[“基本假设和分析框架”](https://www.darioamodei.com/essay/machines-of-loving-grace#basic-assumptions-and-framework)一节。

[^3]: 引自 Amodei [*Peace and governance*](https://www.darioamodei.com/essay/machines-of-loving-grace#4-peace-and-governance) 一节的首段：“二十年前，美国政策制定者认为，与中国开展自由贸易会使中国在富裕起来的同时走向自由化；事实远非如此，如今我们似乎正与一个复兴的威权阵营走向第二次冷战。”

[^4]: Nelson Elhage 等，[“Softmax Linear Units”](https://transformer-circuits.pub/2022/solu/index.html)，*Transformer Circuits Thread*，2022。

[^5]: Trenton Bricken 等，[“Towards Monosemanticity: Decomposing Language Models With Dictionary Learning”](https://transformer-circuits.pub/2023/monosemantic-features)，*Transformer Circuits Thread*，2023。

[^6]: Jack Lindsey 等，[“On the Biology of a Large Language Model”](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)，*Transformer Circuits Thread*，2025。

[^7]: Anthropic，[*Mapping the Mind of a Large Language Model*](https://www.anthropic.com/research/mapping-mind-language-model) 和 [*Tracing the Thoughts of a Large Language Model*](https://www.anthropic.com/research/tracing-thoughts-language-model)。
