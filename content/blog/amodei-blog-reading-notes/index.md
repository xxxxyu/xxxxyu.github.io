+++
title = "Reading Notes of Dario Amodei's Blog"
date = "2025-08-02"
updated = "2026-07-11"
description = "Notes on Dario Amodei's essays about powerful AI, interpretability, and AI policy."
template = "blog-page.html"

[taxonomies]
tags = ["Essay", "Reading"]
+++

> This is still in progress, but the existing notes are ready to read.

I recently read several posts[^1] by [Dario Amodei](https://www.darioamodei.com/), CEO of [Anthropic](https://www.anthropic.com/), and found them insightful as both an AI Ph.D. candidate and a frequent user of Anthropic's Claude models.

They broadened my academic and industry perspectives, often across disciplines.
These notes capture the thoughts that came to me while reading.

Amodei's posts covered (in my reading order):

- [*On DeepSeek and Export Controls*](https://www.darioamodei.com/post/on-deepseek-and-export-controls) (Jan. 2025)
- [*The Urgency of Interpretability*](https://www.darioamodei.com/post/the-urgency-of-interpretability) (Apr. 2025)
- [*Machines of Loving Grace*](https://www.darioamodei.com/essay/machines-of-loving-grace) (Oct. 2024)

## Envisioning Powerful AI

In [*Machines of Loving Grace*](https://www.darioamodei.com/essay/machines-of-loving-grace), Amodei describes what *"powerful AI"* might look like (he prefers this phrase to *AGI*) and how it could change society in the 5–10 years after its arrival. He predicts that *"it could come as early as 2026."* In my summary, his vision resembles today's LLMs in form, though not necessarily in implementation, and has the following properties:

- **Super intelligent.** It is *"smarter than a Nobel Prize winner"* in terms of pure intelligence. For example, it can prove unsolved mathematical theorems and write extremely good novels.
- **Works with interfaces.** Beyond chat, it can use all the interfaces a human needs for virtual work.
- **Autonomous.** It can work on assigned tasks for long periods, asking humans only for necessary clarification.
- **Interacts virtually.** It doesn't have a physical embodiment, but can control (and even create) physical tools through a computer.
- **Massive scale and high speed.** Resources repurposed from training allow millions of instances, which learn and act much faster than humans (e.g., by 10–100×).
- **Independent yet cooperative.** The millions of instances can work both independently on unrelated tasks, and cooperatively on one task, with some of them fine-tuned to be especially good at particular tasks (I would call them *experts*).

He summarizes this as a *"country of geniuses in a datacenter"*: capable of solving difficult problems quickly, but still constrained by the outside world. These constraints include the speed of physical processes, the need for data (e.g., in particle physics), intrinsic complexity (e.g., [*the three-body problem*](https://en.wikipedia.org/wiki/Three-body_problem)), laws and human willingness, and physical limits on transistor density and energy per erased bit.

He therefore borrows the economic idea of *"marginal returns to intelligence"* to reason about a world with very powerful AI.
His predictions build on this question: in which areas, to what extent, and on what timescale would greater intelligence help?
I find this framing refreshing. Without a reasonable way to measure the impact of intelligence, it is easy to become either overly optimistic or overly pessimistic.

The above basic assumptions and framework[^2], although not the main part of *Machines of Loving Grace*, impress me the most.
After all, the hardest thing is not to infer a reasonable answer given a solid framework, but to propose the framework.

He spends the rest of the essay describing what might happen 5–10 years after powerful AI arrives, across five areas: biology and health; neuroscience and mind; economic development and poverty; peace and governance; and work and meaning.
His central prediction is that **powerful AI will compress 50–100 years of human progress into 5–10 years**, which he calls the *"compressed 21st century."* The open question is when such AI will arrive.

In contrast to the [*Basic assumptions and framework*](https://www.darioamodei.com/essay/machines-of-loving-grace#basic-assumptions-and-framework) section, I find some predictions, such as the discussion of liberal democracy and authoritarianism[^3], more personal.
Even within the same analytical framework, people will understand and imagine some issues differently because of their backgrounds and experiences.
That is an important reason I like reading people from different backgrounds, and why I am keeping these notes.

## Mechanistic Interpretability for Safe AI

As modern AI moves toward the "powerful AI" vision above, questions about governance and *safety* become more important (and defining safety is itself a major question).
Powerful AI systems will need technical safety mechanisms, and interpretability is an important part of that work.
In [*The Urgency of Interpretability*](https://www.darioamodei.com/post/the-urgency-of-interpretability), Amodei discusses the importance and history of mechanistic interpretability, and Anthropic's recent progress on this.

Interpretability matters because, unlike conventional systems programmed with explicit rules, modern deep-learning systems are largely *black boxes*. Even their developers do not fully understand why particular inputs produce particular outputs.
Quoting his co-founder [Chris Olah](https://colah.github.io/about.html), Amodei writes that generative AI systems are *"grown"* (I like this word) more than they are *"built"*: their internal mechanisms emerge from training rather than being directly designed.
These systems may have an increasingly large impact on people, making transparency and interpretability necessary for broader deployment.

*Mechanistic interpretability* reverse-engineers neural networks to understand the algorithms and computational mechanisms they learn internally.
Unlike approaches focused on input-output relationships or high-level behavior, it seeks a low-level account of what happens in layers and neurons.
The field first studied CNNs and now increasingly focuses on LLMs.
Chris Olah, co-founder of Anthropic, has been working on this field since he was at Google and OpenAI.

For an LLM, an early step is to identify interpretable neurons[^4]. Researchers found that while some neurons were immediately understandable, most represented an incoherent mixture of words and concepts. This *superposition* allows a model to express more concepts than it has neurons. To disentangle it, they used *sparse autoencoders*[^5] to find combinations of neurons corresponding to cleaner, human-understandable concepts, which they call *features*. Some are highly subtle, such as "literally or figuratively hedging or hesitating." Using this method, Anthropic identified more than 30 million features in Claude 3 Sonnet.

Features can then be used to probe how a model works. Artificially amplifying a feature can significantly affect behavior (["Golden Gate Claude"](https://www.anthropic.com/news/golden-gate-claude)), while tracking and manipulating groups of features (i.e., *circuits*[^6]) can help explain parts of the reasoning process.
Anthropic's research posts[^7] describe this work in more detail.

I appreciate mechanistic interpretability as a way to reverse-engineer neural networks, including LLMs with billions of parameters, because it offers concrete evidence about how models work internally.
It seems like a promising direction, but I doubt that finding "bugs" inside a huge neural network is the complete path to powerful and safe AI — it would be like eliminating cancer cells in a blue whale.
For now, I do not think our models are powerful enough to be "dangerous," although mechanistic interpretability can also help improve their capabilities.

## AI and Politics

This is a TODO section (also too large a topic to discuss) — I'll complete this part when time allows.

[^1]: I started with a Chinese article summarizing [*On DeepSeek and Export Controls*](https://www.darioamodei.com/post/on-deepseek-and-export-controls), then read the original English post and other writing on [Amodei's homepage](https://www.darioamodei.com/).

[^2]: These are summarized from the [*Basic assumptions and framework*](https://www.darioamodei.com/essay/machines-of-loving-grace#basic-assumptions-and-framework) section of the original post.

[^3]: Quoted from the first paragraph of Amodei's [*Peace and governance*](https://www.darioamodei.com/essay/machines-of-loving-grace#4-peace-and-governance) section: *"Twenty years ago US policymakers believed that free trade with China would cause it to liberalize as it became richer; that very much didn’t happen, and we now seem headed for a second cold war with a resurgent authoritarian bloc."*

[^4]: Nelson Elhage et al., ["Softmax Linear Units"](https://transformer-circuits.pub/2022/solu/index.html), *Transformer Circuits Thread*, 2022.

[^5]: Trenton Bricken et al., ["Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"](https://transformer-circuits.pub/2023/monosemantic-features), *Transformer Circuits Thread*, 2023.

[^6]: Jack Lindsey et al., ["On the Biology of a Large Language Model"](https://transformer-circuits.pub/2025/attribution-graphs/biology.html), *Transformer Circuits Thread*, 2025.

[^7]: Anthropic, [*Mapping the Mind of a Large Language Model*](https://www.anthropic.com/research/mapping-mind-language-model) and [*Tracing the Thoughts of a Large Language Model*](https://www.anthropic.com/research/tracing-thoughts-language-model).
