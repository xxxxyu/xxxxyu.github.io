# Research taxonomy — English draft

The structured draft is `data/research-taxonomy.en.toml`. It is deliberately
not connected to the rendered map yet: terminology is the current review step,
before contribution sizing and the visual redesign. Chinese content is unchanged.

## Structure

Use a controlled vocabulary with stable IDs and one navigation parent per term.
This is an organizational tree, not a claim that fields have exclusive scientific
boundaries. Works can reference multiple terms across branches. Model families
and deployment contexts form separate facets, not siblings of research methods.

```
Efficient Inference
  Memory Management
    KV Cache Management
  Quantization
    Low-bit Inference
  Kernel Optimization
    Matrix Multiplication
  Scheduling
    Continuous Batching
  Adaptive Inference
  Inference Runtimes
  Heterogeneous Computing

Robot Learning
  Failure Detection
  Failure Recovery
  Policy Adaptation
  Closed-loop Control
  Robot Simulation

LLM Agents and Reasoning
  LLM Reasoning
    Test-time Scaling
  In-context Learning
  Test-time Adaptation
    In-context Adaptation
    Interaction Memory
  Causal Representation Learning
  Agent Memory
  Personal Agents
  Security and Privacy
  Self-improvement
    Reflection
  Program Synthesis

Data Processing
  Context Sensing
  Stream Processing
  Graph Mining
  Near-memory Computing
```

The four headings are navigation groups, not four prescribed circles. In particular,
Near-memory Computing may ultimately move under a separate Computer Architecture
branch; the draft avoids adding a large visual region for one work. Likewise,
Test-time Scaling is applicable beyond LLM reasoning and can be reused by Zeva.
Zeva's Interaction Memory is deliberately separate from Agent Memory: its
deployment-time in-context causal adaptation is closer to test-time adaptation
and scaling than to a persistent agent-memory subsystem.
The final map should foreground populated problem clusters, not tree roots.

Context facets: Edge AI, Large Language Models (LLMs), Vision-Language-Action
Models (VLAs), World-Action Models (WAMs), Embodied Agents, Personal Agents,
Video Analytics. Implementation names such as vLLM/GPTQv2 and publication genre
such as Review stay in descriptions or metadata.

## Proposed work assignments

Primary terms should be visible on a work preview (usually two or three).
Secondary terms are for detail, search and weaker cross-cluster relationships.

| Work | Primary terms | Secondary terms |
| --- | --- | --- |
| FlexNN | Memory Management; Scheduling | Adaptive Inference |
| Vec-LUT | Low-bit Inference; Kernel Optimization | Matrix Multiplication |
| DWI | Adaptive Inference | — |
| Squeezer | Scheduling | — |
| DIMMining | Graph Mining; Near-memory Computing | — |
| OxyGen | KV Cache Management; Scheduling | Continuous Batching; Inference Runtimes |
| Embodied.cpp | Inference Runtimes; Heterogeneous Computing | Kernel Optimization; Closed-loop Control |
| Cosmos Lite | Low-bit Inference; Kernel Optimization; Inference Runtimes | Robot Simulation; Scheduling |
| GPTQv2 × vLLM | Low-bit Inference; Kernel Optimization | — |
| ActProbe | Failure Detection | — |
| Zeva | In-context Adaptation; Interaction Memory; Causal Representation Learning | Policy Adaptation; Test-time Scaling |
| Zetta | Self-improvement; Failure Recovery; Closed-loop Control | Failure Detection; Robot Simulation; Scheduling; Heterogeneous Computing; Quantization |
| EmbodiSkill | Reflection; Self-improvement | — |
| Personal LLM Agents | Personal Agents; Efficient Inference | Security and Privacy |
| ChainStream | Context Sensing; Program Synthesis | Stream Processing |
| Length-constrained reasoning | LLM Reasoning; Test-time Scaling | — |
| Doctoral overview | Overview link, excluded from similarity | — |

## Sources and confidence

Source records and per-work notes in the TOML distinguish blog tags/body,
paper abstracts/body, repository metadata and a project README. A normalized
editorial label is not presented as an author-supplied keyword.

Zeva explicitly lists **Embodied Foundation Models, Interaction Memory,
Test-time Scaling, Causal Representation** as Index Terms in its v1 HTML.
These verbatim terms are retained separately from normalized assignments.

No formal keyword block was found in the inspected HTML of OxyGen, ActProbe,
EmbodiSkill, Zetta, Embodied.cpp, ChainStream or Personal LLM Agents. Their
assignments use abstracts and sections. The reasoning paper uses its arXiv
abstract. FlexNN uses the official project README because ACM access returned
403. DWI and Squeezer publisher responses did not provide usable content;
their assignments remain title-only, as does DIMMining in this review.

Paper versions can differ from current site metadata. External v1 papers were
used for subject evidence only; no author lists or publication metadata changed.

## Relationship rules for the next implementation step

- Shared specific terms provide stronger evidence than a shared parent theme.
- Primary-primary matches outweigh primary-secondary and secondary-secondary
  matches. A work's own ancestor and descendant tags must not be double-counted.
- Common model/context labels aid filtering but cannot create a relationship
  alone. Two works using LLMs need not be neighbours.
- Contribution size must not influence topical distance.
- Broad surveys/overviews should not become artificial hubs; use the documented
  contribution scope and exclude the doctoral overview from ranking.
- Display the common term as the relationship explanation. No exact numerical
  similarity is proposed until the vocabulary is reviewed.

## Decisions resolved / remaining

- **Resolved:** Use **Self-improvement** as the shared term for Zetta and
  EmbodiSkill. **Self-evolving Agents** can remain a searchable alias or prose
  phrase, but is not the canonical tree label.
- **Resolved:** Zeva's interaction memory is intentionally kept separate from
  Agent Memory. Its deployment-time in-context causal adaptation is closer to
  test-time adaptation/scaling than to a persistent agent-memory subsystem.
  The draft uses **In-context Adaptation** as the work-level label and
  **Test-time Adaptation** as its parent.
- **Resolved:** Personal LLM Agents maps the whole survey through Personal
  Agents, while
   retaining Efficient Inference and Security and Privacy as documented
   subtopics. Quantization and Memory Management are not shown as taxonomy
   assignments for this node.

Remaining editorial questions:

- Zetta's Z-Infra systems topics remain secondary and can be shown in the detail
  view without competing with its primary embodied self-improvement themes.
- Personal LLM Agents shows Personal Agents and Efficient Inference as its two
  primary themes; Security and Privacy remains secondary.
5. Before finalizing DWI, Squeezer and DIMMining, verify the paper keyword blocks
   or abstracts. Their current labels are conservative title-based candidates.
