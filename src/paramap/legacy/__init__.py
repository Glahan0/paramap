"""
Archived: domain-conditioned selective parameter loading.

This was the project's original approach -- profile which FFN neurons activate
for which conversation topic, then keep only the topic-relevant blocks in VRAM
and drop the rest. It is **lossy**: parameters are discarded, so output quality
degrades.

It was retired as the core mechanism after measurement (see
`legacy/experiments/05_evaluation.py`) showed, on Qwen2.5-0.5B at a 50% block
budget and matched budgets across strategies:

  - in-domain perplexity rose by ~750% against a <= 5% target, and
  - domain-conditioned selection did not reliably beat a domain-agnostic
    importance ranking (it won 1 of 4 domains at a 50% budget), and
  - removing blocks produced no measurable speed-up at this model size.

The live project now pursues lossless tiered residency instead: every parameter
is kept, but only a budgeted subset is resident in VRAM at any moment. See
`paramap.residency`.

Nothing here is imported by the live code path. It is kept so the work and its
measurements remain inspectable.
"""
