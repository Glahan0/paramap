# ParamMap

**Run local LLMs under a VRAM budget you choose, without changing what they say.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Bilingual: EN/TR](https://img.shields.io/badge/docs-EN%20%7C%20TR-orange.svg)](README.tr.md)

Tell ParamMap how much VRAM it may use. It keeps as many decoder layers in GPU
memory as that allows, holds the rest in host RAM, and streams each one in just
before it runs — overlapping the transfer with computation on a side CUDA
stream. Every parameter is kept, the arithmetic is unchanged, and the generated
tokens are **identical** to running the model entirely on the GPU.

What you trade is generation speed, and the trade is predictable: per token the
runtime moves `streamed_bytes` across the bus, so

```
tokens/s <= pcie_bandwidth / streamed_bytes_per_token
```

`paramap plan` reports that ceiling before you commit to anything.

---

## What it is for

- **Stop a model from monopolising the GPU.** Cap it at 3 GB and use the rest
  for something else.
- **Make room for context.** Weights that moved to RAM are VRAM the KV cache
  can have.
- **Run a model bigger than your VRAM.** Weights are loaded to host RAM first
  and never need to fit on the GPU all at once.

---

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\activate      # Windows
pip install -e .
```

```bash
# What would a 3 GB budget cost? (nothing is loaded onto the GPU to answer this)
paramap plan --model Qwen/Qwen2.5-1.5B-Instruct --vram 3GB

# Generate under that budget
paramap run --model Qwen/Qwen2.5-1.5B-Instruct --vram 3GB -i

# Sweep budgets and verify the output never changes
paramap bench --model Qwen/Qwen2.5-1.5B-Instruct --budgets 3GB,2GB,1.5GB
```

As a library:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from paramap import plan_residency, StreamingModel

name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval()

plan = plan_residency(model, "500MB", max_context_tokens=2048)
print(plan.describe())

with StreamingModel(model, plan, device=torch.device("cuda")):
    out = model.generate(**tokenizer("Hello", return_tensors="pt").to("cuda"),
                         max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---

## Measured results

Qwen2.5-0.5B-Instruct (bf16, 942 MB of weights), RTX 3060 8 GB, PCIe at
11.9 GB/s measured. 48 generated tokens, greedy:

| Budget | Layers resident / streamed | Peak VRAM | VRAM saved | tokens/s | vs baseline | Output |
|---|---|---|---|---|---|---|
| baseline | 24 / 0 | 963.7 MB | — | 23.8 | — | identical |
| 900 MB | 17 / 7 | 819.9 MB | 14.9% | 21.5 | −9.5% | identical |
| 800 MB | 13 / 11 | 704.8 MB | 26.9% | 21.6 | −9.0% | identical |
| 700 MB | 10 / 14 | 618.6 MB | 35.8% | 20.7 | −13.1% | identical |
| 600 MB | 6 / 18 | 503.6 MB | 47.7% | 19.3 | −18.8% | identical |
| 500 MB | 3 / 21 | 417.3 MB | 56.7% | 18.2 | −23.3% | identical |
| 420 MB | 0 / 24 | 331.2 MB | 65.6% | 17.0 | −28.6% | identical |

So on this setup: **about two thirds of the weight VRAM freed for under a third
of the speed.** At full streaming the runtime reaches 94% of the bus-imposed
ceiling, meaning the prefetch pipeline leaves little on the table.

### Qwen2.5-3B-Instruct

bf16, 5.75 GB of weights, same RTX 3060, PCIe at 13.4 GB/s with pinned memory.
Each streamed layer is 154 MB. "Bus use" is streamed bytes per token × tokens/s
divided by 13.4 GB/s:

| Budget | Layers resident / streamed | Peak VRAM | VRAM saved | tokens/s | vs baseline | Bus use | Output |
|---|---|---|---|---|---|---|---|
| baseline | 36 / 0 | 5.87 GB | — | 18.6 | — | — | identical |
| 5 GB | 27 / 9 | 4.84 GB | 17.5% | 7.8 | −57.9% | 81% | identical |
| 4 GB | 20 / 16 | 3.82 GB | 34.9% | 5.1 | −72.4% | 94% | identical |
| 3 GB | 13 / 23 | 2.79 GB | 52.4% | 3.7 | −80.3% | 98% | identical |
| 2 GB | 6 / 30 | 1.77 GB | 69.9% | 2.9 | −84.7% | 100% | identical |
| 1.5 GB | 3 / 33 | 1.33 GB | 77.4% | 2.6 | −86.0% | 99% | identical |
| 1.15 GB | 0 / 36 | 909 MB | 84.9% | 2.3\* | −87.6% | 95% | identical |

\* From a separate run with all 36 layers pinned. In the sweep, host RAM ran
low, some layers fell back to pageable memory, and it measured 1.6 tokens/s.

The saving grows with model size, as expected (85% vs 66%), because anchors are
a smaller share. **But the speed cost grows too, and it is set by physics, not
by the runtime.** The engine is at 94–100% of the bus ceiling, so

    tokens/s ≤ PCIe bandwidth / bytes streamed per token

and a 3B layer is 154 MB against a 0.5B layer's 18 MB. Streaming dense bf16
weights over PCIe is a good trade when only a few layers have to move, and an
expensive one when most of the model does.

Two honest caveats:

- **The floor is set by what cannot be streamed.** Embeddings and the output
  head ("anchors") are touched every token and stay resident: 260 MB on 0.5B
  (28% of the model), 594 MB on 3B (10%).
- **These numbers are from one GPU.** Run `paramap bench` on your own model and
  machine rather than trusting the tables.

---

## How the budget is spent

In this order, because each item is worthless without the one before it:

1. **Anchors** — embeddings, final norm, output head. Needed every token.
2. **KV cache** for `--context` tokens, sized from the key/value head count so
   grouped-query models are not overcharged.
3. **Activation reserve** — transient tensors. `auto` scales with model size;
   override with `--activation-reserve` once `bench` has told you the real peak.
4. **Staging slots** — the GPU buffers streamed layers rotate through. Two is
   enough to overlap a transfer with the previous layer's compute.
5. **Resident layers** — as many as the remainder holds.

If step 1–3 already exceed the budget, ParamMap refuses with an explanation
rather than failing later with an out-of-memory error.

---

## Project layout

```
src/paramap/
├── residency.py      # budget planner + streaming runtime  <- the core
├── benchmark.py      # VRAM / tokens-per-second / output-identity measurement
├── cli.py            # paramap plan | run | bench
├── arch/             # per-architecture layer access (Qwen and Llama-style layouts)
├── device/           # hardware probe and real PCIe bandwidth measurement
├── i18n/             # EN/TR message catalogue
└── legacy/           # archived earlier approach, see below
tests/
├── test_residency.py # losslessness and budget-accounting invariants
└── test_legacy.py    # keeps the archived work verifiable
legacy/               # archived plans and experiments
```

Supported models: any causal LM whose decoder layers are a `ModuleList` under
`model.model.layers`. Measured on Qwen2.5 only; Llama, Mistral and Gemma use
the same layout and are expected to work, but are not yet tested. Other
architectures need a `ModelArchAdapter` subclass; see `arch/base.py`.

---

## What was archived, and why

The project began as something different: profile which FFN neurons activate
for which conversation topic, then keep only the topic-relevant parameter
blocks in VRAM and **drop** the rest. That is lossy, and measurement retired
it. On Qwen2.5-0.5B, with every selection strategy held to an identical block
budget:

- in-domain perplexity rose by roughly **750%** against a ≤5% target;
- topic-conditioned selection did **not** reliably beat a domain-agnostic
  importance ranking — it won 1 of 4 domains at a 50% budget;
- and dropping blocks produced **no** measurable speed-up at this model size.

The code, its tests and the experiment that produced those numbers are kept
under `paramap.legacy` and `legacy/`, and nothing there is on the live import
path. `legacy/experiments/05_evaluation.py` reproduces the measurements.

---

## Development

```bash
pip install -e ".[dev,legacy]"   # legacy: gguf, for the archived code
pytest -q
```

The test that matters most is `test_streaming_output_is_identical`: it runs on
CPU, needs no large model, and asserts that streamed execution changes nothing.
If it ever fails, the runtime is no longer lossless.

---

*Türkçe dokümantasyon: [README.tr.md](README.tr.md)*
