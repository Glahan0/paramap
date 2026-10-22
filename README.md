# ParamMap

**Run local LLMs under a VRAM budget you choose.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Bilingual: EN/TR](https://img.shields.io/badge/docs-EN%20%7C%20TR-orange.svg)](README.tr.md)

Tell ParamMap how much VRAM it may use. It keeps as many decoder layers in GPU
memory as that allows and holds the rest in host RAM. Every parameter is kept;
nothing is pruned. The layers that did not fit run one of two ways:

- **fast** (the CLI default): they are **computed on the CPU**, straight from
  RAM. This needs no VRAM at all and is usually quicker, but CPU kernels round
  differently from the GPU, so greedy output can occasionally flip a near-tie
  token.
- **exact** (`--exact`): they are **streamed** to the GPU just before they run,
  overlapping the transfer with compute on a side CUDA stream. The arithmetic
  is unchanged and the generated tokens are **identical** to running the model
  entirely on the GPU.

`paramap plan` picks the faster of the two for your machine, tells you which
it picked, and estimates the speed before you commit to anything. In exact mode
the bus sets a hard ceiling:

```
tokens/s <= pcie_bandwidth / streamed_bytes_per_token
```

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
# What would a 3 GB budget cost? (at most one layer touches the GPU, briefly, to time it)
paramap plan --model Qwen/Qwen2.5-1.5B-Instruct --vram 3GB

# Generate under that budget
paramap run --model Qwen/Qwen2.5-1.5B-Instruct --vram 3GB -i

# Sweep budgets: VRAM, speed, and how far the output matches the all-GPU run
paramap bench --model Qwen/Qwen2.5-1.5B-Instruct --budgets 3GB,2GB,1.5GB

# Any of the above with a guarantee that the output is identical
paramap run --model Qwen/Qwen2.5-1.5B-Instruct --vram 3GB --exact
```

As a library:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from paramap import plan_residency, StreamingModel

name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval()

# mode="exact" (the library default) streams; mode="fast" may use the CPU.
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

### Fast mode on Qwen2.5-3B

Same model and GPU; Ryzen 5 5500 (6 cores, AVX2) with dual-channel DDR4-3200,
which reads at 35 GB/s, against 13.4 GB/s over PCIe. A 3B layer takes 5.9 ms on
the CPU and 11.5 ms to cross the bus, so the planner chose the CPU every time:

| Budget | GPU / CPU layers | Peak VRAM | VRAM saved | tokens/s | exact mode | Output vs all-GPU |
|---|---|---|---|---|---|---|
| baseline | 36 / 0 | 5.87 GB | — | 18.4 | 18.6 | — |
| 5 GB | 28 / 8 | 4.84 GB | 17.5% | **10.5** | 7.8 | first 23/48 tokens |
| 4 GB | 21 / 15 | 3.82 GB | 35.0% | **7.7** | 5.1 | identical |
| 3 GB | 14 / 22 | 2.79 GB | 52.4% | **6.1** | 3.7 | first 23/48 tokens |
| 2 GB | 7 / 29 | 1.76 GB | 69.9% | **5.0** | 2.9 | identical |
| 1.5 GB | 4 / 32 | 1.33 GB | 77.4% | **4.6** | 2.6 | identical |
| 1.15 GB | 2 / 34 | 1.03 GB | 82.4% | **4.4** | 2.3 | first 23/48 tokens |
| 0.8 GB | 0 / 36 | 607 MB | 89.9% | **3.4** | won't fit | first 20/48 tokens |

Fast mode is 35–91% quicker at every budget and goes lower, since CPU layers
keep their KV cache in RAM and need at most one staging slot. Where greedy
output differed, it was at a near-tie: at the first differing token the
all-GPU model's top two candidates were 0.125 logits apart.

**Prompts.** Without native bf16 support a CPU layer costs ~3 ms per prompt
token, so fast plans keep one staging slot and run CPU layers on the GPU for
multi-token passes, handing their KV cache back to the CPU afterwards. At 2 GB:

| Prompt | first token, fast | fast without the slot | exact |
|---|---|---|---|
| 25 tokens | 0.54 s | 2.43 s | 0.36 s |
| 512 tokens | 0.68 s | 43.1 s | 0.40 s |
| 2000 tokens | 1.21 s | 185.6 s | 0.74 s |

At budgets too small for the slot (0.8 GB here) prompts fall back to the CPU,
and `paramap plan` says so.

**Quality.** WikiText-2 test, 16,384 tokens teacher-forced in 512-token chunks
(prompt path) and 2,048 tokens fed one at a time through the KV cache (decode
path), each compared with the all-GPU run. For scale, the first row stays
entirely on the GPU and only swaps the attention kernel (sdpa → eager):

| Configuration | prompt PPL | prompt KL (nats/token) | decode PPL | decode KL | decode top-1 agreement |
|---|---|---|---|---|---|
| all-GPU baseline | 10.256 | — | 14.941 | — | — |
| GPU, eager attention (reference) | +0.142% | 1.75e-3 | +0.161% | 2.28e-3 | 97.45% |
| fast 2 GB (29 layers on CPU) | **identical** | 0 | +0.139% | 1.44e-3 | 97.89% |
| fast 0.8 GB (all 36 on CPU) | +0.036% | 1.08e-3 | −0.028% | 1.44e-3 | 97.79% |

Fast mode moves the output less than swapping one GPU attention kernel for
another does, and perplexity moves within ±0.15% in both directions: rounding,
not a quality loss. It is still not identical; use `--exact` when that
matters.

The planner does not mix the two. Streaming some layers while the CPU computes
others looked like it should overlap, but both read the same RAM, and every
mixed plan measured slower than the pure CPU plan at the same budget.

The planner only gives up identical output when it predicts the CPU is at
least 15% faster. On Qwen2.5-0.5B the two measured within 3% of each other
while the CPU plan's output diverged after 3 tokens, so fast mode picks
streaming there, and on a machine with slow RAM and a fast bus it will do the
same. To decide, it times one layer on the CPU and one on the GPU; its
estimates landed within 1% of the measured speed (3B at 5 GB: 9.8 estimated,
9.9 measured; 0.5B at 600 MB: 19.6 vs 19.7).

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
3. **Activation reserve** — transient tensors of the prompt pass, which grow
   with prompt length. `auto` is sized from measurement (32 MB plus four MLP
   rows per `--context` token); override with `--activation-reserve`.
4. **Staging slots** — the GPU buffers streamed layers rotate through. Two is
   enough to overlap a transfer with the previous layer's compute.
5. **Resident layers** — as many as the remainder holds.

In fast mode, when the CPU wins, step 4 reserves a single slot (used only to
run prompts on the GPU) and step 2 only covers the resident layers: CPU layers
keep their KV cache in host RAM.

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
