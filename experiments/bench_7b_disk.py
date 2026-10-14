"""
Qwen2.5-7B (bf16, 14.2 GB) on an 8 GB GPU and a 16 GB machine: the first
model here that fits neither in VRAM nor in free RAM.

Weights are mapped from the safetensors files (paramap.loader), so each layer
lives in one of three places: VRAM, pinned RAM, or the OS file cache / disk.
Sweeps budgets in exact and fast mode and records free RAM alongside, since
how much of the model the file cache can hold decides the disk traffic.

Exact budgets are compared with each other (there is no all-in-VRAM run to
compare with); fast budgets likewise among themselves.

    python experiments/bench_7b_disk.py [--tokens 32] [--out results.txt]
"""

import argparse
import contextlib
import io
import sys
import time

import psutil
import torch

from paramap.benchmark import benchmark_budgets, format_results
from paramap.loader import load_model_mmap

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def ram_line() -> str:
    vm = psutil.virtual_memory()
    return f"RAM: {vm.available / 2**30:.2f} GB available of {vm.total / 2**30:.2f} GB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--exact-budgets", default="7GB,6GB,5GB,4GB")
    parser.add_argument("--fast-budgets", default="7GB,6GB,5GB,4GB")
    parser.add_argument("--out", default="experiments/bench_7b_disk.txt")
    args = parser.parse_args()

    log = open(args.out, "w", encoding="utf-8")

    def say(text: str = "") -> None:
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    say(f"{MODEL} | {torch.cuda.get_device_name(0)} | {time.strftime('%Y-%m-%d %H:%M')}")
    say(ram_line())

    start = time.perf_counter()
    model, directory = load_model_mmap(MODEL, torch.bfloat16)
    say(f"Mapped in {time.perf_counter() - start:.1f} s")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)

    for mode, budgets in (("exact", args.exact_budgets), ("fast", args.fast_budgets)):
        if not budgets:
            continue
        say()
        say(f"=== {mode} === ({ram_line()})")
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            results = benchmark_budgets(
                model,
                tokenizer,
                [b for b in budgets.split(",") if b],
                device=torch.device("cuda"),
                max_new_tokens=args.tokens,
                mode=mode,
            )
        say(captured.getvalue().rstrip())
        say()
        say(format_results(results))
        say(ram_line())

    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
