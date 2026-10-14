"""
Command line interface for ParamMap.

Three things you can ask it:

  paramap plan  -- what would a given VRAM budget cost me? (loads the model
                   into RAM, and at most one layer briefly onto the GPU to time
                   it, so it is safe to ask before committing)
  paramap run   -- generate under a budget, with telemetry
  paramap bench -- sweep several budgets and report the VRAM/speed trade-off,
                   checking at each step that the output has not changed
"""

import argparse
import sys
from typing import List, Optional

import torch

from .benchmark import DEFAULT_PROMPT, benchmark_budgets, format_results, run_generation
from .residency import (
    InsufficientVRAMError,
    StreamingModel,
    format_bytes,
    parse_bytes,
    plan_residency,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _mode(args: argparse.Namespace) -> str:
    return "exact" if args.exact else "fast"


def _pick_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(model_name: str, device: torch.device, dtype: Optional[str]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype:
        torch_dtype = getattr(torch, dtype)
    else:
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading {model_name} ({torch_dtype}) into host RAM...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Deliberately loaded on the CPU: the residency plan decides what reaches
    # the GPU, so a model larger than VRAM never has to fit there first.
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    model.eval()
    return model, tokenizer


def _report_device(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        print(f"Device: {device} (no CUDA; streaming will not help here)")
        return
    free, total = torch.cuda.mem_get_info(device.index or 0)
    print(
        f"Device: {torch.cuda.get_device_name(device.index or 0)} | "
        f"VRAM {format_bytes(free)} free of {format_bytes(total)}"
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> int:
    device = _pick_device(args.device)
    _report_device(device)
    model, _tokenizer = _load(args.model, device, args.dtype)

    plan = plan_residency(
        model,
        args.vram,
        max_context_tokens=args.context,
        staging_slots=args.staging_slots,
        bandwidth_gb_s=args.bandwidth,
        activation_reserve=args.activation_reserve,
        mode=_mode(args),
    )
    print()
    print(plan.describe())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    device = _pick_device(args.device)
    _report_device(device)
    model, tokenizer = _load(args.model, device, args.dtype)

    plan = plan_residency(
        model,
        args.vram,
        max_context_tokens=args.context,
        staging_slots=args.staging_slots,
        bandwidth_gb_s=args.bandwidth,
        activation_reserve=args.activation_reserve,
        mode=_mode(args),
    )
    print()
    print(plan.describe())
    print()

    with StreamingModel(model, plan, device=device) as streaming:
        prompts: List[str] = [args.prompt] if args.prompt else []
        if not prompts and not args.interactive:
            prompts = [DEFAULT_PROMPT]

        for prompt in prompts:
            _generate_and_report(model, tokenizer, prompt, device, args, streaming)

        if args.interactive:
            print("Interactive mode. Blank line or /exit to quit.")
            while True:
                try:
                    prompt = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not prompt or prompt in ("/exit", "/quit"):
                    break
                _generate_and_report(model, tokenizer, prompt, device, args, streaming)

    return 0


def _generate_and_report(model, tokenizer, prompt, device, args, streaming) -> None:
    _ids, text, tokens_per_s, peak = run_generation(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=0,
        repeats=1,
    )
    print(f"\n{text}\n")
    line = f"  {tokens_per_s:.1f} tokens/s"
    if peak:
        line += f" | peak VRAM {format_bytes(peak)}"
    if streaming.plan.streamed_layers:
        line += (
            f" | {len(streaming.plan.streamed_layers)}/{streaming.plan.num_layers}"
            f" layers streamed ({format_bytes(streaming.plan.bytes_streamed_per_token)}/token)"
        )
    if streaming.plan.cpu_layers:
        line += f" | {len(streaming.plan.cpu_layers)}/{streaming.plan.num_layers} layers on CPU"
    print(line)


def cmd_bench(args: argparse.Namespace) -> int:
    device = _pick_device(args.device)
    _report_device(device)
    model, tokenizer = _load(args.model, device, args.dtype)

    budgets = [b.strip() for b in args.budgets.split(",") if b.strip()]
    print(f"\nSweeping {len(budgets)} budget(s): {', '.join(budgets)}\n")

    results = benchmark_budgets(
        model,
        tokenizer,
        budgets,
        device=device,
        prompt=args.prompt or DEFAULT_PROMPT,
        max_new_tokens=args.max_new_tokens,
        max_context_tokens=args.context,
        staging_slots=args.staging_slots,
        bandwidth_gb_s=args.bandwidth,
        mode=_mode(args),
    )

    print()
    print(format_results(results))

    print()
    # Only exact plans promise identical output; a CPU-computed layer that
    # flips a near-tie token is expected, not a bug.
    broken = [
        r.label
        for r in results
        if r.identical_to_baseline is False and (r.plan is None or r.plan.is_exact)
    ]
    differed = [r.label for r in results if r.identical_to_baseline is False and r not in broken]
    if broken:
        print(f"WARNING: output changed at these exact budgets: {', '.join(broken)}")
        print("Exact mode is supposed to be lossless; please report this.")
        return 1
    if differed:
        print(
            f"Output differed from the all-in-VRAM baseline at: {', '.join(differed)}. "
            "These plans compute layers on the CPU, which rounds differently; "
            "use --exact for identical output."
        )
        return 0
    print("Output was identical to the all-in-VRAM baseline at every budget.")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paramap",
        description=(
            "Run local LLMs under an explicit VRAM budget. Weights beyond the "
            "budget are held in host RAM and streamed in as needed; the output "
            "is identical to running the model entirely on the GPU."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
        sub.add_argument(
            "--vram",
            default="4GB",
            help="VRAM budget for this run, e.g. 3GB / 1500MB (default: 4GB)",
        )
        sub.add_argument(
            "--context",
            type=int,
            default=2048,
            help="Max context length to reserve KV cache for (default: 2048)",
        )
        sub.add_argument(
            "--staging-slots",
            type=int,
            default=2,
            help="GPU staging buffers; 2 is enough to overlap transfer with compute",
        )
        sub.add_argument(
            "--activation-reserve",
            default="auto",
            help="VRAM held back for transient tensors (default: auto, scales with the model)",
        )
        sub.add_argument(
            "--bandwidth",
            type=float,
            default=None,
            help="Host-to-device GB/s; measured on this machine when omitted",
        )
        sub.add_argument("--device", default=None, help="e.g. cuda, cuda:1, cpu")
        sub.add_argument("--dtype", default=None, help="e.g. bfloat16, float16, float32")
        sub.add_argument(
            "--exact",
            action="store_true",
            help=(
                "Guarantee output identical to running everything on the GPU: only "
                "stream layers, never compute them on the CPU (slower)"
            ),
        )

    plan_parser = subparsers.add_parser(
        "plan", help="Show what a VRAM budget would cost, without generating"
    )
    add_common(plan_parser)
    plan_parser.set_defaults(func=cmd_plan)

    run_parser = subparsers.add_parser("run", help="Generate text under a VRAM budget")
    add_common(run_parser)
    run_parser.add_argument("--prompt", default=None)
    run_parser.add_argument("--max-new-tokens", type=int, default=128)
    run_parser.add_argument(
        "-i", "--interactive", action="store_true", help="Keep prompting after the first answer"
    )
    run_parser.set_defaults(func=cmd_run)

    bench_parser = subparsers.add_parser(
        "bench", help="Sweep budgets and report the VRAM/speed trade-off"
    )
    add_common(bench_parser)
    bench_parser.add_argument(
        "--budgets",
        default="900MB,700MB,500MB",
        help="Comma-separated budgets to compare against an all-in-VRAM baseline",
    )
    bench_parser.add_argument("--prompt", default=None)
    bench_parser.add_argument("--max-new-tokens", type=int, default=64)
    bench_parser.set_defaults(func=cmd_bench)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    # Windows consoles default to a legacy code page, which mangles any
    # non-ASCII model output. Generated text is the point of this tool, so make
    # the streams UTF-8 where the platform allows it.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except InsufficientVRAMError as exc:
        print(f"\nThat budget will not work:\n  {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
