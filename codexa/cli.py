import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        from codexa.dashboard.server import main as dashboard_main

        dashboard_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "context":
        from codexa.context import main as context_main

        context_main(sys.argv[2:])
        return
    import torch

    from codexa.benchmark import benchmark
    from codexa.config import Config
    from codexa.data.packing import prepare, read_documents, split_documents
    from codexa.inference.generate import from_checkpoint
    from codexa.mathematics.arithmetic import accuracy, problems
    from codexa.tokenizer.bpe import CodexaTokenizer
    from codexa.training.trainer import Trainer

    parser = argparse.ArgumentParser(
        description="Codexa: locally implemented, randomly initialized language models"
    )
    parser.add_argument("--threads", type=int, default=4)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "dashboard", add_help=False, help="Local read-only live/historical dashboard"
    )
    commands.add_parser(
        "context", add_help=False, help="Context transition preparation and bounded benchmarks"
    )
    tok = (
        commands.add_parser("tokenizer")
        .add_subparsers(dest="action", required=True)
        .add_parser("train")
    )
    tok.add_argument("--corpus", required=True)
    tok.add_argument("--vocab-size", type=int, default=512)
    tok.add_argument("--output", required=True)
    data = (
        commands.add_parser("data")
        .add_subparsers(dest="action", required=True)
        .add_parser("prepare")
    )
    data.add_argument("--corpus", required=True)
    data.add_argument("--tokenizer", required=True)
    data.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", default="configs/smoke.yaml")
    train.add_argument("--data", required=True)
    train.add_argument("--tokenizer", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--resume")
    train.add_argument(
        "--until", type=int, help="Stop at this optimizer step without changing the schedule"
    )
    gen = commands.add_parser("generate")
    sources = gen.add_mutually_exclusive_group(required=True)
    sources.add_argument("--checkpoint")
    sources.add_argument("--export")
    gen.add_argument("--prompt", default="")
    gen.add_argument("--device", default="auto")
    gen.add_argument("--max-tokens", type=int, default=None)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--greedy", action="store_true")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--no-cache", action="store_true")
    gen.add_argument("--max-sequence", type=int)
    gen.add_argument("--precision", choices=["fp32", "bf16", "auto"], default=None)
    bench = commands.add_parser("benchmark")
    bench.add_argument("--config", default="configs/smoke.yaml")
    bench.add_argument("--device", default="auto")
    bench.add_argument("--sequence", type=int, default=64)
    bench.add_argument("--steps", type=int, default=5)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--accumulation", type=int, default=1)
    bench.add_argument("--output")
    math = commands.add_parser("math").add_subparsers(dest="action", required=True)
    mg = math.add_parser("generate")
    mg.add_argument("--split", choices=["train", "val", "test"], default="train")
    mg.add_argument("--count", type=int, default=10)
    mg.add_argument("--seed", type=int, default=42)
    mg.add_argument("--output", required=True)
    me = math.add_parser("evaluate")
    me.add_argument("--problems", required=True)
    me.add_argument(
        "--predictions", required=True, help="JSON object mapping problem IDs to answer strings"
    )
    exp = commands.add_parser("export")
    exp.add_argument("--checkpoint", required=True)
    exp.add_argument("--metadata", required=True)
    exp.add_argument("--output", required=True)
    rel = commands.add_parser("release").add_subparsers(dest="action", required=True)
    fixture = rel.add_parser("fixture")
    fixture.add_argument("--corpus", required=True)
    fixture.add_argument("--output", required=True)
    pre = rel.add_parser("preflight")
    pre.add_argument("--artifact", required=True)
    pre.add_argument("--verification-report", required=True)
    ev = commands.add_parser("evaluate")
    evs = ev.add_mutually_exclusive_group(required=True)
    evs.add_argument("--checkpoint")
    evs.add_argument("--export")
    ev.add_argument("--data", required=True)
    ev.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "tokenizer":
        docs, _ = split_documents(read_documents(args.corpus))
        CodexaTokenizer.train(docs, args.vocab_size).save(args.output)
    elif args.command == "data":
        prepare(args.corpus, args.tokenizer, args.output)
    elif args.command == "train":
        trainer = Trainer(
            Config.load(args.config), args.data, args.tokenizer, args.output, args.device
        )
        if args.resume:
            trainer.resume(args.resume)
        trainer.train(until=args.until)
    elif args.command in ("release", "export", "evaluate"):
        from codexa.release.commands import dispatch

        dispatch(args)
    elif args.command == "generate":
        if args.export:
            from codexa.inference.generate import generate
            from codexa.release.artifacts import load_export

            model, tokenizer, _ = load_export(
                args.export, args.device if args.device != "auto" else "cpu"
            )
            print(
                tokenizer.decode(
                    generate(
                        model,
                        [2, *tokenizer.encode(args.prompt)],
                        max_tokens=args.max_tokens if args.max_tokens is not None else 50,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        greedy=args.greedy,
                        seed=args.seed,
                        use_cache=not args.no_cache,
                        max_sequence=args.max_sequence,
                        precision=args.precision or "fp32",
                    )
                )
            )
            return
        print(
            from_checkpoint(
                args.checkpoint,
                args.prompt,
                args.device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                greedy=args.greedy,
                seed=args.seed,
                use_cache=False if args.no_cache else None,
                max_sequence=args.max_sequence,
                precision=args.precision,
            )
        )
    elif args.command == "benchmark":
        try:
            result = benchmark(
                Config.load(args.config),
                args.device,
                args.sequence,
                args.steps,
                args.warmup,
                args.accumulation,
            )
        except torch.OutOfMemoryError:
            result = {
                "measured": False,
                "status": "out_of_memory",
                "config": args.config,
                "sequence": args.sequence,
                "accumulation": args.accumulation,
            }
        import platform

        from codexa.data.packing import fingerprint

        provenance: dict[str, Any] = {
            "config_sha256": fingerprint(args.config),
            "python": platform.python_version(),
            "cuda_runtime": torch.version.cuda,
            "threads": torch.get_num_threads(),
        }
        if (Path.cwd() / ".git").exists():
            from codexa.repository import source_identity

            provenance["source"] = source_identity()
        result["provenance"] = provenance
        text = json.dumps(result, indent=2)
        print(text)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text + "\n")
    elif args.command == "math":
        if args.action == "generate":
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                "".join(
                    json.dumps(p.to_dict()) + "\n"
                    for p in problems(args.split, args.count, args.seed)
                )
            )
        else:
            rows = [json.loads(line) for line in Path(args.problems).read_text().splitlines()]
            print(
                json.dumps(accuracy(rows, json.loads(Path(args.predictions).read_text())), indent=2)
            )


if __name__ == "__main__":
    main()
