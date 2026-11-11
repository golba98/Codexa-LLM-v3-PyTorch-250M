import argparse
import json
from pathlib import Path
from typing import Any, BinaryIO, cast


def main() -> None:
    import sys

    if sys.argv[1:3] == ["data", "stream-probe"]:
        from codexa.data.stream_probe import main as stream_probe_main

        sys.argv[1:3] = []
        stream_probe_main()
        return
    if sys.argv[1:3] == ["data", "language"]:
        from codexa.data.language import main as language_data_main

        language_data_main(sys.argv[3:])
        return
    if sys.argv[1:3] == ["language", "v1"]:
        from codexa.evaluation.language import main as language_v1_main

        language_v1_main(sys.argv[3:])
        return
    if sys.argv[1:3] == ["data", "v2"]:
        from codexa.data.v2 import main as v2_main

        v2_main(sys.argv[3:])
        return
    if sys.argv[1:2] == ["language"]:
        from codexa.evaluation.benchmark import main as language_main

        language_main(sys.argv[2:])
        return
    if sys.argv[1:2] == ["continuation"]:
        from codexa.continuation import main as continuation_main

        continuation_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        from codexa.dashboard.server import main as dashboard_main

        dashboard_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "context":
        from codexa.context import main as context_main

        context_main(sys.argv[2:])
        return
    if (
        len(sys.argv) > 2
        and sys.argv[1] in ("data", "tokenizer")
        and (
            sys.argv[2] in ("curate", "verify", "freeze", "sample", "evaluate")
            or "--dataset" in sys.argv
        )
    ):
        from codexa.data.commands import main as data_main

        data_main(sys.argv[1:])
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
    tokenizer_actions = commands.add_parser("tokenizer").add_subparsers(
        dest="action", required=True
    )
    tok = tokenizer_actions.add_parser("train", help="Fit BPE; production mode uses --dataset")
    for action in ("evaluate", "freeze"):
        tokenizer_actions.add_parser(action, help="Production tokenizer " + action)
    tok.add_argument("--corpus", required=True)
    tok.add_argument("--vocab-size", type=int, default=512)
    tok.add_argument("--output", required=True)
    data_actions = commands.add_parser("data").add_subparsers(dest="action", required=True)
    data = data_actions.add_parser("prepare", help="Tokenize; production mode uses --dataset")
    for action in ("sample", "curate", "verify"):
        data_actions.add_parser(action, help="Production dataset " + action)
    data.add_argument("--corpus", required=True)
    data.add_argument("--tokenizer", required=True)
    data.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", default="configs/smoke.yaml")
    train_data = train.add_mutually_exclusive_group(required=True)
    train_data.add_argument("--data")
    train_data.add_argument("--stream-config")
    train.add_argument("--tokenizer", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--resume")
    train.add_argument(
        "--forward-only",
        action="store_true",
        help="Verify configured streaming, model and checkpoint without updates",
    )
    train.add_argument(
        "--parent-checkpoint", help="Read-only parent load for a forward-only stream check"
    )
    train.add_argument(
        "--max-bytes", type=int, help="Required network cap for a forward-only streaming check"
    )
    train.add_argument(
        "--stream-check-windows",
        type=int,
        default=1,
        help="Bounded production-loader batches for a forward-only check",
    )
    train.add_argument("--checkpoint-volume-uuid")
    train.add_argument("--deterministic", action="store_true")
    train.add_argument(
        "--ui",
        choices=["auto", "terminal", "none"],
        default="auto",
        help="Auto: Rich terminal dashboard on a capable TTY; none: headless logs",
    )
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
    ev.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "tokenizer":
        docs, _ = split_documents(read_documents(args.corpus))
        CodexaTokenizer.train(docs, args.vocab_size).save(args.output)
    elif args.command == "data":
        prepare(args.corpus, args.tokenizer, args.output)
    elif args.command == "train":
        config = Config.load(args.config)
        streams = None
        if args.parent_checkpoint and not args.forward_only:
            raise ValueError("Parent loading here is restricted to forward-only checks")
        if not args.forward_only and (args.max_bytes is not None or args.stream_check_windows != 1):
            raise ValueError("Loader-check bounds require --forward-only")
        if args.forward_only:
            if not 1 <= args.stream_check_windows <= 2048:
                raise ValueError("Forward-only loader check exceeds bounded batch count")
            if (
                not args.stream_config
                or not args.max_bytes
                or not 0 < args.max_bytes <= 64 * 1024**2
            ):
                raise ValueError(
                    "Forward-only stream checks require a configuration and at most 64 MiB network cap"
                )
            if Path(args.output).exists():
                raise FileExistsError("Use a fresh forward-only output directory")
            from codexa.safety import forward_only

            forward_only()
        if args.stream_config:
            from codexa.data.provider_cache import CachedProvider
            from codexa.data.stream_policy import configured_streams
            from codexa.data.stream_probe import ByteBudget
            from codexa.data.streaming import open_provider

            budget = ByteBudget(args.max_bytes) if args.forward_only else None
            streams = configured_streams(
                args.stream_config,
                args.tokenizer,
                config.training.sequence,
                config.training.seed,
                require_launch=not args.forward_only,
                opener=(lambda shard: cast(BinaryIO, budget.wrap(open_provider(shard))))
                if budget
                else CachedProvider(
                    Path(args.output).parent / "stream-cache", args.checkpoint_volume_uuid
                ),
            )
        if args.deterministic:
            torch.use_deterministic_algorithms(True)
        trainer = Trainer(
            config,
            args.data or args.stream_config,
            args.tokenizer,
            args.output,
            args.device,
            checkpoint_volume_uuid=args.checkpoint_volume_uuid,
            stream_data=streams,
        )
        if args.stream_config:
            trainer.dataset_name = json.loads(Path(args.stream_config).read_text())["dataset"]
        if args.resume:
            trainer.resume(args.resume)
        if args.forward_only:
            if args.parent_checkpoint:
                from codexa.training.checkpoint import load

                parent = load(args.parent_checkpoint)
                if Config.from_dict(parent["config"]).model != config.model:
                    raise ValueError("Parent architecture differs from stream check")
                trainer.model.load_state_dict(parent["model"])
                trainer.optimizer.load_state_dict(parent["optimizer"])
            trainer.model.eval()
            import time

            if trainer.device.type == "cuda":
                torch.cuda.synchronize()
            loading_start = time.perf_counter()
            x, y = trainer.train_data.batch(
                trainer.sampler.next(config.training.batch_size), trainer.device
            )
            first_loaded = time.perf_counter()
            for _ in range(args.stream_check_windows - 1):
                trainer.train_data.batch(
                    trainer.sampler.next(config.training.batch_size), trainer.device
                )
            if trainer.device.type == "cuda":
                torch.cuda.synchronize()
            loading_finished = time.perf_counter()
            with torch.inference_mode(), trainer.amp():
                logits = trainer.model(x)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite stream forward")
            validation = trainer.validate()
            trainer.checkpoint("stream-check.pt")
            import hashlib

            batch_identity = hashlib.sha256(
                json.dumps(
                    torch.cat((x[:, :1], y), dim=1).cpu().tolist(), separators=(",", ":")
                ).encode()
            ).hexdigest()
            saved_stream = trainer.sampler.state_dict()
            following = trainer.sampler.next(config.training.batch_size)
            following_identity = hashlib.sha256(
                json.dumps(following, separators=(",", ":")).encode()
            ).hexdigest()
            trainer.sampler.load_state_dict(saved_stream)
            print(
                json.dumps(
                    {
                        "status": "forward-only",
                        "validation_loss": validation,
                        "optimizer_updates": 0,
                        "backward_calls": 0,
                        "network_bytes": budget.used if budget else 0,
                        "stream_windows": trainer.train_data.windows,
                        "batch_sha256": batch_identity,
                        "next_batch_sha256": following_identity,
                        "first_loader_batch_seconds": first_loaded - loading_start,
                        "subsequent_loader_seconds": loading_finished - first_loaded,
                        "subsequent_loader_target_positions": (args.stream_check_windows - 1)
                        * config.training.batch_size
                        * config.training.sequence,
                    }
                )
            )
            return
        trainer.train(until=args.until, ui=args.ui)
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
