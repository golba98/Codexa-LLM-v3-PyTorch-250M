import argparse
import json
import subprocess

from tokenizers import Tokenizer

from codexa.architecture.model import Codexa
from codexa.config import Config
from codexa.data.packing import read_documents, split_documents
from codexa.tokenizer.bpe import CodexaTokenizer
from codexa.training.checkpoint import load
from codexa.training.trainer import seed_all

from .artifacts import export_model, load_export
from .preflight import preflight


def dispatch(args: argparse.Namespace) -> None:
    if args.command == "release" and args.action == "preflight":
        result = preflight(args.artifact, args.verification_report)
        print(json.dumps(result, indent=2))
        return
    if args.command == "release":
        seed_all(42)
        docs, _ = split_documents(read_documents(args.corpus))
        tokenizer = CodexaTokenizer.train(docs, 512)
        model = Codexa(Config().model)
        metadata = {
            "stage": "fixture",
            "version": "0.0.0-fixture.1",
            "training_origin": "random_initialization",
            "training_tokens": 0,
            "datasets": "Original CC0 smoke corpus used only to train BPE; no model training",
            "compute": "CPU random initialization only",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "evaluation": "Serialization tests only; no capability score",
            "uses": "Local release pipeline testing only; not a trained model",
            "weight_license": "Not approved for distribution; fixture only",
        }
        print(export_model(model, tokenizer, metadata, args.output))
        return
    if args.command == "evaluate" and args.export:
        model, _, _ = load_export(args.export)
    else:
        state = load(args.checkpoint)
        model = Codexa(Config.from_dict(state["config"]).model)
        model.load_state_dict(state["model"])
        tokenizer = CodexaTokenizer(Tokenizer.from_str(state["tokenizer_json"]))
    if args.command == "evaluate":
        from codexa.evaluation.language_model import evaluate

        print(json.dumps(evaluate(model, args.data, args.max_batches, args.split), indent=2))
    else:
        from pathlib import Path

        metadata = json.loads(Path(args.metadata).read_text())
        if metadata.get("training_tokens") != state["tokens"]:
            raise ValueError("Training token claim differs from checkpoint")
        metadata.setdefault("data_fingerprints", state.get("fingerprints", {}))
        print(export_model(model, tokenizer, metadata, args.output))
