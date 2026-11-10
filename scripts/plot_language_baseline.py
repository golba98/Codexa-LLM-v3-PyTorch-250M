"""Render preserved baseline measurements; uses Matplotlib separately from model pins."""

import argparse
import hashlib
import json
import platform
from pathlib import Path


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.baseline / "summary.json").read_text())
    proxy = json.loads((args.baseline / "answer-proxies.json").read_text())["scores"]
    lengths = json.loads((args.baseline / "length-controls.json").read_text())["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax = axes[0]
    labels = ["Candidate sum", "Candidate mean", "Free greedy proxy", "Free sampled proxy"]
    for i, split in enumerate(("dev", "test")):
        values = [
            summary[f"{split}:fact:candidate:42"]["correct_by_sum"],
            summary[f"{split}:fact:candidate:42"]["correct_by_mean"],
            proxy[f"{split}:fact:greedy"]["accuracy"] * 100,
            proxy[f"{split}:fact:sample"]["accuracy"] * 100,
        ]
        positions = [j + (i - 0.5) * 0.34 for j in range(4)]
        bars = ax.bar(positions, values, width=0.32, label=split, color=("#2463a0", "#df8a28")[i])
        ax.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=3)
    ax.set_xticks(range(4), labels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent (100 factual items per split)")
    ax.set_title("Candidate ranking overstates free completion")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax = axes[1]
    x = [64, 128, 256]
    for split, color in (("dev", "#2463a0"), ("test", "#df8a28")):
        values = [lengths[f"{split}:sample:42"]["prefix_loops"][str(n)] for n in x]
        all_values = [
            [lengths[f"{split}:sample:{seed}"]["prefix_loops"][str(n)] for seed in (42, 123, 2026)]
            for n in x
        ]
        ax.plot(x, values, "o-", label=f"{split}, seed 42", color=color)
        ax.fill_between(
            x, [min(v) for v in all_values], [max(v) for v in all_values], color=color, alpha=0.15
        )
    ax.axhline(5, color="#527b4a", linestyle="--", label="Proposed maximum 5%")
    ax.set_xticks(x)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Generated prefix length (tokens)")
    ax.set_ylabel("Outputs with detected severe loops (%)")
    ax.set_title("Sampled repetition increases with length")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Codexa v3: unchanged 500M-token checkpoint", fontsize=14)
    fig.supxlabel(
        "Read-only baseline. Free answers use a lexical opening proxy, not human factual accuracy. Shading spans three seeds; it is not a confidence interval.",
        fontsize=9,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"language-baseline.{extension}", dpi=180)
    plt.close(fig)
    metadata = {
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "summary_sha256": hashlib.sha256((args.baseline / "summary.json").read_bytes()).hexdigest(),
        "outputs_sha256": hashlib.sha256(
            (args.baseline / "outputs.jsonl").read_bytes()
        ).hexdigest(),
        "scope": "CPU plotting of preserved measurements; no model execution or dependency changes",
    }
    (args.output / "plot-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
