"""Explicit lock refresh. Existing output pins are retained by uv unless reviewed otherwise."""

import argparse
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--profile", choices=["cpu", "cuda"], required=True)
a = p.parse_args()
output = "requirements-cpu-lock.txt" if a.profile == "cpu" else "requirements-lock.txt"
subprocess.run(
    [
        "uv",
        "pip",
        "compile",
        "pyproject.toml",
        "--extra",
        "dev",
        "--extra",
        "release",
        "--python-version",
        "3.11" if a.profile == "cpu" else "3.13",
        "--constraint",
        "release/torch-" + a.profile + ".txt",
        "--extra-index-url",
        "https://download.pytorch.org/whl/" + ("cpu" if a.profile == "cpu" else "cu130"),
        "--emit-index-url",
        "--output-file",
        output,
    ],
    check=True,
)
