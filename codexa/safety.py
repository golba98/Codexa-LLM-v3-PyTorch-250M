"""Explicit forward-only guard used by preparation and verification, never training."""

import os
from typing import Any

ATTEMPTS: list[str] = []
_installed = False


def require_training_allowed() -> None:
    if os.environ.get("CODEXA_NO_TRAINING") == "1":
        ATTEMPTS.append("training entry point")
        raise RuntimeError("Training is prohibited in this preparation process")


def forward_only() -> None:
    global _installed
    if _installed:
        return
    import torch
    import torch._dynamo  # Register inspected callables before installing distinct guards.

    def guard(label: str) -> Any:
        def blocked(*args: Any, **kwargs: Any) -> Any:
            ATTEMPTS.append(label)
            raise RuntimeError("Forward-only guard: " + label)

        return blocked

    def replace(target: Any, name: str, value: Any) -> None:
        setattr(target, name, value)

    replace(torch.Tensor, "backward", guard("Tensor.backward"))
    torch.autograd.backward = guard("autograd.backward")
    torch.autograd.grad = guard("autograd.grad")
    for name in dir(torch.optim):
        cls = getattr(torch.optim, name)
        if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer):
            replace(cls, "step", guard(name + ".step"))
    os.environ["CODEXA_NO_TRAINING"] = "1"
    torch.set_grad_enabled(False)
    _installed = True
