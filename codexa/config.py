from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    layers: int = 2
    dim: int = 64
    heads: int = 4
    intermediate: int = 176
    vocab_size: int = 512
    context: int = 64
    dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    rope_scaling: str = "none"
    rope_factor: float = 1.0
    attention_backend: str = "auto"

    def __post_init__(self) -> None:
        if min(self.layers, self.dim, self.heads, self.intermediate, self.context) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.dim % self.heads or (self.dim // self.heads) % 2:
            raise ValueError("Width must divide heads and head dimension must be even")
        if self.vocab_size < 260 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid vocabulary or dropout")
        if self.rope_scaling not in ("none", "linear") or self.rope_factor < 1:
            raise ValueError("Invalid RoPE scaling")
        if self.rope_scaling == "none" and self.rope_factor != 1:
            raise ValueError("Scaling factor requires linear scaling")
        if self.attention_backend not in ("auto", "flash", "efficient", "math"):
            raise ValueError("Invalid attention backend")
        if self.rope_base <= 0 or self.norm_eps <= 0:
            raise ValueError("RoPE base and norm epsilon must be positive")


@dataclass
class TrainConfig:
    steps: int = 200
    batch_size: int = 2
    accumulation: int = 1
    sequence: int = 64
    lr: float = 0.003
    min_lr: float = 0.0003
    warmup: int = 10
    weight_decay: float = 0.1
    clip: float = 1.0
    seed: int = 42
    validate_every: int = 25
    checkpoint_every: int = 50
    validation_batches: int = 4
    precision: str = "auto"
    gradient_checkpointing: bool = False
    checkpoint_every_tokens: int | None = None
    checkpoint_keep: int = 2

    def __post_init__(self) -> None:
        if type(self.checkpoint_keep) is not int or self.checkpoint_keep < 2:
            raise ValueError("Retain at least two recovery checkpoints")
        if self.checkpoint_every_tokens is not None and (
            type(self.checkpoint_every_tokens) is not int or self.checkpoint_every_tokens <= 0
        ):
            raise ValueError("Token checkpoint frequency must be a positive integer")
        if (
            min(
                self.steps,
                self.batch_size,
                self.accumulation,
                self.sequence,
                self.validate_every,
                self.checkpoint_every,
                self.validation_batches,
            )
            <= 0
        ):
            raise ValueError("Training counts must be positive")
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if not 0 <= self.warmup < self.steps or not 0 <= self.min_lr <= self.lr:
            raise ValueError("Invalid learning rate schedule")
        if (
            self.clip <= 0
            or self.weight_decay < 0
            or self.precision not in ("auto", "fp32", "bf16")
        ):
            raise ValueError("Invalid optimizer or precision configuration")


@dataclass
class InferenceConfig:
    max_sequence: int | None = None
    max_tokens: int = 50
    use_cache: bool = True
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if (
            self.max_tokens < 0
            or (self.max_sequence is not None and self.max_sequence < 1)
            or self.precision not in ("fp32", "bf16", "auto")
        ):
            raise ValueError("Invalid inference settings")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self) -> None:
        if (
            self.inference.max_sequence is not None
            and self.inference.max_sequence > self.model.context
        ):
            raise ValueError("Inference maximum exceeds model context")
        if self.training.sequence > self.model.context:
            raise ValueError("Training sequence exceeds context")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Config":
        return cls(
            ModelConfig(**value["model"]),
            TrainConfig(**value["training"]),
            InferenceConfig(**value.get("inference", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))
