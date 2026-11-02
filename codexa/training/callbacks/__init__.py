"""Read-only observer API; callback errors never replace training failures."""

from .telemetry import EventBus, TrainingEvent

__all__ = ["EventBus", "TrainingEvent"]
