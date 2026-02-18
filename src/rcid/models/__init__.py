"""Model adapters and loaders."""

from rcid.models.adapter import (
    LLaMA3Adapter,
    ModelAdapter,
    Qwen3Adapter,
    get_adapter,
)

__all__ = [
    "ModelAdapter",
    "Qwen3Adapter",
    "LLaMA3Adapter",
    "get_adapter",
    "load_teacher",
    "load_student",
    "load_student_from_checkpoint",
]


def __getattr__(name: str) -> object:
    """Lazy imports for heavy dependencies (transformers)."""
    if name == "load_teacher":
        from rcid.models.teacher import load_teacher
        return load_teacher
    if name == "load_student":
        from rcid.models.student import load_student
        return load_student
    if name == "load_student_from_checkpoint":
        from rcid.models.student import load_student_from_checkpoint
        return load_student_from_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
