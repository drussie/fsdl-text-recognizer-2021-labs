# lab3/text_recognizer/lit_models/__init__.py
from .base import BaseLitModel
from .ctc import CTCLitModel


try:
    from .ctc import CTCLitModel  # legacy; not needed when using BaseLitModel + --loss=ctc_loss
except Exception:  # pragma: no cover
    CTCLitModel = None  # type: ignore
