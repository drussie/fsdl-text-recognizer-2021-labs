# lab3/text_recognizer/lit_models/ctc.py
# Thin wrapper kept for backwards compatibility. All CTC logic lives in BaseLitModel.

from __future__ import annotations
from .base import BaseLitModel


class CTCLitModel(BaseLitModel):
    """CTC training wrapper.

    Intentionally empty: BaseLitModel already:
      - sets up nn.CTCLoss with a safe blank index (last class by default),
      - computes CPU fallback for CTC on MPS,
      - handles variable-length targets (-1 padded) in _ctc_step,
      - logs train/val/test losses.
    """
    pass
