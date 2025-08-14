# lab3/text_recognizer/lit_models/metrics.py
from __future__ import annotations

from typing import List, Sequence

import torch
from torchmetrics import Metric

try:
    import editdistance  # fast Levenshtein
except Exception:  # pragma: no cover
    editdistance = None


def _levenshtein(a: str, b: str) -> int:
    """Tiny DP fallback if editdistance isn't available."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,       # deletion
                dp[j - 1] + 1,   # insertion
                prev + cost,     # substitution
            )
            prev = cur
    return dp[m]


class CharacterErrorRate(Metric):
    """Average character error rate (CER) over a batch/list of strings.

    Accumulates total edit distance and total reference characters across updates:
        CER = sum(edit_distance(pred, target)) / sum(len(target))
    """
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Sequence[str], targets: Sequence[str]) -> None:  # type: ignore[override]
        if len(preds) != len(targets):
            raise ValueError("preds and targets must have the same length")

        err_sum = 0
        tot = 0
        for p, t in zip(preds, targets):
            if not isinstance(p, str) or not isinstance(t, str):
                raise TypeError("preds and targets must be sequences of strings")
            e = editdistance.eval(p, t) if editdistance is not None else _levenshtein(p, t)
            err_sum += e
            tot += len(t)

        self.errors += torch.tensor(float(err_sum), device=self.errors.device)
        self.total += torch.tensor(float(tot), device=self.total.device)

    def compute(self) -> torch.Tensor:  # type: ignore[override]
        if self.total.item() == 0:
            return torch.tensor(0.0, device=self.total.device)
        return self.errors / self.total
