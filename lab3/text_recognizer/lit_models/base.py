# lab3/text_recognizer/lit_models/base.py
# Modern Lightning-compatible BaseLitModel with robust CTC handling.
# - Uses torchmetrics directly (no pl.metrics)
# - If --loss=ctc_loss, reserves BLANK as the **last class index** by default
#   (or uses data_config['blank_index'] if provided) to avoid label collisions.
# - Works with variable-width line images packed in a batch (labels padded with -1).

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

try:
    # torchmetrics is the modern way
    from torchmetrics.classification import Accuracy
except Exception:  # pragma: no cover
    Accuracy = None  # type: ignore


class BaseLitModel(pl.LightningModule):
    """Generic Lightning wrapper for our models (classification or CTC)."""

    def __init__(self, args: Any, model: nn.Module):
        super().__init__()
        self.model = model
        self.args = args

        # ---- Optimizer hyperparams ----
        self.optimizer_name = getattr(args, "optimizer", "Adam")
        self.lr = float(getattr(args, "lr", 1e-3))
        self.one_cycle_max_lr = getattr(args, "one_cycle_max_lr", None)
        self.one_cycle_total_steps = getattr(args, "one_cycle_total_steps", None)

        # ---- Task/loss selection ----
        self.loss_name = (getattr(args, "loss", None) or "").lower()
        self.is_ctc = self.loss_name == "ctc_loss"

        if self.is_ctc:
            # Determine BLANK index. Prefer explicit data_config from the model.
            blank: Optional[int] = None
            try:
                dc = getattr(self.model, "data_config", None)
                if isinstance(dc, dict):
                    blank = dc.get("blank_index", None)
            except Exception:
                pass
            if blank is None:
                # Fallback: assume last class is the blank
                num_classes = getattr(self.model, "num_classes", None)
                if num_classes is not None:
                    blank = int(num_classes) - 1
            if blank is None:
                blank = 0  # ultimate fallback (not ideal, but prevents crashes)

            self.blank_index = int(blank)
            self.ctc_loss = nn.CTCLoss(blank=self.blank_index, zero_infinity=True)
        else:
            # Standard classification
            self.ce_loss = nn.CrossEntropyLoss()
            # Set up accuracy if available and the model exposes class count
            self.num_classes = int(getattr(self.model, "num_classes", 0) or 0)
            if Accuracy is not None and self.num_classes > 1:
                try:
                    self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
                    self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
                    self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
                except TypeError:
                    # Very old torchmetrics
                    self.train_acc = Accuracy()
                    self.val_acc = Accuracy()
                    self.test_acc = Accuracy()
            else:
                self.train_acc = None
                self.val_acc = None
                self.test_acc = None

    # ------------------------------------------------------------------
    # Lightning API
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        if self.is_ctc:
            x, y = batch  # y padded with -1 to max length
            logits = self(x)  # (B, T, C) expected from line models
            loss = self._ctc_step(logits, y, split="train")
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        else:
            x, y = batch
            logits = self(x)
            loss = self.ce_loss(logits, y)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            if self.train_acc is not None:
                preds = torch.argmax(logits, dim=1)
                self.train_acc.update(preds, y)
                self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
            return loss

    def validation_step(self, batch, batch_idx: int):
        if self.is_ctc:
            x, y = batch
            logits = self(x)
            loss = self._ctc_step(logits, y, split="val")
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        else:
            x, y = batch
            logits = self(x)
            loss = self.ce_loss(logits, y)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            if self.val_acc is not None:
                preds = torch.argmax(logits, dim=1)
                self.val_acc.update(preds, y)
                self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
            return loss

    def test_step(self, batch, batch_idx: int):
        if self.is_ctc:
            x, y = batch
            logits = self(x)
            loss = self._ctc_step(logits, y, split="test")
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        else:
            x, y = batch
            logits = self(x)
            loss = self.ce_loss(logits, y)
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            if self.test_acc is not None:
                preds = torch.argmax(logits, dim=1)
                self.test_acc.update(preds, y)
                self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
            return loss

    def configure_optimizers(self):
        # Robust optimizer resolution (accepts "adam" or "Adam")
        opt_cls = getattr(torch.optim, self.optimizer_name, None)
        if opt_cls is None:
            title_name = self.optimizer_name.title()  # adamw -> Adamw
            opt_cls = getattr(torch.optim, title_name, torch.optim.Adam)
        optimizer = opt_cls(self.parameters(), lr=self.lr)

        if self.one_cycle_max_lr is not None and self.one_cycle_total_steps is not None:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(self.one_cycle_max_lr),
                total_steps=int(self.one_cycle_total_steps),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "monitor": "val_loss",
                },
            }
        return optimizer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ctc_step(self, logits: torch.Tensor, y_padded: torch.Tensor, split: str) -> torch.Tensor:
        """Compute CTC loss.
        Args:
            logits: (B, T, C) raw scores
            y_padded: (B, Lmax) with -1 where padded
        Returns: scalar loss
        """
        # Normalize shape to (B,T,C). Some models return (B,C,T).
        if logits.dim() != 3:
            raise RuntimeError(f"Expected 3D logits for CTC, got shape {tuple(logits.shape)}")

        B, D1, D2 = logits.shape
        num_classes_guess = int(getattr(self.model, "num_classes", 0) or 0)
        if num_classes_guess > 0:
            if D1 == num_classes_guess and D2 != num_classes_guess:
                # (B, C, T) -> (B, T, C)
                logits = logits.transpose(1, 2).contiguous()
            elif D2 == num_classes_guess and D1 != num_classes_guess:
                # already (B, T, C)
                pass
            else:
                # Heuristic: if ambiguous, assume the larger dim is time
                if D1 > D2:
                    logits = logits.transpose(1, 2).contiguous()
        else:
            # Fallback heuristic if we can't guess the class dim
            if D1 > D2:
                logits = logits.transpose(1, 2).contiguous()

        B, T, C = logits.shape
        # Convert to log-probs and TNC as required by CTCLoss
        log_probs = F.log_softmax(logits, dim=-1)            # (B, T, C)
        log_probs = log_probs.permute(1, 0, 2).contiguous()  # (T, B, C)

        # Input lengths: each sequence has full T time-steps from the model
        input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)

        # Targets: flatten valid labels (y != -1)
        target_lengths = (y_padded != -1).sum(dim=1).to(torch.long)  # (B,)
        targets = y_padded[y_padded != -1].to(torch.long)            # (sum L_i,)
        # Sanity: blank must be within [0, C-1]
        if not (0 <= self.blank_index < C):
            raise RuntimeError(
                f"CTC blank index {self.blank_index} out of range for C={C}. "
                "Ensure your model's num_classes includes the blank (usually last)."
            )

        # --- MPS-friendly CTC ---
        # On Apple Silicon, the CTC kernel isn't implemented on MPS yet.
        # Do the heavy forward/backward on GPU, then compute CTCLoss on CPU.
        if log_probs.device.type == "mps":
            loss = self.ctc_loss(
                log_probs.cpu(),
                targets.cpu(),
                input_lengths.cpu(),
                target_lengths.cpu(),
            )
            return loss.to(self.device)
        else:
            loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
            return loss

    # Optional utility for decoding greediest path (could be used in validation logging)
    def ctc_greedy_decode(self, logits: torch.Tensor) -> List[List[int]]:
        """Greedy CTC decode: argmax over classes, collapse repeats, drop blank."""
        B, T, C = logits.shape
        probs = logits.softmax(dim=-1)
        pred = probs.argmax(dim=-1)  # (B, T)
        decoded: List[List[int]] = []
        for b in range(B):
            prev = -1
            seq: List[int] = []
            for t in range(T):
                k = int(pred[b, t].item())
                if k == self.blank_index:
                    prev = -1
                    continue
                if k != prev:
                    seq.append(k)
                    prev = k
            decoded.append(seq)
        return decoded

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default="Adam")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=None)
        parser.add_argument("--loss", type=str, default=None,
                            help="Use 'ctc_loss' for sequence models like LineCNN*")
        return parser
