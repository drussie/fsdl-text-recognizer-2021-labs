#!/usr/bin/env python3
"""MNIST DataModule compatible with Lightning 2.x + our BaseDataModule."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import random_split
from torchvision.datasets import MNIST as TVMNIST
import torchvision.transforms as T

from .base_data_module import BaseDataModule, load_and_print_info


class MNIST(BaseDataModule):
    """MNIST digits (28x28 grayscale, 10 classes)."""

    def __init__(self, args):
        super().__init__(args)
        # validation fraction (of the official training split)
        self.val_fraction: float = float(getattr(args, "val_fraction", 0.1))
        # mapping for pretty printing / decodes
        self.mapping = [str(i) for i in range(10)]
        # cache the download root
        self._root: Path = self.data_dir / "downloaded" / "MNIST"

    @classmethod
    def add_to_argparse(cls, parser):
        # inherit generic loader args from BaseDataModule
        parser = super().add_to_argparse(parser)
        group = parser.add_argument_group("MNIST Args")
        group.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of train set used for validation.")
        return parser

    # -------- Lightning hooks --------
    def prepare_data(self):
        """Download once on a single process."""
        TVMNIST(root=str(self._root), train=True, download=True)
        TVMNIST(root=str(self._root), train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Create train/val/test datasets."""
        transform = T.ToTensor()

        full_train = TVMNIST(root=str(self._root), train=True, download=False, transform=transform)
        n_total = len(full_train)
        n_val = int(self.val_fraction * n_total)
        n_train = n_total - n_val
        self.train_dataset, self.val_dataset = random_split(
            full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )

        self.test_dataset = TVMNIST(root=str(self._root), train=False, download=False, transform=transform)

        # optional: print a quick summary like the original labs
        load_and_print_info(self.train_dataset, self.val_dataset, self.test_dataset, limit=0)

    # -------- Model/config interface --------
    def config(self):
        """Small dict the model expects."""
        return {
            "input_dims": (1, 28, 28),   # channels, height, width
            "num_classes": 10,
            "output_dim": 10,            # alias some models look for
            "mapping": self.mapping,
        }
