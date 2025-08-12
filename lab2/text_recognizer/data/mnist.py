# lab2/text_recognizer/data/mnist.py
from __future__ import annotations

import argparse
from typing import Optional

import torch
from torch.utils.data import Subset
from torchvision.datasets import MNIST as TVMNIST
import torchvision.transforms as T

MEAN, STD = (0.1307,), (0.3081,)

# load_and_print_info is nice-to-have; make it optional
try:
    from .base_data_module import BaseDataModule, load_and_print_info  # type: ignore
    _HAVE_PRINT = True
except Exception:  # pragma: no cover
    from .base_data_module import BaseDataModule  # type: ignore
    _HAVE_PRINT = False


class MNIST(BaseDataModule):
    """28×28 grayscale digits with a simple config() API used by the models."""

    def __init__(self, args):
        super().__init__(args)
        # data split
        self.val_fraction: float = float(getattr(args, "val_fraction", 0.1))

        # augmentation + normalization knobs
        self.augment: bool = bool(getattr(args, "augment", True))
        self.aug_degrees: float = float(getattr(args, "aug_degrees", 10.0))
        self.aug_translate: float = float(getattr(args, "aug_translate", 0.10))  # fraction of W/H
        self.aug_scale_min: float = float(getattr(args, "aug_scale_min", 0.90))
        self.aug_scale_max: float = float(getattr(args, "aug_scale_max", 1.10))
        self.normalize: bool = bool(getattr(args, "normalize", True))

        self._root = self.data_dir / "downloaded" / "MNIST"
        self.mapping = [str(i) for i in range(10)]

    @classmethod
    def add_to_argparse(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().add_to_argparse(parser)
        parser.add_argument("--val_fraction", type=float, default=0.1)

        # BooleanOptionalAction lets you pass --no-augment / --no-normalize if you want
        try:
            BoolAction = argparse.BooleanOptionalAction
        except AttributeError:
            # Fallback for very old Python; keep interface usable via {0,1}
            BoolAction = None

        if BoolAction is not None:
            parser.add_argument("--augment", action=BoolAction, default=True)
            parser.add_argument("--normalize", action=BoolAction, default=True)
        else:
            parser.add_argument("--augment", type=int, choices=[0, 1], default=1)
            parser.add_argument("--normalize", type=int, choices=[0, 1], default=1)

        parser.add_argument("--aug_degrees", type=float, default=10.0)
        parser.add_argument("--aug_translate", type=float, default=0.10, help="fraction of width/height")
        parser.add_argument("--aug_scale_min", type=float, default=0.90)
        parser.add_argument("--aug_scale_max", type=float, default=1.10)
        return parser

    # -------- Lightning hooks --------
    def prepare_data(self) -> None:
        # Ensure files exist (Lightning calls on rank 0)
        TVMNIST(root=str(self._root), train=True, download=True)
        TVMNIST(root=str(self._root), train=False, download=True)

    def _build_transforms(self):
        """Create train/eval transforms based on CLI args."""
        # Train
        train_tfms: list = []
        if self.augment:
            train_tfms.append(
                T.RandomAffine(
                    degrees=self.aug_degrees,
                    translate=(self.aug_translate, self.aug_translate),
                    scale=(self.aug_scale_min, self.aug_scale_max),
                    fill=0,  # MNIST is black background
                )
            )
        train_tfms.append(T.ToTensor())
        if self.normalize:
            train_tfms.append(T.Normalize((0.1307,), (0.3081,)))

        # Eval/Val/Test (no augmentation)
        eval_tfms: list = [T.ToTensor()]
        if self.normalize:
            eval_tfms.append(T.Normalize((0.1307,), (0.3081,)))

        return T.Compose(train_tfms), T.Compose(eval_tfms)

    def setup(self, stage: Optional[str] = None) -> None:
        train_transform, eval_transform = self._build_transforms()

        if getattr(self.args, "augment", False):
            train_transform = T.Compose([
                T.RandomCrop(28, padding=2),
                T.RandomRotation(10, fill=0),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ])
        else:
            train_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ])

        eval_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

        # Build separate bases so train has aug but val does not
        full_train_aug = TVMNIST(root=str(self._root), train=True, download=False, transform=train_transform)
        full_train_plain = TVMNIST(root=str(self._root), train=True, download=False, transform=eval_transform)

        n_total = len(full_train_aug)
        n_val = int(self.val_fraction * n_total)
        n_train = n_total - n_val

        # Deterministic split
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(n_total, generator=g)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        self.train_dataset = Subset(full_train_aug, train_idx)
        self.val_dataset = Subset(full_train_plain, val_idx)

        # Test set (no augmentation)
        self.test_dataset = TVMNIST(root=str(self._root), train=False, download=False, transform=eval_transform)

        if _HAVE_PRINT:
            load_and_print_info(self.train_dataset, self.val_dataset, self.test_dataset, limit=0)

    # -------- Model/config interface --------
    def config(self):
        return {
            "input_dims": (1, 28, 28),
            "num_classes": 10,
            "mapping": self.mapping,
        }
