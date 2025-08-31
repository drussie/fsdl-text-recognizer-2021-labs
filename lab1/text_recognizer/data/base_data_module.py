#!/usr/bin/env python3
"""
Base DataModule with fast DataLoader defaults for Apple Silicon / Lightning 2.x.

Includes backwards-compat helpers expected by older lab code:
- BaseDataModule.data_dirname()
- load_and_print_info(...)
"""

import os
from pathlib import Path
from typing import Optional, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from text_recognizer import util


class BaseDataModule(pl.LightningDataModule):
    """
    Minimal base class most Lab 1 DataModules inherit from.

    - Adds CLI args for batch size & DataLoader performance knobs
    - Provides consistent train/val/test DataLoader builders

    Subclasses are expected to set:
      self.train_dataset / self.val_dataset / self.test_dataset
    and optionally define a custom self.collate_fn or override hooks.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Generic knobs (with sensible defaults)
        self.batch_size: int = int(getattr(args, "batch_size", 64))
        self.num_workers: int = int(getattr(args, "num_workers", os.cpu_count() or 4) or 0)
        self.prefetch_factor: int = int(getattr(args, "prefetch_factor", 4) or 2)

        # Data root (Path). Prefer args.data_dir; otherwise use repo util.
        self.data_dir: Path = Path(getattr(args, "data_dir", None) or self.data_dirname())

        # Placeholders to be filled by subclasses in setup()
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # Optional custom collate function (subclasses may override)
        self.collate_fn = getattr(self, "collate_fn", None)

    # ----- CLI args -----
    @classmethod
    def add_to_argparse(cls, parser):
        group = parser.add_argument_group("DataLoader Args")
        group.add_argument("--batch_size", type=int, default=64, help="Per-device batch size.")
        group.add_argument(
            "--num_workers",
            type=int,
            default=os.cpu_count(),
            help="torch DataLoader workers (>=8 recommended on M-series).",
        )
        group.add_argument(
            "--prefetch_factor",
            type=int,
            default=4,
            help="Batches prefetched per worker (effective when num_workers > 0).",
        )
        group.add_argument(
            "--data_dir",
            type=str,
            default=None,
            help="Override dataset root directory (defaults to util.data_dirname()/data_dir).",
        )
        return parser

    # ----- Back-compat helper (class method) -----
    @classmethod
    def data_dirname(cls) -> Path:
        """
        Back-compat: return the root data directory as a Path.

        Older labs expose util.data_dirname(); some repos expose util.data_dir().
        Fallback to <cwd>/data if neither exists.
        """
        if hasattr(util, "data_dirname"):
            return Path(util.data_dirname())
        if hasattr(util, "data_dir"):
            return Path(util.data_dir())
        return Path.cwd() / "data"

    # ----- Optional hooks (subclasses typically override) -----
    def prepare_data(self):
        """Download/prepare data. Do not assign state here."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Create datasets. Subclasses should set train/val/test datasets here."""
        pass

    # ----- Internal helper for DataLoader kwargs -----
    def _dl_kwargs(self, shuffle: bool = False):
        kw = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,  # still a good default with MPS
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        if self.num_workers > 0:
            kw.update(
                persistent_workers=True,
                prefetch_factor=self.prefetch_factor,
            )
        return kw

    # ----- Standard Lightning hooks -----
    def train_dataloader(self):
        assert self.train_dataset is not None, "train_dataset not set in setup()"
        return DataLoader(self.train_dataset, **self._dl_kwargs(shuffle=True))

    def val_dataloader(self):
        assert self.val_dataset is not None, "val_dataset not set in setup()"
        return DataLoader(self.val_dataset, **self._dl_kwargs(shuffle=False))

    def test_dataloader(self):
        assert self.test_dataset is not None, "test_dataset not set in setup()"
        return DataLoader(self.test_dataset, **self._dl_kwargs(shuffle=False))

    # (Optional) some labs call this
    def config(self):
        """Subclasses usually override to return a dict with dataset-specific info."""
        return {}


# ---- Backwards-compat helper expected by older lab code ----

def _shape_of(x: Any):
    try:
        return tuple(getattr(x, "shape", None) or [])
    except Exception:
        return None


def load_and_print_info(*datasets, limit: int = 3):
    """
    Print quick stats about one or more datasets and return them unchanged.
    Accepts (x, y) samples or single tensors/images.
    Usage-compatible with older FSDL lab code that imported this helper.
    """
    for idx, ds in enumerate(datasets, 1):
        name = getattr(ds, "__class__", type(ds)).__name__
        try:
            n = len(ds)  # may raise if not sized
        except Exception:
            n = "?"
        print(f"[info] Dataset {idx}: {name} | size={n}")

        # Peek a few samples if possible
        if isinstance(n, int) and n > 0:
            for i in range(min(limit, n)):
                try:
                    sample = ds[i]
                except Exception:
                    break
                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    x, y = sample[0], sample[1]
                    print(f"  - sample[{i}]: x_shape={_shape_of(x)} y={y}")
                else:
                    print(f"  - sample[{i}]: shape={_shape_of(sample)} type={type(sample).__name__}")

    # Return exactly what was passed in, to preserve call sites
    if len(datasets) == 1:
        return datasets[0]
    return datasets
