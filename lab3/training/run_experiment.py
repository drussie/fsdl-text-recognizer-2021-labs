# lab3/training/run_experiment.py
# Works with modern PyTorch Lightning; supports SentenceGenerator by rendering
# on-the-fly into line images and encoding labels. Adds <BLANK> for CTC.

from __future__ import annotations

import argparse
import importlib
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont

import pytorch_lightning as pl

# ---------------- Utils ----------------

def _seed_everything(seed: int = 42):
    try:
        pl.seed_everything(seed, workers=True)
    except Exception:
        pl.seed_everything(seed)


def _import_class(qualname: str):
    """Import class from 'pkg.mod.Class'."""
    module, cls = qualname.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


# ---------------- Simple text -> image helpers ----------------

def _render_line_gray(text: str, *, height: int = 28, pad_w: int = 8) -> Image.Image:
    """Render a single-line grayscale image (white background, black text)."""
    font = ImageFont.load_default()
    # measure with textbbox (new API)
    dummy = Image.new("L", (1, 1), 255)
    d = ImageDraw.Draw(dummy)
    left, top, right, bottom = d.textbbox((0, 0), text, font=font)
    w0, h0 = right - left, bottom - top

    # scale width roughly to requested height
    scale = max(1.0, (height - 4) / max(1, h0))
    w = int(max(1, math.ceil(w0 * scale))) + 2 * pad_w
    img = Image.new("L", (w, height), 255)
    d = ImageDraw.Draw(img)
    y = max(0, (height - int(h0 * scale)) // 2)
    x = pad_w
    d.text((x, y), text, font=font, fill=0)
    return img


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.uint8)
    t = torch.from_numpy(arr).float() / 255.0
    return t.unsqueeze(0)  # (1, H, W)

# ---------------- Synthetic dataset from SentenceGenerator ----------------

@dataclass
class _GenCfg:
    img_height: int = 28
    max_length: int = 24
    retries: int = 50


class RenderedLinesFromGenerator(Dataset):
    """Calls a text generator on-the-fly and returns (image_tensor, label_tensor)."""

    def __init__(self, generator, *, length: int, cfg: _GenCfg, mapping: List[str]):
        self.generator = generator
        self.length = int(length)
        self.cfg = cfg

        # build char->id table (assumes mapping are single chars + special tokens)
        self.mapping = list(mapping)
        self.char2id = {c: i for i, c in enumerate(self.mapping) if len(c) == 1}
        self.unk_id = self.mapping.index("<UNK>") if "<UNK>" in self.mapping else len(self.mapping) - 2

    def __len__(self) -> int:
        return self.length

    def _encode(self, s: str) -> torch.Tensor:
        ids = [self.char2id.get(c, self.unk_id) for c in s]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int):
        last_err = None
        for _ in range(max(1, self.cfg.retries)):
            try:
                s = self.generator.generate(max_length=self.cfg.max_length)
                if not isinstance(s, str) or len(s.strip()) == 0:
                    raise RuntimeError("Empty string from generator")
                s = s.lower()
                img = _render_line_gray(s, height=self.cfg.img_height)
                x = _pil_to_tensor(img)  # (1,H,W)
                y = self._encode(s)      # (L,)
                return x, y
            except Exception as e:
                last_err = e
        raise RuntimeError(f"SentenceGenerator kept failing; last error: {repr(last_err)}")


# ---------------- Collate: pad images and labels ----------------

def _pad_and_pack(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]], ignore_index: int = -1):
    xs, ys = zip(*batch)
    B = len(xs)
    C, H = xs[0].shape[0], xs[0].shape[1]
    maxW = max(x.shape[2] for x in xs)
    maxL = max(y.numel() for y in ys)

    xbat = xs[0].new_ones((B, C, H, maxW))  # white background = 1.0
    ybat = torch.full((B, maxL), fill_value=ignore_index, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        W = x.shape[2]
        xbat[i, :, :, :W] = x
        ybat[i, : y.numel()] = y
    return xbat, ybat   # ybat is LongTensor padded with -1


# ---------------- Loader kwargs ----------------

def _loader_kwargs(args, *, is_train: bool):
    nw = int(getattr(args, "num_workers", 0) or 0)
    kw = dict(
        batch_size=int(args.batch_size),
        shuffle=bool(is_train),
        num_workers=nw,
        pin_memory=False,
        persistent_workers=(nw > 0),
        drop_last=False,
        collate_fn=_pad_and_pack,
    )
    pf = getattr(args, "prefetch_factor", None)
    if nw > 0 and pf is not None:
        kw["prefetch_factor"] = int(pf)
    return kw


# ---------------- Config + mapping helpers ----------------

def _alphabet_default() -> List[str]:
    import string
    # lowercase letters + digits + common punctuation + space
    chars = list(string.ascii_lowercase + string.digits + " .,'-!?;:")
    # add specials at the end in this order: <UNK>, <BLANK>
    return chars + ["<UNK>", "<BLANK>"]


def _build_data_config(args, *, mapping: List[str], sample_x: torch.Tensor) -> dict:
    H, W = sample_x.shape[-2], sample_x.shape[-1]
    T_est = max(8, W // 4)  # rough width downsample of ~4x for LineCNNs
    T = int(getattr(args, "output_timesteps", T_est) or T_est)
    return {
        "input_dims": (1, H, W),
        "output_dims": (T, len(mapping)),
        "num_classes": len(mapping),
        "mapping": mapping,
        "blank_index": len(mapping) - 1,  # last index is <BLANK>
    }


# ---------------- CLI ----------------

def _setup_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # Trainer
    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--accelerator", type=str, default=None)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", type=str, default=None)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--num_sanity_val_steps", type=int, default=2)

    # Data / model
    p.add_argument("--data_class", type=str, default="sentence_generator.SentenceGenerator",
                   help="ModulePath.ClassName under text_recognizer.data (or fully-qualified)")
    p.add_argument("--model_class", type=str, default="LineCNNSimple",
                   help="Class under text_recognizer.models (or fully-qualified)")

    # Loader knobs
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)

    # Synthetic specifics
    p.add_argument("--sentence_max_length", type=int, default=24)
    p.add_argument("--line_image_height", type=int, default=28)

    # Optimization
    p.add_argument("--optimizer", type=str, default="Adam")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--loss", type=str, default="ctc_loss", choices=["ctc_loss", "cross_entropy"])

    # Model extras
    p.add_argument("--conv_dim", type=int, default=64)
    p.add_argument("--fc_dim", type=int, default=256)
    p.add_argument("--output_timesteps", type=int, default=None)
    return p


# ---------------- Main ----------------

def main():
    _seed_everything(42)
    args = _setup_parser().parse_args()

    if (args.accelerator or "").lower() == "mps" and args.loss == "ctc_loss":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1":
            print("[warn] Using MPS with CTC. Consider `export PYTORCH_ENABLE_MPS_FALLBACK=1`.")

    # Resolve data class under text_recognizer.data unless fully-qualified
    dc = args.data_class
    if "." not in dc or not dc.startswith("text_recognizer."):
        dc = f"text_recognizer.data.{dc}"
    data_cls = _import_class(dc)
    data_obj = data_cls()  # SentenceGenerator() typically takes no args

    # Build mapping and datasets
    mapping = _alphabet_default()                         # [..., "<UNK>", "<BLANK>"]
    gen_cfg = _GenCfg(img_height=int(args.line_image_height),
                      max_length=int(args.sentence_max_length))
    train_ds = RenderedLinesFromGenerator(data_obj, length=20000, cfg=gen_cfg, mapping=mapping)
    val_ds   = RenderedLinesFromGenerator(data_obj, length=2000,  cfg=gen_cfg, mapping=mapping)
    test_ds  = RenderedLinesFromGenerator(data_obj, length=2000,  cfg=gen_cfg, mapping=mapping)

    print(f"[info] Dataset 1: {train_ds.__class__.__name__} | size={len(train_ds)}")
    print(f"[info] Dataset 2: {val_ds.__class__.__name__} | size={len(val_ds)}")
    print(f"[info] Dataset 3: {test_ds.__class__.__name__} | size={len(test_ds)}")

    # Infer config from one sample
    sample_x, _ = train_ds[0]
    data_config = _build_data_config(args, mapping=mapping, sample_x=sample_x)

    # Model
    mc = args.model_class
    if "." not in mc:
        mc = f"text_recognizer.models.{mc}"
    model_cls = _import_class(mc)
    model = model_cls(data_config=data_config, args=args)

    # Lightning wrapper (our BaseLitModel handles both CE and CTC)
    from text_recognizer import lit_models
    lit_model = lit_models.BaseLitModel(args=args, model=model)

    # Loaders
    train_loader = DataLoader(train_ds, **_loader_kwargs(args, is_train=True))
    val_loader   = DataLoader(val_ds,   **_loader_kwargs(args, is_train=False))
    test_loader  = DataLoader(test_ds,  **_loader_kwargs(args, is_train=False))

    # Trainer
    tr_kwargs = dict(
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
    )
    if args.accelerator:
        tr_kwargs["accelerator"] = args.accelerator
    if args.devices:
        tr_kwargs["devices"] = args.devices
    if args.precision:
        tr_kwargs["precision"] = args.precision
    trainer = pl.Trainer(**tr_kwargs)

    # Train & test
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
