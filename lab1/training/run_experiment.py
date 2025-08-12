#!/usr/bin/env python3
"""Experiment-running framework (Lightning 2.x compatible)."""

import argparse
import importlib
import numpy as np
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner

from text_recognizer import lit_models

warnings.filterwarnings("ignore", message=".*torch.cuda.amp.GradScaler.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
# Reproducibility
# np.random.seed(42)
# torch.manual_seed(42)
# seed_everything(42, workers=True)


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _resolve_lit_model_class(loss: str):
    """Pick a LitModel class based on loss name, with BaseLitModel as fallback."""
    if not loss:
        return lit_models.BaseLitModel
    lookup = {
        "ctc": "CTCLitModel",
        "transformer": "TransformerLitModel",
    }
    name = lookup.get(loss.lower())
    return getattr(lit_models, name) if name and hasattr(lit_models, name) else lit_models.BaseLitModel


def _setup_parser() -> argparse.ArgumentParser:
    """Set up ArgumentParser with trainer + data/model/litmodel args."""
    parser = argparse.ArgumentParser(
        description="FSDL Lab1 Runner (Lightning 2.x)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    # ---- Trainer args (explicit; replaces add_argparse_args / from_argparse_args)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accelerator", type=str, default="auto", help="auto/mps/cpu/gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default=None, help='e.g. "16-mixed"')
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    parser.add_argument("--auto_lr_find", action="store_true", help="Use PL Tuner to find LR before training")

    # ---- Basic experiment args
    parser.add_argument("--data_class", type=str, default="MNIST")
    parser.add_argument("--model_class", type=str, default="MLP")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Parse partial to discover data/model classes, then extend parser
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"text_recognizer.data.{temp_args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{temp_args.model_class}")

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    # Final help flag
    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.

    Example:
        python -m lab1.training.run_experiment --max_epochs=3 --model_class=MLP --data_class=MNIST
    """
    seed_everything(42, workers=True)
    parser = _setup_parser()
    args = parser.parse_args()

    # Resolve classes
    data_class = _import_class(f"text_recognizer.data.{args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{args.model_class}")

    # Instantiate data & model
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    # Choose LitModel (fallback to BaseLitModel if a specialized one isn't available)
    # BaseLitModel.add_to_argparse typically adds `--loss`; handle gracefully if absent.
    loss_name = getattr(args, "loss", None)
    lit_model_class = _resolve_lit_model_class(loss_name)

    # Build / load Lightning module
    if args.load_checkpoint:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    # Logging & callbacks
    logger = TensorBoardLogger(save_dir="training/logs", name="")
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=10),
        ModelCheckpoint(
            dirpath="training/logs",
            filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer (Lightning 2.x)
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
    )

    # Optional: Auto LR finder via Tuner (replaces old Trainer(auto_lr_find=...))
    if args.auto_lr_find:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(lit_model, datamodule=data)
        new_lr = lr_finder.suggestion()
        # Try common attribute names; ignore if not present
        if hasattr(lit_model, "hparams") and hasattr(lit_model.hparams, "lr"):
            lit_model.hparams.lr = new_lr
        elif hasattr(lit_model, "lr"):
            lit_model.lr = new_lr
        elif hasattr(lit_model, "learning_rate"):
            lit_model.learning_rate = new_lr
        print(f"[auto_lr_find] Using suggested lr={new_lr}")

    # Train & test
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
