# lab2/training/run_experiment.py
import argparse
import importlib
import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner

from text_recognizer import lit_models


# set plain seeds too (some labs do both)
np.random.seed(42)
torch.manual_seed(42)


def _import_class(path: str) -> type:
    """Import 'pkg.mod.Class' -> Class."""
    module, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def _setup_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)

    # Trainer args (superset of what we actually use)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", type=str, default=None)  # e.g. "16-mixed"
    p.add_argument("--log_every_n_steps", type=int, default=10)
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--num_sanity_val_steps", type=int, default=2)
    p.add_argument("--auto_lr_find", action="store_true", default=False)

    # Which data/model to use
    p.add_argument("--data_class", type=str, default="MNIST")
    p.add_argument("--model_class", type=str, default="MLP")
    p.add_argument("--load_checkpoint", type=str, default=None)

    # Pull in datamodule/model specific CLI
    tmp, _ = p.parse_known_args()
    data_cls = _import_class(f"text_recognizer.data.{tmp.data_class}")
    model_cls = _import_class(f"text_recognizer.models.{tmp.model_class}")

    data_group = p.add_argument_group("Data Args")
    data_cls.add_to_argparse(data_group)

    model_group = p.add_argument_group("Model Args")
    model_cls.add_to_argparse(model_group)

    lit_group = p.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_group)

    p.add_argument("--help", "-h", action="help")
    return p


def main():
    seed_everything(42, workers=True)

    parser = _setup_parser()
    args = parser.parse_args()

    data_cls = _import_class(f"text_recognizer.data.{args.data_class}")
    model_cls = _import_class(f"text_recognizer.models.{args.model_class}")

    data = data_cls(args)
    model = model_cls(data_config=data.config(), args=args)

    # choose lit wrapper
    lit_model_cls = lit_models.BaseLitModel
    lit_model = (
        lit_model_cls.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
        if args.load_checkpoint
        else lit_model_cls(args=args, model=model)
    )

    logger = TensorBoardLogger(save_dir="training/logs", name="")
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=10),
        ModelCheckpoint(filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}",
                        monitor="val_loss", mode="min"),
    ]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        callbacks=callbacks,
        logger=logger,
    )

    # LR finder (Lightning 2.x)
    if args.auto_lr_find:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(lit_model, datamodule=data)
        new_lr = lr_finder.suggestion()
        if new_lr is not None:
            print(f"[auto_lr_find] Using suggested lr={new_lr}")
            # many labs store LR in hparams
            if hasattr(lit_model, "hparams"):
                try:
                    lit_model.hparams.lr = float(new_lr)
                except Exception:
                    pass

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
