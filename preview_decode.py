# preview_decode.py — batch padding fix for variable-width line images
#
# What this does
#  - Loads a CTC LineCNN checkpoint you trained with run_experiment.py
#  - Renders a few sample strings into 1×H×W tensors (W varies per string)
#  - **Pads** each tensor to the max width in the batch so stacking works
#  - Runs the model and prints a greedy CTC decode for each sample
#
# Usage (PowerShell on Windows, from your repo root):
#   $env:PYTHONPATH = "$PWD\lab3;$PWD"
#   python preview_decode.py --ckpt "lightning_logs\version_17\checkpoints\epoch=29-step=1200.ckpt" \
#       --samples 8 --img_height 28 --output_timesteps 64 --device cuda
#
# You can also pass explicit text with --texts "hello world|deep learning|pytorch lightning"
#
from __future__ import annotations

import argparse
import importlib
from types import SimpleNamespace
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------
# Helpers
# ---------------------------------------------

def render_line_gray(text: str, *, height: int = 28, pad_w: int = 8) -> torch.Tensor:
    """Render a single-line grayscale image (white bg, black text) to a 1×H×W float tensor in [0,1]."""
    text = text or ""
    try:
        font = ImageFont.truetype("Arial.ttf", size=max(10, height - 6))
    except Exception:
        font = ImageFont.load_default()

    # First measure text bbox to compute width
    dummy = Image.new("L", (1, 1), 255)
    d = ImageDraw.Draw(dummy)
    try:
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)
        w0, h0 = right - left, bottom - top
    except Exception:
        # Fallback for very old Pillow
        w0, h0 = d.textsize(text, font=font)

    W = max(1, w0 + 2 * pad_w)
    H = int(height)
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)
    y = max(0, (H - h0) // 2)
    draw.text((pad_w, y), text, font=font, fill=0)

    t = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                         .view(H, W).numpy()).float() / 255.0
    return t.unsqueeze(0)  # 1×H×W


def pad_stack(imgs: List[torch.Tensor]) -> torch.Tensor:
    """Right-pad each 1×H×W to max W with white (1.0), then stack to B×1×H×maxW."""
    assert len(imgs) > 0
    Hs = {int(t.shape[-2]) for t in imgs}
    if len(Hs) != 1:
        raise ValueError(f"All images must share the same height; got {sorted(Hs)}")
    H = imgs[0].shape[-2]
    maxW = max(int(t.shape[-1]) for t in imgs)
    padded = []
    for t in imgs:
        W = int(t.shape[-1])
        if W < maxW:
            t = F.pad(t, (0, maxW - W, 0, 0), value=1.0)
        padded.append(t)
    return torch.stack(padded, dim=0)  # B×1×H×maxW


# Default charset (should match what you trained with in run_experiment)
DEFAULT_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789 -',.")


def infer_ckpt_dims(state_dict: dict) -> Tuple[int, int]:
    """Return (num_classes, fc_dim) from typical LineCNN layer shapes in the checkpoint."""
    num_classes = None
    fc_dim = None
    # Look for the classifier layer
    for k, v in state_dict.items():
        if k.endswith("fc2.weight") and v.dim() == 2:
            num_classes = int(v.shape[0])
            fc_dim = int(v.shape[1])
            break
    if num_classes is None or fc_dim is None:
        # Fallback: try without the leading 'model.' or other prefixes
        for k, v in state_dict.items():
            if "fc2.weight" in k and v.dim() == 2:
                num_classes = int(v.shape[0])
                fc_dim = int(v.shape[1])
                break
    if num_classes is None:
        raise RuntimeError("Could not infer num_classes from checkpoint (no 'fc2.weight')")
    if fc_dim is None:
        # Try bias size as a hint
        for k, v in state_dict.items():
            if k.endswith("fc1.bias") and v.dim() == 1:
                fc_dim = int(v.shape[0])
                break
    if fc_dim is None:
        raise RuntimeError("Could not infer fc_dim from checkpoint")
    return num_classes, fc_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to Lightning .ckpt file")
    ap.add_argument("--samples", type=int, default=6, help="How many random samples to render")
    ap.add_argument("--texts", type=str, default=None,
                    help="Optional pipe-separated texts, e.g. 'hello|world|abc123'")
    ap.add_argument("--img_height", type=int, default=28)
    ap.add_argument("--output_timesteps", type=int, default=64)
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"])  # type: ignore[arg-type]
    ap.add_argument("--model_class", type=str, default="LineCNNSimple",
                    help="text_recognizer.models class name (default LineCNNSimple)")
    args = ap.parse_args()

    # Decide device
    dev = args.device
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(dev)

    # Load checkpoint as plain dict
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Infer classifier dims from checkpoint
    num_classes, fc_dim = infer_ckpt_dims(sd)
    print(f"[ckpt] inferred num_classes={num_classes}, fc_dim={fc_dim}")

    # Build data_config compatible with LineCNN
    mapping = DEFAULT_CHARS + ["<BLANK>"]
    if num_classes != len(mapping):
        # If your training used a different alphabet, just pad/crop to match
        if num_classes > len(mapping):
            mapping = (DEFAULT_CHARS + ["<BLANK>"] + [f"<extra{i}>" for i in range(num_classes - len(DEFAULT_CHARS) - 1)])[:num_classes]
        else:
            mapping = mapping[:num_classes]
    data_config = {
        "input_dims": (1, int(args.img_height), None),
        "output_dims": (int(args.output_timesteps), num_classes),
        "num_classes": num_classes,
        "mapping": mapping,
    }

    # Import and build the model
    model_cls = getattr(importlib.import_module("text_recognizer.models"), args.model_class)
    model = model_cls(data_config=data_config, args=SimpleNamespace(**{"fc_dim": fc_dim}))

    # Wrap in our Lightning-style BaseLitModel to reuse CTC decoding helpers
    from text_recognizer.lit_models.base import BaseLitModel
    lit = BaseLitModel(args=SimpleNamespace(loss="ctc_loss", optimizer="Adam", lr=1e-3), model=model)

    # Load weights (tolerate key name drift)
    missing, unexpected = lit.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)} (ok for preview)")

    lit.eval().to(device)

    # Choose texts
    if args.texts:
        texts = [t.strip() for t in args.texts.split("|") if t.strip()]
    else:
        seed_samples = [
            "the quick brown fox",
            "pytorch lightning",
            "hello world",
            "deep learning",
            "sequence to sequence",
            "fast and curious",
            "artificial intelligence",
            "recognize this line",
        ]
        # Repeat/crop to desired sample count
        need = int(args.samples)
        texts = (seed_samples * ((need + len(seed_samples) - 1) // len(seed_samples)))[:need]

    # Render -> pad -> stack
    imgs = [render_line_gray(t.lower(), height=args.img_height) for t in texts]
    X = pad_stack(imgs).to(device)

    # Forward
    with torch.no_grad():
        logits = lit.model(X)  # (B, T, C) or (B, C, T) — BaseLitModel handles both
        if logits.dim() != 3:
            raise RuntimeError(f"Model returned {tuple(logits.shape)}; expected 3D logits")
        # Normalize to (B, T, C)
        B, D1, D2 = logits.shape
        C_guess = data_config["num_classes"]
        if D1 == C_guess and D2 != C_guess:
            logits = logits.transpose(1, 2).contiguous()

        # Greedy CTC decode
        decoded = lit.ctc_greedy_decode(logits)

    # Map ids -> chars (ignore blank)
    idx_to_char = {i: ch for i, ch in enumerate(mapping[:-1])}
    def ids_to_text(ids: List[int]) -> str:
        return "".join(idx_to_char.get(i, "?") for i in ids)

    print("\n=== Preview ===")
    for t, ids in zip(texts, decoded):
        print(f"IN : {t}")
        print(f"OUT: {ids_to_text(ids)}\n")


if __name__ == "__main__":
    main()
