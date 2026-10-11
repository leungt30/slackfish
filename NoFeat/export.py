"""
Export a training checkpoint to inference-only files for playing and sharing.

    uv run python -m NoFeat.export Models/Weights/SlackFishV-T2-W-Lichess-6.41M/best.pt
    uv run python -m NoFeat.export --all        # every Models/Weights/*/best.pt

Writes next to the checkpoint:
    model.safetensors   the model weights only (float32), no optimizer state:
                        ~26 MB instead of ~77 MB for the 6.41M models
    config.json         model class and size, target and score scale, plus a
                        summary of how it was trained

Safetensors files hold only tensors, so loading one can't run code, unlike
torch.load on a pickled checkpoint. Keep best.pt / last.pt (with optimizer state)
locally if you want to resume training; the bots only need the two files above.
"""
import argparse
import glob
import json
import os

import torch
from safetensors.torch import save_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_FILE, CONFIG_FILE = "model.safetensors", "config.json"


def export(checkpoint: str) -> str:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    folder = os.path.dirname(os.path.abspath(checkpoint))
    state = {k: v.contiguous() for k, v in ckpt["model_state_dict"].items()}
    save_file(state, os.path.join(folder, WEIGHTS_FILE))

    last = (ckpt.get("history") or [{}])[-1]
    args = ckpt.get("args", {})
    config = {
        "model_class": ckpt.get("model_class", "ChessTransformerV1"),   # older checkpoints are V1
        "model_config": ckpt["model_config"],
        "target": ckpt.get("target", "cp"),
        "target_scale": ckpt["target_scale"],
        "target_clip": ckpt.get("target_clip"),
        "parameters": sum(v.numel() for v in state.values()),
        "training": {
            "epochs": ckpt.get("epoch"),
            "sources": args.get("sources"),
            "augment_prob": args.get("augment_prob"),
            "val_mae_cp": last.get("val_mae_cp"),
            "val_mae_cp_by_source": last.get("val_mae_cp_by_source"),
        },
    }
    with open(os.path.join(folder, CONFIG_FILE), "w") as f:
        json.dump(config, f, indent=2)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="*", help="best.pt files to export")
    parser.add_argument("--all", action="store_true", help="export every Models/Weights/*/best.pt")
    args = parser.parse_args()
    paths = args.checkpoints + (sorted(glob.glob(os.path.join(ROOT, "Models", "Weights", "*", "best.pt")))
                                if args.all else [])
    if not paths:
        parser.error("give a checkpoint or --all")
    for path in paths:
        folder = export(path)
        size = os.path.getsize(os.path.join(folder, WEIGHTS_FILE)) / 1e6
        print(f"{os.path.relpath(folder, ROOT)}: {WEIGHTS_FILE} ({size:.1f} MB) + {CONFIG_FILE}")


if __name__ == "__main__":
    main()
