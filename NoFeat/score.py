"""
Score models on one validation set: mean absolute error in centipawns (both sides clipped
to +-1500, as NoFeat.train reports it), overall and by source, so models trained on
different splits can be compared on the same positions.

    uv run python -m NoFeat.score data/nofeat_shuffled Models/Weights/SlackFishV-T3-*/model.safetensors

The validation set is a shuffled folder's own (NoFeat.shuffle_data: its last positions),
or for an .npz / packed folder the split NoFeat.train makes with --seed and --val-frac.
A model file is an exported model.safetensors (with its config.json) or a training
checkpoint. Runs in float32; --limit scores only the first N validation positions.
"""
import argparse
import os

import numpy as np
import torch

from NoFeat import model as models
from NoFeat.evaluator import load_model
from NoFeat.train import batch_squares, load_data


def validation_set(data, args):
    """Indices of the validation positions, as NoFeat.train chooses them."""
    if "val" in data:
        n = len(data["eval"])
        return torch.arange(n - int(data["val"]), n)
    n = len(data["eval"])   # NoFeat.train's split of a whole (not --limit'ed) file
    dtype = torch.int32 if n > 50_000_000 else torch.int64
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed), dtype=dtype)
    return perm[:max(1, int(n * args.val_frac))].clone()


@torch.no_grad()
def score(path, data, idx, device, batch=4096):
    config, state = load_model(path, device)
    model = getattr(models, config.get("model_class", "ChessTransformerV1"))(**config["model_config"]).to(device)
    model.load_state_dict(state)
    model.eval()
    scale, clip = config["target_scale"], config.get("target_clip") or 1500
    squares, castling, turn = (torch.from_numpy(data[k]) for k in ("squares", "castling", "turn"))
    evals, errors = torch.from_numpy(data["eval"]), []
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        pred = model(batch_squares(squares, b, device), castling[b].to(device).long(), turn[b].to(device).long())
        cp = (pred.float().cpu() * scale).clamp(-clip, clip)
        errors.append((cp - evals[b].clamp(-clip, clip)).abs())
    return torch.cat(errors)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="a shuffled folder, or the .npz / packed folder the models were trained on")
    p.add_argument("models", nargs="+", help="model.safetensors or .pt files")
    p.add_argument("--val-frac", type=float, default=0.005, help="for an unshuffled --data: as in training")
    p.add_argument("--seed", type=int, default=42, help="for an unshuffled --data: as in training")
    p.add_argument("--limit", type=int, help="score only the first N validation positions")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args.data)
    idx = validation_set(data, args)[:args.limit]
    names = [str(x) for x in data["source_names"]]
    source = torch.from_numpy(np.asarray(data["source"][idx.numpy()]))
    present = [(i, name) for i, name in enumerate(names) if (source == i).any()]
    print(f"{len(idx):,} validation positions of {args.data}: "
          + ", ".join(f"{name} {int((source == i).sum()):,}" for i, name in present))
    for path in args.models:
        err = score(path, data, idx, device)
        by_source = "  ".join(f"{name} {err[source == i].mean():.2f}" for i, name in present)
        print(f"{os.path.relpath(path):70s} MAE {err.mean():.2f} cp   ({by_source})", flush=True)


if __name__ == "__main__":
    main()
