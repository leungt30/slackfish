"""
Train the NoFeat transformer on data produced by NoFeat.prepare_data.

    uv run python -m NoFeat.train --out Models/Weights/SlackFishV-T2 --epochs 10
    uv run python -m NoFeat.train --out Models/Weights/SlackFishV-T2 --resume Models/Weights/SlackFishV-T2/last.pt --epochs 20
    uv run python -m NoFeat.train --out /tmp/smoke --data data/nofeat_small.npz --epochs 2   # smoke test

Targets are clipped to +-CLIP centipawns and divided by SCALE before MSE.
Checkpoints (model, optimizer, config, losses) go to --out: last.pt after every
epoch and every --save-every steps within one, best.pt when validation loss
improves. --resume picks up at the exact step it stopped (each epoch's shuffle
is seeded, so the order is reproduced). Resuming with a larger --epochs
stretches the cosine LR schedule to the new length.

Augmentation: each training position is colour-swapped (board mirrored, colours,
castling and turn swapped, target negated) with probability --augment-prob, so
the model sees "White is winning" and "Black is winning" equally often. The
validation set is scored both as-is and mirrored; a large gap between the two
MAEs means the model evaluates the two colours inconsistently.

Data sources: --sources picks which CSVs to train on (games, random, tactics).
The validation split is drawn from every source, and MAE is reported per
source, so excluded sources (e.g. tactics) still show how the model handles them.
best.pt is chosen by validation MSE on the training sources only.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from NoFeat.augment import mirror_tokens
from NoFeat.model import MODELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/nofeat.npz")
    p.add_argument("--out", required=True, help="folder for this model, e.g. Models/Weights/SlackFishV-T2")
    p.add_argument("--sources", nargs="+", default=["games", "random"],
                   help="data sources to train on: games, random, tactics (default: games random)")
    p.add_argument("--limit", type=int, help="use a random sample of N positions (before the source filter)")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--clip", type=float, default=1500, help="clip targets to +-CLIP cp")
    p.add_argument("--scale", type=float, default=500, help="divide clipped targets by SCALE")
    p.add_argument("--arch", choices=list(MODELS), default="v1",
                   help="model class from NoFeat/model.py (v1 = ChessTransformerV1, v2 = ChessTransformerV2)")
    # Size options default to the chosen class's own defaults
    p.add_argument("--d-model", type=int)
    p.add_argument("--layers", type=int)
    p.add_argument("--heads", type=int)
    p.add_argument("--d-ff", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--augment-prob", type=float, default=0.5,
                   help="chance each training position is colour-swapped (0 = off)")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="skip torch.compile (compiling takes ~1 min but trains ~1.5x faster)")
    p.add_argument("--save-every", type=int, default=1000,
                   help="also write last.pt every N steps within an epoch (0 = only at epoch end)")
    p.add_argument("--resume", help="checkpoint to continue from")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, squares, castling, turn, targets, idx, args, device, mirror=False):
    """Per-position prediction errors (scaled units) for idx, as a CPU tensor."""
    model.eval()
    errors = []
    for i in range(0, len(idx), 1024):
        b = idx[i:i + 1024]
        sq, ca, tu = squares[b].to(device).long(), castling[b].to(device).long(), turn[b].to(device).long()
        y = targets[b].to(device)
        if mirror:
            sq, ca, tu = mirror_tokens(sq, ca, tu)
            y = -y
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(sq, ca, tu).float()
        errors.append((pred - y).cpu())
    model.train()
    return torch.cat(errors) if errors else torch.zeros(0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    data = np.load(args.data)
    total = len(data["eval"])
    n = total if args.limit is None else min(args.limit, total)
    # --limit takes a random sample, since the files are concatenated in order
    keep = (np.sort(np.random.default_rng(args.seed).choice(total, n, replace=False))
            if n < total else slice(None))
    squares = torch.from_numpy(data["squares"][keep])
    castling = torch.from_numpy(data["castling"][keep])
    turn = torch.from_numpy(data["turn"][keep])
    if "source" not in data:
        raise SystemExit(f"{args.data} has no source tags; re-run `uv run python -m NoFeat.prepare_data`")
    source = torch.from_numpy(data["source"][keep]).long()
    source_names = [str(x) for x in data["source_names"]]
    unknown = set(args.sources) - set(source_names)
    if unknown:
        raise SystemExit(f"unknown --sources {sorted(unknown)}; {args.data} has {source_names}")
    use = torch.tensor([name in args.sources for name in source_names])
    targets = torch.from_numpy(np.clip(data["eval"][keep], -args.clip, args.clip) / args.scale).float()

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_idx = train_idx[use[source[train_idx]]]           # train only on the chosen sources
    val_used = use[source[val_idx]]                          # val positions from those sources
    print(f"training on {', '.join(args.sources)}: {len(train_idx):,} train / "
          f"{val_used.sum().item():,} val (+{(~val_used).sum().item():,} val from other sources), device {device}")

    overrides = {k: v for k, v in dict(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                                       d_ff=args.d_ff, dropout=args.dropout).items() if v is not None}
    model = MODELS[args.arch](**overrides).to(device)
    print(f"{type(model).__name__} {model.config}: {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_idx) // args.batch_size   # drop the ragged last batch
    forward = torch.compile(model) if args.compile and device.type == "cuda" else model
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):   # linear warmup, then cosine decay to 10%
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    start_epoch, start_step, best_val, history = 1, 0, float("inf"), []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val = ckpt["best_val"]
        history = ckpt["history"]
        if ckpt.get("step") is not None:   # saved mid-epoch
            start_epoch, start_step = ckpt["epoch"], ckpt["step"]
            print(f"Resumed from {args.resume} (epoch {start_epoch}, step {start_step}/{steps_per_epoch})")
        else:
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from {args.resume} (after epoch {ckpt['epoch']})")

    if args.wandb:
        import wandb
        wandb.init(project="slackfish-nofeat", config=vars(args))

    def checkpoint(epoch, step=None):
        return {
            "epoch": epoch,
            "step": step,   # next step to run within `epoch`, None once the epoch is done
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_class": type(model).__name__,
            "model_config": model.config,
            "target_clip": args.clip,
            "target_scale": args.scale,
            "best_val": best_val,
            "history": history,
            "args": vars(args),
        }

    def save(ckpt, name):   # write then rename, so a crash mid-save can't corrupt the file
        path = os.path.join(args.out, name)
        torch.save(ckpt, path + ".tmp")
        os.replace(path + ".tmp", path)
        return path

    criterion = nn.MSELoss()
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        shuffle = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        order = train_idx[torch.randperm(len(train_idx), generator=shuffle)]
        swap = torch.rand(len(order), generator=shuffle) < args.augment_prob   # same draw on resume
        first = start_step if epoch == start_epoch else 0
        running, t0 = 0.0, time.time()
        bar = tqdm(range(first, steps_per_epoch), desc=f"epoch {epoch}/{args.epochs}",
                   initial=first, total=steps_per_epoch)
        for step in bar:
            b = order[step * args.batch_size:(step + 1) * args.batch_size]
            sq = squares[b].to(device, non_blocking=True).long()
            ca = castling[b].to(device, non_blocking=True).long()
            tu = turn[b].to(device, non_blocking=True).long()
            y = targets[b].to(device, non_blocking=True)
            if args.augment_prob > 0:
                m = swap[step * args.batch_size:(step + 1) * args.batch_size].to(device, non_blocking=True)
                msq, mca, mtu = mirror_tokens(sq, ca, tu)
                sq = torch.where(m[:, None], msq, sq)
                ca = torch.where(m, mca, ca)
                tu = torch.where(m, mtu, tu)
                y = torch.where(m, -y, y)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = forward(sq, ca, tu)
            loss = criterion(pred.float(), y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 10 == 0:   # .item() syncs with the GPU, so don't do it every step
                running = 0.9 * running + 0.1 * loss.item() if running else loss.item()
            if step % 50 == 0:
                bar.set_postfix(loss=f"{running:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
                if args.wandb:
                    wandb.log({"train_loss": running, "lr": scheduler.get_last_lr()[0]})
            if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < steps_per_epoch:
                save(checkpoint(epoch, step + 1), "last.pt")

        err = evaluate(model, squares, castling, turn, targets, val_idx, args, device)
        mirror_err = evaluate(model, squares, castling, turn, targets, val_idx[val_used], args, device, mirror=True)
        val_mse = (err[val_used] ** 2).mean().item()
        val_mae_cp = err[val_used].abs().mean().item() * args.scale
        mirror_mae_cp = mirror_err.abs().mean().item() * args.scale
        by_source = {name: err[source[val_idx] == i].abs().mean().item() * args.scale
                     for i, name in enumerate(source_names) if (source[val_idx] == i).any()}
        history.append({"epoch": epoch, "train_loss": running, "val_mse": val_mse, "val_mae_cp": val_mae_cp,
                        "val_mirror_mae_cp": mirror_mae_cp, "val_mae_cp_by_source": by_source})
        per_source = "  ".join(f"{k}{'' if k in args.sources else '*'} {v:.0f}" for k, v in by_source.items())
        print(f"epoch {epoch}: train {running:.4f}  val MSE {val_mse:.4f}  val MAE {val_mae_cp:.0f} cp  "
              f"mirrored {mirror_mae_cp:.0f} cp  ({time.time() - t0:.0f}s)\n"
              f"  MAE by source: {per_source}   (* = not trained on)")
        if args.wandb:
            wandb.log({"epoch": epoch, "val_mse": val_mse, "val_mae_cp": val_mae_cp,
                       "val_mirror_mae_cp": mirror_mae_cp,
                       **{f"val_mae_cp/{k}": v for k, v in by_source.items()}})

        is_best = val_mse < best_val
        best_val = min(best_val, val_mse)
        ckpt = checkpoint(epoch)
        save(ckpt, "last.pt")
        if is_best:
            print(f"  new best -> {save(ckpt, 'best.pt')}")


if __name__ == "__main__":
    main()
