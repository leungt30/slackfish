"""
Train the NoFeat transformer on data produced by NoFeat.prepare_data.

    uv run python -m NoFeat.train --out Models/Weights/SlackFishV-T2 --epochs 10
    uv run python -m NoFeat.train --out Models/Weights/SlackFishV-T2 --resume Models/Weights/SlackFishV-T2/last.pt --epochs 20
    uv run python -m NoFeat.train --out /tmp/smoke --data data/nofeat_small.npz --epochs 2   # smoke test

Targets (--target):
  cp       centipawns clipped to +-CLIP and divided by SCALE, trained with MSE.
  winprob  the model outputs a logit; the target is sigmoid(cp / WP_SCALE), the
           win probability for White, trained with binary cross-entropy. Lopsided
           positions saturate near 0/1 instead of dominating the loss, and forced
           mates need no clipping. In play, logit * WP_SCALE is used as a
           centipawn-like score.
Either way the log reports MAE in centipawns (clipped to +-CLIP) so runs compare.
Checkpoints (model, optimizer, config, losses) go to --out: last.pt after every
epoch and every --save-every steps within one, best.pt when validation loss
improves. --resume picks up at the exact step it stopped (each epoch's shuffle
is seeded, so the order is reproduced). Resuming with a larger --epochs
stretches the LR schedule to the new length.

LR schedule: linear warmup over --warmup-steps, then down to --lr-floor x --lr by
the last step, either along a cosine (--schedule cosine, the default) or, with
--schedule wsd (warmup-stable-decay), flat at --lr and then a straight line over
the last --decay-frac of the steps. A WSD run resumed with a larger --epochs
before its decay starts carries on as if it had been planned that long.

Augmentation: each training position is colour-swapped (board mirrored, colours,
castling and turn swapped, target negated) with probability --augment-prob, so
the model sees "White is winning" and "Black is winning" equally often. The
validation set is scored both as-is and mirrored; a large gap between the two
MAEs means the model evaluates the two colours inconsistently. --arch v4 sees every
position from the side to move, so a colour swap doesn't change its input: augmentation
is turned off and the mirrored MAE is the plain one (not scored twice).

Precision (--precision): mixed precision on CUDA. auto uses bf16 on GPUs with
native bf16 (Ampere and newer) and fp16 with loss scaling on older ones such as
Turing (RTX 20xx), where bf16 is emulated and several times slower.

Data (--data): an .npz from NoFeat.prepare_data, or a packed folder from it
(squares stored two per byte, unpacked on the GPU per batch), which is how all
~410M Lichess positions fit in RAM. --val-every N also scores a fixed
--val-subset of the validation set every N steps, so a long single epoch still
gives a learning curve (stored as "curve" in the checkpoint).

Data sources: --sources picks which CSVs to train on (games, random, tactics).
The validation split is drawn from every source, and MAE is reported per
source, so excluded sources (e.g. tactics) still show how the model handles them.
best.pt is chosen by validation MSE on the training sources only.

Multiple GPUs: launch with torchrun, one process per GPU (DistributedDataParallel):

    uv run torchrun --nproc-per-node 4 -m NoFeat.train --out ... (same options)

--batch-size stays the global batch: each GPU takes an equal slice of every step's
batch and gradients are averaged, so a run on N GPUs follows the same steps and LR
schedule as on one, and checkpoints resume on any number of GPUs. A larger global
batch (with a larger --lr) is what makes more GPUs pay off. Validation is split
across the GPUs; only rank 0 logs and writes checkpoints. A packed --data folder is
memory-mapped, so the processes share one copy of it in the page cache.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from NoFeat.augment import mirror_tokens
from NoFeat.encode import unpack_squares
from NoFeat.model import MODELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/nofeat.npz")
    p.add_argument("--out", required=True, help="folder for this model, e.g. Models/Weights/SlackFishV-T2")
    p.add_argument("--sources", nargs="+", default=["games", "random"],
                   help="data sources to train on: games, random, tactics (default: games random)")
    p.add_argument("--limit", type=int, help="use a random sample of N positions (before the source filter)")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--val-every", type=int, default=0,
                   help="also score --val-subset validation positions every N steps (0 = only per epoch)")
    p.add_argument("--val-subset", type=int, default=200_000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine",
                   help="after warmup: cosine decay, or wsd (warmup-stable-decay: flat, then a linear decay)")
    p.add_argument("--decay-frac", type=float, default=0.2,
                   help="wsd: decay over this last fraction of all steps (default 0.2)")
    p.add_argument("--lr-floor", type=float, default=0.1,
                   help="final LR as a fraction of --lr (default 0.1)")
    p.add_argument("--target", choices=["cp", "winprob"], default="cp",
                   help="regress clipped centipawns (cp) or win probability (winprob)")
    p.add_argument("--clip", type=float, default=1500, help="clip cp targets (and reported MAE) to +-CLIP")
    p.add_argument("--scale", type=float, default=500, help="cp target: divide clipped targets by SCALE")
    p.add_argument("--wp-scale", type=float, default=400,
                   help="winprob target: win probability = sigmoid(cp / WP_SCALE)")
    p.add_argument("--arch", choices=list(MODELS), default="v1",
                   help="model class from NoFeat/model.py (v1 ... v4 = ChessTransformerV1 ... V4)")
    # Size options default to the chosen class's own defaults
    p.add_argument("--d-model", type=int)
    p.add_argument("--layers", type=int)
    p.add_argument("--heads", type=int)
    p.add_argument("--d-ff", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--spatial-bias", action="store_true",
                   help="v3 / v4 only: learned per-head attention bias (66x66 / 64x64) in every layer")
    p.add_argument("--smolgen", action="store_true",
                   help="v3 + --spatial-bias only: add a position-dependent bias (Lc0's smolgen) to it")
    p.add_argument("--augment-prob", type=float, default=0.5,
                   help="chance each training position is colour-swapped (0 = off)")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="skip torch.compile (compiling takes ~1 min but trains ~1.5x faster)")
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="mixed precision on CUDA (auto: bf16 if the GPU supports it natively, else fp16)")
    p.add_argument("--save-every", type=int, default=1000,
                   help="also write last.pt every N steps within an epoch (0 = only at epoch end)")
    p.add_argument("--resume", help="checkpoint to continue from")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, squares, castling, turn, targets, idx, args, device, mirror=False, world=1, rank=0):
    """(predictions, targets) for idx in model units, as CPU tensors. With several
    processes each scores its share of idx and the predictions are gathered."""
    model.eval()
    preds = []
    shard = idx.tensor_split(world)[rank]
    for i in range(0, len(shard), 1024):
        b = shard[i:i + 1024]
        sq, ca, tu = batch_squares(squares, b, device), castling[b].to(device).long(), turn[b].to(device).long()
        if mirror:
            sq, ca, tu = mirror_tokens(sq, ca, tu)
        with autocast(args, device):
            preds.append(model(sq, ca, tu).float())
    model.train()
    pred = torch.cat(preds) if preds else torch.zeros(0, device=device)
    if world > 1:
        pred = gather_shards(pred, len(idx), world)
    y = targets[idx]
    return pred.cpu(), -y if mirror else y


def gather_shards(part, n, world):
    """Every process's part of idx.tensor_split(world), concatenated in order."""
    sizes = [n // world + (r < n % world) for r in range(world)]
    padded = part.new_zeros(sizes[0])   # all_gather needs equal sizes; shard 0 is the largest
    padded[:len(part)] = part
    out = part.new_empty(world * sizes[0])
    dist.all_gather_single(out, padded)
    return torch.cat([out[r * sizes[0]:r * sizes[0] + sizes[r]] for r in range(world)])


def load_data(path):
    """The prepared arrays as {name: numpy array}, from an .npz or a packed folder.

    A packed folder is memory-mapped (copy-on-write, never written), so several
    training processes share one copy in the page cache. It is read through once
    first so training doesn't page it in from disk at random."""
    if not os.path.isdir(path):
        return np.load(path)
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n"]

    def read(name, dtype, shape):
        file = os.path.join(path, f"{name}.bin")
        with open(file, "rb", buffering=0) as f:
            buf = bytearray(1 << 26)
            while f.readinto(buf):
                pass
        return np.memmap(file, dtype=dtype, mode="c", shape=shape)
    return {"squares": read("squares", np.uint8, (n, 32)), "castling": read("castling", np.uint8, n),
            "turn": read("turn", np.uint8, n), "eval": read("eval", np.float32, n),
            "source": read("source", np.uint8, n), "source_names": np.array(meta["source_names"])}


CHUNK = 50_000_000   # positions per step of the chunked startup work below


class Targets:
    """The training targets in model units, computed from the evals as they're indexed
    (targets[idx] is a float tensor), so each process doesn't keep 4 bytes per position
    of its own: the evals stay in the shared memory map of a packed --data folder."""

    def __init__(self, evals, args):
        self.evals, self.args = evals, args

    def __len__(self):
        return len(self.evals)

    def __getitem__(self, idx):
        evals = self.evals[idx]
        if self.args.target == "winprob":   # logit of the win probability; mates (+-32000 cp) saturate on their own
            return evals / self.args.wp_scale
        return evals.clamp(-self.args.clip, self.args.clip) / self.args.scale


def in_sources(idx, source, use):
    """Whether each position in idx is from a source in use, as a bool mask built a chunk
    at a time (source[idx].long() in one go would be 8 bytes per position)."""
    mask = torch.empty(len(idx), dtype=torch.bool)
    for i in range(0, len(idx), CHUNK):
        mask[i:i + CHUNK] = use[source[idx[i:i + CHUNK]].long()]
    return mask


def setup_distributed():
    """(rank, world size, device). Under torchrun (WORLD_SIZE > 1) this joins the
    NCCL process group, one GPU per process; otherwise it's a single process."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world == 1:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    return dist.get_rank(), world, device


def batch_squares(squares, b, device):
    """Square tokens long[B, 64] for indices b, unpacking packed (two per byte) storage."""
    sq = squares[b].to(device, non_blocking=True)
    return (unpack_squares(sq) if sq.shape[1] == 32 else sq).long()


def resolve_precision(args, device):
    """Turn --precision auto into bf16 / fp16 / fp32 for this device."""
    if device.type != "cuda":
        return "fp32"
    if args.precision == "auto":
        return "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
    return args.precision


def autocast(args, device):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision)
    return torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None)


def unit_cp(args):
    """Centipawns per model output unit."""
    return args.wp_scale if args.target == "winprob" else args.scale


def val_metrics(pred, y, args):
    """(loss-space MSE, per-position absolute error in centipawns clipped to +-clip).

    The MSE is in the target's own space: scaled cp for --target cp, win probability
    for --target winprob (so it is comparable only between runs with the same target).
    """
    if args.target == "winprob":
        mse = (torch.sigmoid(pred) - torch.sigmoid(y)) ** 2
    else:
        mse = (pred - y) ** 2
    to_cp = lambda t: (t * unit_cp(args)).clamp(-args.clip, args.clip)
    return mse, (to_cp(pred) - to_cp(y)).abs()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rank, world, device = setup_distributed()
    log = print if rank == 0 else lambda *a, **k: None
    if args.batch_size % world:
        raise SystemExit(f"--batch-size {args.batch_size} must divide evenly across {world} GPUs")
    local_batch = args.batch_size // world   # this process's slice of every step's batch
    args.precision = resolve_precision(args, device)
    os.makedirs(args.out, exist_ok=True)

    data = load_data(args.data)
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
    source = torch.from_numpy(data["source"][keep])   # uint8: .long() only on chunks, to save RAM
    source_names = [str(x) for x in data["source_names"]]
    unknown = set(args.sources) - set(source_names)
    if unknown:
        raise SystemExit(f"unknown --sources {sorted(unknown)}; {args.data} has {source_names}")
    use = torch.tensor([name in args.sources for name in source_names])
    targets = Targets(torch.from_numpy(data["eval"][keep]), args)

    del data
    # int32 indices halve the index arrays' memory on huge datasets (small ones keep
    # int64 so their validation split matches earlier runs)
    index_dtype = torch.int32 if n > 50_000_000 else torch.int64
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed), dtype=index_dtype)
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = perm[:n_val].clone(), perm[n_val:]   # train_idx stays a view of perm
    del perm
    chosen = in_sources(train_idx, source, use)              # train only on the chosen sources
    if not chosen.all():   # (when every position is used, skip a full copy of the indices)
        train_idx = train_idx[chosen]
    del chosen
    val_used = in_sources(val_idx, source, use)              # val positions from those sources
    val_subset = val_idx[val_used][:args.val_subset]         # for --val-every
    gpus = f"{world} GPUs, {local_batch} positions each per step" if world > 1 else f"device {device}"
    log(f"training on {', '.join(args.sources)}: {len(train_idx):,} train / "
        f"{val_used.sum().item():,} val (+{(~val_used).sum().item():,} val from other sources), "
        f"{gpus}, {args.precision}")

    overrides = {k: v for k, v in dict(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                                       d_ff=args.d_ff, dropout=args.dropout).items() if v is not None}
    if args.spatial_bias:
        if args.arch not in ("v3", "v4"):
            raise SystemExit("--spatial-bias needs --arch v3 or v4")
        overrides["spatial_bias"] = True
    symmetric = args.arch == "v4"   # model(mirror(x)) == -model(x) by construction
    if symmetric and args.augment_prob > 0:
        log(f"--arch v4 sees every position from the side to move, so colour swaps change "
            f"nothing: --augment-prob {args.augment_prob} -> 0")
        args.augment_prob = 0.0
    if args.smolgen:
        if not args.spatial_bias or args.arch != "v3":
            raise SystemExit("--smolgen needs --arch v3 --spatial-bias")
        overrides["smolgen"] = True
    model = MODELS[args.arch](**overrides).to(device)
    log(f"{type(model).__name__} {model.config}: {sum(p.numel() for p in model.parameters()):,} parameters")
    if world > 1:
        torch.manual_seed(args.seed + rank)   # different dropout per GPU; DDP copies rank 0's weights

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_idx) // args.batch_size   # drop the ragged last batch
    # model stays the bare module (checkpoints, validation); forward averages gradients across GPUs
    forward = torch.compile(model) if args.compile and device.type == "cuda" else model
    if world > 1:
        forward = DistributedDataParallel(forward, device_ids=[device.index])
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):   # linear warmup, then cosine or WSD decay to --lr-floor (multiples of --lr)
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        floor = args.lr_floor
        if args.schedule == "wsd":   # flat at --lr, then linear decay over the last --decay-frac of steps
            decay_start = total_steps - max(1, round(args.decay_frac * total_steps))
            progress = (step - decay_start) / max(1, total_steps - decay_start)
            return 1.0 - (1 - floor) * min(1.0, max(0.0, progress))
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    scaler = torch.amp.GradScaler(device.type, enabled=args.precision == "fp16")   # fp16 needs loss scaling

    start_epoch, start_step, best_val, history, curve = 1, 0, float("inf"), [], []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_val = ckpt["best_val"]
        history = ckpt["history"]
        curve = ckpt.get("curve", [])
        if ckpt.get("step") is not None:   # saved mid-epoch
            start_epoch, start_step = ckpt["epoch"], ckpt["step"]
            log(f"Resumed from {args.resume} (epoch {start_epoch}, step {start_step}/{steps_per_epoch})")
        else:
            start_epoch = ckpt["epoch"] + 1
            log(f"Resumed from {args.resume} (after epoch {ckpt['epoch']})")

    args.wandb = args.wandb and rank == 0
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
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "model_class": type(model).__name__,
            "model_config": model.config,
            "target": args.target,
            "target_clip": args.clip,
            "target_scale": unit_cp(args),   # model output * target_scale = centipawn-like score
            "best_val": best_val,
            "history": history,
            "curve": curve,   # --val-every points
            "args": vars(args),
        }

    def save(ckpt, name):   # write then rename, so a crash mid-save can't corrupt the file
        path = os.path.join(args.out, name)
        if rank == 0:
            torch.save(ckpt, path + ".tmp")
            os.replace(path + ".tmp", path)
        return path

    def validate(idx, mirror=False):
        return evaluate(model, squares, castling, turn, targets, idx, args, device, mirror, world, rank)

    if args.target == "winprob":   # soft-label BCE: target is a probability, prediction a logit
        criterion = lambda pred, y: nn.functional.binary_cross_entropy_with_logits(pred, torch.sigmoid(y))
    else:
        criterion = nn.MSELoss()
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        shuffle = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        order = train_idx[torch.randperm(len(train_idx), generator=shuffle, dtype=index_dtype)]
        swap = torch.empty(len(order), dtype=torch.bool)   # same draw on resume
        for i in range(0, len(order), CHUNK):   # in chunks: one float per position would be 4 bytes each
            swap[i:i + CHUNK] = torch.rand(len(swap[i:i + CHUNK]), generator=shuffle) < args.augment_prob
        first = start_step if epoch == start_epoch else 0
        running, t0 = 0.0, time.time()
        bar = tqdm(range(first, steps_per_epoch), desc=f"epoch {epoch}/{args.epochs}",
                   initial=first, total=steps_per_epoch, disable=rank != 0)
        for step in bar:
            lo = step * args.batch_size + rank * local_batch   # this GPU's slice of the step's batch
            b = order[lo:lo + local_batch]
            sq = batch_squares(squares, b, device)
            ca = castling[b].to(device, non_blocking=True).long()
            tu = turn[b].to(device, non_blocking=True).long()
            y = targets[b].to(device, non_blocking=True)
            if args.augment_prob > 0:
                m = swap[lo:lo + local_batch].to(device, non_blocking=True)
                msq, mca, mtu = mirror_tokens(sq, ca, tu)
                sq = torch.where(m[:, None], msq, sq)
                ca = torch.where(m, mca, ca)
                tu = torch.where(m, mtu, tu)
                y = torch.where(m, -y, y)

            with autocast(args, device):
                pred = forward(sq, ca, tu)
            loss = criterion(pred.float(), y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)   # clip the real gradients, not the scaled ones
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)   # skipped if fp16 gradients overflowed
            scaler.update()
            scheduler.step()

            if step % 10 == 0:   # .item() syncs with the GPU, so don't do it every step
                step_loss = loss.detach()
                if world > 1:   # the whole batch's loss, not just this GPU's slice
                    dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
                running = 0.9 * running + 0.1 * step_loss.item() if running else step_loss.item()
            if step % 50 == 0:
                bar.set_postfix(loss=f"{running:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
                if args.wandb:
                    wandb.log({"train_loss": running, "lr": scheduler.get_last_lr()[0]})
            if args.val_every and (step + 1) % args.val_every == 0 and step + 1 < steps_per_epoch:
                sq_err, cp_err = val_metrics(*validate(val_subset), args)
                sub_source = source[val_subset]
                by_source = {name: cp_err[sub_source == i].mean().item()
                             for i, name in enumerate(source_names) if (sub_source == i).any()}
                curve.append({"epoch": epoch, "step": step + 1, "train_loss": running,
                              "val_mse": sq_err.mean().item(), "val_mae_cp": cp_err.mean().item(),
                              "val_mae_cp_by_source": by_source})
                if rank == 0:
                    tqdm.write(f"step {step + 1}: train {running:.4f}  val subset MSE {curve[-1]['val_mse']:.4f}  "
                               f"MAE {curve[-1]['val_mae_cp']:.0f} cp  ("
                               + "  ".join(f"{k} {v:.0f}" for k, v in by_source.items()) + ")")
                if args.wandb:
                    wandb.log({"val_subset_mse": curve[-1]["val_mse"], "val_subset_mae_cp": curve[-1]["val_mae_cp"]})
            if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < steps_per_epoch:
                save(checkpoint(epoch, step + 1), "last.pt")

        sq_err, cp_err = val_metrics(*validate(val_idx), args)
        val_mse = sq_err[val_used].mean().item()
        val_mae_cp = cp_err[val_used].mean().item()
        if symmetric:   # the mirrored predictions are exactly the negated ones
            mirror_mae_cp = val_mae_cp
        else:
            _, mirror_cp_err = val_metrics(*validate(val_idx[val_used], mirror=True), args)
            mirror_mae_cp = mirror_cp_err.mean().item()
        val_source = source[val_idx]
        by_source = {name: cp_err[val_source == i].mean().item()
                     for i, name in enumerate(source_names) if (val_source == i).any()}
        history.append({"epoch": epoch, "train_loss": running, "val_mse": val_mse, "val_mae_cp": val_mae_cp,
                        "val_mirror_mae_cp": mirror_mae_cp, "val_mae_cp_by_source": by_source})
        per_source = "  ".join(f"{k}{'' if k in args.sources else '*'} {v:.0f}" for k, v in by_source.items())
        log(f"epoch {epoch}: train {running:.4f}  val MSE {val_mse:.4f}  val MAE {val_mae_cp:.0f} cp  "
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
            log(f"  new best -> {save(ckpt, 'best.pt')}")

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
