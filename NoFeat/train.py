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
is seeded, so the order is reproduced), and refuses to start if a setting that
decides the order (ORDER_ARGS: data, sources, limit, val-frac, seed, batch size)
or the number of training positions differs from the checkpoint's. Resuming with
a larger --epochs stretches the LR schedule to the new length.

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

Shuffled folders (NoFeat.shuffle_data, for data bigger than RAM): the positions are
already in random order, so validation is the folder's fixed last positions
(--val-frac doesn't apply), training the rest (--limit: the first N of them), and
each epoch reads them a few shards at a time (ShardStream: --shard-size positions per
shard, the shard order shuffled per epoch and positions within each window of
--window-shards shards). No per-position index arrays are built, so a process needs
~1.6 GB of RAM however big the data, and the page cache only the current window. It
trains on every source in the folder (one can't be skipped while reading in order).

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
    p.add_argument("--val-frac", type=float, default=0.02,
                   help="validation share (a shuffled --data folder has its own, fixed one)")
    p.add_argument("--shard-size", type=int, default=1 << 20,
                   help="shuffled --data: positions per shard (default 1,048,576)")
    p.add_argument("--window-shards", type=int, default=8,
                   help="shuffled --data: shards shuffled together at a time (default 8)")
    p.add_argument("--val-every", type=int, default=0,
                   help="also score --val-subset validation positions every N steps (0 = only per epoch)")
    p.add_argument("--val-subset", type=int, default=200_000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                   help="muon: Muon for the 2D weight matrices inside the layers, AdamW for the rest")
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
                   help="v3 / v4 + --spatial-bias only: add a position-dependent bias (Lc0's smolgen) to it")
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
    first so training doesn't page it in from disk at random; a shuffled folder
    (NoFeat.shuffle_data) isn't, since training reads it a few shards at a time,
    and also returns "val" (its last val positions are the validation set)."""
    if not os.path.isdir(path):
        return np.load(path)
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    n, shuffled = meta["n"], meta.get("format") == "packed-shuffled-v1"

    def read(name, dtype, shape):
        file = os.path.join(path, f"{name}.bin")
        if not shuffled:
            with open(file, "rb", buffering=0) as f:
                buf = bytearray(1 << 26)
                while f.readinto(buf):
                    pass
        return np.memmap(file, dtype=dtype, mode="c", shape=shape)
    data = {"squares": read("squares", np.uint8, (n, 32)), "castling": read("castling", np.uint8, n),
            "turn": read("turn", np.uint8, n), "eval": read("eval", np.float32, n),
            "source": read("source", np.uint8, n), "source_names": np.array(meta["source_names"])}
    if shuffled:
        data.update(val=meta["val"], train_sources=list(meta["train_source_counts"]))
    return data


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


def make_optimizers(model, args):
    """[AdamW], or for --optimizer muon [Muon for the 2D weight matrices inside the layers
    (attention and feed-forward), AdamW for the rest: embeddings, norms, biases, the
    attention bias tables and the output head]. Muon's updates are scaled to AdamW's size
    (match_rms_adamw), so both take the same --lr, schedule and --weight-decay."""
    if args.optimizer == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)]
    inside = lambda name: name.startswith(("encoder.layers.", "blocks."))
    matrices = [p for n, p in model.named_parameters() if p.ndim == 2 and inside(n)]
    rest = [p for n, p in model.named_parameters() if not (p.ndim == 2 and inside(n))]
    return [torch.optim.Muon(matrices, lr=args.lr, weight_decay=args.weight_decay, adjust_lr_fn="match_rms_adamw"),
            torch.optim.AdamW(rest, lr=args.lr, weight_decay=args.weight_decay)]


# What decides each epoch's data order (and the train / validation split), so a resumed
# run must match them: the rest of a half-done epoch is then exactly its unseen positions
ORDER_ARGS = ("data", "sources", "limit", "val_frac", "seed", "batch_size", "shard_size", "window_shards")


def order_changes(ckpt, args, train_positions):
    """{setting: (checkpoint's, this run's)} for every ORDER_ARGS setting (and the number of
    training positions, if the checkpoint recorded it) that differs; empty if none do."""
    saved = ckpt.get("args", {})
    norm = {"data": os.path.realpath, "sources": sorted}
    changed = {k: (saved[k], getattr(args, k)) for k in ORDER_ARGS if k in saved
               and norm.get(k, lambda v: v)(saved[k]) != norm.get(k, lambda v: v)(getattr(args, k))}
    if ckpt.get("train_positions") not in (None, train_positions):
        changed["training positions"] = (ckpt["train_positions"], train_positions)
    return changed


def shuffled_split(data, args):
    """Tensors and split for a shuffled folder (NoFeat.shuffle_data): the positions are
    already in random order, so the training set is the first ones (--limit of them)
    and validation the folder's fixed last ones; nothing per position is built."""
    names = [str(x) for x in data["source_names"]]
    if set(args.sources) - set(names):
        raise SystemExit(f"unknown --sources {sorted(set(args.sources) - set(names))}; {args.data} has {names}")
    if set(data["train_sources"]) - set(args.sources):   # it's read in order, so it can't skip any
        raise SystemExit(f"{args.data} is shuffled and trains on all of its sources: "
                         f"--sources {' '.join(data['train_sources'])}")
    n, n_val = len(data["eval"]), int(data["val"])
    train = n - n_val if args.limit is None else min(args.limit, n - n_val)
    tensors = [torch.from_numpy(data[k]) for k in ("squares", "castling", "turn", "source")]
    stream = ShardStream(train, args.seed, args.shard_size, args.window_shards, args.augment_prob)
    use = torch.tensor([name in args.sources for name in names])
    return (*tensors, Targets(torch.from_numpy(data["eval"]), args), train, torch.arange(n - n_val, n),
            stream, names, use)


class ShardStream:
    """The training order for a shuffled folder, without any per-position arrays: the
    training positions (already in random order) are split into shards of shard_size;
    each epoch shuffles the shard order and then the positions within each window of
    window_shards shards. Everything comes from seeded generators, so every process
    builds the same order on its own and any step's batch can be rebuilt (resuming),
    and only the current window is held (its shuffled indices and colour-swap draws)."""

    def __init__(self, n, seed, shard_size, window_shards, augment_prob):
        self.n, self.seed, self.size, self.k, self.augment_prob = n, seed, shard_size, window_shards, augment_prob
        self.n_shards = -(-n // shard_size)
        self.epoch = self.window = None

    def _plan(self, epoch):   # this epoch's shard order and the stream position where each window starts
        if self.epoch != epoch:
            shards = torch.randperm(self.n_shards, generator=torch.Generator().manual_seed(self.seed * 1000 + epoch))
            sizes = torch.full((self.n_shards,), self.size)
            sizes[-1] = self.n - (self.n_shards - 1) * self.size
            per_window = sizes[shards].split(self.k)
            starts = torch.cumsum(torch.tensor([0] + [int(w.sum()) for w in per_window]), 0)
            self.epoch, self.shards, self.starts, self.window = epoch, shards, starts, None
        return self.starts

    def _load(self, epoch, w):
        if self.window is None or self.window[:2] != (epoch, w):
            idx = torch.cat([torch.arange(s * self.size, min((s + 1) * self.size, self.n))
                             for s in self.shards[w * self.k:(w + 1) * self.k].tolist()])
            g = torch.Generator().manual_seed((self.seed * 1000 + epoch) * 100_000 + w)
            idx = idx[torch.randperm(len(idx), generator=g)]
            self.window = (epoch, w, idx, torch.rand(len(idx), generator=g) < self.augment_prob)
        return self.window[2:]

    def take(self, epoch, start, count):
        """(indices, colour-swap draws) for positions start .. start + count of epoch's order."""
        starts, idx, swap = self._plan(epoch), [], []
        while count > 0:
            w = int(torch.searchsorted(starts, start, right=True)) - 1
            window_idx, window_swap = self._load(epoch, w)
            off = start - int(starts[w])
            k = min(count, len(window_idx) - off)
            idx.append(window_idx[off:off + k])
            swap.append(window_swap[off:off + k])
            start, count = start + k, count - k
        return torch.cat(idx), torch.cat(swap)


def as_list(state):
    return state if isinstance(state, list) else [state]


def from_list(states):
    return states[0] if len(states) == 1 else states


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
    if "val" in data:   # a shuffled folder: no index arrays, see ShardStream
        squares, castling, turn, source, targets, train_count, val_idx, stream, source_names, use = \
            shuffled_split(data, args)
    else:
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
        train_count, stream = len(train_idx), None
    val_used = in_sources(val_idx, source, use)              # val positions from those sources
    val_subset = val_idx[val_used][:args.val_subset]         # for --val-every
    gpus = f"{world} GPUs, {local_batch} positions each per step" if world > 1 else f"device {device}"
    log(f"training on {', '.join(args.sources)}: {train_count:,} train / "
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
        if not args.spatial_bias or args.arch not in ("v3", "v4"):
            raise SystemExit("--smolgen needs --arch v3 or v4 with --spatial-bias")
        overrides["smolgen"] = True
    model = MODELS[args.arch](**overrides).to(device)
    log(f"{type(model).__name__} {model.config}: {sum(p.numel() for p in model.parameters()):,} parameters")
    if world > 1:
        torch.manual_seed(args.seed + rank)   # different dropout per GPU; DDP copies rank 0's weights

    optimizers = make_optimizers(model, args)
    steps_per_epoch = train_count // args.batch_size   # drop the ragged last batch
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
    schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lr_at) for o in optimizers]
    scheduler = schedulers[0]   # (they all follow the same schedule)
    scaler = torch.amp.GradScaler(device.type, enabled=args.precision == "fp16")   # fp16 needs loss scaling

    start_epoch, start_step, best_val, history, curve = 1, 0, float("inf"), [], []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        changed = order_changes(ckpt, args, train_count)
        if changed:   # the rest of the epoch would be a different set of positions
            raise SystemExit(f"--resume {args.resume} would change which positions each step trains on: "
                             + "; ".join(f"{k} {old!r} -> {new!r}" for k, (old, new) in changed.items()))
        model.load_state_dict(ckpt["model_state_dict"])
        for o, state in zip(optimizers, as_list(ckpt["optimizer_state_dict"]), strict=True):
            o.load_state_dict(state)
        for s, state in zip(schedulers, as_list(ckpt["scheduler_state_dict"]), strict=True):
            s.load_state_dict(state)
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
            # one optimizer (AdamW): its state as before; Muon + AdamW: a list of both
            "optimizer_state_dict": from_list([o.state_dict() for o in optimizers]),
            "scheduler_state_dict": from_list([s.state_dict() for s in schedulers]),
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
            "train_positions": train_count,   # checked on --resume, with ORDER_ARGS
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
        if stream is None:   # one permutation of the training positions per epoch
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
            if stream is None:
                b, step_swap = order[lo:lo + local_batch], swap[lo:lo + local_batch]
            else:
                b, step_swap = stream.take(epoch, lo, local_batch)
            sq = batch_squares(squares, b, device)
            ca = castling[b].to(device, non_blocking=True).long()
            tu = turn[b].to(device, non_blocking=True).long()
            y = targets[b].to(device, non_blocking=True)
            if args.augment_prob > 0:
                m = step_swap.to(device, non_blocking=True)
                msq, mca, mtu = mirror_tokens(sq, ca, tu)
                sq = torch.where(m[:, None], msq, sq)
                ca = torch.where(m, mca, ca)
                tu = torch.where(m, mtu, tu)
                y = torch.where(m, -y, y)

            with autocast(args, device):
                pred = forward(sq, ca, tu)
            loss = criterion(pred.float(), y)

            for o in optimizers:
                o.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            for o in optimizers:
                scaler.unscale_(o)   # clip the real gradients, not the scaled ones
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in optimizers:
                scaler.step(o)   # skipped if fp16 gradients overflowed
            scaler.update()
            for s in schedulers:
                s.step()

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
