"""
Shuffle one or more packed folders (NoFeat.prepare_data) into one shuffled packed folder,
for training that reads the data in chunks instead of holding it all in memory.

    uv run python -m NoFeat.shuffle_data data/nofeat_lichess_all --out data/nofeat_shuffled
    uv run python -m NoFeat.shuffle_data data/nofeat_lichess_all data/nofeat_games_2026-08 \
        --out data/nofeat_shuffled_2m

The positions of all the inputs are written in one uniformly random order, and the last
--val-frac of them are the validation set (a random sample, fixed for every run on this
folder). NoFeat.train reads such a folder in shards (see its docstring): it needs no
per-position index arrays and only the shards it is on in the page cache, so the data
only has to fit on disk.

Out of core, so the inputs can be far bigger than RAM: pass 1 reads them in order and
deals each position to one of --buckets temporary files at random; pass 2 loads one
bucket at a time, shuffles it in memory and appends it to the output. Concatenating
randomly filled buckets, each shuffled, is a uniformly random permutation. Memory is
one bucket (~40 bytes per position) plus one input chunk; disk needs room for the
buckets and the output (2x the data) while it runs.

Output: the packed-v1 files (squares two per byte, castling, turn, eval, source) and
meta.json with "format": "packed-shuffled-v1", "n", "val" (the last val positions are
the validation set) and "train_source_counts".
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

FILES = {"squares": (np.uint8, 32), "castling": (np.uint8, 1), "turn": (np.uint8, 1),
         "eval": (np.float32, 1), "source": (np.uint8, 1)}
CHUNK = 10_000_000   # positions read per step of pass 1


def read_meta(path):
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("format") != "packed-v1":
        raise SystemExit(f"{path}: not a packed-v1 folder (format {meta.get('format')!r})")
    return meta


def column(path, name, n, mode="r"):
    dtype, width = FILES[name]
    shape = (n, width) if width > 1 else (n,)
    return np.memmap(os.path.join(path, f"{name}.bin"), dtype=dtype, mode=mode, shape=shape)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="packed-v1 folders from NoFeat.prepare_data")
    p.add_argument("--out", required=True)
    p.add_argument("--val-frac", type=float, default=0.005, help="last fraction kept for validation (default 0.005)")
    p.add_argument("--buckets", type=int, help="temporary buckets (default: ~20M positions each)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    metas = [read_meta(path) for path in args.inputs]
    names = metas[0]["source_names"]
    if any(m["source_names"] != names for m in metas):
        raise SystemExit("the inputs tag their sources differently: re-encode them with the same prepare_data")
    total = sum(m["n"] for m in metas)
    n_buckets = args.buckets or max(1, -(-total // 20_000_000))
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "buckets.tmp")
    os.makedirs(tmp, exist_ok=True)
    print(f"{total:,} positions from {len(args.inputs)} folder(s) -> {n_buckets} buckets")

    # Pass 1: deal every position to a random bucket (sequential reads, appending writes)
    start = time.time()
    files = {(b, name): open(os.path.join(tmp, f"{b}.{name}.bin"), "wb")
             for b in range(n_buckets) for name in FILES}
    sizes = np.zeros(n_buckets, np.int64)
    for path, meta in zip(args.inputs, metas):
        cols = {name: column(path, name, meta["n"]) for name in FILES}
        for i in range(0, meta["n"], CHUNK):
            bucket = rng.integers(0, n_buckets, min(CHUNK, meta["n"] - i))
            order = np.argsort(bucket, kind="stable")
            bounds = np.searchsorted(bucket[order], np.arange(n_buckets + 1))
            parts = {name: np.asarray(col[i:i + len(bucket)])[order] for name, col in cols.items()}
            for b in range(n_buckets):
                for name, part in parts.items():
                    files[b, name].write(part[bounds[b]:bounds[b + 1]].tobytes())
            sizes += np.diff(bounds)
        print(f"  pass 1: {path} ({meta['n']:,}) done, {time.time() - start:.0f}s", flush=True)
    for f in files.values():
        f.close()

    # Pass 2: shuffle each bucket in memory and append it to the output
    out = {name: open(os.path.join(args.out, f"{name}.bin"), "wb") for name in FILES}
    for b in range(n_buckets):
        perm = rng.permutation(sizes[b])
        for name, (dtype, width) in FILES.items():
            part = np.fromfile(os.path.join(tmp, f"{b}.{name}.bin"), dtype=dtype)
            out[name].write((part.reshape(-1, width) if width > 1 else part)[perm].tobytes())
            os.remove(os.path.join(tmp, f"{b}.{name}.bin"))
    for f in out.values():
        f.close()
    shutil.rmtree(tmp)
    print(f"  pass 2: {n_buckets} buckets shuffled, {time.time() - start:.0f}s")

    n_val = max(1, int(total * args.val_frac))
    counts = np.bincount(column(args.out, "source", total)[:total - n_val], minlength=len(names))
    meta = {"format": "packed-shuffled-v1", "n": total, "val": n_val, "source_names": names,
            "train_source_counts": {name: int(c) for name, c in zip(names, counts) if c},
            "inputs": [os.path.abspath(path) for path in args.inputs], "seed": args.seed}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"Saved {args.out}: {total:,} positions, the last {n_val:,} for validation; training on "
          + ", ".join(f"{k} {v:,}" for k, v in meta["train_source_counts"].items()))


if __name__ == "__main__":
    main()
