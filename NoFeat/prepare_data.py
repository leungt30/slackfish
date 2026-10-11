"""
Encode the raw evaluation CSVs into one compact .npz for the transformer.

    uv run python -m NoFeat.prepare_data                       # all rows
    uv run python -m NoFeat.prepare_data --limit 50000 --out data/nofeat_small.npz
    uv run python -m NoFeat.prepare_data --lichess /archive/lichess/lichess_db_eval.jsonl.zst \
        --out /data/slackfish/nofeat_all          # a packed folder (see below)

Lichess (--lichess): the Lichess evaluation database (database.lichess.org,
lichess_db_eval.jsonl.zst, CC0), streamed through `zstd -dc` so it is never
unpacked to disk. Each position uses its deepest evaluation's best line (scores
are from White's perspective, like the CSVs). --lichess-frac keeps a random
fraction of positions and --min-depth drops shallow evaluations. Its positions
are tagged "lichess" and are added to the CSV sources (--sources).

Output formats (--out):
  *.npz     the arrays below, built in memory. Fine for the CSVs (16.6M positions).
  a folder  "packed": the same arrays as raw files written as they are encoded, so
            memory stays small, with squares packed two per byte (uint8[N, 32], see
            encode.pack_squares) to halve their size: ~38 bytes per position, so all
            of Lichess (~410M positions, ~16 GB) still fits in RAM for training.
            Files: squares.bin castling.bin turn.bin eval.bin source.bin meta.json.

Every position is tagged with the file it came from, so training can pick which
sources to use (NoFeat.train --sources) without re-encoding.

Output arrays (N rows):
    squares       uint8[N, 64]   square tokens (see encode.py)
    castling      uint8[N]       castling token
    turn          uint8[N]       0 white to move, 1 black to move
    eval          float32[N]     centipawns from White's perspective;
                                 mate in n -> +-(MATE_CP - |n|)
    source        uint8[N]       index into source_names
    source_names  str[S]         e.g. ["games", "random", "tactics"]
"""
import argparse
import json
import os
import subprocess
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from NoFeat.encode import encode_fens, pack_squares

MATE_CP = 32000
# source name -> CSV. Order matters: a source's index is stored per position.
SOURCES = {
    "games": "data/chessData.csv",       # positions from real games, Stockfish 11 depth 22
    "random": "data/random_evals.csv",   # positions after random moves
    "tactics": "data/tactic_evals.csv",  # puzzle positions (sharp, many forced mates)
}
SOURCE_NAMES = list(SOURCES) + ["lichess"]   # lichess: the --lichess JSONL, not a CSV
LICHESS_CHUNK = 50_000


def parse_eval(s: str) -> float:
    s = str(s).strip()
    if "#" in s:
        mate = s.split("#")[1]
        n = int(mate)
        winning = not mate.startswith("-")   # "#+0" / "#-0" are already mate
        return (MATE_CP - abs(n)) * (1 if winning else -1)
    return float(s)


def mate_cp(n: int) -> float:
    """Lichess mate in n (positive: White mates) -> centipawns, as parse_eval does."""
    return (MATE_CP - abs(n)) * (1 if n > 0 else -1)


def encode_lichess_lines(args):
    """Raw JSONL lines -> (squares, castling, turn, evals) for the positions whose
    deepest evaluation reaches min_depth."""
    lines, min_depth = args
    fens, evals = [], []
    for line in lines:
        row = json.loads(line)
        best = max(row["evals"], key=lambda e: e["depth"])
        pv = best["pvs"][0]
        if best["depth"] < min_depth or pv.get("mate") == 0:
            continue
        fens.append(row["fen"])
        evals.append(pv["cp"] if "cp" in pv else mate_cp(pv["mate"]))
    return *encode_fens(fens), np.array(evals, dtype=np.float32)


def read_lichess(path, frac, min_depth, seed, limit, pool):
    """Stream the Lichess .jsonl.zst and yield (squares, castling, turn, evals) parts
    for a random frac of its positions."""
    rng = np.random.default_rng(seed)
    zstd = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, bufsize=1 << 24)

    def chunks():
        read = kept = 0
        batch = []
        for line in zstd.stdout:
            read += 1
            if frac >= 1 or rng.random() < frac:
                batch.append(line)
                kept += 1
            if len(batch) == LICHESS_CHUNK:
                yield batch, min_depth
                batch = []
            if read % 10_000_000 == 0:
                print(f"  lichess: read {read:,} lines, sampled {kept:,}", flush=True)
            if limit and kept >= limit:
                break
        if batch:
            yield batch, min_depth

    yield from pool.imap(encode_lichess_lines, chunks())
    zstd.kill()


class NpzSink:
    """Collects parts in memory and saves one .npz."""

    def __init__(self, path):
        self.path, self.parts = path, []

    def add(self, squares, castling, turn, evals, source):
        self.parts.append((squares, castling, turn, evals, source))

    def close(self, names):
        squares, castling, turn, evals, source = (np.concatenate([p[i] for p in self.parts]) for i in range(5))
        np.savez(self.path, squares=squares, castling=castling, turn=turn, eval=evals,
                 source=source, source_names=np.array(names))
        return evals


class PackedSink:
    """Appends each part to raw files in a folder as it arrives (see the docstring)."""
    FILES = ("squares", "castling", "turn", "eval", "source")

    def __init__(self, path):
        self.path, self.n = path, 0
        os.makedirs(path, exist_ok=True)
        self.files = {name: open(os.path.join(path, f"{name}.bin"), "wb") for name in self.FILES}
        self.eval_sample = []   # every 100th eval, for the summary

    def add(self, squares, castling, turn, evals, source):
        arrays = dict(squares=pack_squares(squares), castling=castling, turn=turn,
                      eval=evals.astype(np.float32), source=source)
        for name, array in arrays.items():
            self.files[name].write(np.ascontiguousarray(array).tobytes())
        self.eval_sample.append(evals[::100])
        self.n += len(evals)

    def close(self, names):
        for f in self.files.values():
            f.close()
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump({"format": "packed-v1", "n": self.n, "source_names": names}, f)
        return np.concatenate(self.eval_sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES),
                        help="which CSVs to encode (default: all; choose at training time instead)")
    parser.add_argument("--limit", type=int, help="max rows per file")
    parser.add_argument("--out", default="data/nofeat.npz")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--lichess", help="also encode this lichess_db_eval.jsonl.zst")
    parser.add_argument("--lichess-frac", type=float, default=1.0,
                        help="keep this random fraction of Lichess positions (default: all)")
    parser.add_argument("--min-depth", type=int, default=20,
                        help="drop Lichess positions evaluated below this depth (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="for the Lichess sample")
    args = parser.parse_args()
    if args.lichess and not os.path.exists(args.lichess):
        raise SystemExit(f"{args.lichess} not found")

    names = SOURCE_NAMES
    dfs = []
    for name in args.sources:
        path = SOURCES[name]
        if not os.path.exists(path):
            print(f"{path} not found, skipping '{name}'")
            continue
        df = pd.read_csv(path, nrows=args.limit, usecols=["FEN", "Evaluation"])
        df["source"] = names.index(name)
        print(f"{name:8s} {path}: {len(df):,} rows")
        dfs.append(df)
    df = (pd.concat(dfs, ignore_index=True) if dfs
          else pd.DataFrame({"FEN": [], "Evaluation": [], "source": []}))
    fens = df["FEN"].tolist()

    os.makedirs(os.path.dirname(args.out.rstrip("/")) or ".", exist_ok=True)
    sink = NpzSink(args.out) if args.out.endswith(".npz") else PackedSink(args.out)
    start = time.time()
    chunks = [fens[i:i + 50_000] for i in range(0, len(fens), 50_000)]
    with Pool(args.workers) as pool:
        parts = pool.map(encode_fens, chunks) or [encode_fens([])]
        sink.add(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                 np.concatenate([p[2] for p in parts]),
                 df["Evaluation"].map(parse_eval).to_numpy(np.float32), df["source"].to_numpy(np.uint8))
        print(f"Encoded {len(fens):,} positions in {time.time() - start:.0f}s")
        del parts, df, fens

        if args.lichess:
            start, count = time.time(), 0
            lichess = names.index("lichess")
            for squares, castling, turn, evals in read_lichess(args.lichess, args.lichess_frac,
                                                               args.min_depth, args.seed, args.limit, pool):
                sink.add(squares, castling, turn, evals, np.full(len(evals), lichess, np.uint8))
                count += len(evals)
            print(f"lichess  {args.lichess}: {count:,} positions (depth >= {args.min_depth}) "
                  f"in {time.time() - start:.0f}s")

    evals = sink.close(names)
    mates = np.abs(evals) > MATE_CP - 1000
    sampled = "" if isinstance(sink, NpzSink) else " (stats from a 1% sample)"
    print(f"Saved {args.out}: {getattr(sink, 'n', len(evals)):,} positions, {mates.mean():.1%} forced mates, "
          f"median |eval| {np.median(np.abs(evals[~mates])):.0f} cp{sampled}")


if __name__ == "__main__":
    main()
