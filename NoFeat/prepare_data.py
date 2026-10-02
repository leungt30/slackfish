"""
Encode the raw evaluation CSVs into one compact .npz for the transformer.

    uv run python -m NoFeat.prepare_data                       # all rows
    uv run python -m NoFeat.prepare_data --limit 50000 --out data/nofeat_small.npz

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
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from NoFeat.encode import encode_fens

MATE_CP = 32000
# source name -> CSV. Order matters: a source's index is stored per position.
SOURCES = {
    "games": "data/chessData.csv",       # positions from real games, Stockfish 11 depth 22
    "random": "data/random_evals.csv",   # positions after random moves
    "tactics": "data/tactic_evals.csv",  # puzzle positions (sharp, many forced mates)
}


def parse_eval(s: str) -> float:
    s = str(s).strip()
    if "#" in s:
        mate = s.split("#")[1]
        n = int(mate)
        winning = not mate.startswith("-")   # "#+0" / "#-0" are already mate
        return (MATE_CP - abs(n)) * (1 if winning else -1)
    return float(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES),
                        help="which CSVs to encode (default: all; choose at training time instead)")
    parser.add_argument("--limit", type=int, help="max rows per file")
    parser.add_argument("--out", default="data/nofeat.npz")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    names = list(SOURCES)
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
    df = pd.concat(dfs, ignore_index=True)
    fens = df["FEN"].tolist()

    start = time.time()
    chunks = [fens[i:i + 50_000] for i in range(0, len(fens), 50_000)]
    with Pool(args.workers) as pool:
        parts = pool.map(encode_fens, chunks)
    squares = np.concatenate([p[0] for p in parts])
    castling = np.concatenate([p[1] for p in parts])
    turn = np.concatenate([p[2] for p in parts])
    print(f"Encoded {len(fens):,} positions in {time.time() - start:.0f}s")

    evals = df["Evaluation"].map(parse_eval).to_numpy(np.float32)
    source = df["source"].to_numpy(np.uint8)

    np.savez(args.out, squares=squares, castling=castling, turn=turn, eval=evals,
             source=source, source_names=np.array(names))
    mates = np.abs(evals) > MATE_CP - 1000
    print(f"Saved {args.out}: {len(evals):,} positions, {mates.mean():.1%} forced mates, "
          f"median |eval| {np.median(np.abs(evals[~mates])):.0f} cp")


if __name__ == "__main__":
    main()
