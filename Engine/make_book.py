"""Write a Polyglot (.bin) opening book from prepared lines.

    python Engine/make_book.py IN.jsonl OUT.bin [--weight 100]

IN.jsonl has one line per position, {"moves": [uci, ...], "uci": "e7e5"}: the moves played from the
start position to reach it, and the move to play there (Engine/bonga_prep output: Stockfish's best
move for each position in an opponent's observed openings). A position reached by two lines
(transposition) is written once; if its lines disagree the later one wins. python-chess can read
Polyglot books but not write them, so the 16-byte entries (key, move, weight, learn) are packed here."""
import argparse
import json
import struct

import chess
import chess.polyglot

PROMOTION = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}


def polyglot_move(board: chess.Board, move: chess.Move) -> int:
    to = move.to_square
    if board.is_castling(move):   # Polyglot writes castling as the king taking its own rook
        to = chess.square(7 if chess.square_file(move.to_square) > 4 else 0, chess.square_rank(move.from_square))
    return to | move.from_square << 6 | PROMOTION[move.promotion] << 12


def build(records, weight: int = 100) -> bytes:
    entries = {}
    for r in records:
        board = chess.Board()
        for u in r["moves"]:
            board.push_uci(u)
        move = chess.Move.from_uci(r["uci"])
        if move not in board.legal_moves:
            raise ValueError(f"illegal book move {r['uci']} after {' '.join(r['moves'])}")
        entries[chess.polyglot.zobrist_hash(board)] = polyglot_move(board, move)
    return b"".join(struct.pack(">QHHI", key, move, weight, 0) for key, move in sorted(entries.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("dest")
    ap.add_argument("--weight", type=int, default=100)
    a = ap.parse_args()
    data = build([json.loads(line) for line in open(a.source) if line.strip()], a.weight)
    open(a.dest, "wb").write(data)
    print(f"wrote {len(data) // 16} positions to {a.dest}")
