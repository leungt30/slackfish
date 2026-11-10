import json
import os
import sys
import tempfile
import unittest

import chess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Engine"))
from bots import LayeredBook, OpeningBook, open_books  # noqa: E402
from make_book import build  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def book_file(directory, name, records, weight=100):
    path = os.path.join(directory, name)
    open(path, "wb").write(build(records, weight))
    return path


class BookTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        self.dir.cleanup()

    def test_written_book_is_read_back(self):
        records = [{"moves": [], "uci": "e2e4"}, {"moves": ["e2e4"], "uci": "c7c5"},
                   {"moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1", "f6e4"], "uci": "f1e1"}]
        book = OpeningBook(book_file(self.dir.name, "a.bin", records), max_plies=30)
        board = chess.Board()
        self.assertEqual(book.pick(board, self.rng), chess.Move.from_uci("e2e4"))
        board.push_uci("e2e4")
        self.assertEqual(book.pick(board, self.rng), chess.Move.from_uci("c7c5"))

    def test_castling_and_promotion_encoding(self):
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        for uci in ("e1g1", "e1c1"):
            self.assertEqual(self._round_trip(board, uci), chess.Move.from_uci(uci))
        board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        self.assertEqual(self._round_trip(board, "a7a8n"), chess.Move.from_uci("a7a8n"))

    def _round_trip(self, board, uci):
        import chess.polyglot
        from make_book import polyglot_move
        path = os.path.join(self.dir.name, "c.bin")
        import struct
        move = chess.Move.from_uci(uci)
        open(path, "wb").write(struct.pack(">QHHI", chess.polyglot.zobrist_hash(board), polyglot_move(board, move), 1, 0))
        with chess.polyglot.open_reader(path) as r:
            return r.find(board).move

    def test_illegal_move_is_refused(self):
        with self.assertRaises(ValueError):
            build([{"moves": [], "uci": "e2e5"}])

    def test_first_book_with_the_position_plays_and_the_rest_fall_back(self):
        first = book_file(self.dir.name, "a.bin", [{"moves": ["e2e4", "e7e5"], "uci": "e1e2"}])
        second = book_file(self.dir.name, "b.bin", [{"moves": [], "uci": "d2d4"},
                                                   {"moves": ["e2e4", "e7e5"], "uci": "g1f3"}])
        book = open_books([first, second], max_plies=30)
        self.assertIsInstance(book, LayeredBook)
        board = chess.Board()
        self.assertEqual(book.pick(board, self.rng), chess.Move.from_uci("d2d4"))   # only the second knows it
        board.push_uci("e2e4"), board.push_uci("e7e5")
        self.assertEqual(book.pick(board, self.rng), chess.Move.from_uci("e1e2"))   # both: the first wins
        board.push_uci("e1e2")
        self.assertFalse(book.covers(board))                                         # neither
        self.assertIsNone(book.pick(board, self.rng))
        self.assertIn(">", str(book))

    def test_single_path_is_a_plain_book(self):
        path = book_file(self.dir.name, "a.bin", [{"moves": [], "uci": "e2e4"}])
        self.assertIsInstance(open_books([path]), OpeningBook)

    def test_shipped_bongaclang_book_has_the_prepared_lines(self):
        path = os.path.join(ROOT, "Models", "Books", "bongaclang.bin")
        book = open_books([path], max_plies=30)
        board = chess.Board()
        for u in "e2e4 e7e5 e1e2".split():
            board.push_uci(u)
        self.assertEqual(book.pick(board, self.rng), chess.Move.from_uci("g8f6"))
        board = chess.Board()
        for u in "d2d4 e7e5 d4e5 e8e7".split():   # 1.d4 e5 2.dxe5 Ke7
            board.push_uci(u)
        self.assertTrue(book.covers(board))
        lines = [json.loads(line) for line in open(os.path.join(ROOT, "Models", "Books", "bongaclang-lines.jsonl"))]
        self.assertEqual(build(lines), open(path, "rb").read())   # the book is what the lines file says


if __name__ == "__main__":
    unittest.main()
