"""T3: the search, with a material-count evaluator in place of the network."""
import threading
import time
import unittest

import chess

from helpers import MaterialEvaluator, make_bot, random_boards
from bots import MATE_SCORE, position_key, terminal_score


class Search(unittest.TestCase):
    def test_mate_in_one(self):
        board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        move, score = make_bot(depth=1).search(board)
        self.assertEqual(move, chess.Move.from_uci("a1a8"))
        self.assertEqual(score, MATE_SCORE - 1)

    def test_mate_in_two(self):
        board = chess.Board("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1")   # 1.Ra7 (or Rb7) and mate next move
        move, score = make_bot(depth=3).search(board)
        self.assertEqual(score, MATE_SCORE - 3)
        board.push(move)
        for reply in list(board.legal_moves):   # every defence still gets mated at once
            board.push(reply)
            self.assertEqual(make_bot(depth=1).search(board)[1], MATE_SCORE - 1, reply)
            board.pop()

    def test_mate_for_black_is_negative(self):
        board = chess.Board("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1")
        move, score = make_bot(depth=1).search(board)
        self.assertEqual(move, chess.Move.from_uci("a8a1"))
        self.assertEqual(score, -(MATE_SCORE - 1))

    def test_queen_preferred_when_promotions_tie(self):
        board = chess.Board("6k1/4P3/6K1/8/8/8/8/8 w - - 0 1")   # e8=Q and e8=R both mate
        for depth in (1, 3):
            move, _ = make_bot(depth=depth).search(board)
            self.assertEqual(move, chess.Move.from_uci("e7e8q"), f"depth {depth}")

    def test_wins_hanging_queen(self):
        board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
        move, score = make_bot(depth=2).search(board)
        self.assertEqual(move, chess.Move.from_uci("d1d5"))
        self.assertGreater(score, 0)

    def test_position_seen_before_is_a_draw(self):
        board = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):   # back to the start: its second occurrence
            board.push_uci(uci)
        self.assertEqual(terminal_score(board, 1), 0.0)
        board.pop()
        self.assertIsNone(terminal_score(board, 1))

    def test_timed_search_always_returns_a_legal_move(self):
        bot = make_bot(movetime=0.01)
        for board in random_boards(40, seed=3):
            if board.is_game_over():
                continue
            move, _ = bot.search(board)
            self.assertIn(move, board.legal_moves, board.fen())

    def test_new_game_clears_search_state(self):
        bot = make_bot(depth=3)
        bot.search(chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"))
        self.assertTrue(bot.tt and bot.cache)
        bot.history[(0, 1)] = 5.0
        bot.pondered[(1, 0.0)] = ([], [], 1, None)
        bot.new_game()
        self.assertFalse(bot.tt or bot.history or bot.killers or bot.pondered)
        self.assertIsNone(bot.last_search)
        self.assertTrue(bot.cache)   # network scores depend only on the position: kept
        bot.new_game(keep_cache=False)
        self.assertFalse(bot.cache)


class Pondering(unittest.TestCase):
    # Black to move; after any reply White has the same clear best move (Qxd5 wins the queen)
    FEN = "4k3/p7/8/3q4/8/8/8/3QK3 b - - 0 1"

    def wait_for(self, condition, timeout=10.0):
        end = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > end:
                self.fail("timed out")
            time.sleep(0.01)

    def test_pondered_reply_is_played_from_the_pondered_search(self):
        board = chess.Board(self.FEN)
        bot = make_bot(depth=2)
        ponder = bot.ponder(board)
        self.wait_for(lambda: bot.pondered)
        searched = {key for key, _ in bot.pondered}
        reply = next(m for m in board.legal_moves if (board.push(m), position_key(board) in searched, board.pop())[1])
        board.push(reply)
        ponder.finish(board)
        move, score = bot.search(board)
        self.assertTrue(bot.last_pondered)
        fresh_move, fresh_score = make_bot(depth=2).search(board)
        self.assertEqual(score, fresh_score)
        self.assertEqual(move, fresh_move)

    def test_ponder_miss_searches_normally(self):
        board = chess.Board(self.FEN)
        bot = make_bot(depth=2)
        ponder = bot.ponder(board)
        time.sleep(0.05)
        ponder.finish(None)
        board.push_uci("a7a6")
        bot.pondered.pop((position_key(board), bot._margin(bot.temperature)), None)   # as if not reached
        move, _ = bot.search(board)
        self.assertFalse(bot.last_pondered)
        self.assertEqual(move, chess.Move.from_uci("d1d5"))

    def test_finish_without_a_board_stops_quickly(self):
        bot = make_bot(movetime=1.0)   # timed: pondering deepens with no depth limit
        ponder = bot.ponder(chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 3 3"))
        time.sleep(0.2)
        start = time.monotonic()
        ponder.finish(None)
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertFalse(ponder.thread.is_alive())

    def test_stop_event_aborts_a_search(self):
        from bots import SearchAborted
        bot = make_bot(depth=30)
        stop = threading.Event()
        threading.Timer(0.1, stop.set).start()
        with self.assertRaises(SearchAborted):
            bot.search(chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"), stop)


if __name__ == "__main__":
    unittest.main()
