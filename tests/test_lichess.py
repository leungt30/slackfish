"""T4: the Lichess bot's game thread and challenge rules, against a fake Lichess.

Nothing here touches the network: FakeLichess plays the server's side of a game (the
game stream and the move / draw endpoints), and the bot is the stub-evaluator Bot."""
import contextlib
import io
import os
import re
import shutil
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import chess
import chess.pgn
import requests

from helpers import FailingEvaluator, make_bot
import lichess
import mine_hard
import ponderstats

ME = "slackfish"


class FakeLichess:
    """One game's server side. We play White from initial_fen; `replies` are the
    opponent's moves in order, each a UCI string or (uci, extra state fields), e.g.
    ("e7e5", {"bdraw": True}) for a draw offer with that move. When they run out the
    opponent resigns. fail_posts: POSTs of our moves to fail with a ConnectionError
    first; drop_after: events after which the first connection drops."""

    def __init__(self, replies=(), initial_fen="startpos", clock_ms=1000, fail_posts=0, drop_after=None,
                 bad_first_state=False):
        self.replies, self.initial_fen, self.clock_ms = list(replies), initial_fen, clock_ms
        self.fail_posts, self.drop_after, self.bad_first_state = fail_posts, drop_after, bad_first_state
        self.moves, self.posts, self.connects, self.extra = [], [], 0, {}
        self.over = False
        self.lock = threading.Lock()

    def quiet_for(self):
        return float("inf")

    def get(self, path):
        return {"id": ME, "username": "SlackFish", "title": "BOT"}

    def state(self, **status):
        state = {"type": "gameState", "moves": " ".join(self.moves), "wtime": self.clock_ms, "btime": self.clock_ms,
                 "winc": 0, "binc": 0, "status": "started"}
        state.update(self.extra)
        state.update(status)
        self.extra = {}
        return state

    def post(self, path, data=None):
        if "/move/" in path and self.fail_posts:
            self.fail_posts -= 1
            raise requests.ConnectionError("connection reset")
        with self.lock:
            self.posts.append(path)
            if "/move/" in path:
                self.moves.append(path.rsplit("/", 1)[1])

    def our_turn(self):
        board = chess.Board() if self.initial_fen == "startpos" else chess.Board(self.initial_fen)
        for uci in self.moves:
            board.push_uci(uci)
        return board.turn == chess.WHITE

    def stream(self, path):
        self.connects += 1
        first = self.connects == 1
        state = self.state()
        if self.connects > 10:   # the bot keeps failing: end the game rather than hang the test
            state["status"] = "aborted"
        if first and self.bad_first_state:
            state["moves"] = "e2e5"   # not a legal move: on_state raises
        yield {"type": "gameFull", "id": "game1234", "rated": False, "speed": "blitz", "createdAt": 1_700_000_000_000,
               "initialFen": self.initial_fen, "clock": {"initial": self.clock_ms, "increment": 0},
               "white": {"id": ME, "name": "SlackFish"}, "black": {"id": "opp", "name": "Opp", "rating": 2000},
               "state": state}
        events, seen = 1, len(self.moves)
        idle = 0
        while True:
            if first and self.drop_after is not None and events >= self.drop_after:
                raise requests.ConnectionError("stream dropped")
            if len(self.moves) != seen:   # echo our move
                seen = len(self.moves)
            elif not self.our_turn():
                if not self.replies:
                    self.over = True
                    yield self.state(status="resign", winner="white")
                    return
                reply = self.replies.pop(0)
                uci, extra = (reply, {}) if isinstance(reply, str) else reply
                self.moves.append(uci)
                self.extra = extra
                seen = len(self.moves)
            else:   # our turn and no move from us: Lichess sends keep-alives until our flag falls
                idle += 1
                if idle > 50:
                    self.over = True
                    yield self.state(status="outoftime", winner="black")
                    return
                yield None
                continue
            events += 1
            yield self.state()


def game_args(**overrides):
    args = dict(bot=("stub", 2), device="cpu", ponder_replies=None, no_ponder=True, no_tablebase=True,
                moves_left=None)
    args.update(overrides)
    return types.SimpleNamespace(**args)


class LichessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        for name, value in (("GAMES_DIR", self.tmp), ("PAUSE_FILE", os.path.join(self.tmp, "pause")),
                            ("STOP_FILE", os.path.join(self.tmp, "stop"))):
            patcher = mock.patch.object(lichess, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.evaluator = None   # None: material count; set to FailingEvaluator() for a broken GPU
        patcher = mock.patch.object(lichess, "load_bot", lambda name, **kw: make_bot(
            depth=kw.get("depth", 2), movetime=kw.get("movetime"), evaluator=self.evaluator))
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(lichess.time, "sleep", lambda s: None)   # reconnect waits
        patcher.start()
        self.addCleanup(patcher.stop)

    def play(self, li, **args):
        player = lichess.GamePlayer(li, "game1234", ME, game_args(**args), None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            player.run()
        return player, out.getvalue()


class GamePlayerTest(LichessTest):
    def test_one_move_per_turn(self):
        li = FakeLichess(replies=["e7e5", "b8c6", "g8f6"])
        player, _ = self.play(li)
        self.assertTrue(li.over)
        moves = [p for p in li.posts if "/move/" in p]
        self.assertEqual(len(moves), 4)   # our 1st-4th moves; the opponent then resigns
        self.assertEqual(player.result[0], 1.0)

    def test_failed_move_post_is_sent_again_after_reconnecting(self):
        li = FakeLichess(replies=["e7e5"], fail_posts=1)
        _, log = self.play(li)
        self.assertEqual(li.connects, 2)
        self.assertIn("connection lost", log)
        self.assertTrue(li.over)
        self.assertEqual(len([p for p in li.posts if "/move/" in p]), 2)

    def test_dropped_stream_handles_the_current_position_again(self):
        li = FakeLichess(replies=["e7e5", "b8c6"], drop_after=2)
        _, log = self.play(li)
        self.assertGreaterEqual(li.connects, 2)
        self.assertTrue(li.over, log)

    def test_unexpected_state_doesnt_stop_play(self):
        li = FakeLichess(replies=["e7e5"], bad_first_state=True)
        _, log = self.play(li)
        self.assertIn("reading the game again", log)
        self.assertEqual(li.connects, 2)
        self.assertTrue(li.over)

    def test_failing_search_still_moves(self):
        """F1: a search that raises every time (e.g. a GPU error) must not lose the game on time."""
        self.evaluator = FailingEvaluator()
        li = FakeLichess(replies=["e7e5", "b8c6"])
        player, log = self.play(li)
        self.assertTrue(li.over)
        self.assertEqual(player.result[0], 1.0, log)   # the opponent resigned; we didn't flag
        self.assertEqual(len([p for p in li.posts if "/move/" in p]), 3)
        self.assertTrue(player.search_failed)
        self.assertIn("without a search", log)

    def test_draw_offer_accepted_when_level(self):
        li = FakeLichess(replies=[("e7e5", {"bdraw": True})])
        self.play(li)
        self.assertIn("/api/bot/game/game1234/draw/yes", li.posts)

    def test_draw_offer_rule(self):
        """Accept a draw offer only when our last score is under DRAW_BELOW for us."""
        cases = [(None, chess.WHITE, "no"),   # no evaluation yet (e.g. a book move)
                 (0.0, chess.WHITE, "yes"), (lichess.DRAW_BELOW - 1, chess.WHITE, "yes"),
                 (lichess.DRAW_BELOW, chess.WHITE, "no"), (500.0, chess.WHITE, "no"),
                 (500.0, chess.BLACK, "yes"), (-500.0, chess.BLACK, "no")]   # scores are White's view
        for score, color, answer in cases:
            with self.subTest(score=score, color=color):
                li = FakeLichess()
                player = lichess.GamePlayer(li, "game1234", ME, game_args(), None)
                player.color, player.score, player.start = color, score, chess.Board()
                player.opponent, player.ai_level, player.moves_left = "Opp", None, 12
                them = "b" if color == chess.WHITE else "w"
                state = dict(li.state(), moves="e2e4" if color == chess.WHITE else "", **{f"{them}draw": True})
                with mock.patch.object(player, "play"), contextlib.redirect_stdout(io.StringIO()):
                    player.on_state(state)
                self.assertIn(f"/api/bot/game/game1234/draw/{answer}", li.posts)

    def test_takebacks_declined(self):
        li = FakeLichess(replies=[("e7e5", {"btakeback": True})])
        self.play(li)
        self.assertIn("/api/bot/game/game1234/takeback/no", li.posts)

    def test_pgn_and_log_match_the_analysis_tools(self):
        """The move-log contract: mine_hard reads the PGN comments, ponderstats the log lines."""
        li = FakeLichess(replies=["e7e5", "b8c6", "g8f6", "f8c5"])
        player, log = self.play(li, no_ponder=False)
        with open(os.path.join(self.tmp, "game1234.pgn")) as f:
            game = chess.pgn.read_game(f)
        self.assertEqual(game.headers["Result"], "1-0")
        ours = [node for node in game.mainline() if node.turn() == chess.BLACK]   # positions after our moves
        self.assertEqual(len(ours), 5)
        for node in ours:
            comment = node.comment
            self.assertRegex(comment, mine_hard.SCORE)
            self.assertRegex(comment, mine_hard.DEPTH)
            self.assertRegex(comment, mine_hard.TIME)
        pondered = [line for line in log.splitlines() if "ponder " in line and re.search(r"\d+\.\S+\s+\(", line)]
        self.assertTrue(pondered, log)
        for line in pondered:
            self.assertRegex(line, ponderstats.MOVE)
        timestamped = "11-09 12:00:00 " + pondered[0]   # the log as Timestamped writes it
        self.assertRegex(timestamped, ponderstats.MOVE)


class FallbackMoveTest(unittest.TestCase):
    def test_mate_then_captures_then_any(self):
        self.assertEqual(lichess.fallback_move(chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")).uci(), "a1a8")
        self.assertEqual(lichess.fallback_move(chess.Board("4k3/8/8/3q4/8/8/3p4/3RK3 w - - 0 1")).uci(), "d1d2")
        board = chess.Board()
        self.assertIn(lichess.fallback_move(board), board.legal_moves)


def challenge(limit=180, increment=2, rated=False, variant="standard", title=None, kind="clock"):
    tc = {"type": kind, "limit": limit, "increment": increment} if kind == "clock" else {"type": kind}
    return {"id": "c1", "variant": {"key": variant}, "timeControl": tc, "rated": rated, "speed": "blitz",
            "challenger": {"name": "someone", "title": title}}


def runner_args(**overrides):
    args = dict(book=None, challenge=None, stockfish=None, min_base=30, max_base=1800, allow_no_increment=False,
                casual_only=False, no_bots=False, max_games=1)
    args.update(overrides)
    return types.SimpleNamespace(**args)


class DeclineReasonTest(LichessTest):
    def runner(self, li=None, **args):
        with contextlib.redirect_stdout(io.StringIO()):
            return lichess.BotRunner(li or FakeLichess(), runner_args(**args))

    def test_rules(self):
        cases = [
            (challenge(), {}, None),
            (challenge(variant="chess960"), {}, "standard"),
            (challenge(variant="fromPosition"), {}, None),
            (challenge(kind="correspondence"), {}, "timecontrol"),
            (challenge(limit=15, increment=0), {"allow_no_increment": True}, "toofast"),
            (challenge(limit=3600), {}, "tooslow"),
            (challenge(limit=60, increment=0), {}, "timecontrol"),
            (challenge(limit=60, increment=0), {"allow_no_increment": True}, None),
            (challenge(rated=True), {"casual_only": True}, "casual"),
            (challenge(title="BOT"), {"no_bots": True}, "nobot"),
            (challenge(title="BOT"), {}, None),
        ]
        for c, args, expected in cases:
            with self.subTest(tc=c["timeControl"], variant=c["variant"], args=args):
                self.assertEqual(self.runner(**args).decline_reason(c), expected)

    def test_later_when_busy_paused_or_network_unstable(self):
        runner = self.runner()
        runner.games["g"] = object()
        self.assertEqual(runner.decline_reason(challenge()), "later")
        runner = self.runner()
        open(lichess.PAUSE_FILE, "w").close()
        self.assertEqual(runner.decline_reason(challenge()), "later")
        os.remove(lichess.PAUSE_FILE)
        li = FakeLichess()
        li.quiet_for = lambda: 10.0   # a connection error 10 s ago
        self.assertEqual(self.runner(li).decline_reason(challenge()), "later")


class SeriesTest(LichessTest):
    def finished(self, **flags):
        player = types.SimpleNamespace(game_id="g1", result=(0.0, "Opp", None), abandoned=False, search_failed=False,
                                       run=lambda: None)
        vars(player).update(flags)
        with contextlib.redirect_stdout(io.StringIO()):
            runner = lichess.BotRunner(FakeLichess(), runner_args(challenge=["Opp"]))
            runner.games["g1"] = player
            runner.challenged.add("g1")
            runner.play(player)
        return runner

    def test_series_continues_after_a_normal_game(self):
        runner = self.finished()
        self.assertIsNone(runner.halted)
        self.assertEqual(runner.results["Opp"], [0.0])

    def test_series_halts_after_a_failed_search_or_lost_connection(self):
        self.assertIn("search failed", self.finished(search_failed=True).halted)
        self.assertIn("connection", self.finished(abandoned=True).halted)

    def test_game_thread_crash_is_logged_not_raised(self):
        def crash():
            raise RuntimeError("boom")
        runner = self.finished(run=crash)
        self.assertNotIn("g1", runner.games)


if __name__ == "__main__":
    unittest.main()
