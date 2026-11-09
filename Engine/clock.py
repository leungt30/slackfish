"""
Clocks and per-move time budgets, shared by the Lichess bot (Engine/lichess.py), the
benchmarks (Engine/match.py, Engine/tournament.py) and the GUI. No model or network
imports, so any of them can use it cheaply.

The budget is slackfish-bot's Lichess one: 1/MOVES_LEFT of the clock plus the increment,
minus LAG for the network, never more than a third of what's left. match.py and
tournament.py use the same budget, so their clocked games are played the way the bot
plays on Lichess. The GUI keeps its own gentler budget (see Game._budget in gui.py).
"""
import time

import chess

LAG = 0.3   # seconds kept back from each move's budget for the network round trip
MOVES_LEFT = 12   # --moves-left for blitz and slower (blitz games ended with 160-245 s unused at 25)
MOVES_LEFT_BULLET = 12   # ... and bullet (1+1 games ended with ~44 s of 60 unused at 20)


def game_over(board: chess.Board) -> bool:
    """Draw only once a repetition/50-move draw has actually happened, not merely
    because one side *could* claim one with its next move."""
    return board.is_game_over() or board.is_repetition(3) or board.is_fifty_moves()


def speed_of(clock):
    """Lichess's speed for a clock (base, increment) in seconds: from the estimated
    seconds a side, base + 40 x increment."""
    base, inc = clock
    est = base + 40 * inc
    return ("ultraBullet" if est < 30 else "bullet" if est < 180 else "blitz" if est < 480
            else "rapid" if est < 1500 else "classical")


def moves_left_for(clock):
    """The default --moves-left for a clock (base, increment): MOVES_LEFT_BULLET for
    Lichess's bullet and ultraBullet speeds, else MOVES_LEFT."""
    return MOVES_LEFT_BULLET if speed_of(clock) in ("ultraBullet", "bullet") else MOVES_LEFT


def move_budget(left, increment, moves_left=MOVES_LEFT, lag=LAG):
    """Seconds to think on this move: 1/moves_left of the clock plus the increment,
    minus network lag, never more than a third of what's left."""
    left = max(0.0, left - lag)
    return min(left / moves_left + increment, left / 3)


def parse_clock(spec, unit=60.0):
    """'M+I' -> (M * unit, I) seconds: '3+2' -> (180.0, 2.0); with unit=1, '10+1' -> (10.0, 1.0)."""
    base, _, inc = str(spec).partition("+")
    return float(base) * unit, float(inc or 0)


def flag_result(board, flagged):
    """The result when `flagged` runs out of time: a loss, or a draw if the other side
    has too little material to mate."""
    if board.has_insufficient_material(not flagged):
        return "1/2-1/2"
    return "0-1" if flagged == chess.WHITE else "1-0"


def score(result, color):
    """color's score for a result string ("*" and unknown results count as a draw)."""
    white = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
    return white if color == chess.WHITE else 1 - white


def play_clocked(board, choose, clock, max_moves, on_move=None):
    """Play board out with both sides on clock (base, increment) seconds. choose(board,
    budget, left) returns the side to move's move, given this move's budget (move_budget)
    and left, each side's seconds before the move; the time it takes is charged to that
    side. on_move(move) is called after each move is played. Moves the game forward in
    board and returns (result, termination); a game still going after max_moves full
    moves is a draw."""
    base, inc = clock
    left = {chess.WHITE: base, chess.BLACK: base}
    moves_left = moves_left_for(clock)
    while not game_over(board) and board.fullmove_number <= max_moves:
        start = time.monotonic()
        move = choose(board, move_budget(left[board.turn], inc, moves_left), dict(left))
        left[board.turn] -= time.monotonic() - start
        if left[board.turn] < 0:
            return flag_result(board, board.turn), "time forfeit"
        left[board.turn] += inc
        board.push(move)
        if on_move is not None:
            on_move(move)
    outcome = board.outcome(claim_draw=game_over(board))
    if outcome is None:
        return "1/2-1/2", f"move {max_moves} adjudicated draw"
    return outcome.result(), outcome.termination.name.lower()
