"""
Play SlackFish games: bot vs bot, human vs bot, or human vs human.

Players are given as NAME[:DEPTH[:TEMPERATURE]], where NAME is "human" or a bot
from bots.BOTS. Bots play their best move by default (deterministic); a temperature
in centipawns makes them sample from a softmax over every legal move's score instead.

    uv run python Engine/play.py --white human --black v2:2
    uv run python Engine/play.py --white t2:2 --black t1-3.25m:2 --temperature 50 --seed 1
    uv run python Engine/play.py --white t2:2:30 --black t2:2      # only white samples
    uv run python Engine/play.py --white v5:1 --black v2:3 --pgn games/v5_vs_v2.pgn
    uv run python Engine/play.py --white v2 --black v1 --depth 2 --fen "<fen>"

Or from Python:

    from play import play_game, HumanPlayer
    from bots import load_bot
    play_game(HumanPlayer(), load_bot("v2", depth=2))
"""
import argparse
import time

import chess
import chess.pgn

from bots import BOTS, load_bot


class HumanPlayer:
    """Reads moves from the terminal in UCI (e2e4) or SAN (e4, Nf3) notation."""

    def __init__(self, name: str = "Human"):
        self.name = name

    def __str__(self):
        return self.name

    def choose_move(self, board: chess.Board):
        while True:
            text = input(f"{self.name} ({'white' if board.turn else 'black'}) move "
                         "[e2e4 / e4 / moves / undo / quit]: ").strip()
            if text == "quit":
                raise KeyboardInterrupt
            if text == "undo":
                return "undo"
            if text == "moves":
                print(" ".join(board.san(m) for m in board.legal_moves))
                continue
            try:
                return board.parse_uci(text)
            except ValueError:
                pass
            try:
                return board.parse_san(text)
            except ValueError:
                print("Illegal or unrecognised move, try again.")


def make_player(spec: str, default_depth: int = 1, weights: str | None = None,
                default_temperature: float = 0.0, seed: int | None = None):
    """'human' -> HumanPlayer, 'v2' / 'v2:3' / 'v2:3:50' -> Bot (name:depth:temperature)."""
    name, depth, temperature = (spec.split(":") + ["", ""])[:3]
    if name == "human":
        return HumanPlayer()
    return load_bot(name, depth=int(depth) if depth else default_depth, weights=weights,
                    temperature=float(temperature) if temperature else default_temperature, seed=seed)


def game_over(board: chess.Board) -> bool:
    """Draw only once a repetition/50-move draw has actually happened, not merely
    because one side *could* claim one with its next move."""
    return board.is_game_over() or board.is_repetition(3) or board.is_fifty_moves()


def play_game(white, black, fen: str | None = None, verbose: bool = True,
              pgn_path: str | None = None, max_moves: int | None = None) -> chess.pgn.Game:
    """Play one game between two players (anything with choose_move(board)).

    Returns the game as a PGN object; also writes it to pgn_path if given.
    """
    board = chess.Board(fen) if fen else chess.Board()
    players = {chess.WHITE: white, chess.BLACK: black}
    human_playing = any(isinstance(p, HumanPlayer) for p in players.values())
    show_board = verbose or human_playing

    if show_board:
        print(f"{white} (white) vs {black} (black)\n")
        print(board, "\n")

    try:
        while not game_over(board):
            if max_moves and board.fullmove_number > max_moves:
                break
            player = players[board.turn]
            start = time.perf_counter()
            move = player.choose_move(board)

            if move == "undo":
                # Take back the opponent's reply and your own last move
                for _ in range(min(2, len(board.move_stack))):
                    board.pop()
                print(board, "\n")
                continue

            label = f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'} {board.san(move)}"
            board.push(move)
            if show_board:
                info = ""
                if hasattr(player, "nodes"):
                    info = (f"  ({time.perf_counter() - start:.1f}s, {player.nodes} positions evaluated, "
                            f"{player.cache_hits} from cache)")
                print(f"{label}  [{player}]{info}")
                if getattr(player, "temperature", 0) > 0 and player.last_distribution:
                    moves, _, probs = player.last_distribution
                    top = sorted(zip(probs, moves), key=lambda pm: -pm[0])[:5]
                    board.pop()   # SAN needs the position before the move
                    print("    " + "  ".join(f"{board.san(m)} {p:.0%}" for p, m in top))
                    board.push(move)
                print(board, "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nGame stopped.")

    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = str(white)
    game.headers["Black"] = str(black)
    outcome = board.outcome(claim_draw=game_over(board))
    game.headers["Result"] = outcome.result() if outcome else "*"
    if fen:
        game.headers["FEN"] = fen
        game.headers["SetUp"] = "1"

    print(f"Result: {game.headers['Result']}"
          + (f" ({outcome.termination.name.lower()})" if outcome else ""))
    if pgn_path:
        with open(pgn_path, "w") as f:
            print(game, file=f)
        print(f"PGN saved to {pgn_path}")
    return game


def main():
    parser = argparse.ArgumentParser(
        description="Play a SlackFish game.",
        epilog=f"Available bots: {', '.join(BOTS)}. Use NAME:DEPTH or NAME:DEPTH:TEMPERATURE per player.")
    parser.add_argument("--white", default="human", help="human or bot name, e.g. v2 or v2:3")
    parser.add_argument("--black", default="v2", help="human or bot name, e.g. v5 or v5:2")
    parser.add_argument("--depth", type=int, default=1,
                        help="search depth for bots without an explicit :DEPTH (default 1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="sample moves from softmax(score / T), T in centipawns, for bots "
                             "without an explicit :TEMPERATURE (default 0 = always the best move)")
    parser.add_argument("--seed", type=int, help="random seed for sampled moves, for reproducible games")
    parser.add_argument("--white-weights", help="override checkpoint path for white's bot")
    parser.add_argument("--black-weights", help="override checkpoint path for black's bot")
    parser.add_argument("--fen", help="start from this position instead of the initial one")
    parser.add_argument("--pgn", default="game.pgn", help="where to save the game (default game.pgn)")
    parser.add_argument("--gif", help="also save an animated GIF of the game to this path")
    parser.add_argument("--png", help="also save a PNG of the final position to this path")
    parser.add_argument("--max-moves", type=int, help="stop after this many full moves")
    parser.add_argument("--quiet", action="store_true", help="don't print the board in bot-only games")
    args = parser.parse_args()

    white = make_player(args.white, args.depth, args.white_weights, args.temperature, args.seed)
    black = make_player(args.black, args.depth, args.black_weights, args.temperature,
                        None if args.seed is None else args.seed + 1)
    game = play_game(white, black, fen=args.fen, verbose=not args.quiet,
                     pgn_path=args.pgn, max_moves=args.max_moves)

    if args.gif or args.png:
        from export import game_gif, game_png
        board = game.end().board()   # final position, with the full move stack
        # Show the board from the human's side if Black is the only human
        flipped = isinstance(black, HumanPlayer) and not isinstance(white, HumanPlayer)
        names = dict(white=str(white), black=str(black), flipped=flipped)
        if args.gif:
            game_gif(board, args.gif, **names)
            print(f"GIF saved to {args.gif}")
        if args.png:
            game_png(board, args.png, **names)
            print(f"PNG saved to {args.png}")


if __name__ == "__main__":
    main()
