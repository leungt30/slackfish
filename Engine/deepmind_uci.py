"""
DeepMind's searchless chess transformers (Ruoss et al. 2024, "Amortized Planning with
Large-Scale Transformers", github.com/google-deepmind/searchless_chess) as a UCI engine,
so Engine/match.py can benchmark our bots against them. Not one of our bots: it lives
outside Engine/bots.py and runs in its own environment (JAX + Haiku), set up by
scripts/setup-deepmind.sh.

Each move scores every legal move with one batched forward pass (the model predicts the
win probability after each move) and plays the best one, with no search. Moves that
allow a threefold repetition count as 0.5, as in DeepMind's ActionValueEngine.

    external/searchless_chess/.venv/bin/python Engine/deepmind_uci.py --model 270M
    uv run python Engine/match.py t2-sb-wsd-lichessgames-6.97m deepmind:270M --movetime 1

--model is 9M, 136M or 270M (the action-value models; the paper puts the 270M at a
Lichess blitz Elo of 2895 against humans).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(os.path.dirname(HERE), "external", "searchless_chess")
MODELS = {"9M": (8, 256), "136M": (8, 1024), "270M": (16, 1024)}   # layers, width; 8 heads each
STEP = 6_400_000   # the released checkpoints
BATCH = 64         # legal moves per forward pass (padded); a position has at most 218


def build(model, root):
    """An ActionValueEngine like DeepMind's _build_neural_engine, batched over legal moves."""
    sys.path.insert(0, os.path.dirname(root))   # imports as searchless_chess.src...
    from jax import random as jrandom
    import numpy as np
    from searchless_chess.src import tokenizer, training_utils, transformer, utils
    from searchless_chess.src.engines import neural_engines

    layers, width = MODELS[model]
    buckets = 128
    predictor = transformer.build_transformer_predictor(config=transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS, output_size=buckets,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
        num_heads=8, num_layers=layers, embedding_dim=width,
        apply_post_ln=True, apply_qk_layernorm=False, use_causal_mask=False))
    params = training_utils.load_parameters(
        checkpoint_dir=os.path.join(root, "checkpoints", model),
        params=predictor.initial_params(rng=jrandom.PRNGKey(1), targets=np.ones((1, 1), dtype=np.uint32)),
        step=STEP)
    _, values = utils.get_uniform_buckets_edges_values(buckets)
    return neural_engines.ActionValueEngine(
        return_buckets_values=values,
        predict_fn=neural_engines.wrap_predict_fn(predictor=predictor, params=params, batch_size=BATCH))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="270M")
    p.add_argument("--dir", default=os.environ.get("SEARCHLESS_CHESS", DEFAULT_DIR),
                   help="the searchless_chess clone with checkpoints/ (default external/searchless_chess, or $SEARCHLESS_CHESS)")
    args = p.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # share the GPU with our bots
    import chess

    engine, board = None, chess.Board()
    for line in sys.stdin:
        cmd = line.split()
        if not cmd:
            continue
        if cmd[0] == "uci":
            print(f"id name DeepMind searchless {args.model}\nid author Ruoss et al. (Google DeepMind)\nuciok")
        elif cmd[0] == "isready":
            if engine is None:
                engine = build(args.model, os.path.abspath(args.dir))
                engine.play(chess.Board())   # compile once, before the clock starts
            print("readyok")
        elif cmd[0] == "ucinewgame":
            board = chess.Board()
        elif cmd[0] == "position":
            moves = cmd.index("moves") if "moves" in cmd else len(cmd)
            board = chess.Board() if cmd[1] == "startpos" else chess.Board(" ".join(cmd[2:moves]))
            for uci in cmd[moves + 1:]:
                board.push_uci(uci)
        elif cmd[0] == "go":   # time limits are ignored: one forward pass per move
            print(f"bestmove {engine.play(board).uci()}")
        elif cmd[0] == "quit":
            break
        sys.stdout.flush()


if __name__ == "__main__":
    main()
