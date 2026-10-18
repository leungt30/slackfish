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
    uv run python Engine/match.py t3-sg64-muon-lichessgames-16.0m deepmind:270M --movetime 1

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
BATCH_SIZES = (8, 16, 32, 48, 64)   # legal moves are padded to the next size (one compile each)


def build(model, root, precision="fp32"):
    """An ActionValueEngine like DeepMind's _build_neural_engine, batched over legal moves.
    precision fp16 casts the weights to half precision (much faster on GPUs with fp16
    tensor cores, such as the RTX 20 series)."""
    sys.path.insert(0, os.path.dirname(root))   # imports as searchless_chess.src...
    import jax
    from jax import numpy as jnp, random as jrandom
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
    import haiku as hk
    import jmp
    run = lambda params, targets: predictor.predict(params=params, targets=targets, rng=None)
    predict32 = jax.jit(run)
    predict16 = jax.jit(lambda params, targets: run(params, targets))   # its own trace cache

    def forward(batch):
        if precision == "fp32":
            return np.asarray(predict32(params, batch))
        # Mixed precision: the linear layers (most of the compute) multiply in fp16, with their
        # outputs, the residual stream, layer norms and attention in fp32. A few moves still
        # overflow to NaN in fp16 (~1 in 200 positions has one); those rows are redone in fp32.
        with hk.mixed_precision.push_policy(hk.Linear, jmp.get_policy("params=float32,compute=float16,output=float32")):
            out = np.array(predict16(params, batch))
        bad = np.isnan(out).any(axis=(1, 2))
        if bad.any():
            out[bad] = np.asarray(predict32(params, batch))[bad]
        return out

    for size in BATCH_SIZES:   # compile every batch size (both precisions) before the clock starts
        dummy = np.zeros((size, tokenizer.SEQUENCE_LENGTH + 2), dtype=np.int32)
        predict32(params, dummy)
        if precision == "fp16":
            forward(dummy)

    def predict_fn(sequences):
        """Like DeepMind's wrap_predict_fn, but padding each batch to the next of BATCH_SIZES
        instead of always 64 (positions average ~25-35 legal moves)."""
        outputs = []
        for i in range(0, len(sequences), BATCH_SIZES[-1]):
            chunk = sequences[i:i + BATCH_SIZES[-1]]
            size = next(b for b in BATCH_SIZES if b >= len(chunk))
            outputs.append(forward(np.pad(chunk, ((0, size - len(chunk)), (0, 0))))[:len(chunk)])
        return np.concatenate(outputs)

    _, values = utils.get_uniform_buckets_edges_values(buckets)
    return neural_engines.ActionValueEngine(return_buckets_values=values, predict_fn=predict_fn)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="270M")
    p.add_argument("--dir", default=os.environ.get("SEARCHLESS_CHESS", DEFAULT_DIR),
                   help="the searchless_chess clone with checkpoints/ (default external/searchless_chess, or $SEARCHLESS_CHESS)")
    p.add_argument("--precision", choices=("fp32", "fp16"), default="fp16",
                   help="fp16: linear layers in half precision, NaN rows redone in fp32; the same moves as "
                        "fp32 on 400 test positions, 2.4x faster on the RTX 2080 Ti (default fp16)")
    args = p.parse_args()
    # Share the GPU with our bots and other workers: allocate only what's needed, ~1.2 GB for
    # the 270M at the same speed (JAX's default pool grew to ~4.3 GB, and three at once failed)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
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
                engine = build(args.model, os.path.abspath(args.dir), args.precision)
                engine.play(chess.Board())
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
