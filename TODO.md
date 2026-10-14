# TODO

## Spatial attention bias: next variants

**Done:** static per-head 66 x 66 bias (`--spatial-bias`, `MultiHeadSelfAttention` in
`NoFeat/model.py`), trained as T2-SB-W-Lichess-6.97M: full validation 104 vs 106 cp,
13-18 cp ahead through ~300K steps, then the plain model mostly caught up; 68/128
(+22 Elo) against T2-W-Lichess. Now the default bot. Kernel cost ~5% (memory-efficient
attention still runs with a float `attn_mask`). Results in the README Experiments section.

**Next, position-dependent bias (like Lc0's "smolgen"):** generate each head's bias from
the current position with a small network, so the geometry can depend on the pieces
(open vs. closed diagonals; the static table can only learn fixed patterns).

- Lc0's shape: compress each square's vector to 32 numbers, flatten (64 x 32 = 2,048),
  dense to a 256-number position summary, a 256 -> 256 layer per head, then one
  256 -> 64 x 64 output layer shared by all layers. At our sizes (width 256, 16 heads,
  8 layers) that is ~14M parameters, more than the whole 6.41M model: too big.
- Scaled down: 8 numbers per square (512), a 128-number summary, 64 per head, shared
  64 -> 66 x 66 output layer: ~1.9M parameters (~3x the static bias).
- Compute: ~4M multiply-adds per layer per position (~10-15% of a layer's feed-forward),
  and the bias becomes [B, H, 66, 66] per batch instead of one shared table. Expect
  roughly 15-25% slower training; benchmark first.
- Keep the static table too (smolgen output added to it), starting from zero output.
- Test: same full-data run as T2-SB, compare curves and a 128-game match against T2-SB.

**Also possible, shared relative bias:** one learned value per (rank offset, file offset)
pair (15 x 15 = 225 per head) instead of a full 66 x 66 table, plus entries for the
castling/turn tokens: ~20x fewer parameters, and each relationship ("a knight's move
away") is learned once for the whole board. Could show the head start sooner; compare
against T2-SB.

## 64 tokens instead of 66 (inference speed)

**Observation:** profiling inference on the RTX 500 laptop GPU (batch of 32, the size the
search sends), memory-efficient attention is 36% of GPU time, matrix multiplies 41%,
LayerNorm + GELU 18%. Its kernel works in 64 x 64 tiles, so the 66-token sequence (64
squares + castling + side to move) costs 2 x 2 tiles per head, nearly half of it padding.
Quantization doesn't help (fp8 / int8 via torchao: no faster at search batch sizes,
25-60x the error of float16), nor do other SDPA backends or `torch.compile`.

**Idea:** fold castling rights and the side to move into the square tokens instead of
giving them their own tokens: e.g. add a learned "side to move" vector to every square
embedding, and castling-rights vectors to the king and rook squares (or to all squares).
Then the sequence is exactly 64 and attention fits one tile; the spatial bias table
becomes 64 x 64 (rank/file geometry only).

- Needs retraining (new token layout); compare validation MAE with T2-SB-W-Lichess and
  measure positions/s in `Engine/bots.py` at batch 20-64. Expect a sizeable cut in
  attention time (it's the padding), so maybe 10-20% overall; benchmark before training.
- En passant is already encoded in the square tokens, so nothing else needs a token.

## Learning rate for bigger models

**Observation:** T3 (V3, 14.4M, width 384) trailed T2-W-Lichess-6.41M (width 256) at the
same step on the same data: about 15 cp worse validation MAE from 110K to 200K steps,
narrowing to 3 cp at 400K as the cosine schedule lowered the learning rate. Training loss
lagged too (0.600 vs 0.594 at 110K), so it is slower optimization, not overfitting.

**Suspected cause:** both used `--lr 3e-4`. Adam moves every weight by about the same
amount regardless of layer width, so wider layers' outputs change more per step: the
same learning rate is effectively larger for a wider model (the idea behind muP, which
scales the learning rate down with width).

**How to test:** two short T3 runs, identical except the learning rate, e.g.
`--arch v3 --limit 50000000 --epochs 1` with `--lr 3e-4` and `--lr 1.5e-4` (~3 h of GPU
time each on the 2080 Ti). If the lower learning rate leads early on, scale the learning rate down
with width for future large runs (roughly lr x 256 / d_model as a starting point).

## Other ideas

- Fewer heads at the current size: 6.41M model with `--heads 8` (head size 32) vs 16
  heads, same data. Trains 18% faster; does it lose accuracy?
- Bigger model on the full data: tried (T3-W-Lichess-14.4M), no gain; retry only with a
  width-scaled learning rate (above).
- Steeper win-probability curve (`--wp-scale` ~200) and true win/draw/loss labels.
- ~~Time management in search~~: done. A bot with a clock (GUI `--bot-clock`, the
  Lichess bridge) searches by time, see `Bot` in `Engine/bots.py`.
