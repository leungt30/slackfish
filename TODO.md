# TODO

**Baseline for the ablations below:** t2-sb-wsd-lichessgames-6.97m (the default until smolgen):
the t2-sb architecture on the Lichess evaluation database + September 2026 games (828M
positions), 4 GPUs, batch 2048, lr 1.2e-3, WSD to 1%, 4.2 h; val MAE 95.0 (lichess) /
94.6 cp (games). Ablations use its exact recipe (same data, seed and so the same validation
split), then a 256-game depth-2 match against it, plus an equal-time match when the change
affects speed. A ~3 minute run of each goes first, to check the whole pipeline.

## High priority: predict the next move as a secondary objective

Add a policy head that predicts the move to play, trained alongside the evaluation (a
weighted sum of the two losses), as Leela Chess Zero's networks do. Expected to help the
evaluation itself (move prediction forces the network to work out what each piece can
do) and to give the search a move ordering, or a way to look at fewer moves.

- Targets are in data we already have, but not in the prepared files: the Lichess
  evaluation database has Stockfish's best move (the first move of the best line) for
  every position, and the games have the move that was played (plus the next position's
  eval). `NoFeat.prepare_data` needs to store a move per position (a from-to index fits
  in 2 bytes, ~1.7 GB for 828M positions) and the data re-encoded (~1 h).
- Decide the move encoding (64 x 64 from-to plus underpromotions, or Lc0's 1858 moves)
  and how to mirror moves in the colour-swap augmentation.
- Test as an ablation of the default: same recipe with the policy loss, compare val MAE,
  move accuracy, and a 256-game match.

## Ablations of t2-sb-wsd-lichessgames (same data and recipe; 256-game matches against it)

- **Smolgen** (done, the new default `t2-sg-wsd-lichessgames-8.87m`): val 88.2 vs 94.75 cp,
  +96 Elo at depth 2, +67 at 0.5 s/move (+90 with the CUDA-graph evaluator), 23% slower to train.
- **8 heads instead of 16** (done, `t2-h8-wsd-lichessgames-6.69m`): val 95.0 cp, +27 Elo (−10
  to +65) at depth 2: a tie; 10% faster training, and 10-12% faster search with CUDA graphs.
- **64 tokens instead of 66** (done, `--arch v4`, not published): the model flips
  Black-to-move positions inside `forward()`, castling rights become rook tokens 14 / 15, no
  augmentation. Val 92.6 vs 94.75 cp; +26 Elo (−10 to +63) at depth 2, +51 (+14 to +89) at
  0.5 s/move with the CUDA-graph evaluator, where it searches 20-27% faster; 22% faster training.
- **Search speed with CUDA graphs** (done): smolgen −23% (from −36% eager), 8 heads +10-12%
  (from ±0), 64 tokens +20-27%. Smolgen's equal-time win rose to +90.

## Done: T3 with everything that helped (the new default)

`t3-sg64-muon-lichessgames-16.0m`: `--arch v4 --d-model 384 --heads 8 --d-ff 1536 --spatial-bias
--smolgen --optimizer muon`, the WSD recipe at lr 1.2e-3, full data, no augmentation; 15,985,153
parameters, 7.5 h. Val 83.4 cp (vs 88.2 for the smolgen default); against it, +117 Elo at depth
2 and +111 at 0.5 s/move (searches about as fast). Changes four things at once, so each part's
share is unknown; next candidates: next-move prediction on top of this model, and isolating Muon
(the default's architecture with Muon alone).

**Second epoch** (done, `SlackFishV-T3-SG64-Muon-LichessGames-2ep-16.0M`, not published): resumed
from the step-320k snapshot (before the decay) with `--epochs 2`. Val 80.7 vs 83.4 cp, but +8 Elo
(−27 to +43) at depth 2 and +15 (−19 to +49) at 0.5 s/move against the 1-epoch model: not clearly
stronger (a ~1,000-game match would tell). More data looks like the better use of 9 hours.

## Next: more data

**Done, the memory part:** shuffled folders (`NoFeat.shuffle_data`, README "Shuffled folders"):
training reads the data in chunks, ~1.6 GB per process at any size, the same speed. The data now
only needs disk. Next: encode more months of games (one month ~435M positions, ~16 GB), shuffle
them with the current data into one folder, and train T3 on it; compare with the 1-epoch T3 by
each model's own held-out score and by matches. Cross-month duplicates: `prepare_data`
deduplicates within one run, so encode the months together, or accept repeats between months
(a position recurs mostly in openings, which are dropped before ply 15). Encoding: its
deduplication peaked at 65 GB for one month; encode a month at a time if several don't fit.

## Deferred

- **Learning rate for T3 (width 384):** T3 with Muon at 1.2e-3 trained well (above); a lower or
  higher rate is untested.

## Spatial attention bias: next variants

**Done:** static per-head 66 x 66 bias (`--spatial-bias`, `MultiHeadSelfAttention` in
`NoFeat/model.py`), trained as T2-SB-W-Lichess-6.97M: full validation 104 vs 106 cp,
13-18 cp ahead through ~300K steps, then the plain model mostly caught up; 68/128
(+22 Elo) against T2-W-Lichess. Was the default until t2-sb-wsd-lichessgames. Kernel cost ~5% (memory-efficient
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

## Lichess bot fixes

- **Aborted games count toward --games.** An abort (opponent never moved) is logged as
  "aborted: drew" and counted in the series total and the score summary. Skip aborts in
  `BotRunner.play` / `results`, and don't count them toward `self.started`.
- **Tablebase underpromotion.** `tablebase_move` plays Lichess's first move, ranked by DTZ,
  which can be an underpromotion when promoting to a queen also wins (64.f8=R in
  lichess.org/Vag0zGhD). Prefer the queen promotion when it has the same category (win),
  like the search's queen-first rule.
- **Per-opponent cap.** `--challenge A,B,C --games 15` cycles in order, so a skipped turn
  shifts the cycle and one opponent gets 6 games, another 4. Add `--games-each N`.
- **Connection retries.** About 12 "POST ... failed (ConnectionError), retrying" in ~60 games
  on the desktop; all recovered within seconds. Watch for it costing time in bullet.

## Muon optimizer

**Done, first test:** `--optimizer muon` in `NoFeat/train.py` (`torch.optim.Muon`, 2D matrices
inside the layers; the rest on AdamW; `match_rms_adamw`, so the same `--lr`). ChessTransformerV1,
80M positions, WSD at lr 1.2e-3: val MAE 121.2 cp vs 150.3 with AdamW, ahead at every checkpoint,
7% slower per step. Next: in the T3 run above; then on the default's architecture alone, and a
higher learning rate for Muon. The notes below were the plan.

**Idea:** train the weight matrices with Muon instead of AdamW. Muon orthogonalizes each
matrix's momentum (a few Newton-Schulz iterations) before stepping, so every direction of
the update gets a similar step size; embeddings, biases and LayerNorm weights stay on AdamW.

**Why:** it set the small-transformer "speedrun" records, reaching AdamW's loss in roughly
1.3-2x fewer steps, and held up at larger scale. Our model is almost all 2-D matrices
(attention and feed-forward), and full-data runs take ~20 h on the 2080 Ti, so fewer steps
to the same accuracy is real time saved. It is also reported to need less learning-rate
retuning across widths (the T3 problem, above).

**Unknowns:** published results are mostly language models on clean data; our labels are
noisy Stockfish evaluations. Newton-Schulz usually runs in bf16, and we train in fp16 on
Turing (check it's stable, or run it in fp32: it's a few small matmuls per matrix per step).

**How to test:** add `--optimizer {adamw,muon}` to `NoFeat/train.py` (`torch.optim.Muon` if
our PyTorch has it, else a ~50-line reference implementation). Two runs on the same 50M
sample (`--limit 50000000 --epochs 1`), same model and schedule (WSD, `--schedule wsd`): compare the
learning curves, then a 128-game match if Muon gets there faster.

**Also worth a look:** schedule-free AdamW (no schedule: built-in weight averaging, an
alternative to WSD) and AdEMAMix (a second, slow momentum; helps long single-pass runs).

## Other ideas

- Fewer heads at the current size: 6.41M model with `--heads 8` (head size 32) vs 16
  heads, same data. Trains 18% faster; does it lose accuracy?
- Bigger model on the full data: tried (T3-W-Lichess-14.4M), no gain; retry only with a
  width-scaled learning rate (above).
- Steeper win-probability curve (`--wp-scale` ~200) and true win/draw/loss labels.
- ~~Time management in search~~: done. A bot with a clock (GUI `--bot-clock`, the
  Lichess bridge) searches by time, see `Bot` in `Engine/bots.py`.
