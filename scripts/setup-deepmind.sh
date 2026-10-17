#!/usr/bin/env bash
# Set up DeepMind's searchless chess transformers as a benchmark opponent for
# Engine/match.py (deepmind:270M etc.; see Engine/deepmind_uci.py). Not one of our bots.
#
#   scripts/setup-deepmind.sh            # the 270M model (2.0 GB download)
#   scripts/setup-deepmind.sh 9M 136M    # others
#
# Clones github.com/google-deepmind/searchless_chess (Apache 2.0) into
# external/searchless_chess (gitignored; SEARCHLESS_CHESS overrides), gives it its own
# Python 3.10 venv with JAX (CUDA 12) and Haiku, and downloads the checkpoints from
# DeepMind's bucket. Our own .venv is untouched. Re-running skips what's already there.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=${SEARCHLESS_CHESS:-external/searchless_chess}
COMMIT=90ae0e6b121673fc3079aaeffa047580bb600c0a   # tested against this revision
UV=$(command -v uv || echo ~/.local/bin/uv)
MODELS=("${@:-270M}")

if [ ! -d "$DIR/.git" ]; then
    git clone -q https://github.com/google-deepmind/searchless_chess.git "$DIR"
fi
git -C "$DIR" checkout -q "$COMMIT"

if [ ! -x "$DIR/.venv/bin/python" ]; then
    "$UV" venv -q --python 3.10 "$DIR/.venv"
fi
# Only what loading and running the models needs (their requirements.txt also has
# training and notebook dependencies). constants.py imports apache_beam and grain.
"$UV" pip install -q --python "$DIR/.venv/bin/python" \
    "jax[cuda12]==0.4.38" "dm-haiku==0.0.13" orbax-checkpoint optax chex jaxtyping \
    grain apache-beam chess numpy scipy absl-py

mkdir -p "$DIR/checkpoints"
for m in "${MODELS[@]}"; do
    case $m in 9M|136M|270M) ;; *) echo "unknown model $m (9M, 136M or 270M)"; exit 1 ;; esac
    if [ ! -d "$DIR/checkpoints/$m" ]; then
        curl -fL --progress-bar -o "$DIR/checkpoints/$m.zip" "https://storage.googleapis.com/searchless_chess/checkpoints/$m.zip"
        (cd "$DIR/checkpoints" && unzip -q "$m.zip" && rm "$m.zip")
    fi
    echo "deepmind:$m ready"
done
