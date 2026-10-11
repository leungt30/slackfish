"""Load a trained NoFeat model and score chess.Board positions."""
import json
import os

import chess
import numpy as np
import torch

from NoFeat.encode import encode_fens
from NoFeat import model as models


def load_model(path: str, device):
    """(model config dict, state dict) from an exported folder / model.safetensors
    (see NoFeat.export), or from a training checkpoint (.pt)."""
    if os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        with open(os.path.join(os.path.dirname(path), "config.json")) as f:
            config = json.load(f)
        return config, load_file(path, device=str(device))
    # A pickled training checkpoint: only load ones you trust (torch.load can run code)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt, ckpt["model_state_dict"]


class NoFeatEvaluator:
    """Same interface as Engine/bots.Evaluator: centipawns from White's perspective."""

    def __init__(self, path: str, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        config, state = load_model(path, self.device)
        cls = getattr(models, config.get("model_class", "ChessTransformerV1"))   # older checkpoints are V1
        self.model = cls(**config["model_config"]).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.scale = config["target_scale"]

    @torch.no_grad()
    def evaluate(self, boards: list[chess.Board]) -> np.ndarray:
        squares, castling, turn = encode_fens([b.fen() for b in boards])
        # float16 on GPU: ~2.6x faster than float32, scores within ~1 cp on average
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            pred = self.model(torch.from_numpy(squares).to(self.device).long(),
                              torch.from_numpy(castling).to(self.device).long(),
                              torch.from_numpy(turn).to(self.device).long())
        return pred.float().cpu().numpy() * self.scale
