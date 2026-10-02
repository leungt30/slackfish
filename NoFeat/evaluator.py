"""Load a trained NoFeat checkpoint and score chess.Board positions."""
import chess
import numpy as np
import torch

from NoFeat.encode import encode_fens
from NoFeat import model as models


class NoFeatEvaluator:
    """Same interface as Engine/bots.Evaluator: centipawns from White's perspective."""

    def __init__(self, path: str, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cls = getattr(models, ckpt.get("model_class", "ChessTransformerV1"))   # older checkpoints are V1
        self.model = cls(**ckpt["model_config"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.scale = ckpt["target_scale"]

    @torch.no_grad()
    def evaluate(self, boards: list[chess.Board]) -> np.ndarray:
        squares, castling, turn = encode_fens([b.fen() for b in boards])
        # float16 on GPU: ~2.6x faster than float32, scores within ~1 cp on average
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            pred = self.model(torch.from_numpy(squares).to(self.device).long(),
                              torch.from_numpy(castling).to(self.device).long(),
                              torch.from_numpy(turn).to(self.device).long())
        return pred.float().cpu().numpy() * self.scale
