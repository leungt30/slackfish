"""Load a trained NoFeat model and score chess.Board positions."""
import json
import os
import threading

import chess
import numpy as np
import torch

from NoFeat.encode import encode_boards
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


# CUDA graph batch sizes: a batch is padded to the next one (larger ones run in pieces)
GRAPH_SIZES = (8, 16, 32, 64, 128, 256, 512)


class NoFeatEvaluator:
    """Same interface as Engine/bots.Evaluator: centipawns from White's perspective.

    On a GPU the model runs as CUDA graphs: recorded once per batch size in GRAPH_SIZES
    and replayed with one launch, instead of PyTorch launching each of its few hundred
    kernels from Python. The search's batches are small (median 5 positions), where
    those launches, not the GPU, set the time: ~2.9 ms a call eager, 1.2 ms replayed at
    batch 8, with identical outputs. SLACKFISH_CUDA_GRAPHS=0 turns it off."""

    def __init__(self, path: str, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        config, state = load_model(path, self.device)
        cls = getattr(models, config.get("model_class", "ChessTransformerV1"))   # older checkpoints are V1
        self.model = cls(**config["model_config"]).to(self.device)
        self.model.load_state_dict(state)
        # float16 weights on GPU, not autocast: autocast turns off nn.TransformerEncoder's
        # fused inference path, so this is ~1.3-1.45x faster, and closer to float32
        # (0.11 cp mean error vs 0.17). bfloat16 is as fast but ~10x less accurate.
        if self.device.type == "cuda":
            self.model.half()
        self.model.eval()
        self.scale = config["target_scale"]
        self.graphs = {}
        self.lock = threading.Lock()   # the graphs share input/output buffers (pondering is a thread)
        if self.device.type == "cuda" and os.environ.get("SLACKFISH_CUDA_GRAPHS", "1") != "0":
            self._record_graphs()

    @torch.no_grad()
    def _record_graphs(self):
        """One graph per batch size, largest first, all sharing one memory pool."""
        pool = None
        for size in sorted(GRAPH_SIZES, reverse=True):
            inputs = (torch.zeros((size, 64), dtype=torch.long, device=self.device),
                      torch.zeros(size, dtype=torch.long, device=self.device),
                      torch.zeros(size, dtype=torch.long, device=self.device))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):   # warm-up runs, as CUDA graph capture requires
                for _ in range(2):
                    self.model(*inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
                output = self.model(*inputs)
            pool = graph.pool()
            self.graphs[size] = (graph, inputs, output)

    @torch.no_grad()
    def evaluate(self, boards: list[chess.Board]) -> np.ndarray:
        squares, castling, turn = encode_boards(boards)
        if not self.graphs:
            pred = self.model(torch.from_numpy(squares).to(self.device).long(),
                              torch.from_numpy(castling).to(self.device).long(),
                              torch.from_numpy(turn).to(self.device).long())
            return pred.float().cpu().numpy() * self.scale
        out = np.empty(len(boards), dtype=np.float32)
        largest = GRAPH_SIZES[-1]
        with self.lock:
            for start in range(0, len(boards), largest):
                end = min(start + largest, len(boards))
                n = end - start
                graph, (sq, ca, tu), output = self.graphs[next(s for s in GRAPH_SIZES if s >= n)]
                # rows past n keep earlier inputs: each row is scored on its own, so they're ignored
                sq[:n].copy_(torch.from_numpy(squares[start:end]))
                ca[:n].copy_(torch.from_numpy(castling[start:end]))
                tu[:n].copy_(torch.from_numpy(turn[start:end]))
                graph.replay()
                out[start:end] = output[:n].float().cpu().numpy()
        return out * self.scale
