"""
services/gpu.py — GPU memory management.
"""
import gc
import torch

import app.state as state


def release_gpu_memory() -> None:
    """Giải phóng model inference khỏi GPU VRAM."""
    if state._current_infer_model is not None:
        del state._current_infer_model
        state._current_infer_model = None
    if state._current_infer_tokenizer is not None:
        del state._current_infer_tokenizer
        state._current_infer_tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    print("\n--- GPU memory released ---\n")
