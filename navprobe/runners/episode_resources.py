"""Release per-episode Python and already-initialized CUDA resources."""
from __future__ import annotations

import gc
import traceback

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None  # type: ignore[assignment]


def _release_episode_memory(episode_id: str) -> None:
    # Collection is useful on CPU-only installs too. Never probe CUDA availability:
    # only an existing CUDA context may have allocator caches to release.
    try:
        collected = gc.collect()
        if torch is None:
            print(f"[cleanup] episode={episode_id} gc_collected={collected} cuda=torch-unavailable")
        elif torch.cuda.is_initialized():
            before = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
            torch.cuda.empty_cache()
            after = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
            print(
                f"[cleanup] episode={episode_id} gc_collected={collected} "
                f"cuda_allocated_mib={before[0] / 2**20:.1f}->{after[0] / 2**20:.1f} "
                f"cuda_reserved_mib={before[1] / 2**20:.1f}->{after[1] / 2**20:.1f}"
            )
        else:
            print(f"[cleanup] episode={episode_id} gc_collected={collected} cuda=uninitialized")
    except Exception:
        # A cleanup failure must not replace the episode's result or exception.
        print(f"[cleanup] episode={episode_id} memory cleanup failed:")
        traceback.print_exc()
