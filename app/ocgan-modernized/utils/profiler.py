from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch


def build_profiler(cfg, run_dir: str | Path):
    if not cfg.profiler.enabled:
        return None

    trace_dir = Path(run_dir) / "profiler"
    trace_dir.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if cfg.project.device == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(
        wait=int(cfg.profiler.wait),
        warmup=int(cfg.profiler.warmup),
        active=int(cfg.profiler.active),
        repeat=int(cfg.profiler.repeat),
    )

    profiler = torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=bool(cfg.profiler.record_shapes),
        profile_memory=bool(cfg.profiler.profile_memory),
        with_stack=bool(cfg.profiler.with_stack),
    )
    return profiler


def profiler_context(profiler):
    return profiler if profiler is not None else nullcontext()
