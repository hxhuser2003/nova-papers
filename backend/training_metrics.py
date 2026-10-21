"""Disjoint wall-time and PyTorch allocator peaks for training phases.

rollout_and_scoring includes weight sync, generation, reward and old-policy
logprob scoring. parameter_update includes forward/backward, optimizer and
step callbacks outside that phase. Peaks are PyTorch allocator peaks, NOT
whole-device peaks: vLLM allocations in another process are not included.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path


class PhaseMetrics:
    def __init__(self, path, cuda=None, clock=time.perf_counter):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cuda, self.clock, self.active = cuda, clock, None

    def start(self, phase, step):
        if self.active is not None:
            raise RuntimeError('Stop active phase before starting another')
        if self.cuda is not None:
            self.cuda.synchronize()
            self.cuda.reset_peak_memory_stats()
        self.active = (phase, step, self.clock(), time.time())

    def stop(self):
        if self.active is None:
            return
        if self.cuda is not None:
            self.cuda.synchronize()
        phase, step, started, wall_start = self.active
        row = dict(phase=phase, step=step, seconds=self.clock() - started,
                   wall_start=wall_start, wall_end=time.time(),
                   torch_peak_allocated_bytes=None, torch_peak_reserved_bytes=None)
        if self.cuda is not None:
            row.update(torch_peak_allocated_bytes=self.cuda.max_memory_allocated(),
                       torch_peak_reserved_bytes=self.cuda.max_memory_reserved())
        with self.path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        self.active = None

    @contextmanager
    def phase(self, phase, step):
        previous = self.active
        self.stop()
        self.start(phase, step)
        try:
            yield
        finally:
            self.stop()
            if previous is not None:
                self.start(previous[0], previous[1])
