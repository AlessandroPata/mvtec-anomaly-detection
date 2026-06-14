"""Single-flight background job manager for arena batch runs."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field


class JobBusyError(RuntimeError):
    def __init__(self, current_id: str):
        super().__init__(f"A job is already running: {current_id}")
        self.current_id = current_id


@dataclass
class ArenaJob:
    id: str
    category: str
    variant: str
    images: list
    seed: int = 0
    status: str = "running"        # running | done | cancelled | error
    error: str | None = None
    results: list = field(default_factory=list)
    summary: dict | None = None
    cancel_requested: bool = False
    cond: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def add_result(self, r: dict) -> None:
        with self.cond:
            self.results.append(r)
            self.cond.notify_all()

    def finish(self, status: str, summary: dict | None = None, error: str | None = None) -> None:
        with self.cond:
            self.status = status
            self.summary = summary
            self.error = error
            self.cond.notify_all()

    def wait_results(self, cursor: int, timeout: float = 15.0):
        """Block until there are results past cursor or the job leaves 'running'.
        Returns (new_results, status, summary)."""
        with self.cond:
            if cursor >= len(self.results) and self.status == "running" and timeout > 0:
                self.cond.wait(timeout=timeout)
            return list(self.results[cursor:]), self.status, self.summary


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ArenaJob] = {}
        self._current: ArenaJob | None = None

    def start(self, category: str, variant: str, images: list, runner, seed: int = 0) -> ArenaJob:
        with self._lock:
            if self._current is not None and self._current.status == "running":
                raise JobBusyError(self._current.id)
            job = ArenaJob(uuid.uuid4().hex[:12], category, variant, images, seed=seed)
            self._jobs[job.id] = job
            self._current = job
        threading.Thread(target=runner, args=(job,), daemon=True,
                         name=f"arena-{job.id}").start()
        return job

    def get(self, job_id: str) -> ArenaJob | None:
        return self._jobs.get(job_id)

    @property
    def current(self) -> ArenaJob | None:
        return self._current

    def cancel(self, job_id: str) -> ArenaJob | None:
        job = self._jobs.get(job_id)
        if job is not None and job.status == "running":
            job.cancel_requested = True
        return job
