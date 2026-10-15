"""The gateway's job queue (PRD: Priority and scheduling).

Every job has a latency target from the moment it reached the gateway (Critical 8 s, High 12 s, Normal 16 s, Low
20 s; motion onset tightens it by a quarter). Jobs run earliest due first, so a waiting Low job is not starved by a
stream of fresh High ones, and jobs that arrive together still run in class order. Before each pick the queue is
simulated with a learned cost model (Clef time grows with tokens, tokens with frames); if a job would miss its
target, the lowest-priority job that can still give up frames is lowered a step (4, 2, then 1 fps; High not below
2; Critical not below 2, Clef's training rate, and only to keep another Critical job on time). Person detection on
this site was as good at 1 fps as at 4 (M0); fast action has not been tested.

Critical jobs are never dropped. Normal, Low and Maintenance keep only the newest window per (camera, group). Jobs
past their class's maximum age are dropped and counted. Maintenance jobs have no target: they run on spare time, and
those that wait 45 minutes are promoted to Normal, so each camera still gets an answer within the hour.
"""

from __future__ import annotations

import itertools
import threading
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from .motion import MotionRank


class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    MAINTENANCE = 4


MAX_AGE_S: dict[Priority, float | None] = {
    Priority.CRITICAL: None,
    Priority.HIGH: 60.0,
    Priority.NORMAL: 30.0,
    Priority.LOW: 30.0,
    Priority.MAINTENANCE: 3600.0,
}
TARGET_S: dict[Priority, float | None] = {
    Priority.CRITICAL: 8.0,
    Priority.HIGH: 12.0,
    Priority.NORMAL: 16.0,
    Priority.LOW: 20.0,
    Priority.MAINTENANCE: None,
}
MIN_FPS: dict[Priority, float | None] = {  # how far a job's frame rate may be lowered to keep the queue on time
    Priority.CRITICAL: 2.0,  # Clef's training rate; only ever lowered when another Critical job would be late
    Priority.HIGH: 2.0,
    Priority.NORMAL: 1.0,
    Priority.LOW: 1.0,
    Priority.MAINTENANCE: 1.0,
}
ONSET_TIGHTEN = 0.75
MERGE = {Priority.NORMAL, Priority.LOW, Priority.MAINTENANCE}
PROMOTE_AFTER_S = 2700.0  # before the 1 h Maintenance max age, so a waiting check runs instead of expiring
FPS_CAP: dict[Priority, float] = {Priority.LOW: 2.0, Priority.MAINTENANCE: 2.0}


class CostModel:
    """Clef seconds for a job: a fixed part plus seconds per token, learned from answered requests. Tokens are
    estimated from frames (about 78 a frame at the gateway's frame size) or images (about 120 a tile)."""

    TOKENS_PER_FRAME = 78.0
    TOKENS_PER_IMAGE = 120.0

    def __init__(self, base_s: float = 0.25, per_token_s: float = 0.00046) -> None:
        self.base_s, self.per_token_s = base_s, per_token_s

    def tokens(self, job: "Job", fps: float | None = None) -> float:
        if job.images:
            return 180.0 + self.TOKENS_PER_IMAGE * job.images
        return 200.0 + self.TOKENS_PER_FRAME * (fps if fps is not None else job.fps) * job.seconds

    def seconds(self, job: "Job", fps: float | None = None) -> float:
        return self.base_s + self.per_token_s * self.tokens(job, fps)

    def learn(self, tokens: int, seconds: float) -> None:
        if tokens > 500:
            rate = max(0.0, seconds - self.base_s) / tokens
            self.per_token_s += 0.1 * (rate - self.per_token_s)


def lower_fps(fps: float, floor: float) -> float:
    """One step down: above 2 fps to 2, then to 1."""
    return max(floor, 2.0 if fps > 2.0 else 1.0)


@dataclass
class Job:
    camera_id: str
    group: str  # e.g. "fps=8|zone=back-door"; jobs in one group share a Clef request
    priority: Priority
    window_end: float  # epoch seconds
    motion: MotionRank = MotionRank.ACTIVE
    fps: float = 2.0
    payload: Any = None
    queued_at: float = 0.0
    seq: int = field(default=0, compare=False)
    max_age: float | None = None  # set from the class the job was queued in
    promoted: bool = False  # promoted jobs go first in their new class and never expire
    drop_reason: str = ""  # "merged" or "expired" once dropped
    arrived: float | None = None  # when the request reached the gateway (latency counts from here); default: queued
    seconds: float = 30.0  # video length, for the cost estimate
    images: int = 0  # image count for image jobs (no video)
    requested_fps: float = 0.0  # the frame rate asked for, before any lowering

    def deadline(self) -> float:
        return float("inf") if self.max_age is None else self.window_end + self.max_age

    def due(self) -> float:
        """When this job should be answered: arrival plus its class's target (promoted jobs: overdue already)."""
        arrived = self.queued_at if self.arrived is None else self.arrived
        if self.promoted:
            return arrived
        target = TARGET_S[self.priority]
        if target is None:
            return float("inf")
        return arrived + target * (ONSET_TIGHTEN if self.motion == MotionRank.ONSET else 1.0)

    def key(self) -> tuple:
        return (self.due(), int(self.priority), int(self.motion), self.deadline(), self.seq)


class Scheduler:
    def __init__(self) -> None:
        self._jobs: list[Job] = []
        self._lock = threading.Lock()
        self._seq = itertools.count()
        self.skipped: Counter[str] = Counter()  # "<priority>:<reason>" -> count
        self.done: Counter[str] = Counter()
        self.lowered: Counter[str] = Counter()  # "<priority>" -> jobs sent at a lower frame rate to stay on time
        self.cost = CostModel()
        self._dropped: list[Job] = []

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def add(self, job: Job, now: float) -> None:
        job.queued_at, job.seq, job.max_age = now, next(self._seq), MAX_AGE_S[job.priority]
        cap = FPS_CAP.get(job.priority)
        if cap is not None:
            job.fps = min(job.fps, cap)
        job.requested_fps = job.requested_fps or job.fps
        with self._lock:
            if job.priority in MERGE:
                for old in [j for j in self._jobs if j.camera_id == job.camera_id and j.group == job.group]:
                    if old.window_end <= job.window_end:
                        self._jobs.remove(old)
                        old.drop_reason = "merged"
                        self._dropped.append(old)
                        self.skipped[f"{old.priority.name}:merged"] += 1
            self._jobs.append(job)

    def next(self, now: float) -> Job | None:
        with self._lock:
            for job in self._jobs:
                if job.priority == Priority.MAINTENANCE and now - job.queued_at > PROMOTE_AFTER_S:
                    job.priority, job.promoted, job.max_age = Priority.NORMAL, True, None
            expired = [j for j in self._jobs if now > j.deadline()]
            for job in expired:
                self._jobs.remove(job)
                job.drop_reason = "expired"
                self._dropped.append(job)
                self.skipped[f"{job.priority.name}:expired"] += 1
            if not self._jobs:
                return None
            order = sorted(self._jobs, key=Job.key)
            self._keep_on_time(order, now)
            best = order[0]
            self._jobs.remove(best)
            self.done[best.priority.name] += 1
            return best

    def _keep_on_time(self, order: list[Job], now: float) -> None:
        """Lower frame rates, lowest priority first, until the simulated queue meets every target it can."""
        hopeless: set[int] = set()  # late jobs nothing ahead of them can help: they run late, the rest still count
        for _ in range(4 * len(order)):
            t, late = now, None
            for i, job in enumerate(order):
                t += self.cost.seconds(job)
                if t > job.due() and i not in hopeless:
                    late = i
                    break
            if late is None:
                return
            candidates = [j for j in order[: late + 1] if not j.images and MIN_FPS[j.priority] is not None
                          and j.fps > MIN_FPS[j.priority]
                          and (j.priority != Priority.CRITICAL or order[late].priority == Priority.CRITICAL)]
            if not candidates:
                hopeless.add(late)
                continue
            victim = max(candidates, key=lambda j: (int(j.priority), self.cost.seconds(j)))
            if victim.fps == victim.requested_fps:
                self.lowered[victim.priority.name] += 1
            victim.fps = lower_fps(victim.fps, MIN_FPS[victim.priority])

    def take_dropped(self) -> list[Job]:
        """Jobs merged away or expired since the last call, so their callers can be told."""
        with self._lock:
            dropped, self._dropped = self._dropped, []
            return dropped

    def depth(self) -> dict[str, int]:
        with self._lock:
            return dict(Counter(j.priority.name for j in self._jobs))
